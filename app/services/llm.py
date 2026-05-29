import asyncio
import json
import logging
import re

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

# Reusable HTTP client — avoids connection setup overhead on every call
_openrouter_client: httpx.AsyncClient | None = None

# Per-chunk timeout (seconds).
_CHUNK_TIMEOUT = 45.0


def _make_client() -> httpx.AsyncClient:
    """Create a new httpx AsyncClient for OpenRouter."""
    return httpx.AsyncClient(
        timeout=httpx.Timeout(_CHUNK_TIMEOUT, connect=10.0),
        base_url=settings.openrouter_base_url,
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://resumeparser-production-45b1.up.railway.app",
            "X-Title": "Resume Parser API",
        },
    )


def _get_client() -> httpx.AsyncClient:
    """
    Return a reusable async HTTP client for OpenRouter calls.

    The client is bound to the event loop that was running when it was created.
    To survive test suites (pytest-asyncio creates a new loop per test), we tag
    the client with the loop it was created in and recreate it whenever the
    running loop differs.
    """
    global _openrouter_client

    if _openrouter_client is None or _openrouter_client.is_closed:
        _openrouter_client = _make_client()
        try:
            _openrouter_client._bound_loop = asyncio.get_running_loop()
        except RuntimeError:
            _openrouter_client._bound_loop = None
        return _openrouter_client

    try:
        current_loop = asyncio.get_running_loop()
        bound_loop = getattr(_openrouter_client, '_bound_loop', None)
        if bound_loop is not None and bound_loop is not current_loop:
            # Loop changed — recreate the client for the new loop.
            _openrouter_client = _make_client()
            _openrouter_client._bound_loop = current_loop
    except RuntimeError:
        pass  # No running loop (sync context) — keep existing client.

    return _openrouter_client


# ---------------------------------------------------------------------------
# Text normaliser — runs BEFORE section splitting or LLM calls
#
# PDF extraction introduces many artefacts:
#   • Multiple consecutive blank lines (multi-column layouts)
#   • Mid-word hyphenation ("soft-\nware" → "software")
#   • Windows line-endings
#   • Non-breaking spaces / zero-width chars
#   • Garbage Unicode from font encoding issues
#
# Normalise to clean, single-spaced plain text so the LLM sees consistent
# input regardless of the original resume template or design.
# ---------------------------------------------------------------------------

def _normalise_text(raw: str) -> str:
    """
    Normalise raw PDF/DOCX text into clean, LLM-friendly plain text.

    Operations (in order):
    1. CRLF → LF
    2. Remove zero-width / non-printable chars
    3. Replace non-breaking spaces and other Unicode spaces with regular space
    4. Strip fullwidth/garbled chars from bad font encoding (ï¼, Â artifacts)
    5. Re-join soft-hyphenated line breaks (word-\n → word)
    6. Collapse "MM/YYYY\nto\nCurrent" date fragments onto one line
    7. Strip trailing whitespace from every line
    8. Collapse runs of 3+ blank lines to 2 blank lines
    9. Collapse interior runs of spaces (>2) to a single space
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Remove zero-width and other invisible Unicode control chars (keep \n \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\ufeff]', '', text)

    # Non-breaking space and related Unicode spaces → regular space
    for ch in ('\u00a0', '\u2007', '\u202f', '\u2009', '\u2008', '\u2006',
               '\u2005', '\u2004', '\u2003', '\u2002', '\u2001', '\u2000',
               '\u3000', '\u1680'):
        text = text.replace(ch, ' ')

    # Fullwidth chars (U+FF00–U+FFEF) that appear in bad font encodings.
    text = re.sub(r'[\uff00-\uffef]', '', text)

    # "ï¼" artifact: U+00EF + U+00BC (fraction char) is the Latin-1 mis-decode of
    # UTF-8 sequences like EF BC xx (fullwidth separators). Remove entirely.
    text = re.sub(r'ï[¼½¾⅓⅔⅛⅜⅝⅞][\u200b\u200c\u200d]*', '', text)

    # "Â" (U+00C2) appears when UTF-8 byte 0xC2 is decoded as Latin-1.
    # It appears as a spurious prefix character with no semantic content.
    text = re.sub(r'Â\s?', ' ', text)
    text = text.replace('Â', '')

    # Re-join soft-hyphenated line breaks: "soft-\nware" → "software"
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Collapse MM/YYYY\n[to]\nCurrent|Present|MM/YYYY → single-line date ranges
    # Catches: "03/2016\n\nto\n\nCurrent" → "03/2016 to Current"
    # And: "03/2016\nto\n04/2020" → "03/2016 to 04/2020"
    text = re.sub(
        r'(\d{1,2}/\d{4})\s*\n+\s*(to)\s*\n+\s*(\d{1,2}/\d{4}|Current|Present|Now)',
        r'\1 \2 \3',
        text, flags=re.IGNORECASE
    )
    # Also handle Month YYYY\nto\nCurrent
    text = re.sub(
        r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\s*\n+\s*(to)\s*\n+\s*(\d{1,2}/\d{4}|Current|Present|Now|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})',
        r'\1 \2 \3',
        text, flags=re.IGNORECASE
    )

    # Strip trailing spaces/tabs on each line
    lines = [l.rstrip() for l in text.split('\n')]
    text = '\n'.join(lines)

    # Collapse runs of 3+ blank lines → max 2 blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Collapse interior whitespace runs (not newlines) > 1 space → 1 space
    text = re.sub(r'[^\S\n]{2,}', ' ', text)

    return text.strip()


# ---------------------------------------------------------------------------
# Prompts
#
# Design rules:
#   1. Template-independent: never assume section headers exist.
#      The LLM receives the FULL resume text and must scan it completely.
#   2. Strict JSON-only output — no prose, no markdown fences.
#   3. Explicit null semantics: every field must be returned (null if absent).
#   4. Forbid fabrication: "ONLY extract what is explicitly written".
#   5. Enumerate common label variants so the LLM knows what to look for
#      even if the section header is missing or uses a non-standard name.
#   6. Handling ambiguity: when two values could match, prefer the one
#      closest to the resume header (top of document).
# ---------------------------------------------------------------------------

PROMPT_CHUNK_A = """\
You are a precise resume data extraction engine. Your ONLY job is to read the resume \
text below and output a single valid JSON object — no prose, no markdown, no explanation.

OUTPUT FORMAT — return ONLY this JSON structure, nothing else:
{
  "contact": {
    "first_name": "<GIVEN name only — e.g. 'Nitesh', 'Abhishek'. For 3-word names like \
'Nitesh Kumar Singh', first_name='Nitesh', last_name='Kumar Singh'. Never split the \
surname component. Null if not found.>",
    "last_name": "<FAMILY/SURNAME — everything after the first given name. For 'Nitesh Kumar Singh' \
→ 'Kumar Singh'. For 'Abhishek Suman' → 'Suman'. Null if not found.>",
    "full_name": "<candidate's COMPLETE name exactly as written on the resume, e.g. \
'Nitesh Kumar Singh', 'ABHISHEK SUMAN'. This is the official name — NOT a company or employer name.>",
    "email": "<primary email address — must contain @ and a dot. Take the FIRST one found. \
Null if absent. NEVER return a salary number or date as email.>",
    "alternate_email": "<second email address if a different one appears, else null>",
    "phone": "<PRIMARY mobile/phone number. Include country code if shown (+91, +1, etc.). \
Accept formats: +91-9876543210, 9876543210, (800) 555-1234. \
CRITICAL — REJECT: salary figures like 12,00,000 or 15,00,000; years like 2019, 2022; \
employee IDs; Aadhar numbers (12 digits). If unsure, return null.>",
    "alternate_phone": "<second phone/mobile number only if a DIFFERENT number appears. \
Apply the same rejection rules as 'phone'. Null if only one number found.>",
    "current_location": "<city and state/country exactly as written, e.g. 'Mumbai, Maharashtra', \
'Muzaffarpur, Bihar'. Labels to scan: Location, Address, City, Based in, Residing at, Place, \
Permanent Address. If value is 'Open to relocate', 'Anywhere', or 'PAN India', return null.>",
    "linkedin_url": "<full LinkedIn URL — must start with 'http' or 'linkedin.com/in/'. \
Null if not present.>",
    "web_address": "<personal website, portfolio, or GitHub URL. Must NOT be a LinkedIn URL. \
Null if not present.>"
  },
  "personal": {
    "date_of_birth": "<Convert ANY date format to YYYY-MM-DD. Labels: DOB, Date of Birth, \
Born on, D.O.B. Example: '25th Aug 1986' → '1986-08-25', '15/05/1991' → '1991-05-15'. \
Null if not found.>",
    "gender": "<Normalise EXACTLY to one of: Male | Female | Other. Labels: Sex, Gender. \
'M' → 'Male', 'F' → 'Female'. Null if not found.>",
    "nationality": "<as written, e.g. 'Indian', 'British', 'Malaysian'. Null if not found.>",
    "father_name": "<father's full name. Labels: Father's Name, Father Name, F/O, S/O (son of). \
Null if not found.>",
    "mother_name": "<mother's full name. Labels: Mother's Name, Mother Name, M/O. \
Null if not found.>",
    "aadhar_number": "<exactly 12 digits, may appear as 'XXXX XXXX XXXX'. Null if not found.>",
    "pan_number": "<exactly 10 alphanumeric chars, format like ABCDE1234F. Null if not found.>",
    "passport_number": "<passport document number. Null if not found.>",
    "blood_group": "<e.g. O+, A-, AB+, B+. Null if not found.>",
    "languages_known": "<comma-separated list of HUMAN SPOKEN/WRITTEN languages ONLY — \
e.g. 'English, Hindi, Tamil, French'. \
STRICTLY EXCLUDE programming/scripting languages: Java, Python, C++, SQL, JavaScript, \
R, Go, Swift, Kotlin, etc. These are technical skills, NOT spoken languages. \
Null if no human languages found.>",
    "marital_status": "<Normalise to EXACTLY one of: Single | Married | Divorced | Widowed. \
Null if not found.>"
  },
  "professional_meta": {
    "current_ctc": "<Current salary/CTC EXACTLY as written — do not convert or normalise. \
Examples: '12 LPA', '₹15,00,000 p.a.', '12,00,000-15,00,000 (Negotiable) PER ANNUM'. \
Labels: Current CTC, CTC, Current Salary, Present Salary, Salary, Package, Expected Salary. \
CRITICAL — salary figures look like 12,00,000 or 15,00,000; do NOT confuse with phone numbers. \
Null if not found.>",
    "expected_ctc": "<Expected salary EXACTLY as written. Labels: Expected CTC, Expected Salary, \
Desired CTC, Salary Expectation. Null if not found.>",
    "notice_period": "<exactly as written — e.g. '30 days', '2 months', 'Immediate', \
'1 month'. Labels: Notice Period, Availability, Joining Time, Can join in. Null if not found.>",
    "industry": "<primary industry/sector the candidate works in. Infer from job titles and \
company names if not explicitly stated. E.g. 'Oil & Gas', 'Information Technology', \
'Construction', 'Banking & Finance', 'Healthcare'. Null if genuinely unclear.>",
    "preferred_location": "<preferred work city/region as written. Labels: Preferred Location, \
Location Preference, Open to relocation. If value is 'OPEN' or 'Anywhere', return that value. \
Null if not stated.>",
    "marital_status": "<same as personal.marital_status — fill from either section>",
    "languages_known": "<same as personal.languages_known — human languages only>",
    "nationality": "<same as personal.nationality>",
    "blood_group": "<same as personal.blood_group>",
    "gender": "<same as personal.gender>",
    "date_of_birth": "<same as personal.date_of_birth — YYYY-MM-DD>"
  }
}

CRITICAL RULES — read before extracting:
1. Return ONLY the JSON object above. No text before or after it.
2. Every field must be present. Use null (not empty string, not "N/A") when absent.
3. DO NOT fabricate, guess, or infer values not explicitly in the text.
4. NAME SPLITTING: for Indian 3-word names (Given Middle Surname), first_name = first word, \
   last_name = remaining words. E.g. 'Nitesh Kumar Singh' → first='Nitesh', last='Kumar Singh'.
5. PHONE vs SALARY: phone numbers have 10 digits (India) or include a + prefix. \
   Salary figures like 12,00,000 or 15,00,000 are NOT phone numbers — never return them as phone.
6. LANGUAGES: only spoken human languages in languages_known. Never include Java, Python, SQL, etc.

Resume text (complete):
"""

PROMPT_CHUNK_B = """\
You are a precise resume data extraction engine. Your ONLY job is to read the resume \
text below and output a single valid JSON object — no prose, no markdown, no explanation.

OUTPUT FORMAT — return ONLY this JSON structure, nothing else:
{
  "skills": {
    "primary_skills": ["<5-10 core skills — use what is listed in 'Key Skills', 'Core Competencies', \
'Areas of Expertise', or 'Primary Skills' sections. If no explicit section, pick the 5-10 most \
frequently mentioned skills across the ENTIRE text. Preserve exact capitalisation: 'JavaScript', \
'AWS', 'Primavera P6'. Empty array [] if nothing found.>"],
    "technical_skills": ["<ALL technical items: programming languages, frameworks, libraries, \
databases, cloud platforms (AWS, Azure, GCP), DevOps/CI-CD tools, software (SAP, Primavera, \
AutoCAD, MS Project), testing tools, IDEs. Do NOT include soft skills here. \
Preserve exact capitalisation. [] if none found.>"],
    "general_skills": ["<ALL non-technical skills: soft skills (Leadership, Communication, \
Teamwork), methodologies (Agile, Scrum, Six Sigma, PMP), domain expertise (Project Management, \
HSE Management, Supply Chain). Do NOT include programming tools here. [] if none found.>"],
    "all_skills": ["<deduplicated UNION of the three arrays above — every unique skill \
from primary + technical + general. No duplicates. [] if none found.>"]
  },
  "certifications": [
    {
      "name": "<full certification name EXACTLY as written — e.g. \
'Post Graduate Diploma in Thermal Energy', \
'Professional course certification in Primavera P6 project management', \
'AWS Certified Solutions Architect — Associate'>",
      "issuer": "<issuing organisation EXACTLY as written — e.g. \
'JSW Energy Centre of Excellence', 'Synergy School of Business Skills, Chandigarh', \
'Amazon Web Services'. Null if not stated.>"
    }
  ],
  "awards": [
    {
      "name": "<award/achievement name EXACTLY as written>",
      "year": "<4-digit year as a string, e.g. '2022'. Null if not stated.>"
    }
  ],
  "summary": "<2-sentence professional summary (25-60 words). \
RULE 1: If the resume has an explicit 'Profile Summary', 'Career Objective', 'Professional Summary', \
or 'About Me' section, paraphrase it into 2 tight sentences. \
RULE 2: If no explicit summary exists, write 2 sentences using ONLY the candidate's actual \
role, years of experience, and top skills found in the text. \
Do NOT mention the candidate's name. Write in third-person present tense. \
Do NOT fabricate achievements or skills not in the text.>",
  "projects": [
    {
      "name": "<full project name EXACTLY as written>",
      "duration": "<date range or duration as written, e.g. 'Dec-2019 to Sept-2020'. Null if not stated.>",
      "description": "<one sentence describing the project based ONLY on what is written. \
Do NOT invent technologies or outcomes not mentioned in the text.>"
    }
  ]
}

CRITICAL RULES — read before extracting:
1. Return ONLY the JSON object above. No text before or after it.
2. Absent fields → null (strings/objects) or [] (arrays). Never use "" or "N/A".
3. DO NOT fabricate skills, certifications, or projects not present in the text.
4. SKILLS DEDUPLICATION: if a skill appears in technical_skills, do not repeat it in general_skills. \
   all_skills must be the union with no duplicates.
5. CERTIFICATIONS vs EDUCATION: academic degrees (B.Tech, MBA, B.A., 12th, Diploma) are NOT \
   certifications. Only include named professional/vendor/course certificates.
6. AWARDS vs CERTIFICATIONS: a certificate course is a certification, not an award. \
   Awards are prizes, recognition, 'Employee of the Year', scholarships, contest wins.
7. PROJECTS: include ONLY named, discrete projects. A job role description is NOT a project \
   unless it has a distinct project name.

Resume text (complete):
"""

PROMPT_CHUNK_C = """\
You are a precise resume data extraction engine. Your ONLY job is to read the resume \
text below and output a single valid JSON object — no prose, no markdown, no explanation.

OUTPUT FORMAT — return ONLY this JSON structure, nothing else:
{
  "experience": {
    "experience": [
      {
        "company": "<DIRECT EMPLOYER name exactly as written — the organisation that pays \
the candidate. NOT a client name, NOT a project client, NOT a site name. \
E.g. 'ISGEC HEAVY ENGINEERING LIMITED', 'Excellent Projects (I) Pvt. Ltd.', 'Infosys Ltd.'. \
If the resume shows 'Organization: TCS | Client: Citibank', company = 'TCS'.>",
        "title": "<job title/designation EXACTLY as written — e.g. 'Sr. Planning Engineer', \
'Scaffolding Supervisor', 'Software Engineer'. NOT a department or project name.>",
        "duration": "<date range EXACTLY as written — e.g. 'Mar-2022 to Present', \
'Jan 2020 – Dec 2022', 'Apr 2015 to Mar 2017'. Keep 'Present'/'Current' as-is. \
If only one date appears, use it as the end date. Null if no dates found.>",
        "description": "<ONE sentence summarising what the candidate actually DID in this role, \
based ONLY on bullet points or text written about this role. \
If no description is written (e.g. the resume only lists company + title + dates), \
write: 'Worked as [title] at [company] during [duration].' — do NOT invent responsibilities.>",
        "department": "<department or team name if explicitly stated, else null>"
      }
    ],
    "total_years_of_experience": "<RULE 1 — HIGHEST PRIORITY: scan for explicit phrases like \
'X years of experience', 'X+ years', 'over X years', 'X.X years of professional experience'. \
If found, use EXACTLY that number as a float. E.g. '9+ years' → 9.0, '4.2 years' → 4.2. \
RULE 2 — only if Rule 1 finds nothing: sum the EMPLOYER tenure (not project durations). \
Treat 'Present'/'Current'/'Till Date' as 2025. Round to 1 decimal. \
Null if no experience found.>",
    "number_of_companies": "<Count ONLY organisations where the candidate held a direct \
employment role (has a job title). DO NOT count client organisations, project sites, or \
deployment locations. Example: 'Excellent Projects (I) Pvt. Ltd.' with 8 client site \
deployments = 1 company. 'Infosys | Client: Citibank' = 1 company (Infosys). \
Null if no experience found.>",
    "current_company": "<The EMPLOYER from the most recent experience entry — the one with \
'Present'/'Current' or the latest end date. EXACTLY as written. Null if not found.>",
    "current_designation": "<Job title from the most recent experience entry. \
If no title in experience, look at the first 1-6 words of the resume in all-caps or \
title-case — if it is NOT a section header (not 'Summary', 'Skills', 'Experience', \
'Education', 'Profile', 'Objective', 'Resume', 'CV', 'Curriculum', 'Vitae'), treat it \
as the current designation. Null if not found.>",
    "current_ctc": "<current salary/CTC EXACTLY as written. Labels: Current CTC, CTC, \
Salary, Package, Expected Salary. E.g. '12,00,000-15,00,000 (Negotiable) PER ANNUM'. \
Null if not found.>",
    "expected_ctc": "<expected salary EXACTLY as written. Labels: Expected CTC, Expected Salary, \
Desired CTC. Null if not found.>",
    "notice_period": "<exactly as written. Labels: Notice Period, Availability, Joining Time, \
Can join in. E.g. '30 days', '2 months', 'Immediate'. Null if not found.>",
    "current_employment_status": "<EXACTLY one of: Employed | Unemployed | Freelancer. \
'Employed' if the latest role shows 'Present' or 'Current'. \
'Unemployed' if ALL roles have a past end date. \
'Freelancer' if the candidate explicitly states freelance/self-employed work. \
Null if cannot determine.>",
    "industry": "<primary industry/sector. Infer from job titles and company names if not \
explicitly stated. E.g. 'Oil & Gas', 'Construction', 'Information Technology', \
'Banking & Finance', 'Healthcare', 'Manufacturing'. Null if genuinely unclear.>",
    "preferred_location": "<preferred work location as written. Labels: Preferred Location, \
Location Preference, Open to relocation. If value is 'OPEN' or 'Anywhere', keep it as-is. \
Null if not stated.>"
  },
  "education": {
    "education": [
      {
        "institution": "<college, university, or school name EXACTLY as written>",
        "degree": "<degree title EXACTLY as written — e.g. 'B.Tech', 'M.Sc', 'MBA', \
'B.A', 'Post Graduate Diploma', 'Ph.D', '12th', '10th'>",
        "field_of_study": "<subject/discipline/branch — e.g. 'Mechanical Engineering', \
'Computer Science', 'Thermal Energy', 'Commerce'. Null if not stated.>",
        "start_year": "<4-digit integer start year. Null if not stated.>",
        "end_year": "<4-digit integer end/completion year. If only one year shown, \
put it here. Null if not stated.>",
        "grade": "<CGPA, GPA, percentage, or division EXACTLY as written — e.g. \
'8.5 CGPA', '75%', 'First Class', 'Distinction'. Null if not stated.>"
      }
    ],
    "highest_degree": "<MOST ADVANCED degree found — use standard short form: \
Ph.D | M.Tech | M.Sc | MBA | MCA | PGDM | Post Graduation | B.Tech | B.Sc | B.A | \
BCA | B.Com | Diploma | 12th | 10th. Null if no education found.>",
    "qualification_1": "<highest/most recent degree short form — e.g. 'MBA', 'M.Tech', \
'PG Diploma', 'B.Tech', 'B.A'. Null if no education.>",
    "qualification_1_type": "<EXACTLY one of: Post Graduation | Graduation | Diploma | 12th | 10th. \
Ph.D/M.Tech/MBA/MCA/PGDM/Post Graduate Diploma → 'Post Graduation'; \
B.Tech/B.Sc/BCA/B.Com/B.A/BE → 'Graduation'; \
Diploma/ITI → 'Diploma'; 12th/HSC/Intermediate → '12th'; 10th/SSC/Matriculation → '10th'.>",
    "institute_1": "<institution for qualification_1 EXACTLY as written>",
    "qualification_2": "<second qualification short form. Null if only one qualification.>",
    "qualification_2_type": "<type for qualification_2 — same enum as qualification_1_type. \
Null if only one qualification.>",
    "institute_2": "<institution for qualification_2. Null if only one qualification.>",
    "education_detail": "<concise one-line summary of top 2-3 qualifications — \
e.g. 'B.A from Jai Prakash University Chapra (2011), 12th from B.S.E. Board Patna (2008)'. \
Null if no education found.>"
  }
}

CRITICAL RULES — read before extracting:
1. Return ONLY the JSON object above. No text before or after it.
2. Absent fields → null (not "", not "N/A", not "Not mentioned"). Arrays → [].
3. DO NOT fabricate companies, titles, dates, salaries, degrees, or descriptions.
4. EMPLOYER vs CLIENT: Many resumes show 'Organization: X | Client: Y' or a table with \
   'Contractor | Period | Client Site'. The EMPLOYER is the organisation that employs the \
   candidate (X). The client/site is where they were deployed. Only count employers for \
   number_of_companies. Each distinct employer = 1 company regardless of client deployments.
5. EXPERIENCE ORDERING: most recent first. Ongoing roles ('Present'/'Current') come first.
6. EXPERIENCE DESCRIPTIONS: if the resume provides bullet points for a role, summarise \
   them in one sentence. If NO bullet points exist (just company + title + dates), write \
   'Worked as [title] at [company] ([duration]).' — do NOT invent duties.
7. EDUCATION: include ONLY formal academic degrees. Certificate courses and professional \
   certifications DO NOT belong here (they go in certifications in Chunk B).
8. QUALIFICATION TYPES: strictly use one of Post Graduation | Graduation | Diploma | 12th | 10th.

Resume text (complete):
"""


# ---------------------------------------------------------------------------
# Chunk definitions
#
# Each chunk receives the FULL normalised resume text.
# Template-independent — no section splitting needed for extraction quality.
#
# (chunk_name, prompt_template, max_tokens)
# ---------------------------------------------------------------------------
CHUNKS = [
    ("chunk_a", PROMPT_CHUNK_A, 1100),
    ("chunk_b", PROMPT_CHUNK_B, 2300),
    ("chunk_c", PROMPT_CHUNK_C, 2800),
]


# Maximum resume length to send to each chunk without truncation.
_FULL_TEXT_THRESHOLD = 8000


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------

async def _call_openrouter(prompt: str, max_tokens: int = 200) -> str:
    """Make a single OpenRouter chat-completion call."""
    client = _get_client()
    try:
        response = await client.post(
            "/chat/completions",
            json={
                "model": settings.openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
            },
        )
        response.raise_for_status()
    except httpx.ConnectError:
        logger.error("Cannot connect to OpenRouter")
        raise HTTPException(status_code=503, detail="OpenRouter service is unavailable.")
    except httpx.TimeoutException:
        logger.error("OpenRouter request timed out")
        raise HTTPException(status_code=504, detail="LLM processing timed out.")
    except httpx.HTTPStatusError as e:
        logger.error(f"OpenRouter returned error: {e.response.status_code} — {e.response.text[:300]}")
        raise HTTPException(status_code=502, detail=f"LLM service returned an error: {e.response.status_code}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def _clean_response(raw: str) -> str:
    """Strip thinking tags, code fences, and whitespace from LLM output."""
    cleaned = raw.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _sanitize_nulls(obj):
    """Recursively convert string 'null' / 'N/A' / '' to actual None in parsed JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nulls(item) for item in obj]
    if isinstance(obj, str) and obj.strip().lower() in ("null", "none", "n/a", "na", "not available", "not applicable", ""):
        return None
    return obj


def _extract_json(raw: str) -> dict:
    """Extract a JSON object from raw LLM text and sanitize null strings."""
    cleaned = _clean_response(raw)
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end > start:
            try:
                data = json.loads(cleaned[start:end])
            except json.JSONDecodeError:
                pass
    if data is None:
        logger.warning(f"Failed to parse chunk response: {cleaned[:200]}")
        return {}
    return _sanitize_nulls(data)


# ---------------------------------------------------------------------------
# Post-processing: merge professional_meta into experience data
# ---------------------------------------------------------------------------

def _merge_professional_meta(parsed: dict) -> dict:
    """
    Merge fields from professional_meta into experience data (and personal data).

    The professional_meta prompt extracts CTC/notice/industry from anywhere in the
    document. If the experience prompt missed these (because they were in the header
    or personal section), we fill them in from professional_meta.
    """
    meta = parsed.get("professional_meta", {})
    exp  = parsed.get("experience", {})

    # Fields to backfill into experience if missing there
    meta_to_exp = [
        "current_ctc", "expected_ctc", "notice_period",
        "industry", "preferred_location",
    ]
    for field in meta_to_exp:
        if not exp.get(field) and meta.get(field):
            exp[field] = meta[field]

    # Fields to backfill into personal if missing there
    personal = parsed.get("personal", {})
    meta_to_personal = [
        "marital_status", "languages_known", "nationality",
        "blood_group", "gender", "date_of_birth",
    ]
    for field in meta_to_personal:
        if not personal.get(field) and meta.get(field):
            personal[field] = meta[field]

    parsed["experience"] = exp
    parsed["personal"]   = personal
    return parsed


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_score(parsed: dict) -> dict:
    """
    Compute resume score using the 7-category weighted matrix.

    Score matrix (each category scored 0–10, then weighted):
    ┌─────────────────────────────┬────────┬──────────────────────────────────────────┐
    │ Category                    │ Weight │ Evaluation Criteria                      │
    ├─────────────────────────────┼────────┼──────────────────────────────────────────┤
    │ Contact Information         │  5%    │ Name, email, phone, LinkedIn, portfolio  │
    │ Professional Summary        │ 15%    │ Clear, concise, highlights strengths      │
    │ Work Experience             │ 25%    │ Structured, achievements, measurable KPIs │
    │ Skills                      │ 20%    │ Relevant, categorized, hard + soft skills │
    │ Education & Certifications  │ 10%    │ Complete, formatted, relevant certs       │
    │ Achievements / Projects     │ 15%    │ Projects, awards, quantifiable results    │
    │ Format & Design             │ 10%    │ Clean layout, readable, no typos          │
    └─────────────────────────────┴────────┴──────────────────────────────────────────┘
    """
    contact    = parsed.get("contact", {})
    skills     = parsed.get("skills", {}).get("all_skills", []) or parsed.get("skills", {}).get("skills", [])
    experience = parsed.get("experience", {}).get("experience", [])
    education  = parsed.get("education", {}).get("education", [])
    certs      = parsed.get("certifications", {}).get("certifications", [])
    projects   = parsed.get("projects", {}).get("projects", [])
    awards     = parsed.get("awards", {}).get("awards", [])
    summary    = parsed.get("summary", {}).get("summary", "") or ""

    remarks = []

    # ── 1. Contact Information (weight 5%) ───────────────────────────────────
    contact_score = 0
    has_name  = bool(contact.get("full_name") or contact.get("first_name") or contact.get("name"))
    has_email = bool(contact.get("email"))
    has_phone = bool(contact.get("phone") or contact.get("alternate_phone"))
    has_loc   = bool(contact.get("current_location"))
    has_li    = bool(contact.get("linkedin_url"))

    if has_name:  contact_score += 3
    if has_email: contact_score += 3
    if has_phone: contact_score += 2
    if has_loc:   contact_score += 1
    if has_li:    contact_score += 1
    contact_score = min(10, contact_score)

    if contact_score < 5:
        remarks.append("Contact information is incomplete — add name, email, and phone.")

    # ── 2. Professional Summary (weight 15%) ─────────────────────────────────
    summary_score = 0
    if summary:
        words = len(summary.split())
        if words >= 30:   summary_score = 10
        elif words >= 15: summary_score = 7
        elif words >= 5:  summary_score = 4
        else:             summary_score = 2
    else:
        remarks.append("No professional summary found — add a concise 2–3 sentence summary.")

    # ── 3. Work Experience (weight 25%) ──────────────────────────────────────
    exp_count = len(experience)
    has_desc  = sum(1 for e in experience if e.get("description") and len(e["description"]) > 20)

    if exp_count == 0:
        exp_score = 0
        remarks.append("No work experience found.")
    elif exp_count == 1:
        exp_score = 4
    elif exp_count == 2:
        exp_score = 6
    elif exp_count <= 4:
        exp_score = 8
    else:
        exp_score = 9

    if exp_count > 0 and has_desc == exp_count:
        exp_score = min(10, exp_score + 1)
    if exp_count > 0 and has_desc < exp_count:
        remarks.append("Add measurable impact and descriptions to work experience entries.")

    # ── 4. Skills (weight 20%) ────────────────────────────────────────────────
    skill_count = len(skills)
    if skill_count == 0:
        skills_score = 0
        remarks.append("No skills listed — add both technical and soft skills.")
    elif skill_count <= 3:
        skills_score = 2
        remarks.append("Very few skills listed — aim for at least 8–10 relevant skills.")
    elif skill_count <= 6:  skills_score = 4
    elif skill_count <= 10: skills_score = 6
    elif skill_count <= 15: skills_score = 8
    elif skill_count <= 20: skills_score = 9
    else:                   skills_score = 10

    # ── 5. Education & Certifications (weight 10%) ───────────────────────────
    edu_score = 0
    if education:
        edu = education[0]
        if edu.get("institution"):    edu_score += 2
        if edu.get("degree"):         edu_score += 2
        if edu.get("field_of_study"): edu_score += 1
        if edu.get("end_year"):       edu_score += 1
        if edu.get("grade"):          edu_score += 1
    else:
        remarks.append("No education details found.")

    cert_boost = min(3, len(certs))
    edu_score  = min(10, edu_score + cert_boost)
    if not education:
        edu_score = 0

    # ── 6. Achievements / Projects (weight 15%) ──────────────────────────────
    achieve_score = 0
    project_count = len(projects)
    award_count   = len(awards)

    if project_count == 0 and award_count == 0:
        achieve_score = 0
        remarks.append("No projects or achievements found — add notable projects or awards.")
    else:
        if project_count >= 3:   achieve_score += 6
        elif project_count == 2: achieve_score += 4
        elif project_count == 1: achieve_score += 2

        proj_with_desc = sum(1 for p in projects if p.get("description") and len(p["description"]) > 10)
        if proj_with_desc == project_count and project_count > 0:
            achieve_score = min(10, achieve_score + 1)

        if award_count >= 2:   achieve_score = min(10, achieve_score + 3)
        elif award_count == 1: achieve_score = min(10, achieve_score + 2)

    # ── 7. Format & Design (weight 10%) ──────────────────────────────────────
    populated_sections = sum([
        has_name,
        bool(summary),
        bool(experience),
        bool(skills),
        bool(education),
        bool(projects or awards),
        bool(certs),
    ])
    format_score = min(10, populated_sections + 3)

    # ── Weighted overall ──────────────────────────────────────────────────────
    overall = round(
        contact_score   * 0.05 * 10
        + summary_score * 0.15 * 10
        + exp_score     * 0.25 * 10
        + skills_score  * 0.20 * 10
        + edu_score     * 0.10 * 10
        + achieve_score * 0.15 * 10
        + format_score  * 0.10 * 10
    )
    overall = min(100, overall)

    if overall >= 90:
        grade = "Excellent"
        if not remarks: remarks.append("Professional-ready resume with strong coverage across all sections.")
    elif overall >= 75:
        grade = "Good"
        if not remarks: remarks.append("Good resume — address the minor gaps above to reach Excellent.")
    elif overall >= 50:
        grade = "Average"
        if not remarks: remarks.append("Resume needs improvement in several areas — see remarks above.")
    else:
        grade = "Poor"
        if not remarks: remarks.append("Resume needs a major overhaul — many critical sections are missing.")

    return {
        "overall":                  overall,
        "contact_information":      contact_score,
        "professional_summary":     summary_score,
        "work_experience":          exp_score,
        "skills":                   skills_score,
        "education_certifications": edu_score,
        "achievements_projects":    achieve_score,
        "format_design":            format_score,
        "grade":                    grade,
        "remarks":                  " ".join(remarks),
    }


# ---------------------------------------------------------------------------
# Parallel parsing
# ---------------------------------------------------------------------------

async def _call_chunk(chunk_name: str, prompt: str, max_tok: int, chunk_text: str) -> tuple[str, dict]:
    """Call OpenRouter for a single chunk and return (name, extracted_data)."""
    try:
        logger.info(f"Chunk '{chunk_name}': sending {len(chunk_text)} chars, max_tok={max_tok}")
        raw = await _call_openrouter(prompt + chunk_text, max_tokens=max_tok)
        data = _extract_json(raw)
        logger.info(f"Chunk '{chunk_name}' extracted {len(data)} fields")
        return chunk_name, data
    except Exception as e:
        logger.error(f"Chunk '{chunk_name}' failed: {e}")
        return chunk_name, {}


async def parse_resume(text: str) -> dict:
    """
    Parse resume using 3 parallel LLM calls via OpenRouter.

    Architecture:
    1. Normalise raw text (remove encoding artefacts, collapse whitespace).
    2. Send full text (up to _FULL_TEXT_THRESHOLD) to all 3 chunks in parallel.
       - Chunk A: contact + personal + professional_meta (950 tokens)
       - Chunk B: skills + certifications + awards + summary + projects (2100 tokens)
       - Chunk C: experience + education (2600 tokens)
    3. Merge professional_meta into experience/personal to fill any gaps.
    4. Compute the 7-category score matrix.
    """
    text = _normalise_text(text)
    chunk_text = text[:_FULL_TEXT_THRESHOLD]

    coroutines = [
        _call_chunk(name, prompt, max_tok, chunk_text)
        for name, prompt, max_tok in CHUNKS
    ]

    try:
        results = await asyncio.wait_for(
            asyncio.gather(*coroutines),
            timeout=120.0,
        )
    except asyncio.TimeoutError:
        logger.error("parse_resume: overall 120-second timeout exceeded")
        raise HTTPException(status_code=504, detail="Resume parsing timed out. Try a smaller file.")

    # Unpack 3-chunk results into per-section dicts
    parsed: dict = {}
    for chunk_name, data in results:
        if chunk_name == "chunk_a":
            parsed["contact"]           = data.get("contact", {})
            parsed["personal"]          = data.get("personal", {})
            parsed["professional_meta"] = data.get("professional_meta", {})
        elif chunk_name == "chunk_b":
            parsed["skills"]        = data.get("skills", {})
            parsed["certifications"] = {"certifications": data.get("certifications", [])}
            parsed["awards"]        = {"awards": data.get("awards", [])}
            parsed["summary"]       = {"summary": data.get("summary")}
            parsed["projects"]      = {"projects": data.get("projects", [])}
        elif chunk_name == "chunk_c":
            parsed["experience"] = data.get("experience", {})
            parsed["education"]  = data.get("education", {})

    parsed = _merge_professional_meta(parsed)

    score = _compute_score(parsed)

    contact       = parsed.get("contact", {})
    personal      = parsed.get("personal", {})
    skills_data   = parsed.get("skills", {})
    exp_data      = parsed.get("experience", {})
    edu_data      = parsed.get("education", {})
    cert_data     = parsed.get("certifications", {})
    proj_data     = parsed.get("projects", {})
    award_data    = parsed.get("awards", {})
    summary_data  = parsed.get("summary", {})

    # Build spoken_languages as a list (split the comma string the LLM returns)
    languages_raw = personal.get("languages_known")
    if isinstance(languages_raw, list):
        spoken_languages = [l.strip() for l in languages_raw if l.strip()]
    elif isinstance(languages_raw, str) and languages_raw:
        import re as _re
        spoken_languages = [
            p.strip() for p in _re.split(r'[,;/]|\band\b', languages_raw, flags=_re.IGNORECASE)
            if p.strip()
        ]
    else:
        spoken_languages = []

    return {
        # Contact
        "first_name":         contact.get("first_name"),
        "last_name":          contact.get("last_name"),
        "name":               contact.get("full_name"),
        "full_name":          contact.get("full_name"),
        "email":              contact.get("email"),
        "alternate_email":    contact.get("alternate_email"),
        "phone":              contact.get("phone"),
        "number":             contact.get("alternate_phone"),
        "current_location":   contact.get("current_location"),
        "linkedin_url":       contact.get("linkedin_url"),
        "web_address":        contact.get("web_address"),

        # Personal
        "date_of_birth":      personal.get("date_of_birth"),
        "gender":             personal.get("gender"),
        "nationality":        personal.get("nationality"),
        "father_name":        personal.get("father_name"),
        "mother_name":        personal.get("mother_name"),
        "aadhar_number":      personal.get("aadhar_number"),
        "pan_number":         personal.get("pan_number"),
        "passport_number":    personal.get("passport_number"),
        "blood_group":        personal.get("blood_group"),
        "languages_known":    languages_raw if isinstance(languages_raw, str) else ", ".join(spoken_languages),
        "spoken_languages":   spoken_languages,
        "marital_status":     personal.get("marital_status"),

        # Skills (categorized)
        "skills":             skills_data.get("all_skills", []),
        "primary_skills":     skills_data.get("primary_skills", []),
        "technical_skills":   skills_data.get("technical_skills", []),
        "general_skills":     skills_data.get("general_skills", []),

        # Experience + professional details
        "experience":                exp_data.get("experience", []),
        "total_years_of_experience": exp_data.get("total_years_of_experience"),
        "years_of_experience":       exp_data.get("total_years_of_experience"),
        "number_of_companies":       exp_data.get("number_of_companies"),
        "current_company":           exp_data.get("current_company"),
        "current_designation":       exp_data.get("current_designation"),
        "current_ctc":               exp_data.get("current_ctc"),
        "expected_ctc":              exp_data.get("expected_ctc"),
        "notice_period":             exp_data.get("notice_period"),
        "current_employment_status": exp_data.get("current_employment_status"),
        "industry":                  exp_data.get("industry"),
        "preferred_location":        exp_data.get("preferred_location"),

        # Education
        "education":          edu_data.get("education", []),
        "highest_degree":     edu_data.get("highest_degree"),
        "qualification_1":    edu_data.get("qualification_1"),
        "qualification_1_type": edu_data.get("qualification_1_type"),
        "institute_1":        edu_data.get("institute_1"),
        "qualification_2":    edu_data.get("qualification_2"),
        "qualification_2_type": edu_data.get("qualification_2_type"),
        "institute_2":        edu_data.get("institute_2"),
        "education_detail":   edu_data.get("education_detail"),

        # Other
        "projects":           proj_data.get("projects", []),
        "certifications":     cert_data.get("certifications", []),
        "awards":             award_data.get("awards", []),
        "summary":            summary_data.get("summary"),
        "resume_score":       score,
    }


async def check_openrouter() -> bool:
    """Check if OpenRouter is reachable and the API key is valid."""
    client = _get_client()
    try:
        resp = await client.get("/models")
        return resp.status_code == 200
    except Exception:
        return False
