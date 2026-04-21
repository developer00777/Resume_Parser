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
You are an expert resume parser. Read the complete resume text below and extract \
contact information, personal/biographical details, and professional metadata. \
Return ALL three sections in a single JSON object.

OUTPUT FORMAT — return ONLY a single valid JSON object, no other text:
{
  "contact": {
    "first_name": "<given/first name only, or null>",
    "last_name": "<family/surname only, or null>",
    "full_name": "<complete name as written, or null>",
    "email": "<primary email — must contain @, or null>",
    "alternate_email": "<second email if present, else null>",
    "phone": "<primary phone number with country code if shown, or null>",
    "alternate_phone": "<second/alternate phone number, or null>",
    "current_location": "<city and state/country as written, or null>",
    "linkedin_url": "<full LinkedIn profile URL — must contain 'linkedin.com', or null>",
    "web_address": "<personal website, portfolio, or GitHub URL — NOT LinkedIn, or null>"
  },
  "personal": {
    "date_of_birth": "<YYYY-MM-DD, or null>",
    "gender": "<Male | Female | Other, or null>",
    "nationality": "<nationality or null>",
    "father_name": "<father's full name or null>",
    "mother_name": "<mother's full name or null>",
    "aadhar_number": "<12-digit Aadhar number or null>",
    "pan_number": "<10-char PAN number or null>",
    "passport_number": "<passport ID or null>",
    "blood_group": "<e.g. O+, A-, AB+, or null>",
    "languages_known": "<comma-separated human language list or null>",
    "marital_status": "<Single | Married | Divorced | Widowed or null>"
  },
  "professional_meta": {
    "current_ctc": "<current salary/CTC exactly as written e.g. '12 LPA', '₹15,00,000 p.a.', or null>",
    "expected_ctc": "<expected salary/CTC exactly as written, or null>",
    "notice_period": "<notice period exactly as written e.g. '30 days', '2 months', 'Immediate', or null>",
    "industry": "<primary industry or domain e.g. 'Information Technology', 'Finance', 'Healthcare', or null>",
    "preferred_location": "<preferred work location as written, or null>",
    "marital_status": "<Single | Married | Divorced | Widowed, or null>",
    "languages_known": "<comma-separated language list, or null>",
    "nationality": "<nationality, or null>",
    "blood_group": "<e.g. O+, A-, AB+, or null>",
    "gender": "<Male | Female | Other, or null>",
    "date_of_birth": "<YYYY-MM-DD, or null>"
  }
}

EXTRACTION RULES — contact:
- Scan the ENTIRE text; contact details may appear at the top, bottom, header, or sidebar.
- first_name / last_name: split the candidate's own name (not any employer's name).
- full_name: the candidate's complete name exactly as written (not a company name).
- email: look for the pattern word@domain.tld — take the first one as primary.
- phone: accept formats like +91-9876543210, (800) 555-1234, 9876543210. \
  Reject salary figures (e.g. 12,00,000) and years (e.g. 2019).
- current_location: labels to look for: "Location", "Address", "City", "Based in", \
  "Residing at", "Place". If the value is "Open to relocate" or "Anywhere", return null.
- linkedin_url: extract the full URL starting with http or linkedin.com/in/…
- web_address: GitHub, portfolio, personal site — NOT the LinkedIn URL.

EXTRACTION RULES — personal:
- Personal details often appear in a "Personal Details", "Personal Information", \
  "Bio-Data", or "Declaration" section, but may also appear in a sidebar or footer.
- date_of_birth: convert any date format to YYYY-MM-DD. Labels: "DOB", "Date of Birth", "Born on".
- gender: normalise exactly to "Male", "Female", or "Other".
- aadhar_number: 12 digits, may be written as "XXXX XXXX XXXX".
- pan_number: exactly 10 alphanumeric characters (e.g. ABCDE1234F).
- languages_known: Extract ONLY spoken/written human languages (e.g. English, Hindi, Tamil, French). \
  DO NOT extract programming/scripting languages (Java, Python, SQL, C++, etc.).
- marital_status: normalise to one of Single / Married / Divorced / Widowed.

EXTRACTION RULES — professional_meta:
- Scan the ENTIRE text including headers, footers, sidebars, and personal detail sections.
- current_ctc: look for "Current CTC", "CTC", "Current Salary", "Present Salary", "Package".
- expected_ctc: look for "Expected CTC", "Expected Salary", "Desired CTC", "Salary Expectation".
- notice_period: look for "Notice Period", "Availability", "Joining Time", "Can join in".
- industry: infer from job titles and companies if not explicitly stated.
- preferred_location: look for "Preferred Location", "Location Preference", "Open to relocation".

GENERAL RULES:
- DO NOT invent, guess, or infer any value. If a field is not in the text, return null.

Resume text (complete):
"""

PROMPT_CHUNK_B = """\
You are an expert resume parser. Read the complete resume text below and extract \
skills, certifications, awards, professional summary, and projects. \
Return ALL five sections in a single JSON object.

OUTPUT FORMAT — return ONLY a single valid JSON object, no other text:
{
  "skills": {
    "primary_skills": ["<top 5-10 core skills — most prominent in the resume>"],
    "technical_skills": ["<programming languages, frameworks, databases, tools, platforms, software>"],
    "general_skills": ["<soft skills, domain knowledge, methodologies, management skills>"],
    "all_skills": ["<EVERY skill found — union of the above three lists>"]
  },
  "certifications": [
    {
      "name": "<full certification name exactly as written>",
      "issuer": "<issuing organisation or null>"
    }
  ],
  "awards": [
    {
      "name": "<award or achievement name exactly as written>",
      "year": "<year as written e.g. 2022, or null>"
    }
  ],
  "summary": "<2-sentence professional summary>",
  "projects": [
    {
      "name": "<full project name exactly as written>",
      "duration": "<project duration or date range as written, or null>",
      "description": "<one concise sentence describing the project and technologies used>"
    }
  ]
}

EXTRACTION RULES — skills:
- Scan the ENTIRE text — skills appear in "Skills", "Technical Skills", "IT Skills", \
  "Core Competencies", "Areas of Expertise", "Technologies", "Tools", \
  but ALSO embedded in job descriptions, project descriptions, and summary paragraphs.
- primary_skills: top 5-10 most prominent skills based on: \
  (a) explicitly listed in a "Key Skills" or "Primary Skills" section, \
  (b) mentioned most frequently, or (c) highlighted in the summary/objective.
- technical_skills: all technologies — languages, frameworks, libraries, databases, \
  cloud platforms, DevOps tools, IDEs, testing tools, etc.
- general_skills: soft skills ("Leadership"), methodologies ("Agile", "Scrum"), \
  and domain expertise ("Project Management").
- all_skills: deduplicated union of primary + technical + general.
- Copy skill names exactly as written (preserve capitalisation like "JavaScript", "AWS").
- If no skills found, return empty arrays [].

EXTRACTION RULES — certifications:
- Scan the ENTIRE text — may appear under "Certifications", "Certificates", "Courses", "Training".
- Include vendor certs (AWS, Azure, GCP, Salesforce, Oracle, Microsoft, Cisco, PMP, etc.) \
  and named online courses (Coursera, Udemy, edX, NPTEL, etc.).
- DO NOT include academic degrees (B.Tech, MBA, etc.) — those belong in education.
- issuer: the organisation that issues the certificate. Use null if not stated.
- If none found, return [].

EXTRACTION RULES — awards:
- Scan the ENTIRE text — may appear under "Awards", "Honours", "Recognitions", \
  "Achievements", "Scholarships", "Accomplishments", or as bullet points in job sections.
- Include named awards, scholarships, prizes, "Employee of the Month/Year", patents.
- DO NOT include certifications (e.g. AWS, Salesforce) here.
- year: 4-digit integer as string. Use null if not stated.
- If none found, return [].

EXTRACTION RULES — summary:
- Task 1: If the resume contains an explicit professional summary, objective, or profile \
  statement, extract it verbatim (or lightly paraphrase to 2 sentences).
- Task 2: If no explicit summary exists, compose a 2-sentence summary based SOLELY \
  on the candidate's role, years of experience, and top skills found in the text.
- Do NOT mention the candidate's name. Write in third-person present tense.
- The summary must be 25-60 words.

EXTRACTION RULES — projects:
- Scan the ENTIRE text — may appear under "Projects", "Key Projects", "Personal Projects", \
  "Academic Projects", "Project Experience", or embedded in job descriptions.
- Extract EVERY distinct named project.
- description: one sentence based ONLY on what is written.
- If no projects found, return [].

GENERAL RULES:
- DO NOT invent values. If a field is absent, return null or [].

Resume text (complete):
"""

PROMPT_CHUNK_C = """\
You are an expert resume parser. Read the complete resume text below and extract \
ALL work experience entries AND all formal academic qualifications. \
Return both sections in a single JSON object.

OUTPUT FORMAT — return ONLY a single valid JSON object, no other text:
{
  "experience": {
    "experience": [
      {
        "company": "<employer/company name exactly as written>",
        "title": "<job title/designation exactly as written>",
        "duration": "<start date – end date as written, e.g. Jan 2020 – Present>",
        "description": "<one concise sentence summarising responsibilities and achievements>",
        "department": "<department or team name, or null>"
      }
    ],
    "total_years_of_experience": "<numeric decimal e.g. 5.5, or null>",
    "number_of_companies": "<integer count of distinct employers, or null>",
    "current_company": "<name of the current or most recent employer, or null>",
    "current_designation": "<current or most recent job title, or null>",
    "current_ctc": "<current salary/CTC exactly as written e.g. '12 LPA', '₹15,00,000', or null>",
    "expected_ctc": "<expected salary/CTC exactly as written, or null>",
    "notice_period": "<notice period exactly as written e.g. '30 days', '2 months', 'Immediate', or null>",
    "current_employment_status": "<Employed | Unemployed | Freelancer, or null>",
    "industry": "<primary industry/domain the candidate works in, or null>",
    "preferred_location": "<preferred work location as written, or null>"
  },
  "education": {
    "education": [
      {
        "institution": "<college, university, or school name exactly as written>",
        "degree": "<degree title e.g. B.Tech, M.Sc, MBA, Ph.D, 12th, 10th>",
        "field_of_study": "<subject/discipline/branch e.g. Computer Science, Finance>",
        "start_year": "<4-digit integer or null>",
        "end_year": "<4-digit integer or null>",
        "grade": "<CGPA, GPA, percentage, or division as written, or null>"
      }
    ],
    "highest_degree": "<highest academic qualification e.g. Ph.D | M.Tech | MBA | B.Tech | Diploma | 12th | 10th, or null>",
    "qualification_1": "<most recent/highest degree short form e.g. MBA, M.Tech>",
    "qualification_1_type": "<Post Graduation | Graduation | Diploma | 12th | 10th>",
    "institute_1": "<institution for qualification_1>",
    "qualification_2": "<second qualification short form e.g. B.Tech, B.Sc>",
    "qualification_2_type": "<Post Graduation | Graduation | Diploma | 12th | 10th>",
    "institute_2": "<institution for qualification_2>",
    "education_detail": "<one-line summary e.g. 'MBA from IIM Ahmedabad (2020), B.Tech from IIT Delhi (2018)'>"
  }
}

EXTRACTION RULES — experience:
- Scan the ENTIRE text — experience entries may appear under "Work Experience", \
  "Professional Experience", "Employment History", "Career History", "Work History", \
  or sometimes under "Projects" when the resume is project-based.
- Extract EVERY job/role listed — do not skip older or shorter ones.
- Order entries from MOST RECENT to OLDEST (place ongoing/Present roles first).
- company: employer's full name (not a product name or technology name).
- title: the exact designation/job title.
- duration: copy dates exactly as written. If "Present" or "Current" appears, keep it.
- description: write exactly ONE sentence summarising what the candidate did in that role, \
  based ONLY on what is stated in the text. Do not repeat the company name.
- total_years_of_experience: \
  RULE 1 (highest priority): scan the text for any phrase like "X years of experience", \
  "X+ years", "over X years", "X.X years of professional experience". If found, use \
  EXACTLY that number as a float (e.g. "5 years" → 5.0, "4.2 years" → 4.2). \
  RULE 2 (only if no explicit statement found): calculate from the earliest employer \
  start date to the latest end date (treat "till date"/"present"/"current" as 2024); \
  round to 1 decimal place. \
  DO NOT add up project durations or client engagement periods — use employer tenures only.
- number_of_companies: count ONLY companies where the candidate held a direct employment role \
  (with a Designation/Title). Do NOT count "Client" fields or project clients. \
  Example: Organization=Infosys, Client=Citibank → count only Infosys.
- current_employment_status: infer "Employed" if the latest role says "Present" or "Current"; \
  "Unemployed" if all roles have end dates in the past; "Freelancer" if stated.
- current_ctc / expected_ctc: look for "CTC", "Current CTC", "Salary", "Expected CTC", "Package".
- notice_period: look for "Notice Period", "Joining Time", "Available from".
- preferred_location: look for "Preferred Location", "Location Preference", "Open to".
- current_designation: use the title from the most recent experience entry. \
  If no experience entries have an explicit title, look at the FIRST LINE of the resume — \
  if it is 1–6 words in all-caps or title-case and is NOT a section header word \
  (not "Summary", "Skills", "Experience", "Education", "Profile", "Objective", \
  "Resume", "Curriculum", "Vitae", "CV"), treat it as the job title.
- If no experience found, return "experience": [].

EXTRACTION RULES — education:
- Scan the ENTIRE text — may appear under "Education", "Academic Background", \
  "Qualifications", "Academic Details", or degree info in a summary/objective.
- Include ALL formal academic degrees: B.E., B.Tech, B.Sc, B.Com, B.A., BCA, MCA, \
  M.Tech, M.Sc, M.Com, MBA, PGDM, Post Graduate Diploma, Ph.D, Diploma, 12th/HSC, 10th/SSC.
- DO NOT include professional certifications — those belong in certifications.
- start_year / end_year: 4-digit integers. If only one year shown, put it in end_year.
- grade: accept CGPA, GPA, percentage, or division (e.g. First Class, Distinction).
- highest_degree: the most advanced degree found — use the standard short form.
- qualification_1_type / qualification_2_type: one of \
  "Post Graduation", "Graduation", "Diploma", "12th", "10th". \
  (Ph.D, M.Tech, MBA, MCA → "Post Graduation"; B.Tech, B.Sc, BCA, B.Com → "Graduation"; \
   Diploma → "Diploma"; 12th/HSC → "12th"; 10th/SSC → "10th")
- education_detail: concise one-line string combining degree + institution (+ year) \
  for the top 2-3 qualifications.
- Order education entries from MOST RECENT to OLDEST.
- If no formal education found, return "education": [].

GENERAL RULES:
- DO NOT invent companies, titles, dates, salaries, degrees, or descriptions.

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
    ("chunk_a", PROMPT_CHUNK_A, 950),
    ("chunk_b", PROMPT_CHUNK_B, 2100),
    ("chunk_c", PROMPT_CHUNK_C, 2600),
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

    return {
        # Contact
        "first_name":         contact.get("first_name"),
        "last_name":          contact.get("last_name"),
        "name":               contact.get("full_name"),
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
        "languages_known":    personal.get("languages_known"),
        "marital_status":     personal.get("marital_status"),

        # Skills (categorized)
        "skills":             skills_data.get("all_skills", []),
        "primary_skills":     skills_data.get("primary_skills", []),
        "technical_skills":   skills_data.get("technical_skills", []),
        "general_skills":     skills_data.get("general_skills", []),

        # Experience + professional details
        "experience":                exp_data.get("experience", []),
        "total_years_of_experience": exp_data.get("total_years_of_experience"),
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
