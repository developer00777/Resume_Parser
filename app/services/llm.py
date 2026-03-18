import asyncio
import json
import logging
import re

import httpx
from fastapi import HTTPException

from app.config import settings

logger = logging.getLogger(__name__)

# Reusable HTTP client — avoids connection setup overhead on every call
_ollama_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return a reusable async HTTP client for Ollama calls."""
    global _ollama_client
    if _ollama_client is None or _ollama_client.is_closed:
        _ollama_client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=30.0),
            base_url=settings.ollama_host,
        )
    return _ollama_client


# ---------------------------------------------------------------------------
# Prompts
#
# Design rules for llama3.2:3b (small model):
#   1. NO concrete example values — small models copy them verbatim.
#      Use <angle-bracket> placeholders in the output format line only.
#   2. Lead with WHAT to extract, then HOW to format, then hard rules.
#   3. State "If not found return []" / "return null" explicitly — the model
#      needs to be told the empty-case behaviour or it hallucinates.
#   4. One JSON key per prompt — keeps the output short and predictable.
#   5. Forbid fabrication explicitly ("ONLY from the text", "DO NOT invent").
# ---------------------------------------------------------------------------

PROMPT_CONTACT = """You are a resume parser. Extract the candidate's contact information from the resume text below.

Return ONLY valid JSON with these exact keys (use null for any field not found):
{"name": "<full name>", "email": "<email address>", "phone": "<primary phone>", "number": "<secondary phone or null>", "current_location": "<city and/or country>"}

Rules:
- name: the person's actual full name — NOT their job title, NOT a company name
- email: must contain @ symbol
- phone/number: digits only with optional + prefix — ignore salary figures like 12,00,000
- current_location: city/state/country — if "OPEN" or "Anywhere" is written, use null
- If a field is genuinely absent, use null
- DO NOT invent or guess any value

Resume text:
"""

PROMPT_SKILLS = """You are a resume parser. List every technical skill, tool, technology, software, and methodology explicitly mentioned in the resume text below.

Return ONLY valid JSON:
{"skills": ["<skill1>", "<skill2>", ...]}

Rules:
- Copy skill names exactly as written in the text
- Include hard skills AND soft skills if listed
- DO NOT add skills that are not in the text
- If no skills are found, return {"skills": []}

Resume text:
"""

PROMPT_EXPERIENCE = """You are a resume parser. Extract every work experience entry from the resume text below.

Return ONLY valid JSON:
{"experience": [{"company": "<company name>", "title": "<job title>", "duration": "<start date - end date>", "description": "<one sentence summary of responsibilities>"}]}

Rules:
- Extract ALL jobs/roles listed, not just the most recent
- company: the employer's name exactly as written
- title: the exact job title
- duration: dates as written (e.g. "Jan 2020 - Present")
- description: write ONE sentence summarising what they did — based ONLY on the text
- DO NOT invent companies, titles, or dates
- If no experience found, return {"experience": []}

Resume text:
"""

PROMPT_EDUCATION = """You are a resume parser. Extract formal academic education from the resume text below.

Return ONLY valid JSON:
{"education": [{"institution": "<college or university name>", "degree": "<degree title>", "field_of_study": "<subject or discipline>", "year": "<graduation year or null>", "grade": "<grade/GPA/CGPA or null>"}]}

Rules:
- Include: B.Tech, B.E., B.Sc, M.Tech, M.Sc, MBA, MCA, BCA, Ph.D, Diploma, Bachelor, Master, Post Graduate Diploma, and any other academic degree
- Look in ALL sections including Certifications — degrees are sometimes listed there
- DO NOT include professional course certifications (e.g. Primavera P6, Microsoft Project, Salesforce, AWS) as education entries
- If year or grade is not mentioned, use null
- If no formal education is found, return {"education": []}

Resume text:
"""

PROMPT_CERTIFICATIONS = """You are a resume parser. Extract all professional certifications and courses from the resume text below.

Return ONLY valid JSON:
{"certifications": [{"name": "<certification name>", "issuer": "<issuing organisation or null>"}]}

Rules:
- Include: professional courses, vendor certifications (Salesforce, AWS, Microsoft, Oracle), and any named certification
- DO NOT include academic degrees (B.Tech, MBA, etc.) here — those belong in education
- Copy the certification name exactly as written
- If issuer is not mentioned, use null
- If none found, return {"certifications": []}

Resume text:
"""

PROMPT_PROJECTS = """You are a resume parser. Extract all projects from the resume text below.

Return ONLY valid JSON:
{"projects": [{"name": "<project name>", "duration": "<dates or null>", "description": "<one sentence summary>"}]}

Rules:
- Extract the FULL project name — do not truncate it
- description: one sentence based ONLY on what is written
- If duration is not mentioned, use null
- DO NOT invent project names or details
- If no projects found, return {"projects": []}

Resume text:
"""

PROMPT_AWARDS = """You are a resume parser. Extract all awards, honours, achievements, and scholarships from the resume text below.

Return ONLY valid JSON:
{"awards": [{"name": "<award name>", "year": "<year or null>"}]}

Rules:
- Only include items explicitly listed as awards, achievements, or honours
- If year is not mentioned, use null
- DO NOT invent awards
- If none found, return {"awards": []}

Resume text:
"""

PROMPT_SUMMARY = """You are a resume writer. Write a concise 2-sentence professional summary for the candidate described in the resume text below.

Return ONLY valid JSON:
{"summary": "<2-sentence professional summary>"}

Rules:
- Base the summary ENTIRELY on facts in the text (role, years of experience, key skills, industry)
- Do NOT copy sentences verbatim from the text
- Do NOT mention the candidate's name
- Write in third-person (e.g. "Results-driven engineer with...")

Resume text:
"""

# (chunk_name, prompt_template, max_tokens, text_slice)
# text_slice=(start,end) uses a fixed character slice; None uses the section splitter
CHUNKS = [
    ("contact",        PROMPT_CONTACT,        120, None),
    ("skills",         PROMPT_SKILLS,         300, None),
    ("experience",     PROMPT_EXPERIENCE,     400, None),
    ("education",      PROMPT_EDUCATION,      200, None),
    ("certifications", PROMPT_CERTIFICATIONS, 200, None),
    ("projects",       PROMPT_PROJECTS,       350, None),
    ("awards",         PROMPT_AWARDS,         120, None),
    ("summary",        PROMPT_SUMMARY,        120, None),
]


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

# Header patterns — catches most resume styles (title case, upper case, with
# optional leading whitespace / bullet chars)
_HEADER_PATTERNS = {
    "skills":         r'(?:^|\n)[^\n]{0,10}(?:TECHNICAL\s+SKILLS?|SKILLS?|TECHNOLOGIES|TOOLS\s*&?\s*TECH(?:NOLOGIES)?|CORE\s+COMPETENCIES)[^\n]{0,20}(?:\n|$)',
    "experience":     r'(?:^|\n)[^\n]{0,10}(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT\s+HISTORY|CAREER\s+HISTORY|WORK\s+HISTORY)[^\n]{0,20}(?:\n|$)',
    "education":      r'(?:^|\n)[^\n]{0,10}(?:EDUCATION(?:AL)?\s*(?:BACKGROUND|QUALIFICATION)?|ACADEMIC\s+(?:BACKGROUND|QUALIFICATION|DETAILS)?|QUALIFICATION)[^\n]{0,20}(?:\n|$)',
    "projects":       r'(?:^|\n)[^\n]{0,10}(?:PROJECTS?(?:\s+EXPERIENCE)?|KEY\s+PROJECTS?)[^\n]{0,20}(?:\n|$)',
    "certifications": r'(?:^|\n)[^\n]{0,10}(?:CERTIFICATIONS?|CERTIFICATES?|PROFESSIONAL\s+CERTIFICATIONS?|LICENSES?\s*&?\s*CERTIFICATIONS?)[^\n]{0,20}(?:\n|$)',
    "awards":         r'(?:^|\n)[^\n]{0,10}(?:AWARDS?|HONOURS?|HONORS?|ACHIEVEMENTS?|SCHOLARSHIPS?|RECOGNITION)[^\n]{0,20}(?:\n|$)',
}

# How many chars to allow per section (generous — better to send more than truncate)
_SECTION_CHAR_LIMIT = {
    "skills":         2000,
    "experience":     3000,
    "education":      1500,
    "projects":       2000,
    "certifications": 1500,
    "awards":         1000,
    "contact":        1000,
    "summary":        2000,
}


def _split_resume_sections(text: str) -> dict:
    """
    Split resume text into named sections using header detection.

    Strategy:
    1. Find all section header positions using regex.
    2. Each section's content runs from its header to the next header.
    3. Apply per-section char limits (generous, to avoid truncation).
    4. Run a set of intelligent fallbacks for short/missing sections.
    5. Always build contact from _extract_contact_text (email/phone anchor).
    """
    positions = {}
    for section, pattern in _HEADER_PATTERNS.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            positions[section] = match.start()

    sorted_pos = sorted(positions.items(), key=lambda x: x[1])

    result = {}
    for i, (name, pos) in enumerate(sorted_pos):
        end = sorted_pos[i + 1][1] if i + 1 < len(sorted_pos) else len(text)
        limit = _SECTION_CHAR_LIMIT.get(name, 1500)
        result[name] = text[pos:end].strip()[:limit]

    # ── Fallback: no sections detected at all (plain text / no headers) ──────
    if not result:
        result["skills"]         = text[:2000]
        result["experience"]     = text[:3000]
        result["education"]      = text[:1500]
        result["certifications"] = text[:1500]
        result["projects"]       = text[:2000]
        result["awards"]         = text[:1000]

    # ── Fallback: skills section too short → grab from header to end of text ──
    if "skills" in result and len(result["skills"]) < 80:
        skills_pos = positions.get("skills", 0)
        result["skills"] = text[skills_pos:].strip()[:2000]

    # ── Fallback: experience too short → try merging with projects ────────────
    if result.get("experience") and len(result["experience"]) < 200:
        if result.get("projects"):
            result["experience"] = result["experience"] + "\n\n" + result["projects"]

    # ── Fallback: education too short → degrees are likely under certifications
    if result.get("education") and len(result["education"]) < 120:
        if result.get("certifications"):
            result["education"] = result["education"] + "\n\n" + result["certifications"]

    # ── Fallback: education section missing entirely → search full text ───────
    if "education" not in result:
        result["education"] = text

    # ── Fallback: certifications missing → use end of document ───────────────
    if "certifications" not in result:
        result["certifications"] = result.get("education", text[-2000:])

    # ── Contact: always use anchor-based extractor (not section splitter) ─────
    result["contact"] = _extract_contact_text(text)

    # ── Summary: use first ~2000 chars (intro + profile summary) ─────────────
    result["summary"] = text[:2000]

    return result


def _extract_contact_text(text: str) -> str:
    """
    Locate the contact block anywhere in the document.

    Strategy:
    1. Anchor on the first email address — most reliable signal.
    2. If no email, anchor on the first phone number.
    3. Expand ±500 chars around the anchor to capture the nearby name/location.
    4. Also include the very first 300 chars of the document (name is often at
       the very top in standard resumes, before the email).
    5. Fallback: return first 2000 chars.
    """
    email_match = re.search(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', text)
    # Phone: 7+ digit sequence, optionally with +, spaces, dashes, parens
    # Exclude salary-style numbers (12,00,000) by requiring no commas nearby
    phone_match = re.search(r'(?<![,\d])[\+]?[\d][\d\s\-\(\)\.]{7,}[\d](?![,\d]{3})', text)

    anchor = email_match.start() if email_match else (phone_match.start() if phone_match else None)

    if anchor is not None:
        start = max(0, anchor - 500)
        end   = min(len(text), anchor + 500)
        block = text[start:end]
        # Also prepend the first 300 chars in case name is at the very top
        header = text[:300]
        if header not in block:
            block = header + "\n" + block
        return block[:1000]

    return text[:2000]


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------

async def _call_ollama(prompt: str, max_tokens: int = 100) -> str:
    """Make a single Ollama generate call with JSON mode enabled."""
    client = _get_client()
    try:
        response = await client.post(
            "/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama service")
        raise HTTPException(status_code=503, detail="Ollama service is unavailable. Ensure it is running.")
    except httpx.TimeoutException:
        logger.error("Ollama chunk request timed out")
        raise HTTPException(status_code=504, detail="LLM processing timed out.")
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama returned error: {e.response.status_code}")
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {e.response.status_code}")

    return response.json().get("response", "")


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
    """Recursively convert string 'null' to actual None in parsed JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_nulls(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nulls(item) for item in obj]
    if isinstance(obj, str) and obj.strip().lower() in ("null", "none", "n/a", ""):
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
    skills     = parsed.get("skills", {}).get("skills", [])
    experience = parsed.get("experience", {}).get("experience", [])
    education  = parsed.get("education", {}).get("education", [])
    certs      = parsed.get("certifications", {}).get("certifications", [])
    projects   = parsed.get("projects", {}).get("projects", [])
    awards     = parsed.get("awards", {}).get("awards", [])
    summary    = parsed.get("summary", {}).get("summary", "") or ""

    remarks = []

    # ── 1. Contact Information (weight 5%) ───────────────────────────────────
    contact_score = 0
    if contact.get("name"):                             contact_score += 3
    if contact.get("email"):                            contact_score += 3
    if contact.get("phone") or contact.get("number"):  contact_score += 2
    if contact.get("current_location"):                 contact_score += 1
    contact_score = min(10, contact_score + (1 if contact_score >= 9 else 0))
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

    if exp_count > 0:
        if has_desc == exp_count: exp_score = min(10, exp_score + 1)
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
        if edu.get("year"):           edu_score += 1
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
        bool(contact.get("name")),
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
    """Call Ollama for a single chunk and return (name, extracted_data)."""
    try:
        logger.info(f"Chunk '{chunk_name}': sending {len(chunk_text)} chars, max_tok={max_tok}")
        raw = await _call_ollama(prompt + chunk_text, max_tokens=max_tok)
        data = _extract_json(raw)
        logger.info(f"Chunk '{chunk_name}' extracted {len(data)} fields")
        return chunk_name, data
    except Exception as e:
        logger.error(f"Chunk '{chunk_name}' failed: {e}")
        return chunk_name, {}


async def parse_resume(text: str) -> dict:
    """
    Parse resume using parallel LLM calls + Ollama JSON mode.

    All 8 chunk requests fire concurrently via asyncio.gather.
    Wall-clock time ≈ slowest single chunk, not the sum of all.
    """
    sections = _split_resume_sections(text)

    coroutines = []
    for chunk_name, prompt, max_tok, text_slice in CHUNKS:
        chunk_text = text[text_slice[0]:text_slice[1]] if text_slice else sections.get(chunk_name, text[:800])
        coroutines.append(_call_chunk(chunk_name, prompt, max_tok, chunk_text))

    results = await asyncio.gather(*coroutines)
    parsed = {name: data for name, data in results}

    score = _compute_score(parsed)

    contact      = parsed.get("contact", {})
    skills_data  = parsed.get("skills", {})
    exp_data     = parsed.get("experience", {})
    edu_data     = parsed.get("education", {})
    cert_data    = parsed.get("certifications", {})
    proj_data    = parsed.get("projects", {})
    award_data   = parsed.get("awards", {})
    summary_data = parsed.get("summary", {})

    return {
        "name":             contact.get("name"),
        "email":            contact.get("email"),
        "phone":            contact.get("phone"),
        "number":           contact.get("number"),
        "current_location": contact.get("current_location"),
        "skills":           skills_data.get("skills", []),
        "experience":       exp_data.get("experience", []),
        "education":        edu_data.get("education", []),
        "projects":         proj_data.get("projects", []),
        "certifications":   cert_data.get("certifications", []),
        "awards":           award_data.get("awards", []),
        "summary":          summary_data.get("summary"),
        "resume_score":     score,
    }


async def list_models() -> list[dict]:
    """List available models from Ollama."""
    client = _get_client()
    try:
        response = await client.get("/api/tags")
        response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Ollama service is unavailable.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama returned an error: {e.response.status_code}")

    return response.json().get("models", [])
