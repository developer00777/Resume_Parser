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


# --- Prompts: focused, no placeholder text that LLM can copy ---

PROMPT_CONTACT = """From the text below, extract the person's name, email, phone number, secondary phone number, and location.
Reply ONLY with valid JSON. Use null (not the string "null") for any missing field.
{"name":"John Doe","email":"john@example.com","phone":"+1234567890","number":null,"current_location":"City, State"}

Text:
"""

PROMPT_SKILLS = """Read the text below carefully. List EVERY skill, tool, and technology that appears in it.
Do NOT make up skills. Only list what is explicitly written in the text.
Reply ONLY with JSON: {"skills":["first skill from text","second skill from text"]}

Text:
"""

PROMPT_EXPERIENCE = """Read the text below carefully. Extract each job/role with company name, title, dates, and a 1-sentence description of what they did.
Do NOT copy example text. Write the actual data from the text.
Reply ONLY with JSON: {"experience":[{"company":"actual company","title":"actual title","duration":"actual dates","description":"actual summary of work done"}]}

Text:
"""

PROMPT_EDUCATION = """Extract education details. Look carefully for the degree name, field/major of study, institution name, graduation year, and grade/GPA/CGPA.
Reply ONLY with valid JSON. Use null for genuinely missing fields.
{"education":[{"institution":"MIT","degree":"B.Tech","field_of_study":"Computer Science","year":"2020","grade":"8.5"}]}

Text:
"""

PROMPT_CERTIFICATIONS = """Extract all certifications from the text below.
Do NOT make up certifications. Only list what is in the text.
Reply ONLY with JSON: {"certifications":[{"name":"cert name from text","issuer":"issuing org from text"}]}

Text:
"""

PROMPT_PROJECTS = """Extract all projects from the text below with their name, duration, and a 1-sentence description.
Do NOT copy example text. Write based on actual project details.
Reply ONLY with JSON: {"projects":[{"name":"actual project name","duration":"actual dates","description":"actual 1-sentence summary"}]}

Text:
"""

PROMPT_AWARDS = """Extract all awards, scholarships, and achievements from the text below.
Do NOT make up awards. Only list what is in the text.
Reply ONLY with JSON: {"awards":[{"name":"actual award name","year":"actual year"}]}

Text:
"""

PROMPT_SUMMARY = """Write a 2-sentence professional summary based on the text below.
Do NOT copy example text. Write based on the actual person's details.
Reply ONLY with JSON: {"summary":"your written summary here"}

Text:
"""

# (name, prompt_template, max_tokens, text_slice)
# text_slice=(start,end) for fixed slice, None to use section splitter
CHUNKS = [
    ("contact", PROMPT_CONTACT, 80, (0, 500)),
    ("skills", PROMPT_SKILLS, 200, None),
    ("experience", PROMPT_EXPERIENCE, 200, None),
    ("education", PROMPT_EDUCATION, 120, None),
    ("certifications", PROMPT_CERTIFICATIONS, 120, None),
    ("projects", PROMPT_PROJECTS, 250, None),
    ("awards", PROMPT_AWARDS, 80, None),
    ("summary", PROMPT_SUMMARY, 100, None),
]


def _split_resume_sections(text: str) -> dict:
    """Split resume text into sections. Uses uppercase headers for accurate detection."""
    result = {}

    header_patterns = {
        "skills": r'(?:^|\n)\s*(SKILLS|TECHNICAL\s+SKILLS|TECHNOLOGIES|TOOLS)',
        "experience": r'(?:^|\n)\s*(WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|CAREER)',
        "education": r'(?:^|\n)\s*(EDUCATION|ACADEMIC|QUALIFICATION)',
        "projects": r'(?:^|\n)\s*(PROJECTS|PROJECT\s+EXPERIENCE)',
        "certifications": r'(?:^|\n)\s*(CERTIFICATIONS?|CERTIFICATES?)',
        "awards": r'(?:^|\n)\s*(AWARDS?|SCHOLARSHIPS?|ACHIEVEMENTS?|HONORS?)',
    }

    positions = {}
    for section, pattern in header_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            positions[section] = match.start()

    sorted_pos = sorted(positions.items(), key=lambda x: x[1])

    for i, (name, pos) in enumerate(sorted_pos):
        end = sorted_pos[i + 1][1] if i + 1 < len(sorted_pos) else len(text)
        result[name] = text[pos:end].strip()[:1000]

    # Fallback: if we couldn't find sections, use the full text
    if not result:
        result["skills"] = text[:1500]
        result["experience"] = text[:1500]
        result["education"] = text[:1500]

    # Skills section might be at the very end — ensure we capture it fully
    if "skills" in result and len(result["skills"]) < 100:
        # Section was found but very short — grab to end of text
        skills_pos = positions.get("skills", 0)
        result["skills"] = text[skills_pos:].strip()[:1500]

    # Experience: if section is very short, the real work details might be under PROJECTS
    if "experience" in result and len(result["experience"]) < 200 and "projects" in result:
        result["experience"] = result["experience"] + "\n\n" + result["projects"]

    result["contact"] = text[:500]
    result["summary"] = text[:1500]

    return result


async def _call_ollama(prompt: str, max_tokens: int = 100) -> str:
    """
    Make a single Ollama generate call with JSON mode enabled.

    Ollama JSON mode (format="json") constrains token sampling so the model
    always produces syntactically valid JSON — no code fences, no prose,
    no truncated objects. This eliminates the need for regex cleanup and
    fallback parsing on every response.
    """
    client = _get_client()
    try:
        response = await client.post(
            "/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "format": "json",          # ← Ollama JSON mode: guaranteed valid JSON output
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama service")
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable. Ensure it is running.",
        )
    except httpx.TimeoutException:
        logger.error("Ollama chunk request timed out")
        raise HTTPException(
            status_code=504,
            detail="LLM processing timed out.",
        )
    except httpx.HTTPStatusError as e:
        logger.error(f"Ollama returned error: {e.response.status_code}")
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {e.response.status_code}",
        )

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
    if isinstance(obj, str) and obj.strip().lower() == "null":
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

    Interpretation:
      90–100 → Excellent / Professional-ready
      75–89  → Good / Needs minor improvements
      50–74  → Average / Needs improvement
      <50    → Poor / Needs major overhaul
    """
    contact      = parsed.get("contact", {})
    skills       = parsed.get("skills", {}).get("skills", [])
    experience   = parsed.get("experience", {}).get("experience", [])
    education    = parsed.get("education", {}).get("education", [])
    certs        = parsed.get("certifications", {}).get("certifications", [])
    projects     = parsed.get("projects", {}).get("projects", [])
    awards       = parsed.get("awards", {}).get("awards", [])
    summary      = parsed.get("summary", {}).get("summary", "") or ""

    remarks = []

    # ── 1. Contact Information (weight 5%) — score 0–10 ──────────────────────
    contact_score = 0
    if contact.get("name"):         contact_score += 3
    if contact.get("email"):        contact_score += 3
    if contact.get("phone") or contact.get("number"): contact_score += 2
    if contact.get("current_location"): contact_score += 1
    # LinkedIn / portfolio would be +1 but not extracted yet — max achievable is 9→10
    contact_score = min(10, contact_score + (1 if contact_score >= 9 else 0))
    if contact_score < 5:
        remarks.append("Contact information is incomplete — add name, email, and phone.")

    # ── 2. Professional Summary (weight 15%) — score 0–10 ────────────────────
    summary_score = 0
    if summary:
        words = len(summary.split())
        if words >= 30:   summary_score = 10   # rich, detailed
        elif words >= 15: summary_score = 7    # decent
        elif words >= 5:  summary_score = 4    # too brief
        else:             summary_score = 2
    else:
        remarks.append("No professional summary found — add a concise 2–3 sentence summary.")

    # ── 3. Work Experience (weight 25%) — score 0–10 ─────────────────────────
    exp_count = len(experience)
    has_desc  = sum(1 for e in experience if e.get("description") and len(e["description"]) > 20)
    has_dates = sum(1 for e in experience if e.get("duration"))

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

    # Bonus: descriptions present (+1), dates present (+1)
    if exp_count > 0:
        if has_desc == exp_count:  exp_score = min(10, exp_score + 1)
        if has_dates == exp_count: exp_score = min(10, exp_score + 0)  # already counted in base

    if exp_count > 0 and has_desc < exp_count:
        remarks.append("Add measurable impact and descriptions to work experience entries.")

    # ── 4. Skills (weight 20%) — score 0–10 ──────────────────────────────────
    skill_count = len(skills)
    if skill_count == 0:
        skills_score = 0
        remarks.append("No skills listed — add both technical and soft skills.")
    elif skill_count <= 3:
        skills_score = 2
        remarks.append("Very few skills listed — aim for at least 8–10 relevant skills.")
    elif skill_count <= 6:
        skills_score = 4
    elif skill_count <= 10:
        skills_score = 6
    elif skill_count <= 15:
        skills_score = 8
    elif skill_count <= 20:
        skills_score = 9
    else:
        skills_score = 10

    # ── 5. Education & Certifications (weight 10%) — score 0–10 ──────────────
    edu_score = 0
    if education:
        edu = education[0]
        if edu.get("institution"):   edu_score += 2
        if edu.get("degree"):        edu_score += 2
        if edu.get("field_of_study"): edu_score += 1
        if edu.get("year"):          edu_score += 1
        if edu.get("grade"):         edu_score += 1
    else:
        remarks.append("No education details found.")

    # Certifications boost (up to +3)
    cert_boost = min(3, len(certs))
    edu_score  = min(10, edu_score + cert_boost)

    if not education:
        edu_score = 0

    # ── 6. Achievements / Projects (weight 15%) — score 0–10 ─────────────────
    achieve_score = 0
    project_count = len(projects)
    award_count   = len(awards)

    if project_count == 0 and award_count == 0:
        achieve_score = 0
        remarks.append("No projects or achievements found — add notable projects or awards.")
    else:
        # Projects: up to 6 points
        if project_count >= 3:   achieve_score += 6
        elif project_count == 2: achieve_score += 4
        elif project_count == 1: achieve_score += 2

        # Projects with descriptions: +1
        proj_with_desc = sum(1 for p in projects if p.get("description") and len(p["description"]) > 10)
        if proj_with_desc == project_count and project_count > 0:
            achieve_score = min(10, achieve_score + 1)

        # Awards: up to 3 points
        if award_count >= 2:     achieve_score = min(10, achieve_score + 3)
        elif award_count == 1:   achieve_score = min(10, achieve_score + 2)

    # ── 7. Format & Design (weight 10%) — score 0–10 ─────────────────────────
    # Inferred from data completeness — cannot visually inspect the PDF from here
    # Scored based on: sections present, data populated, no null overload
    populated_sections = sum([
        bool(contact.get("name")),
        bool(summary),
        bool(experience),
        bool(skills),
        bool(education),
        bool(projects or awards),
        bool(certs),
    ])
    format_score = min(10, populated_sections + 3)  # base 3 for passing file validation

    # ── Weighted overall (each raw score × weight → sum out of 100) ──────────
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

    # ── Interpretation band ───────────────────────────────────────────────────
    if overall >= 90:
        grade = "Excellent"
        if not remarks:
            remarks.append("Professional-ready resume with strong coverage across all sections.")
    elif overall >= 75:
        grade = "Good"
        if not remarks:
            remarks.append("Good resume — address the minor gaps above to reach Excellent.")
    elif overall >= 50:
        grade = "Average"
        if not remarks:
            remarks.append("Resume needs improvement in several areas — see remarks above.")
    else:
        grade = "Poor"
        if not remarks:
            remarks.append("Resume needs a major overhaul — many critical sections are missing.")

    return {
        "overall":                 overall,
        "contact_information":     contact_score,
        "professional_summary":    summary_score,
        "work_experience":         exp_score,
        "skills":                  skills_score,
        "education_certifications": edu_score,
        "achievements_projects":   achieve_score,
        "format_design":           format_score,
        "grade":                   grade,
        "remarks":                 " ".join(remarks),
    }


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

    Improvements over sequential approach:
    - asyncio.gather fires all 8 chunk requests concurrently. Since each
      request is pure async I/O (no CPU work in Python), all 8 are in-flight
      simultaneously. Ollama queues them internally but the event loop doesn't
      block between calls — wall-clock time ≈ slowest single chunk instead of
      sum of all chunks.
    - Ollama JSON mode (format="json") guarantees valid JSON output, so
      _extract_json only needs json.loads() — no regex cleanup fallbacks needed.
    """
    sections = _split_resume_sections(text)

    # Build coroutines for all chunks — one per extraction category
    coroutines = []
    for chunk_name, prompt, max_tok, text_slice in CHUNKS:
        chunk_text = text[text_slice[0]:text_slice[1]] if text_slice else sections.get(chunk_name, text[:800])
        coroutines.append(_call_chunk(chunk_name, prompt, max_tok, chunk_text))

    # Fire all 8 requests concurrently — total time ≈ slowest chunk, not sum
    results = await asyncio.gather(*coroutines)
    parsed = {name: data for name, data in results}

    # Compute rule-based score from extracted data
    score = _compute_score(parsed)

    # Merge all chunks into final result
    contact = parsed.get("contact", {})
    skills_data = parsed.get("skills", {})
    exp_data = parsed.get("experience", {})
    edu_data = parsed.get("education", {})
    cert_data = parsed.get("certifications", {})
    proj_data = parsed.get("projects", {})
    award_data = parsed.get("awards", {})
    summary_data = parsed.get("summary", {})

    return {
        "name": contact.get("name"),
        "email": contact.get("email"),
        "phone": contact.get("phone"),
        "number": contact.get("number"),
        "current_location": contact.get("current_location"),
        "skills": skills_data.get("skills", []),
        "experience": exp_data.get("experience", []),
        "education": edu_data.get("education", []),
        "projects": proj_data.get("projects", []),
        "certifications": cert_data.get("certifications", []),
        "awards": award_data.get("awards", []),
        "summary": summary_data.get("summary"),
        "resume_score": score,
    }


async def list_models() -> list[dict]:
    """List available models from Ollama."""
    client = _get_client()
    try:
        response = await client.get("/api/tags")
        response.raise_for_status()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama service is unavailable.",
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama returned an error: {e.response.status_code}",
        )

    result = response.json()
    return result.get("models", [])
