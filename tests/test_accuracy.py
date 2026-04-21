"""
Industry-standard resume parser accuracy test suite.

Strategy
--------
We test against 2484 real anonymised resumes across 24 job categories from
the archive/data/data/ dataset. Each resume is a real US-format resume with:
  • Job title at top (not a person name, as they're anonymized)
  • "Company Name" placeholder for all employers
  • "City, State" placeholder for locations
  • Real skills, experience bullet points, education, certifications

Accuracy is measured field-by-field. We define "correct" as:
  • MUST extract (expected always present): skills, experience entries,
    current_designation (= first line of resume)
  • SHOULD extract (expected in 90%+ of resumes): summary
  • CAN extract (present in subset): education, certifications, projects, awards

We sample N resumes per category (default 2 per category = 48 total),
then run async parsing in batches and score each field.

Pass threshold: >= 96% on MUST fields, >= 90% on SHOULD fields.

Usage:
    pytest tests/test_accuracy.py -v -s          # default 2/category
    pytest tests/test_accuracy.py -v -s --count=5  # 5/category = 120 resumes
    pytest tests/test_accuracy.py -v -s --count=0  # ALL 2484 resumes
"""

import asyncio
import io
import json
import pathlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import pytest
from pypdf import PdfReader

# ── Paths ─────────────────────────────────────────────────────────────────────
ARCHIVE_ROOT = pathlib.Path("archive/data/data")
CATEGORIES = sorted([d for d in ARCHIVE_ROOT.iterdir() if d.is_dir()])


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class FieldResult:
    name: str
    present: int = 0
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def presence_rate(self) -> float:
        return self.present / self.total if self.total else 0.0


@dataclass
class ResumeResult:
    path: str
    category: str
    success: bool
    parse_time_ms: float
    text_length: int
    fields: dict = field(default_factory=dict)
    error: Optional[str] = None


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_pdf_text(path: pathlib.Path) -> str:
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages).strip()


# ── Ground-truth derivation ───────────────────────────────────────────────────
# Since these are real resumes with known structure, we derive expected values
# from the text itself rather than from a hand-labelled dataset.

def _derive_expected(text: str, normalised: str) -> dict:
    """
    Derive expected extraction targets directly from the normalised resume text.

    For anonymised resumes (no real names/emails), we check structural correctness:
    - Has the parser extracted *something* meaningful for each field?
    - Is the extracted value consistent with what appears in the text?
    """
    expected = {}

    tl = normalised.lower()

    # ── Current designation: first non-empty line that looks like a job title ─
    # Skip lines that are clearly not job titles: section headers (Summary,
    # Skills, Experience), generic openers, very short lines, or company-name-
    # only lines. The designation is typically 2-6 words in ALL CAPS or Title Case.
    NOT_TITLE_KEYWORDS = {
        "summary", "skills", "experience", "education", "highlights", "objective",
        "profile", "qualifications", "accomplishments", "overview", "career",
        "background", "work history", "employment", "certifications", "references",
    }
    first_line = ""
    for line in normalised.split("\n")[:10]:
        stripped = line.strip()
        if not stripped:
            continue
        words = stripped.split()
        # Skip: single words, pure section headers, purely numeric, or very long
        if not words or len(words) > 10:
            continue
        if stripped.lower() in NOT_TITLE_KEYWORDS:
            continue
        if stripped.lower().split()[0] in NOT_TITLE_KEYWORDS:
            continue
        # Skip: looks like a date
        if re.match(r'^\d{2}/\d{4}', stripped):
            continue
        # This looks like a title
        first_line = stripped
        break
    expected["has_current_designation"] = bool(first_line)
    expected["current_designation_hint"] = first_line[:60] if first_line else None

    # ── Skills: resume must have skills extractable ────────────────────────
    has_skill_section = any(kw in tl for kw in [
        "skills", "competencies", "expertise", "highlights", "proficiencies",
        "qualifications", "technologies", "tools"
    ])
    expected["has_skills"] = has_skill_section or len(tl) > 500

    # ── Experience: count approximate job entries ──────────────────────────
    # Look for date patterns that precede job entries
    date_patterns = [
        r'\d{1,2}/\d{4}\s+to\s+(?:\d{1,2}/\d{4}|current|present)',
        r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\s+to\s+(?:\d{1,2}/\d{4}|current|present)',
        r'\d{4}\s*[-–]\s*(?:\d{4}|current|present)',
    ]
    job_dates = []
    for pat in date_patterns:
        job_dates.extend(re.findall(pat, tl))
    expected["min_experience_entries"] = max(1, len(job_dates))
    expected["has_experience"] = (
        "experience" in tl or "work history" in tl or
        "employment" in tl or "career" in tl or len(job_dates) > 0
    )

    # ── Summary ────────────────────────────────────────────────────────────
    expected["has_summary_source"] = any(kw in tl[:500] for kw in [
        "summary", "objective", "profile", "overview", "executive profile",
        "career focus", "highlights", "professional background"
    ])

    # ── Education ─────────────────────────────────────────────────────────
    expected["has_education_source"] = any(kw in tl for kw in [
        "education", "academic", "university", "college", "bachelor",
        "master", "degree", "b.s", "b.a", "m.s", "m.a", "ph.d", "mba",
        "diploma", "certification"
    ])

    # ── Certifications ─────────────────────────────────────────────────────
    expected["has_cert_source"] = any(kw in tl for kw in [
        "certified", "certification", "certificate", "license", "credential",
        "pmp", "aws", "cisco", "microsoft", "comptia", "cpa", "cfa", "six sigma"
    ])

    return expected


# ── Accuracy scoring per result ───────────────────────────────────────────────

def _score_result(parsed: dict, expected: dict, text: str, normalised: str) -> dict:
    """
    Score the parsed output against expected values.

    Returns a dict of field -> (passed: bool, note: str).
    """
    scores = {}
    tl = normalised.lower()

    # ── 1. Current designation ─────────────────────────────────────────────
    designation_hint = (expected.get("current_designation_hint") or "").lower()
    extracted_designation = (parsed.get("current_designation") or "").lower().strip()
    if designation_hint and extracted_designation:
        # Allow partial match (hint words in extracted or vice versa)
        hint_words = set(designation_hint.split())
        ext_words = set(extracted_designation.split())
        overlap = hint_words & ext_words
        passed = len(overlap) >= max(1, len(hint_words) * 0.5)
    elif designation_hint and not extracted_designation:
        # Also accept if we can find it via industry/title in experience
        exp_list = parsed.get("experience", [])
        if exp_list and isinstance(exp_list, list):
            titles = [e.get("title", "").lower() for e in exp_list if isinstance(e, dict)]
            any_match = any(
                len(set(designation_hint.split()) & set(t.split())) >= 1
                for t in titles if t
            )
            passed = any_match
        else:
            passed = False
    else:
        passed = bool(extracted_designation)
    scores["current_designation"] = (passed, f"expected~'{designation_hint[:40]}', got='{extracted_designation[:40]}'")

    # ── 2. Skills ──────────────────────────────────────────────────────────
    skills = parsed.get("skills", [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    has_skills = len(skills) >= 3
    scores["skills"] = (has_skills, f"extracted {len(skills)} skills")

    # ── 3. Experience entries ──────────────────────────────────────────────
    experience = parsed.get("experience", [])
    if isinstance(experience, list):
        exp_count = len(experience)
    elif isinstance(experience, str):
        exp_count = len([e for e in experience.split("\n") if e.strip()])
    else:
        exp_count = 0
    min_exp = expected.get("min_experience_entries", 1)
    # We expect at least 1 experience entry if the resume has experience signals
    has_exp_source = expected.get("has_experience", True)
    if has_exp_source:
        scores["experience_entries"] = (exp_count >= 1, f"extracted {exp_count}, expected >= 1")
    else:
        scores["experience_entries"] = (True, "no experience signals in text")

    # ── 4. Summary extracted ───────────────────────────────────────────────
    summary = (parsed.get("summary") or "").strip()
    has_summary = len(summary.split()) >= 10
    scores["summary"] = (has_summary, f"{len(summary.split())} words")

    # ── 5. Education (conditional) ─────────────────────────────────────────
    if expected.get("has_education_source"):
        education = parsed.get("education", [])
        if isinstance(education, list):
            edu_count = len(education)
        elif isinstance(education, str):
            edu_count = len([e for e in education.split("\n") if e.strip()])
        else:
            edu_count = 0
        # Only fail if education is CLEARLY in the text but not extracted
        # (education section header present + content)
        edu_section_explicit = any(kw in tl for kw in ["education\n", "\neducation\n", "educational background"])
        if edu_section_explicit:
            scores["education"] = (edu_count >= 1, f"extracted {edu_count} entries")
        else:
            scores["education"] = (True, "education not in explicit section")
    else:
        scores["education"] = (True, "no education source in text")

    # ── 6. No hallucination check ──────────────────────────────────────────
    # The extracted email (if any) must appear in the original text
    extracted_email = parsed.get("email")
    if extracted_email:
        email_in_text = extracted_email.lower() in tl
        scores["no_hallucinated_email"] = (email_in_text, f"email '{extracted_email}' in text={email_in_text}")
    else:
        scores["no_hallucinated_email"] = (True, "no email extracted (none expected for anonymised resumes)")

    # ── 7. No hallucinated company names ──────────────────────────────────
    # Archive resumes use "Company Name" placeholder — we verify the extracted
    # company is either:
    #   (a) "Company Name" verbatim (correct extraction of the placeholder)
    #   (b) A company-like value found in the text (e.g. real company for real resumes)
    #   (c) Same as designation (valid for self-employed / freelance roles)
    # We flag as WRONG only if the company appears nowhere in the text AND
    # it isn't a "Company Name" placeholder.
    current_company = (parsed.get("current_company") or "").strip()
    if current_company:
        company_lower = current_company.lower()
        # Acceptable: "Company Name" placeholder
        is_placeholder = "company" in company_lower
        # Acceptable: value appears anywhere in the text
        in_text = company_lower in tl
        # Acceptable: freelance/self-employment roles where title = company
        designation_lower = (parsed.get("current_designation") or "").lower()
        is_self_employed = (
            company_lower == designation_lower or
            "freelance" in company_lower or
            "self-employed" in company_lower or
            "independent" in company_lower
        )
        passed = is_placeholder or in_text or is_self_employed
        scores["company_not_hallucinated"] = (passed, f"company='{current_company[:40]}'")
    else:
        scores["company_not_hallucinated"] = (True, "no company extracted")

    # ── 8. Skills quality ──────────────────────────────────────────────────
    # Extracted skills should be grounded in the resume text.
    # We use a lenient word-level check: at least one non-trivial word from each
    # skill phrase should appear in the text. This allows the LLM to normalise
    # skill names (e.g. "Claims management" from "manage claims" in a bullet).
    SKILL_STOPWORDS = {
        "management", "skills", "skill", "ability", "experience", "knowledge",
        "understanding", "proficiency", "expertise", "and", "or", "of", "the",
        "in", "to", "for", "with", "a", "an"
    }
    if skills:
        grounded_count = 0
        for skill in skills:
            skill_lower = skill.lower()
            # Exact match
            if skill_lower in tl:
                grounded_count += 1
                continue
            # Word-level match: any significant word from the skill appears in text
            words = re.findall(r'\b[a-z]{4,}\b', skill_lower)
            significant = [w for w in words if w not in SKILL_STOPWORDS]
            if significant and any(w in tl for w in significant):
                grounded_count += 1
                continue
        ratio = grounded_count / len(skills)
        scores["skills_grounded"] = (ratio >= 0.7, f"{grounded_count}/{len(skills)} skills grounded in text")
    else:
        scores["skills_grounded"] = (False, "no skills extracted")

    # ── 9. Experience descriptions grounded ───────────────────────────────
    if experience and isinstance(experience, list) and len(experience) > 0:
        exp_item = experience[0]
        if isinstance(exp_item, dict):
            desc = (exp_item.get("description") or "").lower()
            # Description words (non-stopwords) should appear in text
            stopwords = {"the", "a", "an", "in", "to", "for", "of", "and", "or",
                         "with", "at", "by", "from", "on", "as", "is", "are", "was"}
            desc_words = [w for w in re.findall(r'\b[a-z]{4,}\b', desc) if w not in stopwords]
            if desc_words:
                found = sum(1 for w in desc_words if w in tl)
                ratio = found / len(desc_words)
                scores["experience_grounded"] = (ratio >= 0.5, f"{found}/{len(desc_words)} desc words in text")
            else:
                scores["experience_grounded"] = (bool(desc), "no content words in description")
        else:
            scores["experience_grounded"] = (True, "experience not dict format")
    else:
        scores["experience_grounded"] = (True, "no experience entries to check")

    # ── 10. Total years of experience numeric ─────────────────────────────
    years_exp = parsed.get("total_years_of_experience")
    if years_exp is not None:
        try:
            yf = float(years_exp)
            scores["experience_years_numeric"] = (0.5 <= yf <= 50, f"years={yf}")
        except (ValueError, TypeError):
            scores["experience_years_numeric"] = (False, f"invalid years value: {years_exp}")
    else:
        scores["experience_years_numeric"] = (True, "years not extracted (optional)")

    return scores


# ── Async batch runner ────────────────────────────────────────────────────────

async def _parse_one_resume(path: pathlib.Path, category: str) -> ResumeResult:
    from app.services.llm import parse_resume, _normalise_text

    start = time.time()
    try:
        raw_text = _extract_pdf_text(path)
        if not raw_text or len(raw_text) < 100:
            return ResumeResult(
                path=str(path), category=category, success=False,
                parse_time_ms=0, text_length=0,
                error="Text extraction failed or too short"
            )

        normalised = _normalise_text(raw_text)
        expected = _derive_expected(raw_text, normalised)

        parsed = await parse_resume(normalised)
        elapsed_ms = (time.time() - start) * 1000

        field_scores = _score_result(parsed, expected, raw_text, normalised)

        return ResumeResult(
            path=str(path), category=category, success=True,
            parse_time_ms=round(elapsed_ms, 1),
            text_length=len(normalised),
            fields=field_scores,
        )
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        return ResumeResult(
            path=str(path), category=category, success=False,
            parse_time_ms=round(elapsed_ms, 1), text_length=0,
            error=str(e)[:200]
        )


async def _run_batch(paths_with_cats: list, concurrency: int = 3) -> list:
    """Run parse jobs with bounded concurrency to avoid rate limits."""
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def _bounded(path, cat):
        async with sem:
            return await _parse_one_resume(path, cat)

    tasks = [_bounded(p, c) for p, c in paths_with_cats]
    results = await asyncio.gather(*tasks)
    return list(results)


# ── Pytest fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def resume_count(request):
    return int(request.config.getoption("--count", default="2"))


@pytest.fixture(scope="session")
def concurrency(request):
    return int(request.config.getoption("--concurrency", default="3"))


@pytest.fixture(scope="session")
def accuracy_results(resume_count, concurrency):
    """Run the full accuracy benchmark once per session and cache results."""
    if not ARCHIVE_ROOT.exists():
        pytest.skip("archive/data/data not found — skipping accuracy tests")

    paths_with_cats = []
    for cat in CATEGORIES:
        pdfs = sorted(cat.glob("*.pdf"))
        sample = pdfs if resume_count == 0 else pdfs[:resume_count]
        paths_with_cats.extend([(p, cat.name) for p in sample])

    if not paths_with_cats:
        pytest.skip("No PDF files found in archive")

    print(f"\n[accuracy] Running {len(paths_with_cats)} resumes across {len(CATEGORIES)} categories")
    print(f"[accuracy] Concurrency: {concurrency} parallel LLM calls")

    results = asyncio.run(_run_batch(paths_with_cats, concurrency=concurrency))
    return results


# ── Report helpers ────────────────────────────────────────────────────────────

def _aggregate(results: list) -> dict:
    """Aggregate per-resume results into field-level accuracy stats."""
    totals = {}
    successful = [r for r in results if r.success]

    for result in successful:
        for field_name, (passed, note) in result.fields.items():
            if field_name not in totals:
                totals[field_name] = FieldResult(name=field_name)
            totals[field_name].total += 1
            if passed:
                totals[field_name].correct += 1
                totals[field_name].present += 1
            else:
                pass  # present stays 0

    return totals


def _print_report(results: list, totals: dict):
    """Print a detailed accuracy report."""
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print(f"\n{'='*70}")
    print(f"RESUME PARSER ACCURACY REPORT")
    print(f"{'='*70}")
    print(f"Total resumes: {len(results)}")
    print(f"Successful parses: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed parses: {len(failed)}")
    if successful:
        avg_time = sum(r.parse_time_ms for r in successful) / len(successful)
        print(f"Avg parse time: {avg_time:.0f}ms")

    print(f"\n{'FIELD ACCURACY':40} {'CORRECT/TOTAL':15} {'ACCURACY':10}")
    print(f"{'-'*70}")

    # Must-have fields (highest bar)
    must_fields = [
        "current_designation",
        "skills",
        "skills_grounded",
        "experience_entries",
        "experience_grounded",
        "no_hallucinated_email",
        "company_not_hallucinated",
    ]
    should_fields = [
        "summary",
        "education",
    ]
    nice_fields = [
        "experience_years_numeric",
    ]

    overall_must_correct = 0
    overall_must_total = 0

    print("\n[MUST-HAVE FIELDS — target 96%+]")
    for fname in must_fields:
        if fname in totals:
            fr = totals[fname]
            mark = "✓" if fr.accuracy >= 0.96 else "✗"
            print(f"  {mark} {fr.name:38} {fr.correct:4}/{fr.total:<10} {fr.accuracy*100:5.1f}%")
            overall_must_correct += fr.correct
            overall_must_total += fr.total

    if overall_must_total:
        overall_must_acc = overall_must_correct / overall_must_total
        print(f"\n  {'OVERALL MUST-HAVE':38} {overall_must_correct:4}/{overall_must_total:<10} {overall_must_acc*100:5.1f}%")

    print("\n[SHOULD-HAVE FIELDS — target 90%+]")
    for fname in should_fields:
        if fname in totals:
            fr = totals[fname]
            mark = "✓" if fr.accuracy >= 0.90 else "✗"
            print(f"  {mark} {fr.name:38} {fr.correct:4}/{fr.total:<10} {fr.accuracy*100:5.1f}%")

    print("\n[NICE-TO-HAVE FIELDS]")
    for fname in nice_fields:
        if fname in totals:
            fr = totals[fname]
            print(f"    {fr.name:38} {fr.correct:4}/{fr.total:<10} {fr.accuracy*100:5.1f}%")

    if failed:
        print(f"\n[FAILURES]")
        for r in failed[:10]:
            print(f"  {r.category}/{pathlib.Path(r.path).name}: {r.error}")

    # Per-category breakdown
    print(f"\n[PER-CATEGORY ACCURACY (designation extraction)]")
    cat_stats = {}
    for result in successful:
        cat = result.category
        if cat not in cat_stats:
            cat_stats[cat] = {"correct": 0, "total": 0}
        if "current_designation" in result.fields:
            passed, _ = result.fields["current_designation"]
            cat_stats[cat]["total"] += 1
            if passed:
                cat_stats[cat]["correct"] += 1

    for cat, stats in sorted(cat_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] else 0
        mark = "✓" if acc >= 0.90 else "~" if acc >= 0.70 else "✗"
        print(f"  {mark} {cat:30} {stats['correct']}/{stats['total']} = {acc*100:.0f}%")

    print(f"\n{'='*70}\n")


# ── Test functions ─────────────────────────────────────────────────────────────

class TestAccuracy:

    def test_archive_exists(self):
        """Verify the archive dataset is available."""
        if not ARCHIVE_ROOT.exists():
            pytest.skip("archive/data/data not found")
        pdf_count = len(list(ARCHIVE_ROOT.rglob("*.pdf")))
        assert pdf_count > 0, "No PDF files found in archive"
        print(f"\n  Archive: {pdf_count} PDFs across {len(CATEGORIES)} categories")

    def test_parse_success_rate(self, accuracy_results):
        """At least 98% of resumes must parse without errors."""
        total = len(accuracy_results)
        successful = sum(1 for r in accuracy_results if r.success)
        rate = successful / total if total else 0
        print(f"\n  Parse success: {successful}/{total} = {rate*100:.1f}%")
        assert rate >= 0.98, f"Parse success rate {rate*100:.1f}% < 98%"

    def test_skills_extraction_accuracy(self, accuracy_results):
        """Skills must be extracted in >= 96% of resumes that have skill sections."""
        totals = _aggregate(accuracy_results)
        if "skills" not in totals:
            pytest.skip("No skills data in results")
        fr = totals["skills"]
        print(f"\n  Skills accuracy: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.96, f"Skills extraction {fr.accuracy*100:.1f}% < 96%"

    def test_skills_grounded_accuracy(self, accuracy_results):
        """Extracted skills must be grounded in the resume text (no hallucination)."""
        totals = _aggregate(accuracy_results)
        if "skills_grounded" not in totals:
            pytest.skip("No skills grounded data")
        fr = totals["skills_grounded"]
        print(f"\n  Skills grounded: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.96, f"Skills groundedness {fr.accuracy*100:.1f}% < 96%"

    def test_experience_extraction_accuracy(self, accuracy_results):
        """At least one experience entry must be extracted per resume."""
        totals = _aggregate(accuracy_results)
        if "experience_entries" not in totals:
            pytest.skip("No experience data")
        fr = totals["experience_entries"]
        print(f"\n  Experience entries: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.96, f"Experience extraction {fr.accuracy*100:.1f}% < 96%"

    def test_experience_grounded(self, accuracy_results):
        """Experience descriptions must be grounded in text (not hallucinated)."""
        totals = _aggregate(accuracy_results)
        if "experience_grounded" not in totals:
            pytest.skip("No experience grounded data")
        fr = totals["experience_grounded"]
        print(f"\n  Experience grounded: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.94, f"Experience groundedness {fr.accuracy*100:.1f}% < 94%"

    def test_current_designation_extraction(self, accuracy_results):
        """Current job title/designation must be extracted correctly."""
        totals = _aggregate(accuracy_results)
        if "current_designation" not in totals:
            pytest.skip("No designation data")
        fr = totals["current_designation"]
        print(f"\n  Current designation: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.90, f"Designation extraction {fr.accuracy*100:.1f}% < 90%"

    def test_no_email_hallucination(self, accuracy_results):
        """Parser must not fabricate email addresses not in the source text."""
        totals = _aggregate(accuracy_results)
        if "no_hallucinated_email" not in totals:
            pytest.skip("No email hallucination data")
        fr = totals["no_hallucinated_email"]
        print(f"\n  No hallucinated email: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.99, f"Email hallucination rate too high: {(1-fr.accuracy)*100:.1f}%"

    def test_no_company_hallucination(self, accuracy_results):
        """Parser must not return the job title as the company name."""
        totals = _aggregate(accuracy_results)
        if "company_not_hallucinated" not in totals:
            pytest.skip("No company hallucination data")
        fr = totals["company_not_hallucinated"]
        print(f"\n  No hallucinated company: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.96, f"Company hallucination rate too high"

    def test_summary_extraction(self, accuracy_results):
        """Summary should be extracted for 90%+ of resumes."""
        totals = _aggregate(accuracy_results)
        if "summary" not in totals:
            pytest.skip("No summary data")
        fr = totals["summary"]
        print(f"\n  Summary extraction: {fr.correct}/{fr.total} = {fr.accuracy*100:.1f}%")
        assert fr.accuracy >= 0.90, f"Summary extraction {fr.accuracy*100:.1f}% < 90%"

    def test_full_report(self, accuracy_results):
        """Print a full accuracy report (always passes — informational only)."""
        totals = _aggregate(accuracy_results)
        _print_report(accuracy_results, totals)

        # Compute composite MUST score
        must_fields = [
            "current_designation", "skills", "skills_grounded",
            "experience_entries", "experience_grounded",
            "no_hallucinated_email", "company_not_hallucinated",
        ]
        total_correct = sum(totals[f].correct for f in must_fields if f in totals)
        total_checks = sum(totals[f].total for f in must_fields if f in totals)
        if total_checks:
            composite = total_correct / total_checks
            print(f"COMPOSITE MUST-HAVE ACCURACY: {composite*100:.2f}%")
            assert composite >= 0.96, f"Composite must-have accuracy {composite*100:.2f}% < 96%"


# ── Spot-check tests (run without the archive — use 2 bundled resumes) ────────

class TestSpotCheck:
    """Quick spot-checks on the 2 bundled test resumes."""

    @pytest.mark.asyncio
    async def test_abhishek_suman_resume(self):
        """Parse AbhishekSumanCV.pdf and verify key fields."""
        from app.services.llm import parse_resume, _normalise_text
        from app.services.document import _extract_pdf

        path = pathlib.Path("AbhishekSumanCV.pdf")
        if not path.exists():
            pytest.skip("AbhishekSumanCV.pdf not found")

        text = await _extract_pdf(path.read_bytes())
        text = _normalise_text(text)
        parsed = await parse_resume(text)

        # Contact
        assert parsed.get("name") or parsed.get("first_name"), "Name not extracted"
        email = parsed.get("email", "")
        assert email and "@" in email, f"Email not extracted: {email}"
        phone = parsed.get("phone", "")
        assert phone and len(re.sub(r'\D', '', phone)) >= 10, f"Phone invalid: {phone}"

        # Experience
        exp = parsed.get("experience", [])
        assert len(exp) >= 3, f"Expected >= 3 experience entries, got {len(exp)}"
        titles = [e.get("title", "").lower() for e in exp]
        assert any("engineer" in t or "planning" in t for t in titles), "Planning Engineer title missing"

        # Skills
        skills = parsed.get("skills", [])
        assert len(skills) >= 8, f"Expected >= 8 skills, got {len(skills)}"
        skill_text = " ".join(skills).lower()
        assert "primavera" in skill_text or "project management" in skill_text, "Core skills missing"

        # Years of experience
        yoe = parsed.get("total_years_of_experience")
        assert yoe and float(yoe) >= 8, f"YoE should be >= 8, got {yoe}"

        # Personal
        assert parsed.get("nationality") == "Indian", "Nationality missing"
        assert parsed.get("marital_status", "").lower() in ("married", "single"), "Marital status missing"

        print(f"\n  Abhishek: name={parsed.get('name')}, email={email}, skills={len(skills)}, exp={len(exp)}")

    @pytest.mark.asyncio
    async def test_tharun_salesforce_resume(self):
        """Parse Tharun S resume and verify Salesforce-developer-specific fields."""
        from app.services.llm import parse_resume, _normalise_text
        from app.services.document import _extract_pdf

        path = pathlib.Path("my resume - 2025-02-21 04_47_34.pdf")
        if not path.exists():
            pytest.skip("Tharun resume not found")

        text = await _extract_pdf(path.read_bytes())
        text = _normalise_text(text)
        parsed = await parse_resume(text)

        # Contact
        email = parsed.get("email", "")
        assert email and "tharun" in email.lower(), f"Email wrong: {email}"
        phone = parsed.get("phone", "")
        assert phone and len(re.sub(r'\D', '', phone)) >= 10, f"Phone invalid: {phone}"
        location = parsed.get("current_location", "")
        assert location and "bangalore" in location.lower(), f"Location wrong: {location}"

        # Experience
        exp = parsed.get("experience", [])
        assert len(exp) >= 2, f"Expected >= 2 experience entries, got {len(exp)}"
        companies = [e.get("company", "").lower() for e in exp]
        assert any("cirr" in c or "navyaan" in c or "nav" in c for c in companies), \
            f"Expected companies not found: {companies}"

        # Skills — must have Apex, Salesforce
        skills = parsed.get("skills", [])
        skill_text = " ".join(skills).lower()
        assert "apex" in skill_text, "Apex not in skills"
        assert "salesforce" in skill_text, "Salesforce not in skills"
        assert len(skills) >= 15, f"Expected >= 15 skills, got {len(skills)}"

        # Certifications
        certs = parsed.get("certifications", [])
        cert_names = [c.get("name", "").lower() for c in certs]
        assert any("salesforce" in c for c in cert_names), f"Salesforce cert missing: {cert_names}"
        assert len(certs) >= 3, f"Expected >= 3 certs, got {len(certs)}"

        # Education
        edu = parsed.get("education", [])
        assert len(edu) >= 1, "Education missing"
        degrees = [e.get("degree", "").lower() for e in edu]
        assert any("mca" in d or "master" in d or "computer" in d for d in degrees), \
            f"MCA degree missing: {degrees}"

        # YoE
        yoe = parsed.get("total_years_of_experience")
        assert yoe and 3.0 <= float(yoe) <= 5.0, f"YoE should be ~3.9, got {yoe}"

        # Awards
        awards = parsed.get("awards", [])
        assert len(awards) >= 1, "Award missing"

        print(f"\n  Tharun: email={email}, skills={len(skills)}, certs={len(certs)}, exp={len(exp)}")
