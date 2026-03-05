#!/usr/bin/env python3
"""Test script to parse a resume via the Resume Parser API."""

import sys
import json
import time
import httpx

API_BASE = "http://localhost:8000"
API_KEY = "changeme"
HEADERS = {"X-API-Key": API_KEY}


def test_health():
    """Test the health endpoint."""
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    resp = httpx.get(f"{API_BASE}/health", timeout=10)
    data = resp.json()
    print(f"Status Code : {resp.status_code}")
    print(f"Status      : {data.get('status')}")
    print(f"Ollama      : {data.get('ollama_connected')}")
    print(f"Model       : {data.get('model')}")
    assert resp.status_code == 200, f"Health check failed: {resp.status_code}"
    assert data["ollama_connected"], "Ollama is not connected"
    print("PASSED\n")


def test_models():
    """Test the models endpoint."""
    print("=" * 60)
    print("TEST 2: List Models")
    print("=" * 60)
    resp = httpx.get(f"{API_BASE}/api/v1/models", headers=HEADERS, timeout=10)
    data = resp.json()
    print(f"Status Code : {resp.status_code}")
    print(f"Models      : {[m['name'] for m in data.get('models', [])]}")
    assert resp.status_code == 200, f"Models endpoint failed: {resp.status_code}"
    print("PASSED\n")


def test_auth_rejection():
    """Test that requests without API key are rejected."""
    print("=" * 60)
    print("TEST 3: Auth Rejection (no API key)")
    print("=" * 60)
    resp = httpx.get(f"{API_BASE}/api/v1/models", timeout=10)
    print(f"Status Code : {resp.status_code}")
    assert resp.status_code == 401, f"Expected 401, got {resp.status_code}"
    print("PASSED\n")


def test_parse_resume(file_path: str):
    """Test resume parsing with a real file."""
    print("=" * 60)
    print("TEST 4: Parse Resume")
    print("=" * 60)
    print(f"File: {file_path}")

    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("/")[-1], f, "application/pdf")}
        print("Uploading and parsing... (this may take a minute)")
        start = time.time()
        resp = httpx.post(
            f"{API_BASE}/api/v1/parse",
            headers=HEADERS,
            files=files,
            timeout=httpx.Timeout(1800.0, connect=30.0),
        )
        elapsed = time.time() - start

    print(f"Status Code : {resp.status_code}")
    print(f"Client Time : {elapsed:.2f}s")

    if resp.status_code != 200:
        print(f"ERROR: {resp.text}")
        return

    data = resp.json()
    print(f"Success     : {data.get('success')}")
    print(f"Server Time : {data.get('processing_time_ms', 0):.2f}ms")
    print()

    resume = data.get("data", {})

    print("-" * 40)
    print("CONTACT INFO")
    print("-" * 40)
    print(f"  Name     : {resume.get('name')}")
    print(f"  Email    : {resume.get('email')}")
    print(f"  Phone    : {resume.get('phone')}")
    print(f"  Alt Phone: {resume.get('number')}")
    print(f"  Location : {resume.get('current_location')}")
    print()

    print("-" * 40)
    print("SKILLS")
    print("-" * 40)
    skills = resume.get("skills", [])
    if skills:
        for i, skill in enumerate(skills, 1):
            print(f"  {i:2d}. {skill}")
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("EXPERIENCE")
    print("-" * 40)
    experience = resume.get("experience", [])
    if experience:
        for i, exp in enumerate(experience, 1):
            print(f"  [{i}] {exp.get('title', 'N/A')} at {exp.get('company', 'N/A')}")
            print(f"      Duration: {exp.get('duration', 'N/A')}")
            desc = exp.get("description", "")
            if desc:
                print(f"      {desc[:120]}{'...' if len(desc) > 120 else ''}")
            print()
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("EDUCATION")
    print("-" * 40)
    education = resume.get("education", [])
    if education:
        for i, edu in enumerate(education, 1):
            print(f"  [{i}] {edu.get('degree', 'N/A')} in {edu.get('field_of_study', 'N/A')}")
            print(f"      {edu.get('institution', 'N/A')} ({edu.get('year', 'N/A')})")
            if edu.get("grade"):
                print(f"      Grade: {edu.get('grade')}")
            print()
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("CERTIFICATIONS")
    print("-" * 40)
    certs = resume.get("certifications", [])
    if certs:
        for i, c in enumerate(certs, 1):
            issuer = f" ({c.get('issuer')})" if c.get("issuer") else ""
            print(f"  {i:2d}. {c.get('name', 'N/A')}{issuer}")
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("PROJECTS")
    print("-" * 40)
    projects = resume.get("projects", [])
    if projects:
        for i, p in enumerate(projects, 1):
            dur = f" [{p.get('duration')}]" if p.get("duration") else ""
            print(f"  [{i}] {p.get('name', 'N/A')}{dur}")
            desc = p.get("description", "")
            if desc:
                print(f"      {desc[:120]}{'...' if len(str(desc)) > 120 else ''}")
            print()
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("AWARDS")
    print("-" * 40)
    awards = resume.get("awards", [])
    if awards:
        for i, a in enumerate(awards, 1):
            yr = f" ({a.get('year')})" if a.get("year") else ""
            print(f"  {i:2d}. {a.get('name', 'N/A')}{yr}")
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("SUMMARY")
    print("-" * 40)
    summary = resume.get("summary", "")
    if summary:
        print(f"  {summary}")
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("RESUME SCORE")
    print("-" * 40)
    score = resume.get("resume_score", {})
    if score:
        print(f"  Overall    : {score.get('overall', 'N/A')}/100")
        print(f"  Content    : {score.get('content', 'N/A')}/100")
        print(f"  Experience : {score.get('experience_relevance', 'N/A')}/100")
        print(f"  Skills     : {score.get('skills_match', 'N/A')}/100")
        print(f"  Education  : {score.get('education', 'N/A')}/100")
        if score.get("remarks"):
            print(f"  Remarks    : {score.get('remarks')}")
    else:
        print("  (none extracted)")
    print()

    print("-" * 40)
    print("RAW JSON RESPONSE")
    print("-" * 40)
    print(json.dumps(data, indent=2))
    print()

    assert data.get("success"), "Parse was not successful"
    print("PASSED")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_parse_resume.py <path_to_resume.pdf>")
        sys.exit(1)

    resume_path = sys.argv[1]

    test_health()
    test_models()
    test_auth_rejection()
    test_parse_resume(resume_path)

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
