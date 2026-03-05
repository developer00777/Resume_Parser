"""
Pytest test suite for Resume Parser API.

All external dependencies (Ollama, Salesforce) are mocked so tests run
in CI without any live services.

Run:
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

client = TestClient(app)

VALID_API_KEY = settings.api_key

MOCK_PARSED_RESULT = {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+911234567890",
    "number": None,
    "current_location": "Mumbai, Maharashtra",
    "skills": ["Python", "FastAPI", "Salesforce", "Docker"],
    "experience": [
        {
            "company": "Acme Corp",
            "title": "Software Engineer",
            "duration": "2021 – 2024",
            "description": "Built microservices and REST APIs.",
        }
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field_of_study": "Computer Science",
            "year": "2021",
            "grade": "8.9",
        }
    ],
    "projects": [],
    "certifications": [{"name": "AWS Solutions Architect", "issuer": "Amazon"}],
    "awards": [],
    "summary": "Experienced software engineer with 3 years in Python and cloud.",
    "resume_score": {
        "overall": 82,
        "content": 85,
        "experience_relevance": 80,
        "skills_match": 90,
        "education": 75,
        "remarks": "Well-structured resume.",
    },
}

# Minimal valid PDF bytes (1-page blank PDF)
MINIMAL_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
    b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
    b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
    b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
    b"4 0 obj<</Length 44>>\nstream\nBT /F1 12 Tf 100 700 Td (Test Resume) Tj ET\nendstream\nendobj\n"
    b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
    b"xref\n0 6\n0000000000 65535 f \n"
    b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n0\n%%EOF"
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealth:
    def test_health_ollama_up(self):
        with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value.__aenter__ = AsyncMock(return_value=mock_get.return_value)
            mock_get.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_get.return_value.status_code = 200
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("healthy", "degraded")
        assert "ollama_connected" in body

    def test_health_no_auth_required(self):
        resp = client.get("/health")
        # Should not return 401
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

class TestAuth:
    def test_missing_key_returns_401(self):
        resp = client.post("/api/v1/parse")
        assert resp.status_code == 401

    def test_wrong_key_returns_401(self):
        resp = client.post(
            "/api/v1/parse",
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 401

    def test_docs_no_auth(self):
        resp = client.get("/docs")
        assert resp.status_code == 200

    def test_redoc_no_auth(self):
        resp = client.get("/redoc")
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Parse endpoint (web-app mode)
# ---------------------------------------------------------------------------

class TestParseEndpoint:
    def _post_pdf(self, pdf_bytes: bytes = MINIMAL_PDF):
        return client.post(
            "/api/v1/parse",
            headers={"X-API-Key": VALID_API_KEY},
            files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

    def test_parse_returns_200_with_mocked_llm(self):
        with (
            patch("app.services.document.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.services.llm.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Jane Doe\njane@example.com\nPython, FastAPI"
            mock_llm.return_value = MOCK_PARSED_RESULT
            resp = self._post_pdf()

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["data"]["name"] == "Jane Doe"
        assert body["data"]["email"] == "jane@example.com"
        assert "Python" in body["data"]["skills"]
        assert "processing_time_ms" in body

    def test_parse_wrong_file_type_returns_400(self):
        resp = client.post(
            "/api/v1/parse",
            headers={"X-API-Key": VALID_API_KEY},
            files={"file": ("resume.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_parse_no_file_returns_422(self):
        resp = client.post(
            "/api/v1/parse",
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 422

    def test_parse_docx_accepted(self):
        with (
            patch("app.services.document.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.services.llm.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT
            resp = client.post(
                "/api/v1/parse",
                headers={"X-API-Key": VALID_API_KEY},
                files={
                    "file": (
                        "resume.docx",
                        io.BytesIO(b"PK fake docx"),
                        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                },
            )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Models endpoint
# ---------------------------------------------------------------------------

class TestModelsEndpoint:
    def test_models_returns_list(self):
        with patch("app.services.llm.list_models", new_callable=AsyncMock) as mock_models:
            mock_models.return_value = [{"name": "qwen2.5:3b", "size": 2000000, "modified_at": "2024-01-01"}]
            resp = client.get("/api/v1/models", headers={"X-API-Key": VALID_API_KEY})

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert isinstance(body["models"], list)
        assert body["models"][0]["name"] == "qwen2.5:3b"


# ---------------------------------------------------------------------------
# Salesforce parse-candidate endpoint
# ---------------------------------------------------------------------------

class TestSalesforceEndpoint:
    def test_parse_candidate_mocked(self):
        with (
            patch(
                "app.routes.salesforce.fetch_resume_from_candidate",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("app.routes.salesforce.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.salesforce.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_fetch.return_value = (b"%PDF fake", "resume.pdf")
            mock_ext.return_value = "Jane Doe\njane@example.com"
            mock_llm.return_value = MOCK_PARSED_RESULT

            resp = client.post(
                "/api/v1/salesforce/parse-candidate",
                headers={"X-API-Key": VALID_API_KEY},
                params={"record_id": "a0012345678ABCDE"},
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        sf = body["data"]
        # Should have Salesforce field names
        assert "Phone" in sf
        assert "Current_Location" in sf
        assert "SkillList" in sf
        assert "ResumeRich" in sf

    def test_parse_attachment_mocked(self):
        with (
            patch(
                "app.routes.salesforce.fetch_resume_by_attachment_id",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("app.routes.salesforce.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.salesforce.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_fetch.return_value = (b"%PDF fake", "resume.pdf")
            mock_ext.return_value = "Jane Doe resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT

            resp = client.post(
                "/api/v1/salesforce/parse-attachment",
                headers={"X-API-Key": VALID_API_KEY},
                params={"attachment_id": "068XXXXXXXXXXXX"},
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_parse_url_mocked(self):
        with (
            patch(
                "app.routes.salesforce.fetch_resume_by_url",
                new_callable=AsyncMock,
            ) as mock_fetch,
            patch("app.routes.salesforce.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.salesforce.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_fetch.return_value = (b"%PDF fake", "resume.pdf")
            mock_ext.return_value = "Jane Doe resume"
            mock_llm.return_value = MOCK_PARSED_RESULT

            resp = client.post(
                "/api/v1/salesforce/parse-url",
                headers={"X-API-Key": VALID_API_KEY},
                params={"resume_url": "https://example.my.salesforce.com/resume.pdf"},
            )

        assert resp.status_code == 200
        assert resp.json()["success"] is True


# ---------------------------------------------------------------------------
# Schema / mapping unit tests
# ---------------------------------------------------------------------------

class TestSalesforceMapping:
    def test_map_to_salesforce_fields(self):
        from app.schemas.response import map_to_salesforce

        sf = map_to_salesforce(MOCK_PARSED_RESULT)

        assert sf.Phone == "+911234567890"
        assert sf.Current_Location == "Mumbai, Maharashtra"
        assert sf.City == "Mumbai"
        assert sf.State == "Maharashtra"
        assert sf.CurrentCompany == "Acme Corp"
        assert sf.CurrentDesignation == "Software Engineer"
        assert sf.SkillList == "Python, FastAPI, Salesforce, Docker"
        assert sf.AutoPopulate_Skillset is True
        assert "Acme Corp" in (sf.ResumeRich or "")

    def test_map_empty_parsed(self):
        from app.schemas.response import map_to_salesforce

        empty = {
            "name": None, "email": None, "phone": None, "number": None,
            "current_location": None, "skills": [], "experience": [],
            "education": [], "projects": [], "certifications": [],
            "awards": [], "summary": None,
            "resume_score": {"overall": 0, "content": 0, "experience_relevance": 0,
                             "skills_match": 0, "education": 0, "remarks": None},
        }
        sf = map_to_salesforce(empty)
        assert sf.Phone is None
        assert sf.SkillList is None
        assert sf.AutoPopulate_Skillset is False


# ---------------------------------------------------------------------------
# Score computation unit tests
# ---------------------------------------------------------------------------

class TestScoreComputation:
    def test_compute_score_full_resume(self):
        from app.services.llm import _compute_score

        parsed = {
            "contact": {"name": "Jane", "email": "jane@example.com", "phone": "123", "current_location": "Mumbai"},
            "skills": {"skills": ["Python", "Django", "Docker", "AWS", "SQL", "React", "Go", "Rust", "Java", "Kubernetes"]},
            "experience": {"experience": [
                {"company": "A", "title": "SWE", "duration": "2020-2023", "description": "Built APIs"},
                {"company": "B", "title": "Lead", "duration": "2023-2025", "description": "Led team"},
            ]},
            "education": {"education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS", "year": "2020", "grade": "9.0"}]},
            "certifications": {"certifications": [{"name": "AWS", "issuer": "Amazon"}]},
            "projects": {"projects": []},
            "summary": {"summary": "Senior engineer."},
        }
        score = _compute_score(parsed)
        assert score["overall"] >= 70
        assert score["content"] >= 70
        assert score["skills_match"] >= 75

    def test_compute_score_empty_resume(self):
        from app.services.llm import _compute_score

        parsed = {
            "contact": {}, "skills": {}, "experience": {},
            "education": {}, "certifications": {}, "projects": {}, "summary": {},
        }
        score = _compute_score(parsed)
        assert score["overall"] == 0
        assert "No skills" in score["remarks"] or "No work experience" in score["remarks"] or "Missing" in score["remarks"]
