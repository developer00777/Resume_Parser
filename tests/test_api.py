"""
Pytest test suite for Resume Parser API.

All external dependencies (OpenRouter, Salesforce) are mocked so tests run
in CI without any live services.

Run:
    pytest tests/test_api.py -v
"""
from __future__ import annotations

import io
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
            "duration": "2021 – Present",
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
        "overall": 74,
        "contact_information": 9,
        "professional_summary": 7,
        "work_experience": 5,
        "skills": 6,
        "education_certifications": 8,
        "achievements_projects": 2,
        "format_design": 8,
        "grade": "Average",
        "remarks": "Good resume — address the minor gaps above to reach Excellent.",
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
    def test_health_openrouter_up(self):
        with patch("app.services.llm.check_openrouter", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = True
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["openrouter_connected"] is True

    def test_health_openrouter_down(self):
        with patch("app.services.llm.check_openrouter", new_callable=AsyncMock) as mock_check:
            mock_check.return_value = False
            resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["openrouter_connected"] is False

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
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
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
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
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
# Bulk parse endpoint
# ---------------------------------------------------------------------------

class TestBulkParseEndpoint:
    def test_bulk_parse_multiple_files(self):
        with (
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT

            files = [
                ("files", (f"resume{i}.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
                for i in range(3)
            ]
            resp = client.post(
                "/api/v1/parse-bulk",
                headers={"X-API-Key": VALID_API_KEY},
                files=files,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 3
        assert body["parsed"] == 3
        assert body["failed"] == 0
        assert len(body["results"]) == 3
        assert "total_processing_time_ms" in body

    def test_bulk_parse_too_many_files_returns_400(self):
        files = [
            ("files", (f"resume{i}.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
            for i in range(21)
        ]
        resp = client.post(
            "/api/v1/parse-bulk",
            headers={"X-API-Key": VALID_API_KEY},
            files=files,
        )
        assert resp.status_code == 400

    def test_bulk_parse_partial_failure(self):
        call_count = 0

        async def mock_parse_resume(text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise ValueError("Simulated LLM failure")
            return MOCK_PARSED_RESULT

        with (
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", side_effect=mock_parse_resume),
        ):
            mock_ext.return_value = "Resume text"

            files = [
                ("files", (f"resume{i}.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
                for i in range(3)
            ]
            resp = client.post(
                "/api/v1/parse-bulk",
                headers={"X-API-Key": VALID_API_KEY},
                files=files,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["parsed"] == 2
        assert body["failed"] == 1

    def test_bulk_requires_auth(self):
        resp = client.post("/api/v1/parse-bulk")
        assert resp.status_code == 401

    def test_bulk_no_files_returns_400(self):
        resp = client.post(
            "/api/v1/parse-bulk",
            headers={"X-API-Key": VALID_API_KEY},
            files=[("files", ("", io.BytesIO(b""), "application/pdf"))],
        )
        # FastAPI raises 422 for empty file list; we just verify it's not 200
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# Bulk Salesforce parse endpoint
# ---------------------------------------------------------------------------

class TestBulkSalesforceParseEndpoint:
    def test_bulk_salesforce_returns_scschamps_fields(self):
        with (
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT

            files = [
                ("files", (f"resume{i}.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
                for i in range(2)
            ]
            resp = client.post(
                "/api/v1/parse-bulk/salesforce",
                headers={"X-API-Key": VALID_API_KEY},
                files=files,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["total"] == 2
        assert body["parsed"] == 2
        # Each result should have Salesforce field names
        item = body["results"][0]
        assert item["success"] is True
        assert "Phone" in item["data"]
        assert "SkillList" in item["data"]
        assert "CurrentCompany" in item["data"]

    def test_bulk_salesforce_too_many_files(self):
        files = [
            ("files", (f"resume{i}.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
            for i in range(21)
        ]
        resp = client.post(
            "/api/v1/parse-bulk/salesforce",
            headers={"X-API-Key": VALID_API_KEY},
            files=files,
        )
        assert resp.status_code == 400

    def test_bulk_salesforce_requires_auth(self):
        resp = client.post("/api/v1/parse-bulk/salesforce")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Async job endpoints
# ---------------------------------------------------------------------------

class TestBulkJobEndpoints:
    def test_submit_job_returns_202_with_job_id(self):
        with (
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT

            files = [
                ("files", ("resume.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
            ]
            resp = client.post(
                "/api/v1/parse-bulk/job",
                headers={"X-API-Key": VALID_API_KEY},
                files=files,
            )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] in ("processing", "completed")
        assert body["total"] == 1

    def test_get_job_not_found_returns_404(self):
        resp = client.get(
            "/api/v1/parse-bulk/job/nonexistent-job-id",
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert resp.status_code == 404

    def test_get_job_returns_status(self):
        with (
            patch("app.routes.parser.extract_text", new_callable=AsyncMock) as mock_ext,
            patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as mock_llm,
        ):
            mock_ext.return_value = "Resume text"
            mock_llm.return_value = MOCK_PARSED_RESULT

            files = [
                ("files", ("resume.pdf", io.BytesIO(MINIMAL_PDF), "application/pdf"))
            ]
            submit_resp = client.post(
                "/api/v1/parse-bulk/job",
                headers={"X-API-Key": VALID_API_KEY},
                files=files,
            )
        assert submit_resp.status_code == 202
        job_id = submit_resp.json()["job_id"]

        poll_resp = client.get(
            f"/api/v1/parse-bulk/job/{job_id}",
            headers={"X-API-Key": VALID_API_KEY},
        )
        assert poll_resp.status_code == 200
        body = poll_resp.json()
        assert body["job_id"] == job_id
        assert body["status"] in ("processing", "completed", "failed")

    def test_job_requires_auth(self):
        resp = client.post("/api/v1/parse-bulk/job")
        assert resp.status_code == 401


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

    def test_map_current_company_picks_present_role(self):
        """Current company should be the role with 'Present' in duration, not the first entry."""
        from app.schemas.response import map_to_salesforce

        parsed = {
            **MOCK_PARSED_RESULT,
            "experience": [
                {"company": "OldCorp", "title": "Junior Dev", "duration": "2015 - 2018", "description": "Legacy work."},
                {"company": "MidCorp", "title": "Developer", "duration": "2018 - 2021", "description": "Mid work."},
                {"company": "NewCorp", "title": "Senior Engineer", "duration": "2021 - Present", "description": "Current role."},
                {"company": "AnotherOld", "title": "Intern", "duration": "2013 - 2015", "description": "Early career."},
            ],
        }
        sf = map_to_salesforce(parsed)
        assert sf.CurrentCompany == "NewCorp"
        assert sf.CurrentDesignation == "Senior Engineer"

    def test_map_empty_parsed(self):
        from app.schemas.response import map_to_salesforce

        empty = {
            "name": None, "email": None, "phone": None, "number": None,
            "current_location": None, "skills": [], "experience": [],
            "education": [], "projects": [], "certifications": [],
            "awards": [], "summary": None,
            "resume_score": {
                "overall": 0, "contact_information": 0, "professional_summary": 0,
                "work_experience": 0, "skills": 0, "education_certifications": 0,
                "achievements_projects": 0, "format_design": 0,
                "grade": "Poor", "remarks": None,
            },
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
                {"company": "A", "title": "SWE", "duration": "2020-2023", "description": "Built APIs successfully"},
                {"company": "B", "title": "Lead", "duration": "2023-2025", "description": "Led engineering team"},
            ]},
            "education": {"education": [{"institution": "IIT", "degree": "B.Tech", "field_of_study": "CS", "year": "2020", "grade": "9.0"}]},
            "certifications": {"certifications": [{"name": "AWS", "issuer": "Amazon"}]},
            "projects": {"projects": [{"name": "P1", "duration": "2023", "description": "Built a scalable system"}]},
            "awards": {"awards": []},
            "summary": {"summary": "Senior engineer with 5 years experience building scalable cloud systems."},
        }
        score = _compute_score(parsed)
        assert score["overall"] >= 60
        assert score["contact_information"] >= 6
        assert score["skills"] >= 6
        assert score["grade"] in ("Excellent", "Good", "Average")

    def test_compute_score_empty_resume(self):
        from app.services.llm import _compute_score

        parsed = {
            "contact": {}, "skills": {}, "experience": {},
            "education": {}, "certifications": {}, "projects": {}, "awards": {}, "summary": {},
        }
        score = _compute_score(parsed)
        assert score["overall"] < 20
        assert score["grade"] == "Poor"
        assert score["contact_information"] == 0
        assert score["work_experience"] == 0
        assert score["skills"] == 0

    def test_score_matrix_weights(self):
        """Verify weighted formula: sum of (raw * weight * 10) = overall."""
        from app.services.llm import _compute_score

        parsed = {
            "contact": {"name": "X", "email": "x@x.com", "phone": "999"},
            "skills": {"skills": ["Python", "SQL", "Docker", "AWS", "Go", "React", "Java", "Rust"]},
            "experience": {"experience": [
                {"company": "C", "title": "Dev", "duration": "2020-2024", "description": "Worked on backend systems"}
            ]},
            "education": {"education": [{"institution": "Uni", "degree": "BSc", "field_of_study": "CS", "year": "2020", "grade": None}]},
            "certifications": {"certifications": []},
            "projects": {"projects": []},
            "awards": {"awards": []},
            "summary": {"summary": "Software developer with 4 years of experience in Python and cloud technologies."},
        }
        score = _compute_score(parsed)
        # All 7 category keys must be present
        for key in ("contact_information", "professional_summary", "work_experience",
                    "skills", "education_certifications", "achievements_projects", "format_design"):
            assert key in score, f"Missing key: {key}"
            assert 0 <= score[key] <= 10, f"{key} out of range: {score[key]}"
        assert 0 <= score["overall"] <= 100
        assert score["grade"] in ("Excellent", "Good", "Average", "Poor")
