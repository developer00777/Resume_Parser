"""
HTTP-level tests for the async batch endpoints and for the partial-results
behaviour of the synchronous bulk endpoints.

Redis is mocked here — tests/test_queue.py exercises the real bookkeeping.
"""
from __future__ import annotations

import asyncio
import base64
import io
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)
KEY = {"X-API-Key": settings.api_key}

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


def _files(count=1, name="resume.pdf", content_type="application/pdf"):
    return [
        ("files", (f"{i}-{name}", io.BytesIO(MINIMAL_PDF), content_type))
        for i in range(count)
    ]


@pytest.fixture
def pool():
    """Pretend Redis is connected; queue calls are asserted individually."""
    sentinel = object()
    with patch("app.routes.jobs.queue.get_pool", return_value=sentinel):
        yield sentinel


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------

class TestSubmit:
    def test_submit_returns_202_and_a_batch_id_immediately(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("abc123", 3)
            resp = client.post("/api/v1/parse/salesforce/jobs", headers=KEY, files=_files(3))

        assert resp.status_code == 202
        body = resp.json()
        assert body["batch_id"] == "abc123"
        assert body["status"] == "queued"
        assert body["mode"] == "salesforce"
        assert body["total"] == 3
        assert body["poll_url"] == "/api/v1/parse/jobs/abc123"

    @pytest.mark.parametrize(
        "path,mode",
        [
            ("/api/v1/parse/jobs", "generic"),
            ("/api/v1/parse/salesforce/jobs", "salesforce"),
            ("/api/v1/parse/client/jobs", "client"),
        ],
    )
    def test_each_output_mode_has_a_queue_endpoint(self, pool, path, mode):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post(path, headers=KEY, files=_files(1))

        assert resp.status_code == 202
        assert resp.json()["mode"] == mode
        assert create.await_args.args[1] == mode

    def test_files_are_buffered_before_anything_is_enqueued(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 2)
            client.post("/api/v1/parse/jobs", headers=KEY, files=_files(2))

        buffered = create.await_args.args[2]
        assert len(buffered) == 2
        assert all(entry[2] == MINIMAL_PDF for entry in buffered)

    def test_octet_stream_upload_is_accepted_via_its_extension(self, pool):
        """Salesforce multipart callouts often declare application/octet-stream."""
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post(
                "/api/v1/parse/salesforce/jobs",
                headers=KEY,
                files=_files(1, content_type="application/octet-stream"),
            )

        assert resp.status_code == 202
        assert create.await_args.args[2][0][1] == "application/pdf"

    def test_callback_url_is_passed_through(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post(
                "/api/v1/parse/salesforce/jobs",
                headers=KEY,
                files=_files(1),
                data={"callback_url": "https://example.com/hook", "callback_token": "tok"},
            )

        assert resp.status_code == 202
        assert resp.json()["callback_url"] == "https://example.com/hook"
        assert create.await_args.kwargs["callback_url"] == "https://example.com/hook"
        assert create.await_args.kwargs["callback_token"] == "tok"

    def test_callback_url_outside_the_allowlist_is_rejected(self, pool):
        with patch.object(settings, "job_callback_allowed_hosts", "my.salesforce.com"):
            resp = client.post(
                "/api/v1/parse/salesforce/jobs",
                headers=KEY,
                files=_files(1),
                data={"callback_url": "https://evil.example/hook"},
            )
        assert resp.status_code == 400

    def test_unsupported_type_is_rejected_up_front(self, pool):
        resp = client.post(
            "/api/v1/parse/jobs",
            headers=KEY,
            files=[("files", ("notes.txt", io.BytesIO(b"hello"), "text/plain"))],
        )
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_oversized_file_is_rejected_up_front(self, pool):
        with patch.object(settings, "max_file_size", 10):
            resp = client.post("/api/v1/parse/jobs", headers=KEY, files=_files(1))
        assert resp.status_code == 413

    def test_too_many_files_is_rejected(self, pool):
        with patch.object(settings, "queue_max_files", 2):
            resp = client.post("/api/v1/parse/jobs", headers=KEY, files=_files(3))
        assert resp.status_code == 400
        assert "Maximum allowed per batch is 2" in resp.json()["detail"]

    def test_queue_down_returns_503_not_a_hang(self):
        with patch("app.routes.jobs.queue.get_pool", return_value=None):
            resp = client.post("/api/v1/parse/salesforce/jobs", headers=KEY, files=_files(1))
        assert resp.status_code == 503
        assert "Redis is not reachable" in resp.json()["detail"]

    def test_submit_requires_the_api_key(self):
        resp = client.post("/api/v1/parse/salesforce/jobs", files=_files(1))
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Poll / delete
# ---------------------------------------------------------------------------

class TestPoll:
    def _batch(self, **overrides):
        batch = {
            "batch_id": "abc123",
            "status": "processing",
            "mode": "salesforce",
            "total": 3,
            "completed": 1,
            "parsed": 1,
            "failed": 0,
            "remaining": 2,
            "running": 1,
            "created_at": 1000.0,
            "updated_at": 1010.0,
            "finished_at": None,
            "total_processing_time_ms": 10000.0,
            "results": [
                {"index": 0, "filename": "a.pdf", "success": True,
                 "data": {"Name": "Jane"}, "error": None, "processing_time_ms": 900.0}
            ],
        }
        batch.update(overrides)
        return batch

    def test_poll_returns_progress_and_results_so_far(self, pool):
        with patch("app.routes.jobs.queue.get_batch", new_callable=AsyncMock) as get:
            get.return_value = self._batch()
            resp = client.get("/api/v1/parse/jobs/abc123", headers=KEY)

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "processing"
        assert body["remaining"] == 2
        assert body["results"][0]["data"] == {"Name": "Jane"}

    def test_unknown_batch_is_404(self, pool):
        with patch("app.routes.jobs.queue.get_batch", new_callable=AsyncMock) as get:
            get.return_value = None
            resp = client.get("/api/v1/parse/jobs/nope", headers=KEY)
        assert resp.status_code == 404

    def test_delete_returns_204(self, pool):
        with patch("app.routes.jobs.queue.delete_batch", new_callable=AsyncMock) as delete:
            delete.return_value = True
            resp = client.delete("/api/v1/parse/jobs/abc123", headers=KEY)
        assert resp.status_code == 204

    def test_delete_unknown_batch_is_404(self, pool):
        with patch("app.routes.jobs.queue.delete_batch", new_callable=AsyncMock) as delete:
            delete.return_value = False
            resp = client.delete("/api/v1/parse/jobs/nope", headers=KEY)
        assert resp.status_code == 404

    def test_poll_requires_the_api_key(self):
        resp = client.get("/api/v1/parse/jobs/abc123")
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Synchronous endpoints must degrade to partial results, never 504
# ---------------------------------------------------------------------------

class TestSyncPartialResults:
    """
    The regression this whole change exists for: a bulk request that outruns the
    wall-clock budget used to answer 504 and bin every resume that had parsed.
    """

    def _post(self, path, count=3):
        return client.post(path, headers=KEY, files=_files(count))

    @pytest.mark.parametrize(
        "path", ["/api/v1/parse", "/api/v1/parse/salesforce", "/api/v1/parse/client"]
    )
    def test_slow_batch_returns_200_with_what_finished(self, path):
        calls = {"n": 0}

        async def one_fast_then_hang(_text):
            calls["n"] += 1
            if calls["n"] > 1:
                await asyncio.sleep(30)
            return {"name": "Jane Doe", "email": "jane@example.com"}

        with patch("app.routes.parser.parse_resume", side_effect=one_fast_then_hang), \
             patch("app.routes.parser.extract_text", new_callable=AsyncMock) as extract, \
             patch.object(settings, "bulk_timeout", 0.5), \
             patch.object(settings, "bulk_concurrency", 5):
            extract.return_value = "Jane Doe, Python developer"
            resp = self._post(path, count=3)

        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["total"] == 3
        assert body["parsed"] == 1
        assert body["failed"] == 2
        # The one that finished is intact...
        assert body["results"][0]["success"] is True
        assert body["results"][0]["data"] is not None
        # ...and the rest say what to do about it.
        timed_out = body["results"][1]
        assert timed_out["success"] is False
        assert "Timed out after 0s" in timed_out["error"]
        assert "/jobs" in timed_out["error"]

    def test_fast_batch_is_unaffected(self):
        with patch("app.routes.parser.parse_resume", new_callable=AsyncMock) as parse, \
             patch("app.routes.parser.extract_text", new_callable=AsyncMock) as extract:
            extract.return_value = "Jane Doe"
            parse.return_value = {"name": "Jane Doe"}
            resp = self._post("/api/v1/parse/salesforce", count=2)

        assert resp.status_code == 200
        body = resp.json()
        assert (body["parsed"], body["failed"]) == (2, 0)

    def test_per_file_failure_still_only_fails_that_file(self):
        async def fail_second(text):
            if "second" in text:
                raise ValueError("bad resume")
            return {"name": "Jane Doe"}

        with patch("app.routes.parser.parse_resume", side_effect=fail_second), \
             patch("app.routes.parser.extract_text", new_callable=AsyncMock) as extract:
            extract.side_effect = ["first resume", "second resume"]
            resp = self._post("/api/v1/parse/salesforce", count=2)

        body = resp.json()
        assert (body["parsed"], body["failed"]) == (1, 1)
        assert "bad resume" in body["results"][1]["error"]


# ---------------------------------------------------------------------------
# JSON (base64) submit — the Apex-friendly transport
# ---------------------------------------------------------------------------

class TestSubmitBase64:
    def _body(self, count=1, filename="resume.pdf", **extra):
        payload = {
            "files": [
                {
                    "filename": f"{i}-{filename}",
                    "content_base64": base64.b64encode(MINIMAL_PDF).decode(),
                }
                for i in range(count)
            ]
        }
        payload.update(extra)
        return payload

    def test_json_submit_returns_202(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("b64batch", 2)
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/base64", headers=KEY, json=self._body(2)
            )

        assert resp.status_code == 202
        body = resp.json()
        assert body["batch_id"] == "b64batch"
        assert body["mode"] == "salesforce"
        assert body["poll_url"] == "/api/v1/parse/jobs/b64batch"

    @pytest.mark.parametrize(
        "path,mode",
        [
            ("/api/v1/parse/jobs/base64", "generic"),
            ("/api/v1/parse/salesforce/jobs/base64", "salesforce"),
            ("/api/v1/parse/client/jobs/base64", "client"),
        ],
    )
    def test_every_mode_has_a_json_twin(self, pool, path, mode):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post(path, headers=KEY, json=self._body(1))

        assert resp.status_code == 202
        assert create.await_args.args[1] == mode

    def test_bytes_survive_the_round_trip(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            client.post("/api/v1/parse/jobs/base64", headers=KEY, json=self._body(1))

        filename, content_type, content, _external_id = create.await_args.args[2][0]
        assert content == MINIMAL_PDF
        # Apex need not send a MIME type; the extension settles it.
        assert content_type == "application/pdf"
        assert filename == "0-resume.pdf"

    def test_explicit_content_type_is_honoured(self, pool):
        body = {
            "files": [{
                "filename": "cv.docx",
                "content_base64": base64.b64encode(b"docx-bytes").decode(),
                "content_type": (
                    "application/vnd.openxmlformats-officedocument"
                    ".wordprocessingml.document"
                ),
            }]
        }
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json=body)

        assert resp.status_code == 202
        assert create.await_args.args[2][0][1].endswith("wordprocessingml.document")

    def test_callback_fields_are_passed_through(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/base64",
                headers=KEY,
                json=self._body(
                    1,
                    callback_url="https://example.com/hook",
                    callback_token="tok",
                ),
            )

        assert resp.json()["callback_url"] == "https://example.com/hook"
        assert create.await_args.kwargs["callback_token"] == "tok"

    def test_garbage_base64_is_a_clear_400(self, pool):
        body = {"files": [{"filename": "cv.pdf", "content_base64": "!!!not base64!!!"}]}
        resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json=body)
        assert resp.status_code == 400
        assert "not valid base64" in resp.json()["detail"]

    def test_empty_file_list_is_rejected(self, pool):
        resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json={"files": []})
        assert resp.status_code == 400

    def test_unsupported_extension_is_rejected(self, pool):
        body = {"files": [{
            "filename": "notes.txt",
            "content_base64": base64.b64encode(b"hello").decode(),
        }]}
        resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json=body)
        assert resp.status_code == 400
        assert "Unsupported file type" in resp.json()["detail"]

    def test_oversized_file_is_rejected(self, pool):
        with patch.object(settings, "max_file_size", 10):
            resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json=self._body(1))
        assert resp.status_code == 413

    def test_too_many_files_is_rejected(self, pool):
        with patch.object(settings, "queue_max_files", 2):
            resp = client.post("/api/v1/parse/jobs/base64", headers=KEY, json=self._body(3))
        assert resp.status_code == 400

    def test_json_submit_requires_the_api_key(self):
        resp = client.post("/api/v1/parse/jobs/base64", json=self._body(1))
        assert resp.status_code == 401

    def test_queue_down_returns_503(self):
        with patch("app.routes.jobs.queue.get_pool", return_value=None):
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/base64", headers=KEY, json=self._body(1)
            )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Submit by record id — the bulk path that never touches the Apex heap
# ---------------------------------------------------------------------------

class TestSubmitRecords:
    def test_record_submit_returns_202(self, pool):
        with patch(
            "app.routes.jobs.queue.create_reference_batch", new_callable=AsyncMock
        ) as create:
            create.return_value = ("recbatch", 2)
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [{"ref": "068AAA"}, {"ref": "068BBB", "kind": "attachment"}]},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert (body["batch_id"], body["total"], body["mode"]) == ("recbatch", 2, "salesforce")

    def test_kind_defaults_to_attachment(self, pool):
        with patch(
            "app.routes.jobs.queue.create_reference_batch", new_callable=AsyncMock
        ) as create:
            create.return_value = ("id", 1)
            client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [{"ref": "068AAA"}]},
            )

        assert create.await_args.args[2] == [("attachment", "068AAA", None, None)]

    def test_candidate_and_url_kinds_are_accepted(self, pool):
        with patch(
            "app.routes.jobs.queue.create_reference_batch", new_callable=AsyncMock
        ) as create:
            create.return_value = ("id", 2)
            resp = client.post(
                "/api/v1/parse/jobs/records",
                headers=KEY,
                json={"records": [
                    {"ref": "a0X111", "kind": "candidate"},
                    {"ref": "https://example.com/cv.pdf", "kind": "url", "filename": "cv.pdf"},
                ]},
            )

        assert resp.status_code == 202
        assert create.await_args.args[2] == [
            ("candidate", "a0X111", None, None),
            ("url", "https://example.com/cv.pdf", "cv.pdf", None),
        ]

    def test_unknown_kind_is_rejected(self, pool):
        resp = client.post(
            "/api/v1/parse/salesforce/jobs/records",
            headers=KEY,
            json={"records": [{"ref": "068AAA", "kind": "carrier-pigeon"}]},
        )
        assert resp.status_code == 400
        assert "Unknown kind" in resp.json()["detail"]

    def test_empty_record_list_is_rejected(self, pool):
        resp = client.post(
            "/api/v1/parse/salesforce/jobs/records", headers=KEY, json={"records": []}
        )
        assert resp.status_code == 400

    def test_too_many_records_is_rejected(self, pool):
        with patch.object(settings, "queue_max_files", 2):
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [{"ref": f"068{i}"} for i in range(3)]},
            )
        assert resp.status_code == 400

    def test_callback_is_passed_through(self, pool):
        with patch(
            "app.routes.jobs.queue.create_reference_batch", new_callable=AsyncMock
        ) as create:
            create.return_value = ("id", 1)
            client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [{"ref": "068AAA"}],
                      "callback_url": "https://example.com/hook",
                      "callback_token": "tok"},
            )

        assert create.await_args.kwargs["callback_url"] == "https://example.com/hook"
        assert create.await_args.kwargs["callback_token"] == "tok"

    def test_record_submit_requires_the_api_key(self):
        resp = client.post(
            "/api/v1/parse/salesforce/jobs/records", json={"records": [{"ref": "068AAA"}]}
        )
        assert resp.status_code == 401

    def test_queue_down_returns_503(self):
        with patch("app.routes.jobs.queue.get_pool", return_value=None):
            resp = client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [{"ref": "068AAA"}]},
            )
        assert resp.status_code == 503


class TestExternalIdCorrelation:
    """
    Apex needs to tie each result back to its own record. Filenames collide and
    ordering is fragile, so the caller's own key is echoed through untouched.
    """

    def test_record_submit_carries_external_id(self, pool):
        with patch(
            "app.routes.jobs.queue.create_reference_batch", new_callable=AsyncMock
        ) as create:
            create.return_value = ("id", 1)
            client.post(
                "/api/v1/parse/salesforce/jobs/records",
                headers=KEY,
                json={"records": [
                    {"ref": "068AAA", "external_id": "a0X1000000CandId"}
                ]},
            )

        assert create.await_args.args[2] == [
            ("attachment", "068AAA", None, "a0X1000000CandId")
        ]

    def test_base64_submit_carries_external_id(self, pool):
        with patch("app.routes.jobs.queue.create_batch", new_callable=AsyncMock) as create:
            create.return_value = ("id", 1)
            client.post(
                "/api/v1/parse/jobs/base64",
                headers=KEY,
                json={"files": [{
                    "filename": "cv.pdf",
                    "content_base64": base64.b64encode(MINIMAL_PDF).decode(),
                    "external_id": "a0X1000000CandId",
                }]},
            )

        assert create.await_args.args[2][0][3] == "a0X1000000CandId"

    def test_poll_exposes_external_id_on_each_result(self, pool):
        batch = {
            "batch_id": "b", "status": "completed", "mode": "salesforce",
            "total": 1, "completed": 1, "parsed": 1, "failed": 0,
            "remaining": 0, "running": 0,
            "created_at": 1.0, "updated_at": 2.0, "finished_at": 2.0,
            "total_processing_time_ms": 1000.0,
            "results": [{
                "index": 0, "filename": "cv.pdf", "external_id": "a0X1000000CandId",
                "success": True, "data": {"Name": "Jane"}, "error": None,
                "processing_time_ms": 900.0,
            }],
        }
        with patch("app.routes.jobs.queue.get_batch", new_callable=AsyncMock) as get:
            get.return_value = batch
            resp = client.get("/api/v1/parse/jobs/b", headers=KEY)

        assert resp.json()["results"][0]["external_id"] == "a0X1000000CandId"
