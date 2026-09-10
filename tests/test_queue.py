"""
Batch-store and worker tests, run against fakeredis — no live Redis needed.

These cover the bookkeeping that makes the queue safe to poll: result rows land
independently, `remaining` only reaches zero once, the completion callback can
be claimed exactly once, and a batch abandoned by a dead worker still reaches a
terminal state.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from app.config import settings
from app.services import queue

PDF = "application/pdf"


@pytest.fixture
async def pool():
    """An ArqRedis speaking to an in-process fake server."""
    import fakeredis.aioredis
    import redis.asyncio
    from arq.connections import ArqRedis

    server = fakeredis.FakeServer()
    connection_pool = redis.asyncio.ConnectionPool(
        connection_class=fakeredis.aioredis.FakeConnection, server=server
    )
    client = ArqRedis(connection_pool=connection_pool)
    yield client
    await client.aclose()


async def _make_batch(pool, mode="salesforce", count=2, **kwargs):
    files = [(f"cv{i}.pdf", PDF, f"bytes-{i}".encode()) for i in range(count)]
    return await queue.create_batch(pool, mode, files, **kwargs)


# ---------------------------------------------------------------------------
# Batch lifecycle
# ---------------------------------------------------------------------------

class TestBatchLifecycle:
    async def test_create_batch_stores_files_and_enqueues_one_job_per_file(self, pool):
        batch_id, total = await _make_batch(pool, count=3)

        assert total == 3
        assert await pool.zcard(settings.queue_name) == 3

        for index in range(3):
            entry = await queue.load_file(pool, batch_id, index)
            assert entry["filename"] == f"cv{index}.pdf"
            assert entry["content_type"] == PDF
            assert entry["content"] == f"bytes-{index}".encode()
            assert entry["source"] == queue.SOURCE_INLINE

    async def test_fresh_batch_reports_queued(self, pool):
        batch_id, _ = await _make_batch(pool, count=2)

        batch = await queue.get_batch(pool, batch_id)

        assert batch["status"] == "queued"
        assert (batch["total"], batch["remaining"], batch["completed"]) == (2, 2, 0)
        assert batch["mode"] == "salesforce"
        assert batch["results"] == []

    async def test_partial_progress_reports_processing_with_results_so_far(self, pool):
        batch_id, _ = await _make_batch(pool, count=2)
        await queue.mark_running(pool, batch_id, 1)

        remaining = await queue.record_result(
            pool, batch_id, 0,
            {"index": 0, "filename": "cv0.pdf", "success": True, "data": {"Name": "Jane"},
             "error": None, "processing_time_ms": 1200.0},
        )

        assert remaining == 1
        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "processing"
        assert (batch["completed"], batch["parsed"], batch["failed"]) == (1, 1, 0)
        # A caller can consume finished resumes without waiting for the batch.
        assert batch["results"][0]["data"] == {"Name": "Jane"}

    async def test_batch_completes_once_every_file_reports(self, pool):
        batch_id, _ = await _make_batch(pool, count=2)

        for index in range(2):
            await queue.mark_running(pool, batch_id, 1)
            await queue.record_result(
                pool, batch_id, index,
                {"index": index, "filename": f"cv{index}.pdf", "success": index == 0,
                 "data": None, "error": None if index == 0 else "boom",
                 "processing_time_ms": 10.0},
            )

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert (batch["remaining"], batch["parsed"], batch["failed"]) == (0, 1, 1)
        assert batch["finished_at"] is not None
        assert [r["index"] for r in batch["results"]] == [0, 1]

    async def test_recording_a_result_frees_the_stored_bytes(self, pool):
        batch_id, _ = await _make_batch(pool, count=1)

        await queue.record_result(
            pool, batch_id, 0,
            {"index": 0, "filename": "cv0.pdf", "success": True, "data": {},
             "error": None, "processing_time_ms": 1.0},
            was_running=False,
        )

        assert await queue.load_file(pool, batch_id, 0) is None

    async def test_unknown_batch_is_none(self, pool):
        assert await queue.get_batch(pool, "does-not-exist") is None

    async def test_delete_batch_removes_everything(self, pool):
        batch_id, _ = await _make_batch(pool, count=2)

        assert await queue.delete_batch(pool, batch_id) is True
        assert await queue.get_batch(pool, batch_id) is None
        assert await queue.delete_batch(pool, batch_id) is False


# ---------------------------------------------------------------------------
# Completion callback
# ---------------------------------------------------------------------------

class TestCallbackClaim:
    async def test_claim_succeeds_once_and_only_once(self, pool):
        batch_id, _ = await _make_batch(
            pool, count=1, callback_url="https://example.com/hook", callback_token="tok"
        )

        assert await queue.claim_callback(pool, batch_id) == ("https://example.com/hook", "tok")
        # A retried job must not deliver the same batch twice.
        assert await queue.claim_callback(pool, batch_id) is None

    async def test_no_callback_url_means_no_claim(self, pool):
        batch_id, _ = await _make_batch(pool, count=1)
        assert await queue.claim_callback(pool, batch_id) is None

    def test_allowlist_permits_any_host_when_unset(self):
        with patch.object(settings, "job_callback_allowed_hosts", ""):
            assert queue.callback_url_allowed("https://anything.example/hook") is True

    def test_allowlist_matches_host_and_subdomains(self):
        with patch.object(settings, "job_callback_allowed_hosts", "my.salesforce.com"):
            assert queue.callback_url_allowed("https://my.salesforce.com/x") is True
            assert queue.callback_url_allowed("https://eu5.my.salesforce.com/x") is True
            assert queue.callback_url_allowed("https://evil.example/x") is False


# ---------------------------------------------------------------------------
# Stale-batch reaping
# ---------------------------------------------------------------------------

class TestStaleReaping:
    async def test_abandoned_batch_reaches_a_terminal_state(self, pool):
        """A worker killed mid-job must not leave a poller hanging forever."""
        batch_id, _ = await _make_batch(pool, count=2)
        # One file finished; then the worker died without reporting the other.
        await queue.record_result(
            pool, batch_id, 0,
            {"index": 0, "filename": "cv0.pdf", "success": True, "data": {},
             "error": None, "processing_time_ms": 5.0},
            was_running=False,
        )
        await pool.hset(queue._k_batch(batch_id), "updated_at", repr(time.time() - 10_000))

        batch = await queue.get_batch(pool, batch_id)

        assert batch["status"] == "completed"
        assert (batch["remaining"], batch["parsed"], batch["failed"]) == (0, 1, 1)
        assert "No worker reported this file" in batch["results"][1]["error"]

    async def test_recent_batch_is_not_reaped(self, pool):
        batch_id, _ = await _make_batch(pool, count=2)

        batch = await queue.get_batch(pool, batch_id)

        assert batch["status"] == "queued"
        assert batch["remaining"] == 2


# ---------------------------------------------------------------------------
# Worker task
# ---------------------------------------------------------------------------

class TestWorker:
    async def test_successful_parse_writes_a_mapped_result(self, pool):
        from app import worker

        batch_id, _ = await _make_batch(pool, mode="client", count=1)

        with patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract, \
             patch("app.worker.parse_resume", new_callable=AsyncMock) as parse:
            extract.return_value = "Jane Doe, Python developer"
            parse.return_value = {"name": "Jane Doe", "email": "jane@example.com"}
            await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert batch["results"][0]["success"] is True
        assert batch["results"][0]["data"]["Name"] == "Jane Doe"

    async def test_permanent_failure_is_recorded_not_retried(self, pool):
        from fastapi import HTTPException

        from app import worker

        batch_id, _ = await _make_batch(pool, count=1)

        with patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract:
            extract.side_effect = HTTPException(status_code=422, detail="Failed to parse PDF")
            await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert batch["results"][0]["success"] is False
        assert batch["results"][0]["error"] == "Failed to parse PDF"

    async def test_transient_failure_asks_arq_for_a_retry(self, pool):
        from arq import Retry
        from fastapi import HTTPException

        from app import worker

        batch_id, _ = await _make_batch(pool, count=1)

        with patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract:
            extract.side_effect = HTTPException(status_code=503, detail="OpenRouter down")
            with pytest.raises(Retry):
                await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        # Still outstanding — a retry, not a failure.
        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "queued"
        assert batch["remaining"] == 1

    async def test_transient_failure_on_last_attempt_is_recorded(self, pool):
        from fastapi import HTTPException

        from app import worker

        batch_id, _ = await _make_batch(pool, count=1)

        with patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract:
            extract.side_effect = HTTPException(status_code=503, detail="OpenRouter down")
            await worker.parse_file(
                {"redis": pool, "job_try": settings.job_max_tries}, batch_id, 0
            )

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert batch["results"][0]["error"] == "OpenRouter down"

    async def test_expired_batch_job_is_dropped_quietly(self, pool):
        from app import worker

        # Nothing was ever stored under this id.
        await worker.parse_file({"redis": pool, "job_try": 1}, "gone", 0)

        assert await queue.get_batch(pool, "gone") is None

    async def test_completion_fires_the_callback_once(self, pool):
        from app import worker

        batch_id, _ = await _make_batch(
            pool, count=1, callback_url="https://example.com/hook", callback_token="tok"
        )

        with patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract, \
             patch("app.worker.parse_resume", new_callable=AsyncMock) as parse, \
             patch("app.worker.httpx.AsyncClient") as http:
            extract.return_value = "text"
            parse.return_value = {"name": "Jane Doe"}
            post = http.return_value.__aenter__.return_value.post
            post.return_value = AsyncMock(status_code=200)

            await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        post.assert_awaited_once()
        url, = post.await_args.args
        assert url == "https://example.com/hook"
        assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer tok"
        assert post.await_args.kwargs["json"]["status"] == "completed"


# ---------------------------------------------------------------------------
# Record-reference batches (Apex sends ids, the worker fetches the files)
# ---------------------------------------------------------------------------

class TestReferenceBatches:
    async def test_reference_batch_stores_no_bytes(self, pool):
        batch_id, total = await queue.create_reference_batch(
            pool, "salesforce",
            [("attachment", "068AAA", None), ("candidate", "a0X111", "named.pdf")],
        )

        assert total == 2
        assert await pool.zcard(settings.queue_name) == 2
        # Nothing large was uploaded, so there is no blob to hold.
        assert await pool.get(queue._k_blob(batch_id, 0)) is None

        first = await queue.load_file(pool, batch_id, 0)
        assert first["source"] == queue.SOURCE_ATTACHMENT
        assert first["ref"] == "068AAA"
        assert "content" not in first

        second = await queue.load_file(pool, batch_id, 1)
        assert (second["source"], second["ref"], second["filename"]) == (
            queue.SOURCE_CANDIDATE, "a0X111", "named.pdf"
        )

    async def test_reference_batch_reports_like_any_other(self, pool):
        batch_id, _ = await queue.create_reference_batch(
            pool, "salesforce", [("attachment", "068AAA", None)]
        )

        batch = await queue.get_batch(pool, batch_id)
        assert (batch["status"], batch["total"], batch["mode"]) == ("queued", 1, "salesforce")


class TestWorkerResolve:
    async def test_inline_entry_uses_its_own_bytes(self):
        from app import worker

        content, content_type = await worker._resolve(
            {"filename": "cv.pdf", "content_type": PDF,
             "source": queue.SOURCE_INLINE, "content": b"pdf-bytes"}
        )
        assert (content, content_type) == (b"pdf-bytes", PDF)

    async def test_attachment_entry_is_downloaded_from_salesforce(self):
        from app import worker

        entry = {"filename": "068AAA.pdf", "source": queue.SOURCE_ATTACHMENT, "ref": "068AAA"}
        with patch(
            "app.services.salesforce.fetch_resume_by_attachment_id", new_callable=AsyncMock
        ) as fetch:
            fetch.return_value = (b"downloaded", "Jane_Doe_CV.docx")
            content, content_type = await worker._resolve(entry)

        fetch.assert_awaited_once_with("068AAA")
        assert content == b"downloaded"
        # Salesforce knows the real name and extension; both win.
        assert entry["filename"] == "Jane_Doe_CV.docx"
        assert content_type.endswith("wordprocessingml.document")

    async def test_unknown_source_is_a_clear_error(self):
        from fastapi import HTTPException
        from app import worker

        with pytest.raises(HTTPException) as exc:
            await worker._resolve({"filename": "x.pdf", "source": "carrier-pigeon", "ref": "1"})
        assert exc.value.status_code == 400

    async def test_failed_download_is_recorded_against_the_file(self, pool):
        from fastapi import HTTPException
        from app import worker

        batch_id, _ = await queue.create_reference_batch(
            pool, "salesforce", [("attachment", "068MISSING", None)]
        )

        with patch(
            "app.services.salesforce.fetch_resume_by_attachment_id", new_callable=AsyncMock
        ) as fetch:
            fetch.side_effect = HTTPException(status_code=404, detail="Resume not found")
            await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert batch["results"][0]["error"] == "Resume not found"

    async def test_reference_file_parses_end_to_end(self, pool):
        from app import worker

        batch_id, _ = await queue.create_reference_batch(
            pool, "client", [("attachment", "068AAA", None)]
        )

        with patch(
            "app.services.salesforce.fetch_resume_by_attachment_id", new_callable=AsyncMock
        ) as fetch, \
             patch("app.worker.validate_file_bytes", new_callable=AsyncMock) as extract, \
             patch("app.worker.parse_resume", new_callable=AsyncMock) as parse:
            fetch.return_value = (b"%PDF-fake", "Jane.pdf")
            extract.return_value = "Jane Doe, Python developer"
            parse.return_value = {"name": "Jane Doe"}
            await worker.parse_file({"redis": pool, "job_try": 1}, batch_id, 0)

        batch = await queue.get_batch(pool, batch_id)
        assert batch["status"] == "completed"
        assert batch["results"][0]["success"] is True
        assert batch["results"][0]["filename"] == "Jane.pdf"
        assert batch["results"][0]["data"]["Name"] == "Jane Doe"
