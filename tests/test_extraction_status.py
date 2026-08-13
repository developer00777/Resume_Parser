"""
Tests for OpenRouter retry behaviour and per-chunk extraction status.

The failure this guards against: a bulk request fires 15 concurrent LLM calls,
one gets a 429, its chunk returns nothing, and the response still says
success — so an entire experience section reads as "this candidate has no job
history". Retrying removes most occurrences; the status makes the rest visible
instead of silent.
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.services import llm


def _completion(content: str = '{"contact": {"email": "a@b.com"}}') -> dict:
    return {"choices": [{"message": {"content": content}}]}


def _response(status: int, json_body=None, headers=None) -> httpx.Response:
    return httpx.Response(
        status_code=status,
        json=json_body if json_body is not None else {"error": "boom"},
        headers=headers or {},
        request=httpx.Request("POST", "https://openrouter.test/chat/completions"),
    )


class TestRetry:
    @pytest.mark.asyncio
    async def test_retries_a_rate_limit_then_succeeds(self):
        calls = []

        async def post(*args, **kwargs):
            calls.append(1)
            return _response(429) if len(calls) < 3 else _response(200, _completion())

        with (patch.object(llm, "_get_client") as get_client,
              patch("asyncio.sleep", new_callable=AsyncMock) as sleep):
            get_client.return_value.post = post
            result = await llm._call_openrouter("prompt", max_tokens=10)

        assert len(calls) == 3
        assert "a@b.com" in result
        assert sleep.await_count == 2, "should back off between attempts"

    @pytest.mark.asyncio
    async def test_gives_up_after_the_attempt_limit(self):
        calls = []

        async def post(*args, **kwargs):
            calls.append(1)
            return _response(429)

        with (patch.object(llm, "_get_client") as get_client,
              patch("asyncio.sleep", new_callable=AsyncMock)):
            get_client.return_value.post = post
            with pytest.raises(HTTPException) as exc:
                await llm._call_openrouter("prompt")

        assert len(calls) == llm._MAX_ATTEMPTS
        assert exc.value.status_code == 502

    @pytest.mark.asyncio
    async def test_does_not_retry_an_auth_failure(self):
        # A bad API key will not fix itself; retrying wastes the callout budget.
        calls = []

        async def post(*args, **kwargs):
            calls.append(1)
            return _response(401)

        with (patch.object(llm, "_get_client") as get_client,
              patch("asyncio.sleep", new_callable=AsyncMock) as sleep):
            get_client.return_value.post = post
            with pytest.raises(HTTPException):
                await llm._call_openrouter("prompt")

        assert len(calls) == 1 and sleep.await_count == 0

    @pytest.mark.asyncio
    async def test_retries_a_server_error(self):
        calls = []

        async def post(*args, **kwargs):
            calls.append(1)
            return _response(503) if len(calls) == 1 else _response(200, _completion())

        with (patch.object(llm, "_get_client") as get_client,
              patch("asyncio.sleep", new_callable=AsyncMock)):
            get_client.return_value.post = post
            await llm._call_openrouter("prompt")

        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_repeated_timeouts_stay_inside_the_call_budget(self):
        """
        Three 45s timeouts plus backoff is 139s, which blows parse_resume's 120s
        budget, the bulk request's 110s, and Salesforce's 120s cumulative
        callout ceiling — turning a fast failure into a timeout for the whole
        resume.
        """
        clock = {"now": 0.0}
        attempts = []

        async def post(*args, **kwargs):
            attempts.append(kwargs.get("timeout"))
            clock["now"] += kwargs.get("timeout") or llm._CHUNK_TIMEOUT
            raise httpx.TimeoutException("timed out")

        async def fake_sleep(seconds):
            clock["now"] += seconds

        with (patch.object(llm, "_get_client") as get_client,
              patch.object(llm.time, "monotonic", lambda: clock["now"]),
              patch("asyncio.sleep", side_effect=fake_sleep)):
            get_client.return_value.post = post
            with pytest.raises(HTTPException) as exc:
                await llm._call_openrouter("prompt")

        assert exc.value.status_code == 504
        assert clock["now"] < 110.0, (
            f"total wall time {clock['now']:.1f}s must stay under the bulk budget")
        assert sum(a for a in attempts if a) <= llm._CALL_DEADLINE

    @pytest.mark.asyncio
    async def test_cheap_failures_still_retry_freely(self):
        clock = {"now": 0.0}
        calls = []

        async def post(*args, **kwargs):
            calls.append(1)
            clock["now"] += 0.8  # fast rejection
            return _response(429) if len(calls) < 3 else _response(200, _completion())

        async def fake_sleep(seconds):
            clock["now"] += seconds

        with (patch.object(llm, "_get_client") as get_client,
              patch.object(llm.time, "monotonic", lambda: clock["now"]),
              patch("asyncio.sleep", side_effect=fake_sleep)):
            get_client.return_value.post = post
            result = await llm._call_openrouter("prompt")

        assert len(calls) == 3, "cheap failures should use the full attempt budget"
        assert "a@b.com" in result

    @pytest.mark.parametrize("header,expected", [
        ({"retry-after": "5"}, 5.0),
        ({"retry-after": "0.5"}, 0.5),
        ({}, None),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "600"}, None),   # implausibly long
        ({"retry-after": "0"}, None),
    ])
    def test_honours_a_sane_retry_after(self, header, expected):
        assert llm._retry_after(_response(429, headers=header)) == expected


class TestChunkStatus:
    @pytest.mark.asyncio
    async def test_success_reports_no_error(self):
        with patch.object(llm, "_call_openrouter", new_callable=AsyncMock) as call:
            call.return_value = '{"contact": {"email": "a@b.com"}}'
            name, data, error = await llm._call_chunk("chunk_a", "p", 10, "text")
        assert name == "chunk_a" and error is None
        assert data["contact"]["email"] == "a@b.com"

    @pytest.mark.asyncio
    async def test_failure_reports_why(self):
        with patch.object(llm, "_call_openrouter", new_callable=AsyncMock) as call:
            call.side_effect = HTTPException(502, "LLM service returned an error: 429")
            _, data, error = await llm._call_chunk("chunk_c", "p", 10, "text")
        assert data == {} and "429" in error

    @pytest.mark.asyncio
    async def test_unparseable_output_is_a_failure_not_an_empty_resume(self):
        with patch.object(llm, "_call_openrouter", new_callable=AsyncMock) as call:
            call.return_value = "I'm sorry, I cannot"
            _, data, error = await llm._call_chunk("chunk_b", "p", 10, "text")
        assert data == {} and error is not None


class TestExtractionStatusMapping:
    def test_a_clean_run_is_complete(self):
        status = llm._extraction_status({}, truncated=False, text_length=4000)
        assert status["complete"] is True and status["chunks_ok"] == 3
        assert status["failed_sections"] == []

    def test_a_failed_chunk_names_its_sections(self):
        status = llm._extraction_status({"chunk_c": "429"}, truncated=False, text_length=4000)
        assert status["complete"] is False and status["chunks_ok"] == 2
        assert status["failed_chunks"] == ["chunk_c"]
        assert status["failed_sections"] == ["education", "experience"]

    def test_every_chunk_maps_to_sections(self):
        # A chunk with no mapping fails silently — its fields would be written
        # as blanks after a failure.
        for chunk_name, _prompt, _tokens in llm.CHUNKS:
            assert llm.CHUNK_SECTIONS.get(chunk_name), f"{chunk_name} has no sections"

    def test_truncation_marks_the_parse_incomplete(self):
        status = llm._extraction_status({}, truncated=True, text_length=15000)
        assert status["complete"] is False and status["text_truncated"] is True
        assert status["text_length"] == 15000
        assert status["text_sent"] == llm._FULL_TEXT_THRESHOLD

    def test_a_short_resume_is_not_truncated(self):
        status = llm._extraction_status({}, truncated=False, text_length=1200)
        assert status["text_truncated"] is False and status["text_sent"] == 1200


class TestParseResumeSurfacesStatus:
    @pytest.mark.asyncio
    async def test_status_reaches_the_parsed_result(self):
        async def one_chunk_fails(chunk_name, prompt, max_tok, chunk_text):
            if chunk_name == "chunk_c":
                return chunk_name, {}, "LLM service returned an error: 429"
            return chunk_name, {}, None

        with patch.object(llm, "_call_chunk", side_effect=one_chunk_fails):
            parsed = await llm.parse_resume("Some resume text that is long enough.")

        assert parsed["extraction"]["complete"] is False
        assert parsed["extraction"]["failed_sections"] == ["education", "experience"]

    @pytest.mark.asyncio
    async def test_long_resume_is_flagged_as_truncated(self):
        async def ok(chunk_name, prompt, max_tok, chunk_text):
            return chunk_name, {}, None

        with patch.object(llm, "_call_chunk", side_effect=ok):
            parsed = await llm.parse_resume("word " * 4000)

        assert parsed["extraction"]["text_truncated"] is True
