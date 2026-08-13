# Remaining work

Everything the Salesforce integration needs to function is done. This is the
honest list of what is *not* — kept here rather than claimed complete, because
several items only bite in production and one of them will lose data if it is
forgotten.

Nothing below blocks a **sandbox** deployment. Items marked **Before
production** should be closed before real candidate records are at stake.

---

## Before production

### 1. Apex has never run against an org — **Before production**

The classes compile logically and the tests are shaped correctly, but no
Salesforce runtime has executed them. Deploy to a scratch org or sandbox and run
`sf apex run test` before trusting them. Production deployment needs 75% coverage
org-wide.

### 2. Resumes are still truncated at 8,000 characters — **Before production**

`_FULL_TEXT_THRESHOLD` in `app/services/llm.py`. Long or senior CVs lose their
tail, usually the earliest employment.

This is now *reported* (`extraction.text_truncated`, `text_sent`, `text_length`)
and surfaced onto the record, so nobody is misled — but it is not fixed. The fix
is to page the remainder through an extra chunk and merge, which costs one more
LLM call per long resume.

### 3. The async job store is single-process — **Before production if used**

`_jobs` in `app/routes/parser.py` is a plain in-memory dict:

- never evicted, so it grows without bound while holding candidate PII
- lost on restart
- invisible to other workers, so `/parse/job/{id}` polls the wrong one under
  more than one uvicorn worker

Also `asyncio.create_task()`'s result is not retained, so the task can be
garbage-collected mid-flight.

Either move it to Redis with a TTL, or pin the deployment to one worker and
document the ceiling. **The `/parse/job` endpoints are the only affected ones —
the Salesforce write-back path does not use them.**

### 4. PII goes to a third-party LLM — **Decision, not code**

Aadhaar, PAN, passport number, date of birth and parents' names are extracted by
prompt and sent to OpenRouter. Image-based PDFs are sent whole, base64-encoded.

For Indian candidate data this needs an explicit decision under the DPDP Act:
redact before the call, drop those fields from extraction, or get a data
processing agreement. It is not a patch — it is a call someone has to make and
record.

### 5. Auth hardening — **Before production**

- `API_KEY` defaults to `changeme`. Set a real one.
- The comparison is not constant-time.
- `/docs`, `/redoc` and `/openapi.json` are public, and the middleware's
  `path.startswith("/docs")` also matches `/docsanything`.
- `/health` is unauthenticated and calls OpenRouter on every hit.
- There is no rate limiting anywhere.

---

## Should fix

### 6. Timing has little headroom

One `parse-and-update` is: fetch + extract + up to 75 s of LLM + PATCH, against
Salesforce's 120 s cumulative callout ceiling. The retry budget is bounded
(`_CALL_DEADLINE`) so it cannot overrun, but the margin is thin. Measure the real
p95 against production-sized resumes rather than trusting the arithmetic.

### 7. Salesforce token has no expiry tracking

Cached in memory and refreshed only when something returns 401. Works, but means
one guaranteed failed request per token expiry.

### 8. `update_candidate` returns Salesforce's error verbatim

Deliberate — the error names the offending field, which is the only practical way
to debug a mapping problem. But it is an information leak to the caller. Log it
fully, return something sanitised.

### 9. CI is red, and was before this work

`ruff check` reports 40 pre-existing errors (mostly `E701`/`E741` in
`llm.py`'s score block). Because `lint` gates `test` gates `docker-build`, the
whole pipeline fails.

Worse, `ci.yml` does an unpinned `pip install ruff`, so the rule set drifts on
its own and CI can break with no code change. **Pin the ruff version**, then
clear or ignore the existing 40.

`ci.yml` also still sets `OLLAMA_*` environment variables, which nothing reads.

### 10. `docker-compose.yml` cannot reach Salesforce

It passes no `SF_*` variables through, so every Salesforce endpoint returns 503
under Compose. Railway deployment is unaffected.

### 11. Production image ships the test suite

`docker/Dockerfile` copies `tests/`, installs pytest and reportlab from
`requirements.txt`, generates a sample PDF at build time, and runs as root.

### 12. `docs/API.md` is fiction

Still describes Ollama, 8 LLM calls and v1.0.0. Its §10 Apex sample hardcodes
`changeme` and uses `@future`, which cannot return a value or chain. Anyone
building from it builds the wrong thing. `docs/SALESFORCE_INTEGRATION.md` is the
current one; `API.md` should be rewritten or deleted.

---

## Known-unknown

### 13. `BulkResumeMasker.rar` has not been reviewed

A related codebase was shared but never opened, so it is unknown whether it
duplicates, supersedes or conflicts with this one. Worth resolving before two
parsers end up in production.

### 14. Field mapping is unverified against the real org

`load_specs_from_describe()` resolves names against whatever org it is pointed
at, so it is correct by construction — but no one has yet seen the output for the
target org. Run:

```
POST /api/v1/salesforce/describe/reload
```

and read `unresolved`. Every name on that list is data the parser extracts and
then discards because the org has no such field.
