# Resume Parser - CLAUDE.md

## Project Overview

FastAPI-based REST API that parses PDF/DOCX resume files and extracts structured data using an OpenRouter LLM. Supports two output modes:
- **Web-app mode** — generic JSON response (`/api/v1/parse`)
- **Salesforce mode** — SCSCHAMPS-mapped JSON, pulls resumes directly from Salesforce via OAuth2 (`/api/v1/salesforce/*`)

Both modes are available simultaneously on the same running server.

Each mode has a **synchronous** path (parse inside the request) and an **async
queue** path (accept now, parse on a worker, poll or callback for results). Bulk
loads belong on the queue — see `docs/QUEUE.md`.

## Tech Stack

- **Framework:** FastAPI 0.115.0 + Uvicorn 0.32.0
- **Language:** Python 3.11
- **LLM:** OpenRouter — model `openai/gpt-4o-mini` (extraction + OCR)
- **PDF Parsing:** PyPDF 5.1.0 (text-based); OCR fallback via gpt-4o-mini vision (image-based)
- **DOCX Parsing:** python-docx 1.1.2
- **HTTP Client:** httpx 0.27.2 (async)
- **Job queue:** arq 0.28.0 + Redis 7 (async bulk parsing)
- **Config:** Pydantic Settings + python-dotenv
- **Containerization:** Docker + Docker Compose (app + worker + redis)
- **Testing:** pytest + pytest-asyncio + pytest-httpx
- **CI:** GitHub Actions

## Project Structure

```
app/
├── main.py                  # FastAPI app entry, registers all routers, opens Redis pool
├── config.py                # Pydantic Settings (OpenRouter + app + Salesforce + queue)
├── worker.py                # arq worker: parses ONE queued file per job
├── routes/
│   ├── parser.py            # /api/v1/parse, /api/v1/models  (sync web-app endpoints)
│   ├── jobs.py              # /api/v1/parse/*/jobs           (async queue endpoints)
│   └── salesforce.py        # /api/v1/salesforce/*           (Salesforce endpoints)
├── services/
│   ├── document.py          # PDF/DOCX extraction; OCR fallback for image-based PDFs
│   ├── llm.py               # OpenRouter integration, 3 consolidated prompts, score computation
│   ├── queue.py             # Redis batch store + arq pool (owns every Redis key)
│   └── salesforce.py        # OAuth2 token flow + resume file fetch from SF
├── schemas/response.py      # Pydantic models: ResumeData, SalesforceResumeData,
│                            #   map_to_generic/salesforce/client(), batch schemas
└── middleware/auth.py        # X-API-Key middleware
docker/
├── Dockerfile               # Python 3.11-slim image (serves both app and worker)
└── .dockerignore
docs/
├── API.md                   # Full endpoint reference
└── QUEUE.md                 # Async queue: design, Railway deploy, Apex integration
tests/
├── test_api.py              # Pytest suite (all external deps mocked)
├── test_jobs.py             # Queue endpoints + sync partial-result behaviour
├── test_queue.py            # Batch store + worker, against fakeredis
├── test_parse_resume.py     # Manual integration test script
├── generate_sample_pdf.py   # ReportLab sample PDF generator
├── sample_resume.txt        # Sample resume text
└── sample_resume.pdf        # Generated sample resume
.github/workflows/
└── ci.yml                   # CI: lint → test → docker-build
pytest.ini                   # Pytest config (asyncio_mode=auto)
```

## Commands

### Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

# Queue worker (needs Redis running; the API works without it, queue routes 503)
arq app.worker.WorkerSettings

# Run unit tests (no live services needed — all mocked, queue uses fakeredis)
pytest tests/test_api.py tests/test_jobs.py tests/test_queue.py -v

# Manual integration test (requires running services)
python tests/test_parse_resume.py tests/sample_resume.pdf
```

### Docker

```bash
docker compose up --build            # redis + app + worker
docker compose up -d --scale worker=3
docker compose logs -f worker
docker compose down
```

## API Endpoints

### Web-app (generic)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check + OpenRouter and queue status |
| POST | `/api/v1/parse` | X-API-Key | Upload 1–15 PDF/DOCX → generic JSON |
| POST | `/api/v1/parse/salesforce` | X-API-Key | Upload 1–15 → SCSCHAMPS JSON |
| POST | `/api/v1/parse/client` | X-API-Key | Upload 1–15 → client-1 JSON |
| GET | `/api/v1/models` | X-API-Key | Currently configured model |

Sync endpoints return **200 with partial results** if the batch outruns
`BULK_TIMEOUT` — unfinished files come back as failed items. They no longer 504.

### Async queue (preferred for bulk) — see `docs/QUEUE.md`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/parse/jobs` | X-API-Key | Queue → generic JSON, 202 + `batch_id` |
| POST | `/api/v1/parse/salesforce/jobs` | X-API-Key | Queue → SCSCHAMPS JSON |
| POST | `/api/v1/parse/client/jobs` | X-API-Key | Queue → client-1 JSON |
| GET | `/api/v1/parse/jobs/{batch_id}` | X-API-Key | Status + results so far |
| DELETE | `/api/v1/parse/jobs/{batch_id}` | X-API-Key | Drop a batch early |

Deprecated: `POST /api/v1/parse/job` and `GET /api/v1/parse/job/{job_id}`
(singular) hold state in one process's memory — use the plural `jobs` routes.

### Salesforce

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/salesforce/parse-candidate` | X-API-Key | Record ID → fetch resume from SF → SCSCHAMPS JSON |
| POST | `/api/v1/salesforce/parse-attachment` | X-API-Key | ContentVersion/Attachment ID → SCSCHAMPS JSON |
| POST | `/api/v1/salesforce/parse-url` | X-API-Key | Resume URL (SF or external) → SCSCHAMPS JSON |

## Configuration

All env vars — see `.env.example` for full list.

### Core

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | `""` | OpenRouter API key |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | LLM extraction model |
| `OPENROUTER_OCR_MODEL` | `openai/gpt-4o-mini` | Vision model for image-based PDF OCR |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter base URL |
| `API_KEY` | `changeme` | X-API-Key value |
| `MAX_FILE_SIZE` | `10485760` | Upload limit (bytes) |
| `LOG_LEVEL` | `INFO` | Logging level |

### Queue (async bulk path)

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Railway's Redis plugin injects this |
| `QUEUE_NAME` | `resume_parse` | Must match between app and worker |
| `QUEUE_MAX_FILES` | `100` | Cap per async batch |
| `WORKER_CONCURRENCY` | `8` | Resumes in flight per worker process |
| `JOB_TIMEOUT` | `300` | Per-file budget inside the worker (s) |
| `JOB_MAX_TRIES` | `3` | Attempts per file (transient errors only) |
| `JOB_TTL` | `86400` | Lifetime of bytes and results in Redis (s) |
| `JOB_STALE_SECONDS` | `1800` | Abandoned batch reaped after this |
| `JOB_CALLBACK_ALLOWED_HOSTS` | *(empty = any)* | Comma-separated host suffixes |

### Synchronous bulk

| Variable | Default | Description |
|---|---|---|
| `BULK_MAX_FILES` | `15` | Files per synchronous request |
| `BULK_TIMEOUT` | `110` | Wall-clock budget — keep under Salesforce's 120s callout cap |
| `BULK_CONCURRENCY` | `5` | Files parsed in parallel per request |

### Salesforce (required for SF endpoints)

| Variable | Description |
|---|---|
| `SF_CLIENT_ID` | Connected App Consumer Key |
| `SF_CLIENT_SECRET` | Connected App Consumer Secret |
| `SF_USERNAME` | (Optional) SF username for password flow |
| `SF_PASSWORD` | (Optional) SF password |
| `SF_SECURITY_TOKEN` | (Optional) SF security token |
| `SF_LOGIN_URL` | `https://login.salesforce.com` or `https://test.salesforce.com` |
| `SF_API_VERSION` | `59.0` |

## Salesforce Integration Notes

- **OAuth2 flow:** Prefers `client_credentials` grant. Falls back to `password` grant if `SF_USERNAME`/`SF_PASSWORD` are set.
- **Token caching:** Token is cached in memory; call `invalidate_token()` to force re-auth.
- **Resume fetch priority:** `SCSCHAMPS__Resume_Attachment_Id__c` → `SCSCHAMPS__Resume_URL__c`
- **ContentVersion first, Attachment fallback:** `/sobjects/ContentVersion/{id}/VersionData` then `/sobjects/Attachment/{id}/Body`
- **SCSCHAMPS field mapping** is in `app/schemas/response.py → map_to_salesforce()`. All field names match `SCSCHAMPS__<Field>__c` with prefix/suffix stripped.

## CI/CD (GitHub Actions)

Pipeline: `lint → test → docker-build`

- **lint:** `ruff check` on `app/` and `tests/`
- **test:** `pytest tests/test_api.py` — all external deps mocked (no Ollama/Salesforce needed)
- **docker-build:** verifies `docker/Dockerfile` builds successfully
- Test results uploaded as artifact (`test-results/results.xml`)
- Optional Docker Hub push (uncomment in `.github/workflows/ci.yml`, set secrets)

## Key Architecture Patterns

- **Dual-mode output:** Same parsing pipeline; `map_to_salesforce()` converts the result for SF
- **Sync vs. queue:** The HTTP request either parses inline (bounded by
  `BULK_TIMEOUT`, degrades to partial results) or only *accepts* work and hands it
  to Redis. Salesforce caps a callout at 120s, so no synchronous budget can ever
  fit a large batch — that is why the queue exists.
- **One queued job per file, not per batch:** the unit of retry, timeout and
  failure is a single resume, so one bad PDF cannot sink the other fourteen.
  Results are written to Redis as each file lands and stream back to pollers.
- **Stateless:** Files processed in memory; only the queue persists bytes, and
  only until that file is parsed
- **Async throughout:** httpx async client reused across Ollama calls
- **Graceful degradation:** Empty fields returned on LLM extraction failure (no crash)
- **Auth middleware:** `X-API-Key` on all `/api/v1/*`; public: `/health`, `/docs`, `/redoc`
- **LLM:** temperature=0.0, 3 consolidated prompts fired in parallel (Chunk A: contact+personal+meta 950tok, Chunk B: skills+certs+awards+summary+projects 2100tok, Chunk C: experience+education 2600tok)
- **OCR fallback:** PyPDF first; if <100 chars extracted, sends raw PDF as base64 data URL to gpt-4o-mini vision via OpenRouter
- **Score matrix:** 7-category weighted system (contact 5%, summary 15%, experience 25%, skills 20%, education+certs 10%, achievements 15%, format 10%) — each raw 0–10, overall 0–100 with grade band (Excellent/Good/Average/Poor)
 
 