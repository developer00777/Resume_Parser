# Resume Parser - CLAUDE.md

## Project Overview

FastAPI-based REST API that parses PDF/DOCX resume files and extracts structured data using OpenRouter. Supports three output modes:
- **Web-app mode** — generic JSON response (`/api/v1/parse`)
- **Salesforce read mode** — SCSCHAMPS-mapped JSON, pulls resumes directly from Salesforce via OAuth2 (`/api/v1/salesforce/parse-*`)
- **Salesforce write-back** — parses and PATCHes the fields onto the Candidate record (`/api/v1/salesforce/parse-and-update`)

Both modes are available simultaneously on the same running server.

## Tech Stack

- **Framework:** FastAPI 0.115.0 + Uvicorn 0.32.0
- **Language:** Python 3.11
- **LLM:** OpenRouter — model `openai/gpt-4o-mini` (extraction + OCR)
- **PDF Parsing:** PyPDF 5.1.0 (text-based); OCR fallback via gpt-4o-mini vision (image-based)
- **DOCX Parsing:** python-docx 1.1.2
- **HTTP Client:** httpx 0.27.2 (async)
- **Config:** Pydantic Settings + python-dotenv
- **Containerization:** Docker + Docker Compose
- **Testing:** pytest + pytest-asyncio + pytest-httpx
- **CI:** GitHub Actions

## Project Structure

```
app/
├── main.py                  # FastAPI app entry, registers all routers
├── config.py                # Pydantic Settings (Ollama + app + Salesforce creds)
├── routes/
│   ├── parser.py            # /api/v1/parse, /api/v1/models  (web-app endpoints)
│   └── salesforce.py        # /api/v1/salesforce/*           (Salesforce endpoints)
├── services/
│   ├── document.py          # PDF/DOCX extraction; OCR fallback for image-based PDFs
│   ├── llm.py               # OpenRouter integration, 3 parallel prompts, retry, score computation
│   ├── salesforce.py        # OAuth2 token flow, resume fetch, sObject describe, record PATCH
│   ├── salesforce_coerce.py # Describe-driven field mapping + value coercion
│   └── salesforce_schema.py # Field-mapping lifecycle: startup load, TTL refresh, lazy retry
├── schemas/response.py      # Pydantic models: ResumeData, SalesforceResumeData,
│                            #   map_to_salesforce(), ParseResponse, etc.
└── middleware/auth.py        # X-API-Key middleware
docker/
├── Dockerfile               # Python 3.11-slim image
└── .dockerignore
tests/
├── test_api.py              # Pytest suite (all external deps mocked)
├── test_parse_resume.py     # Manual integration test script
├── generate_sample_pdf.py   # ReportLab sample PDF generator
├── sample_resume.txt        # Sample resume text
└── sample_resume.pdf        # Generated sample resume
.github/workflows/
└── ci.yml                   # CI: lint → test → docker-build
pytest.ini                   # Pytest config (asyncio_mode=auto)
salesforce/                  # SFDX source — deploy with `sf project deploy start -d salesforce/force-app`
└── force-app/main/default/
    ├── classes/             # ResumeParserQueueable, ResumeParserAction (+ tests)
    ├── lwc/resumeParser/    # "Parse Resume" quick-action component
    ├── objects/.../fields/  # Resume_Parse_Status__c
    └── quickActions/        # Parse Resume button
docs/SALESFORCE_INTEGRATION.md   # Step-by-step org setup runbook
```

## Commands

### Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

# Run unit tests (no live services needed — all mocked)
pytest tests/test_api.py -v

# Manual integration test (requires running services)
python tests/test_parse_resume.py tests/sample_resume.pdf
```

### Docker

```bash
docker-compose up --build
docker-compose down
docker-compose logs -f app
```

## API Endpoints

### Web-app (generic)

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | No | Health check + OpenRouter status |
| POST | `/api/v1/parse` | X-API-Key | Upload 1–15 PDF/DOCX → generic JSON |
| POST | `/api/v1/parse/salesforce` | X-API-Key | Upload 1–15 → SCSCHAMPS JSON |
| POST | `/api/v1/parse/job` | X-API-Key | Async bulk submit → job_id (single-process only) |
| GET | `/api/v1/models` | X-API-Key | Configured OpenRouter model |

### Salesforce

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/salesforce/parse-candidate` | X-API-Key | Record ID → fetch resume from SF → SCSCHAMPS JSON |
| POST | `/api/v1/salesforce/parse-attachment` | X-API-Key | ContentVersion/Attachment ID → SCSCHAMPS JSON |
| POST | `/api/v1/salesforce/parse-url` | X-API-Key | Resume URL (SF or allowlisted host) → SCSCHAMPS JSON |
| POST | `/api/v1/salesforce/parse-and-update` | X-API-Key | Record ID → parse → **PATCH the record**. `dry_run=true` validates without writing |
| POST | `/api/v1/salesforce/describe/reload` | X-API-Key | Rebuild the field mapping from a live Describe |
| GET | `/api/v1/salesforce/describe/status` | X-API-Key | Mapping state: age, unresolved and read-only fields |

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
| `SF_EXTERNAL_RESUME_HOSTS` | Comma-separated non-Salesforce hosts `parse-url` may fetch from. The SF token is **never** sent to these. Empty = Salesforce-hosted only |
| `SF_DESCRIBE_TTL_MINUTES` | How long the field mapping stays trusted before re-fetching (default 60, 0 = never) |

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

## Salesforce Write-Back

**The parser owns the write, not Apex.** Apex cannot host this work: identifiers
are case-insensitive (so a wrapper cannot hold both `Resume_Score` and a nested
`resume_score`), `Date.valueOf()` throws on free-text dates and takes the whole
DML with it, restricted picklists reject free text the same way, and heap is
6 MB. So Apex sends a record ID and reads back a summary.

- **Field mapping is derived, not hardcoded.** `load_specs_from_describe()`
  tries `SCSCHAMPS__X__c`, `X__c`, then `X` against the org's real schema. This
  is what makes the same build work against a namespaced managed-package org and
  a locally-customised one.
- **Loaded at startup**, refreshed on `SF_DESCRIBE_TTL_MINUTES`, and retried
  lazily on first use if the startup load failed. Write-back returns **409**
  while no mapping is loaded — writing against assumed names could write to the
  wrong field.
- **Fields are withheld, never blanked.** A field is skipped when it is None,
  when Salesforce owns it (`_ORG_OWNED_FIELDS`), when the LLM chunk that
  produces it failed, or when the org has no updateable field of that name.
- **`extraction`** on every response reports failed chunks, the sections they
  owned, and whether the resume was truncated — so an empty field can be told
  apart from a failed call.

## Key Architecture Patterns

- **Dual-mode output:** Same parsing pipeline; `map_to_salesforce()` converts the result for SF
- **Stateless:** Files processed in memory, nothing persisted to disk
- **Async throughout:** httpx async client reused across OpenRouter calls
- **Graceful degradation:** Empty fields returned on LLM extraction failure (no crash)
- **Auth middleware:** `X-API-Key` on all `/api/v1/*`; public: `/health`, `/docs`, `/redoc`
- **LLM:** temperature=0.0, 3 consolidated prompts fired in parallel (Chunk A: contact+personal+meta 950tok, Chunk B: skills+certs+awards+summary+projects 2100tok, Chunk C: experience+education 2600tok)
- **OCR fallback:** PyPDF first; if <100 chars extracted, sends raw PDF as base64 data URL to gpt-4o-mini vision via OpenRouter
- **Score matrix:** 7-category weighted system (contact 5%, summary 15%, experience 25%, skills 20%, education+certs 10%, achievements 15%, format 10%) — each raw 0–10, overall 0–100 with grade band (Excellent/Good/Average/Poor)
 
 