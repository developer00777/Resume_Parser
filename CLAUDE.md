# Resume Parser - CLAUDE.md

## Project Overview

FastAPI-based REST API that parses PDF/DOCX resume files and extracts structured data using a local Ollama LLM. Supports two output modes:
- **Web-app mode** — generic JSON response (`/api/v1/parse`)
- **Salesforce mode** — SCSCHAMPS-mapped JSON, pulls resumes directly from Salesforce via OAuth2 (`/api/v1/salesforce/*`)

Both modes are available simultaneously on the same running server.

## Tech Stack

- **Framework:** FastAPI 0.115.0 + Uvicorn 0.32.0
- **Language:** Python 3.11
- **LLM:** Ollama (local) — model `llama3.2:3b`
- **PDF Parsing:** PyPDF 5.1.0
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
│   ├── document.py          # PDF/DOCX text extraction & file validation
│   ├── llm.py               # Ollama integration, 8 prompts, score computation
│   └── salesforce.py        # OAuth2 token flow + resume file fetch from SF
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
| GET | `/health` | No | Health check + Ollama status |
| POST | `/api/v1/parse` | X-API-Key | Upload PDF/DOCX → generic JSON |
| GET | `/api/v1/models` | X-API-Key | List Ollama models |

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
| `OLLAMA_MODEL` | `llama3.2:3b` | LLM model |
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama endpoint |
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
- **Stateless:** Files processed in memory, nothing persisted to disk
- **Async throughout:** httpx async client reused across Ollama calls
- **Graceful degradation:** Empty fields returned on LLM extraction failure (no crash)
- **Auth middleware:** `X-API-Key` on all `/api/v1/*`; public: `/health`, `/docs`, `/redoc`
- **LLM:** temperature=0.1, 8 specialized prompts, regex section-splitting before LLM
- **Score matrix:** 7-category weighted system (contact 5%, summary 15%, experience 25%, skills 20%, education+certs 10%, achievements 15%, format 10%) — each raw 0–10, overall 0–100 with grade band (Excellent/Good/Averag  e/Poor)
 
 