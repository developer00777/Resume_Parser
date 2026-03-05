Based on your requirements, here's a comprehensive Product Requirements Document (PRD) for your Resume Parser:
---
Resume Parser PRD
1. Project Overview
Project Name: Resume Parser API  
Type: REST API Service (Dockerized)  
Core Functionality: Parse PDF/Word resumes using Ollama LFM2:2.6B model and return structured JSON data  
Target Users: HR teams, recruitment platforms, ATS systems
---
2. Technical Stack
| Component | Technology |
|-----------|------------|
| API Framework | FastAPI (Python) |
| LLM Integration | Ollama API (local) |
| Document Parsing | PyPDF2 (PDF), python-docx (Word) |
| Authentication | Static API Key (via X-API-Key header) |
| Containerization | Docker + Docker Compose |
---
3. Functional Requirements
3.1 API Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /api/v1/parse | Parse resume (PDF/DOCX) |
| GET | /health | Health check |
| GET | /api/v1/models | List available Ollama models |
3.2 Input Specification
- Content-Type: multipart/form-data
- File Formats: PDF (.pdf), Word (.docx)
- Max File Size: 10MB
- Required Field: file (uploaded document)
3.3 Authentication
- Type: Static API Key
- Header: X-API-Key
- Implementation: Configurable via environment variable API_KEY
3.4 Output Schema (JSON)
{
  success: true,
  data: {
    name: John Doe,
    email: john@example.com,
    phone: +1-234-567-8900,
    location: San Francisco, CA,
    skills: [Python, FastAPI, Docker],
    experience: [
      {
        company: Tech Corp,
        title: Senior Developer,
        duration: 2020 - Present,
        description: Led backend development...
      }
    ],
    education: [
      {
        institution: MIT,
        degree: B.S. Computer Science,
        year: 2020
      }
    ],
    summary: Experienced developer...
  },
  processing_time: 2.5s
}
---
4. Architecture
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─────────────┐      ┌─────────────────────────┐  │
│  │   FastAPI   │──────│      Ollama (LFM2:2.6B)  │  │
│  │   Service   │      │    (host network mode)   │  │
│  └─────────────┘      └─────────────────────────┘  │
│        │                                            │
│  ┌─────────────┐                                    │
│  │   Volume    │                                    │
│  │  (uploads)  │                                    │
│  └─────────────┘                                    │
│                                                     │
└─────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │ Client  │
    │(cURL/   │
    │ Postman)│
    └─────────┘
---
5. Docker Configuration
5.1 Services
- app: FastAPI application (Python 3.11)
- ollama: Ollama service with LFM2:2.6B model pre-loaded
5.2 Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| OLLAMA_HOST | Ollama API URL | http://host.docker.internal:11434 |
| OLLAMA_MODEL | Model name | lfm2:2.6b |
| API_KEY | Authentication key | changeme |
| MAX_FILE_SIZE | Max upload size (bytes) | 10485760 |
| LOG_LEVEL | Logging level | INFO |
---
6. Non-Functional Requirements
| Requirement | Specification |
|-------------|----------------|
| Performance | < 30s per resume (depends on LLM) |
| Scalability | Stateless API (horizontal scaling via container orchestration) |
| Security | File type validation, API key auth, no file storage |
| Error Handling | Graceful degradation with meaningful error messages |
---
7. Project Structure
resume-parser/
├── app/
│   ├── main.py              # FastAPI application
│   ├── routes/
│   │   └── parser.py        # Parse endpoint
│   ├── services/
│   │   ├── document.py      # PDF/DOCX extraction
│   │   └── llm.py           # Ollama integration
│   ├── schemas/
│   │   └── response.py      # Pydantic models
│   ├── config.py            # Configuration
│   └── requirements.txt     # Python dependencies
├── docker/
│   ├── Dockerfile           # FastAPI container
│   └── Dockerfile.ollama   # Ollama container (optional)
├── docker-compose.yml
├── .env.example
└── README.md
---
8. Implementation Phases
Phase 1: Core API
- [ ] FastAPI setup with logging
- [ ] Document parsing (PDF/DOCX)
- [ ] Basic Ollama integration
- [ ] Health endpoint
Phase 2: Authentication & Security
- [ ] API key authentication
- [ ] File type/size validation
- [ ] Rate limiting (optional)
Phase 3: LLM Prompt Engineering
- [ ] Structured output prompt for resume fields
- [ ] Error handling for malformed resumes
- [ ] Processing time optimization
Phase 4: Docker & Deployment
- [ ] Docker Compose setup
- [ ] Ollama container configuration
- [ ] Environment-based configuration
---
9. Acceptance Criteria
1. ✅ API accepts PDF/DOCX files via multipart upload
2. ✅ Returns structured JSON with all resume fields
3. ✅ Authentication via X-API-Key header works
4. ✅ Docker Compose starts both services successfully
5. ✅ Health check returns service status
6. ✅ Invalid files return appropriate error messages
---
Would you like me to proceed with implementation, or would you like to adjust any requirements (e.g., add more output fields, change auth mechanism)?
▣  Plan · minimax/minimax-m2.5 · 2m 11s
right now first make a plan.md file let me audit it manually so that we can move further
Thinking: The user wants me to create a plan.md file for them to audit before moving forward. I'm in plan mode, so I can only read and analyze - no edits allowed. Let me create a comprehensive plan that covers all aspects of the resume parser project.
Let me structure the plan in a way that's easy to audit and covers:
1. Project overview
2. Technical architecture
3. Implementation steps
4. API design
5. Docker setup
6. Security considerations
7. Testing approach
I'll present this as a plan document for the user to review.
Implementation Plan: Resume Parser with Ollama LFM2:2.6B
1. Project Structure
resume-parser/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Configuration management
│   ├── routes/
│   │   ├── __init__.py
│   │   └── parser.py           # Parse endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── document.py         # PDF/DOCX text extraction
│   │   └── llm.py              # Ollama API client
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── response.py         # Pydantic response models
│   └── middleware/
│       ├── __init__.py
│       └── auth.py              # API key authentication
├── docker/
│   ├── Dockerfile              # FastAPI container
│   └── .dockerignore
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   └── test_document.py
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── plan.md                     # This file
└── README.md
---
2. API Design
2.1 Endpoints
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | No | Health check |
| GET | /api/v1/models | Yes | List Ollama models |
| POST | /api/v1/parse | Yes | Parse resume |
2.2 Request/Response Format
POST /api/v1/parse
Request (multipart/form-data):
- file: PDF or DOCX file (max 10MB)
Response (JSON):
{
  success: true,
  data: {
    name: string,
    email: string,
    phone: string,
    location: string,
    skills: [string],
    experience: [{company, title, duration, description}],
    education: [{institution, degree, year}],
    summary: string
  },
  processing_time_ms: 2500
}
---
3. Docker Architecture
3.1 Services (docker-compose.yml)
services:
  app:
    build: ./docker
    ports:
      - "8000:8000"
    environment:
      - OLLAMA_HOST=http://host.docker.internal:11434
      - OLLAMA_MODEL=lfm2:2.6b
      - API_KEY=${API_KEY}
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./uploads:/app/uploads
    restart: unless-stopped
3.2 Ollama Integration
- Ollama runs on host network (not in container) for GPU access
- App connects via host.docker.internal:11434
- Model: lfm2:2.6b (pre-pull on host before deployment)
---
4. Implementation Phases
Phase 1: Foundation
- [ ] Set up FastAPI project structure
- [ ] Create requirements.txt with dependencies
- [ ] Implement config.py (environment-based)
- [ ] Set up logging
Phase 2: Document Processing
- [ ] Implement PDF extraction (PyPDF2)
- [ ] Implement DOCX extraction (python-docx)
- [ ] Add file validation (type, size)
Phase 3: LLM Integration
- [ ] Create Ollama service client
- [ ] Design prompt for structured extraction
- [ ] Handle streaming/async responses
- [ ] Implement response parsing
Phase 4: API & Authentication
- [ ] Implement /health endpoint
- [ ] Implement /api/v1/parse endpoint
- [ ] Add API key authentication middleware
- [ ] Add request validation (Pydantic)
Phase 5: Docker & DevOps
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Create .env.example
- [ ] Add health checks
Phase 6: Testing
- [ ] Write unit tests for document parsing
- [ ] Write integration tests for API
- [ ] Add sample resume for testing
---
5. Dependencies (requirements.txt)
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pypdf==5.1.0
python-docx==1.1.2
httpx==0.27.2
pydantic==2.9.2
pydantic-settings==2.6.0
python-dotenv==1.0.1
---
6. Security Requirements
| Requirement | Implementation |
|-------------|----------------|
| API Authentication | Static API key via X-API-Key header |
| File Validation | Whitelist PDF/DOCX only |
| Size Limit | 10MB max file size |
| No Persistent Storage | Temp files cleaned after processing |
---
7. Configuration Variables
| Variable | Description | Required |
|----------|-------------|----------|
| OLLAMA_HOST | Ollama API URL | Yes |
| OLLAMA_MODEL | Model name (default: lfm2:2.6b) | Yes |
| API_KEY | Authentication key | Yes |
| MAX_FILE_SIZE | Max upload bytes (default: 10485760) | No |
| LOG_LEVEL | Logging level (default: INFO) | No |
---
8. Open Questions for Audit
1. Model Loading: Should the model be pulled in the Docker startup, or assume it's pre-installed on host?
2. Error Handling: What specific error messages should be returned for different failure scenarios?
3. Rate Limiting: Should we add rate limiting per API key?
4. Async Processing: For production, should we use a task queue (Celery/Redis) for long-running parsing?
5. Monitoring: Should we add Prometheus metrics endpoint?
---
This plan is ready for your audit. Let me know if you want to adjust any scope or add more details before implementation begins.
# Resume parser PRD with Ollama LLM2
▣  Plan · minimax/minimax-m2.5 · 1m 8s