# Resume Parser API — Complete Documentation

> **Version:** 1.0.0
> **Base URL (local dev):** `http://localhost:8000`
> **Base URL (Docker):** `http://localhost:8000`
> **Interactive docs:** `http://localhost:8000/docs` (Swagger UI) · `http://localhost:8000/redoc`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Authentication](#3-authentication)
4. [Environment & Configuration](#4-environment--configuration)
5. [Running the API](#5-running-the-api)
6. [Endpoints — System](#6-endpoints--system)
7. [Endpoints — Web App (Generic)](#7-endpoints--web-app-generic)
8. [Endpoints — Salesforce](#8-endpoints--salesforce)
9. [Response Schemas](#9-response-schemas)
10. [Salesforce Integration Guide](#10-salesforce-integration-guide)
11. [Error Reference](#11-error-reference)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Testing](#13-testing)

---

## 1. Overview

The **Resume Parser API** extracts structured candidate data from PDF and DOCX resume files using a locally-running Ollama LLM. It exposes two output modes from the same parsing engine:

| Mode | Who uses it | Output |
|---|---|---|
| **Web App** | Any client that uploads a file directly | Generic JSON (`ResumeData`) |
| **Salesforce** | Salesforce Apex / Flow via OAuth2 | SCSCHAMPS-mapped JSON (`SalesforceResumeData`) |

Both modes run simultaneously on the same server instance.

**Processing pipeline (shared by both modes):**

```
Upload / Fetch file
      ↓
Validate type + size (PDF / DOCX, max 10 MB)
      ↓
Extract raw text (PyPDF / python-docx)
      ↓
Split into sections by regex header detection
      ↓
8 focused LLM calls to Ollama (contact, skills, experience,
  education, certifications, projects, awards, summary)
      ↓
Rule-based resume scoring (0–100)
      ↓
Return JSON
```

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Clients                             │
│   Web App   │   Salesforce Apex   │   Postman / cURL    │
└──────┬──────┴──────────┬──────────┴────────────┬────────┘
       │                 │                        │
       │         X-API-Key header on all requests │
       ▼                 ▼                        ▼
┌────────────────────────────────────────────────────────┐
│                  FastAPI Application                    │
│                                                        │
│  APIKeyMiddleware  ──►  /api/v1/parse                  │
│                         /api/v1/models                 │
│                         /api/v1/salesforce/parse-*     │
│                         /health                        │
├────────────────────────────────────────────────────────┤
│  Services                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │ document.py  │  │    llm.py    │  │salesforce.py│  │
│  │ PDF/DOCX     │  │ Ollama calls │  │ OAuth2 +    │  │
│  │ extraction   │  │ 8 prompts    │  │ file fetch  │  │
│  └──────────────┘  └──────┬───────┘  └──────┬──────┘  │
└─────────────────────────── │ ─────────────── │ ────────┘
                             ▼                 ▼
                    ┌─────────────┐   ┌──────────────────┐
                    │   Ollama    │   │   Salesforce API  │
                    │ :11434      │   │  login.sf.com     │
                    └─────────────┘   └──────────────────┘
```

**Key files:**

| File | Responsibility |
|---|---|
| `app/main.py` | App entry point, registers routers and middleware |
| `app/config.py` | All environment-based config via Pydantic Settings |
| `app/middleware/auth.py` | API key validation on every protected route |
| `app/routes/parser.py` | Generic web-app endpoints |
| `app/routes/salesforce.py` | Salesforce-specific endpoints |
| `app/services/document.py` | File validation and text extraction |
| `app/services/llm.py` | Ollama LLM integration and resume scoring |
| `app/services/salesforce.py` | Salesforce OAuth2 + file download |
| `app/schemas/response.py` | All Pydantic request/response models + `map_to_salesforce()` |

---

## 3. Authentication

All endpoints under `/api/v1/*` require an API key passed in the request header.

```
X-API-Key: <your-api-key>
```

**Public paths (no key required):**
- `GET /health`
- `GET /docs`
- `GET /redoc`
- `GET /openapi.json`

**Response when key is missing or wrong:**

```json
HTTP 401 Unauthorized

{
  "success": false,
  "detail": "Invalid or missing API key."
}
```

**Setting the API key:**
Set `API_KEY=<value>` in your `.env` file (default: `changeme`). Always change this before deploying.

---

## 4. Environment & Configuration

Copy `.env.example` to `.env` and fill in values.

### Core variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama service endpoint |
| `OLLAMA_MODEL` | `qwen2.5:3b` | LLM model to use |
| `API_KEY` | `changeme` | API authentication key |
| `MAX_FILE_SIZE` | `10485760` | Max upload size in bytes (10 MB) |
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG / INFO / WARNING / ERROR) |

### Salesforce variables

Required only for `/api/v1/salesforce/*` endpoints.

| Variable | Description |
|---|---|
| `SF_CLIENT_ID` | Connected App Consumer Key |
| `SF_CLIENT_SECRET` | Connected App Consumer Secret |
| `SF_USERNAME` | (Optional) Salesforce username — only for password OAuth flow |
| `SF_PASSWORD` | (Optional) Salesforce password — only for password OAuth flow |
| `SF_SECURITY_TOKEN` | (Optional) Security token appended to password for non-trusted IPs |
| `SF_LOGIN_URL` | `https://login.salesforce.com` (prod) or `https://test.salesforce.com` (sandbox) |
| `SF_API_VERSION` | `59.0` |

**OAuth flow selection logic:**
- If `SF_USERNAME` and `SF_PASSWORD` are set → **Username-Password grant**
- Otherwise → **Client Credentials grant** (requires the Connected App to have it enabled)

---

## 5. Running the API

### Local development

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Copy and edit config
cp .env.example .env
# Edit .env with your API_KEY and optionally Salesforce creds

# 3. Start Ollama separately (with a downloaded model)
ollama pull qwen2.5:3b
ollama serve

# 4. Run the API
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Compose (recommended)

```bash
# Start FastAPI + Ollama together
docker-compose up --build

# Pull a model into the running Ollama container
docker-compose exec ollama ollama pull qwen2.5:3b

# Stop
docker-compose down
```

Services started:
- `app` — FastAPI on port `8000`
- `ollama` — Ollama inference engine on port `11434` (internal only, not exposed to internet)

---

## 6. Endpoints — System

### `GET /health`

Check if the API and Ollama are running.

**Authentication:** Not required

**Request:**
```bash
curl http://localhost:8000/health
```

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "ollama_connected": true,
  "model": "qwen2.5:3b"
}
```

`status` is `"degraded"` if Ollama is unreachable (API itself is still up).

---

## 7. Endpoints — Web App (Generic)

### `POST /api/v1/parse`

Upload a resume file (PDF or DOCX) and receive structured JSON.

**Authentication:** `X-API-Key` header required

**Request:**
```
Content-Type: multipart/form-data
Body field:   file  (PDF or DOCX, max 10 MB)
```

**cURL example:**
```bash
curl -X POST http://localhost:8000/api/v1/parse \
  -H "X-API-Key: changeme" \
  -F "file=@/path/to/resume.pdf"
```

**Python example:**
```python
import httpx

with open("resume.pdf", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/parse",
        headers={"X-API-Key": "changeme"},
        files={"file": ("resume.pdf", f, "application/pdf")},
        timeout=120,
    )

data = response.json()
print(data["data"]["name"])
print(data["data"]["skills"])
```

**JavaScript / fetch example:**
```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);

const res = await fetch("http://localhost:8000/api/v1/parse", {
  method: "POST",
  headers: { "X-API-Key": "changeme" },
  body: form,
});

const { data } = await res.json();
console.log(data.name, data.skills);
```

**Successful response `200 OK`:**
```json
{
  "success": true,
  "processing_time_ms": 18432.5,
  "data": {
    "name": "Jane Doe",
    "email": "jane@example.com",
    "phone": "+911234567890",
    "number": null,
    "current_location": "Mumbai, Maharashtra",
    "skills": ["Python", "FastAPI", "Docker", "AWS", "PostgreSQL"],
    "experience": [
      {
        "company": "Acme Corp",
        "title": "Senior Software Engineer",
        "duration": "Jan 2022 – Present",
        "description": "Led backend team building microservices in Python and Go."
      }
    ],
    "education": [
      {
        "institution": "IIT Bombay",
        "degree": "B.Tech",
        "field_of_study": "Computer Science",
        "year": "2021",
        "grade": "8.9 CGPA"
      }
    ],
    "projects": [
      {
        "name": "Resume Parser",
        "duration": "3 months",
        "description": "Built an LLM-powered resume parsing API."
      }
    ],
    "certifications": [
      { "name": "AWS Solutions Architect – Associate", "issuer": "Amazon" }
    ],
    "awards": [
      { "name": "Employee of the Quarter", "year": "2023" }
    ],
    "summary": "Experienced backend engineer with 3+ years building scalable APIs and distributed systems.",
    "resume_score": {
      "overall": 84,
      "content": 90,
      "experience_relevance": 85,
      "skills_match": 88,
      "education": 75,
      "remarks": "Well-structured resume with good coverage across all sections."
    }
  }
}
```

---

### `GET /api/v1/models`

List all models currently available in the connected Ollama instance.

**Authentication:** `X-API-Key` header required

**cURL example:**
```bash
curl http://localhost:8000/api/v1/models \
  -H "X-API-Key: changeme"
```

**Response `200 OK`:**
```json
{
  "success": true,
  "models": [
    {
      "name": "qwen2.5:3b",
      "size": 1927897856,
      "modified_at": "2024-11-10T12:00:00Z"
    },
    {
      "name": "llama3.2:3b",
      "size": 2019000320,
      "modified_at": "2024-10-20T08:30:00Z"
    }
  ]
}
```

---

## 8. Endpoints — Salesforce

All three Salesforce endpoints use the same OAuth2 token (cached in memory, auto-refreshed on 401). They return **`SalesforceResumeData`** — field names matching `SCSCHAMPS__<Field>__c` with the `SCSCHAMPS__` prefix and `__c` suffix stripped.

---

### `POST /api/v1/salesforce/parse-candidate`

Fetch a resume from a **Salesforce Candidate record** (`SCSCHAMPS__Candidate__c`) by its record ID, parse it, and return SCSCHAMPS-mapped JSON.

The endpoint reads `SCSCHAMPS__Resume_Attachment_Id__c` from the record first; if empty, falls back to `SCSCHAMPS__Resume_URL__c`.

**Authentication:** `X-API-Key` header required
**Salesforce credentials:** `SF_CLIENT_ID` + `SF_CLIENT_SECRET` must be set in `.env`

**Query parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `record_id` | string | Yes | The `SCSCHAMPS__Candidate__c` record ID (18-char Salesforce ID) |

**cURL example:**
```bash
curl -X POST \
  "http://localhost:8000/api/v1/salesforce/parse-candidate?record_id=a0012345678ABCDE" \
  -H "X-API-Key: changeme"
```

**Response `200 OK`:**
```json
{
  "success": true,
  "processing_time_ms": 22150.0,
  "data": {
    "Phone": "+911234567890",
    "PhoneNumber": "+911234567890",
    "AlternatePhoneNumber": null,
    "Current_Location": "Mumbai, Maharashtra",
    "City": "Mumbai",
    "State": "Maharashtra",
    "CurrentCompany": "Acme Corp",
    "CurrentDesignation": "Senior Software Engineer",
    "CurrentDuration": "Jan 2022 – Present",
    "Designation": "Senior Software Engineer",
    "Company": "Acme Corp",
    "Primary_Skills": "Python\nFastAPI\nDocker\nAWS\nPostgreSQL\nReact\nSQL\nKubernetes\nGo\nTerraform",
    "Technical_Skills": "Python\nFastAPI\nDocker\nAWS\nPostgreSQL\nReact\nSQL\nKubernetes\nGo\nTerraform",
    "SkillList": "Python, FastAPI, Docker, AWS, PostgreSQL, React, SQL, Kubernetes, Go, Terraform",
    "AutoPopulate_Skillset": true,
    "ResumeRich": "<b>Senior Software Engineer</b> at Acme Corp (Jan 2022 – Present)<br/>Led backend team building microservices.",
    "resume_score": {
      "overall": 84,
      "content": 90,
      "experience_relevance": 85,
      "skills_match": 88,
      "education": 75,
      "remarks": "Well-structured resume with good coverage across all sections."
    },
    "Title": null,
    "AadharNumber": null,
    "AlternateEmail": null,
    "LinkedIn_URL": null,
    "Web_address": null,
    "DateOfBirth": null,
    "Gender": null,
    "Preferred_Location": null,
    "Department": null,
    "Years_of_Experience": null,
    "General_Skills": null,
    "Key_Skillsets_del": null,
    "Current_CTC": null,
    "Expected_CTC": null,
    "Notice_Period": null,
    "Available_To_Start": null,
    "Resume_URL": null,
    "Resume_Attachment_Id": null,
    "Resume": null,
    "Candidate_Status": null,
    "Background_Check": null,
    "Source": null,
    "Talent_Id": null,
    "Job_Id": null,
    "job": null,
    "Lead": null,
    "Recruiter": null,
    "converted_from_lead": false
  }
}
```

---

### `POST /api/v1/salesforce/parse-attachment`

Download a resume directly by a Salesforce **ContentVersion ID** or **Attachment ID** and parse it.

Use this when you already have the attachment ID without needing to look up a full candidate record.

**Query parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `attachment_id` | string | Yes | Salesforce ContentVersion or Attachment ID |
| `filename` | string | No | Filename hint for extension detection (default: `resume.pdf`) |

**cURL example:**
```bash
curl -X POST \
  "http://localhost:8000/api/v1/salesforce/parse-attachment?attachment_id=068XXXXXXXXXXXX&filename=resume.pdf" \
  -H "X-API-Key: changeme"
```

**Response:** Same `SalesforceParseResponse` structure as `parse-candidate`.

---

### `POST /api/v1/salesforce/parse-url`

Download a resume from a URL stored in `SCSCHAMPS__Resume_URL__c` (or any URL) and parse it.

Supports:
- Absolute URLs (`https://...`)
- Salesforce-relative paths (`/services/data/...`) — resolved against the instance URL automatically with a Bearer token

**Query parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `resume_url` | string | Yes | Full URL or Salesforce-relative path |
| `filename` | string | No | Filename hint for extension detection (default: `resume.pdf`) |

**cURL example:**
```bash
curl -X POST \
  "http://localhost:8000/api/v1/salesforce/parse-url?resume_url=https://yourorg.my.salesforce.com/services/data/v59.0/sobjects/ContentVersion/068XXX/VersionData" \
  -H "X-API-Key: changeme"
```

**Response:** Same `SalesforceParseResponse` structure as `parse-candidate`.

---

## 9. Response Schemas

### `ResumeData` (generic web-app response)

```
ResumeData
├── name              string | null
├── email             string | null
├── phone             string | null
├── number            string | null    # alternate/secondary phone
├── current_location  string | null
├── skills            string[]
├── experience        Experience[]
│   ├── company       string | null
│   ├── title         string | null
│   ├── duration      string | null
│   └── description   string | null
├── education         Education[]
│   ├── institution   string | null
│   ├── degree        string | null
│   ├── field_of_study string | null
│   ├── year          string | null
│   └── grade         string | null
├── projects          Project[]
│   ├── name          string | null
│   ├── duration      string | null
│   └── description   string | null
├── certifications    Certification[]
│   ├── name          string | null
│   └── issuer        string | null
├── awards            Award[]
│   ├── name          string | null
│   └── year          string | null
├── summary           string | null
└── resume_score      ResumeScore
    ├── overall              int (0–100)
    ├── content              int (0–100)
    ├── experience_relevance int (0–100)
    ├── skills_match         int (0–100)
    ├── education            int (0–100)
    └── remarks              string | null
```

### `SalesforceResumeData` (SCSCHAMPS-mapped response)

All fields correspond to `SCSCHAMPS__<Field>__c` in Salesforce, with prefix and suffix stripped.

```
SalesforceResumeData
├── Title                   string | null   → SCSCHAMPS__Title__c
├── AadharNumber            string | null   → SCSCHAMPS__AadharNumber__c
├── AlternateEmail          string | null   → SCSCHAMPS__AlternateEmail__c
├── AlternatePhoneNumber    string | null   → SCSCHAMPS__AlternatePhoneNumber__c
├── Phone                   string | null   → SCSCHAMPS__Phone__c
├── PhoneNumber             string | null   → SCSCHAMPS__PhoneNumber__c
├── LinkedIn_URL            string | null   → SCSCHAMPS__LinkedIn_URL__c
├── Web_address             string | null   → SCSCHAMPS__Web_address__c
├── DateOfBirth             string | null   → SCSCHAMPS__DateOfBirth__c
├── Gender                  string | null   → SCSCHAMPS__Gender__c
├── City                    string | null   → SCSCHAMPS__City__c
├── State                   string | null   → SCSCHAMPS__State__c
├── Current_Location        string | null   → SCSCHAMPS__Current_Location__c
├── Preferred_Location      string | null   → SCSCHAMPS__Preferred_Location__c
├── CurrentDesignation      string | null   → SCSCHAMPS__CurrentDesignation__c
├── Designation             string | null   → SCSCHAMPS__Designation__c
├── Department              string | null   → SCSCHAMPS__Department__c
├── Company                 string | null   → SCSCHAMPS__Company__c
├── CurrentCompany          string | null   → SCSCHAMPS__CurrentCompany__c
├── CurrentDuration         string | null   → SCSCHAMPS__CurrentDuration__c
├── Years_of_Experience     string | null   → SCSCHAMPS__Years_of_Experience__c
├── Primary_Skills          string | null   → SCSCHAMPS__Primary_Skills__c  (newline-separated, top 10)
├── Technical_Skills        string | null   → SCSCHAMPS__Technical_Skills__c (all skills, newline-sep)
├── General_Skills          string | null   → SCSCHAMPS__General_Skills__c
├── SkillList               string | null   → SCSCHAMPS__SkillList__c  (comma-separated)
├── AutoPopulate_Skillset   bool            → SCSCHAMPS__AutoPopulate_Skillset__c
├── Key_Skillsets_del       string | null   → SCSCHAMPS__Key_Skillsets_del__c
├── Current_CTC             string | null   → SCSCHAMPS__Current_CTC__c
├── Expected_CTC            string | null   → SCSCHAMPS__Expected_CTC__c
├── Notice_Period           string | null   → SCSCHAMPS__Notice_Period__c
├── Available_To_Start      string | null   → SCSCHAMPS__Available_To_Start_c
├── ResumeRich              string | null   → SCSCHAMPS__ResumeRich__c  (HTML summary of experience)
├── Resume_URL              string | null   → SCSCHAMPS__Resume_URL__c
├── Resume_Attachment_Id    string | null   → SCSCHAMPS__Resume_Attachment_Id__c
├── Resume                  string | null   → SCSCHAMPS__Resume__c
├── Candidate_Status        string | null   → SCSCHAMPS__Candidate_Status__c
├── Background_Check        string | null   → SCSCHAMPS__Background_Check__c
├── Source                  string | null   → SCSCHAMPS__Source__c
├── Talent_Id               string | null   → SCSCHAMPS__Talent_Id__c
├── Job_Id                  string | null   → SCSCHAMPS__Job_Id__c
├── job                     string | null   → SCSCHAMPS__job__c
├── Lead                    string | null   → SCSCHAMPS__Lead__c
├── Recruiter               string | null   → SCSCHAMPS__Recruiter__c
├── converted_from_lead     bool            → SCSCHAMPS__converted_from_lead__c
└── resume_score            ResumeScore     (same as generic, for reference — not a SCSCHAMPS field)
```

### `ResumeScore` — scoring logic

| Field | Weight | What it measures |
|---|---|---|
| `content` | 20% | Completeness of contact info + summary presence |
| `skills_match` | 25% | Number of skills listed (0 → 0, 20+ → 100) |
| `experience_relevance` | 30% | Number of roles + presence of descriptions + bonus for certs/projects |
| `education` | 25% | Degree name, field of study, year, grade presence |
| `overall` | — | Weighted average of the above four |

---

## 10. Salesforce Integration Guide

This section is a step-by-step guide for connecting your Salesforce org to the Resume Parser API.

---

### Step 1 — Create a Connected App in Salesforce

1. Go to **Setup → App Manager → New Connected App**
2. Fill in:
   - **Connected App Name:** Resume Parser Integration
   - **API Name:** Resume_Parser_Integration
   - **Contact Email:** your email
3. Under **OAuth Settings**, check **Enable OAuth Settings**
4. **Callback URL:** `https://your-api-domain.com/oauth/callback` (placeholder — not used for server flows)
5. **Selected OAuth Scopes:**
   - `api` — Access and manage your data
   - `refresh_token, offline_access` (if using password flow)
6. Check **Enable Client Credentials Flow** if you want the `client_credentials` grant (no username/password in the API server)
7. Save. Wait 10 minutes for Salesforce to propagate the app.
8. Copy **Consumer Key** → `SF_CLIENT_ID` in `.env`
9. Copy **Consumer Secret** → `SF_CLIENT_SECRET` in `.env`

---

### Step 2 — Whitelist your API server IP in Salesforce

1. Go to **Setup → Security → Network Access**
2. Add your server's IP address as a Trusted IP Range
3. This avoids needing `SF_SECURITY_TOKEN` appended to the password

---

### Step 3 — Set up a Remote Site in Salesforce (for Apex callouts TO your API)

1. Go to **Setup → Security → Remote Site Settings → New**
2. **Remote Site Name:** Resume_Parser_API
3. **Remote Site URL:** `https://your-api-domain.com` (must be HTTPS)
4. Save

---

### Step 4 — Create a Named Credential (recommended)

1. Go to **Setup → Security → Named Credentials → New Legacy**
2. **Label:** Resume Parser API
3. **Name:** Resume_Parser_API
4. **URL:** `https://your-api-domain.com`
5. **Identity Type:** Named Principal
6. **Authentication Protocol:** No Authentication (we handle it via header)
7. Save

---

### Step 5 — Apex Callout: Push a Candidate Record ID to the API

This pattern calls the API from Salesforce, passing a record ID. The API fetches the resume file from Salesforce internally.

```apex
public class ResumeParserService {

    private static final String NAMED_CREDENTIAL = 'callout:Resume_Parser_API';
    private static final String API_KEY = 'changeme'; // Store in Custom Settings or Custom Metadata

    /**
     * Parses the resume attached to a Candidate record.
     * Call this from a Flow, Trigger, or Lightning component.
     * Must be @future or Queueable because it makes a callout.
     */
    @future(callout=true)
    public static void parseCandidate(String recordId) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(NAMED_CREDENTIAL + '/api/v1/salesforce/parse-candidate?record_id=' + recordId);
        req.setMethod('POST');
        req.setHeader('X-API-Key', API_KEY);
        req.setTimeout(120000); // 120s max (Salesforce limit)

        Http http = new Http();
        HttpResponse res = http.send(req);

        if (res.getStatusCode() == 200) {
            ResumeParseResponse parsed = (ResumeParseResponse) JSON.deserialize(
                res.getBody(), ResumeParseResponse.class
            );
            updateCandidateRecord(recordId, parsed.data);
        } else {
            System.debug('Parse failed: ' + res.getStatus() + ' — ' + res.getBody());
        }
    }

    private static void updateCandidateRecord(String recordId, SFResumeData data) {
        SCSCHAMPS__Candidate__c candidate = new SCSCHAMPS__Candidate__c(
            Id = recordId,
            SCSCHAMPS__Phone__c              = data.Phone,
            SCSCHAMPS__PhoneNumber__c        = data.PhoneNumber,
            SCSCHAMPS__Current_Location__c   = data.Current_Location,
            SCSCHAMPS__City__c               = data.City,
            SCSCHAMPS__State__c              = data.State,
            SCSCHAMPS__CurrentCompany__c     = data.CurrentCompany,
            SCSCHAMPS__CurrentDesignation__c = data.CurrentDesignation,
            SCSCHAMPS__CurrentDuration__c    = data.CurrentDuration,
            SCSCHAMPS__SkillList__c          = data.SkillList,
            SCSCHAMPS__Primary_Skills__c     = data.Primary_Skills,
            SCSCHAMPS__Technical_Skills__c   = data.Technical_Skills,
            SCSCHAMPS__AutoPopulate_Skillset__c = data.AutoPopulate_Skillset,
            SCSCHAMPS__ResumeRich__c         = data.ResumeRich
        );
        update candidate;
    }


    // ── Wrapper classes to deserialize the JSON response ─────────────────────

    public class ResumeParseResponse {
        public Boolean success;
        public SFResumeData data;
        public Decimal processing_time_ms;
    }

    public class SFResumeData {
        public String  Phone;
        public String  PhoneNumber;
        public String  AlternatePhoneNumber;
        public String  Current_Location;
        public String  City;
        public String  State;
        public String  CurrentCompany;
        public String  CurrentDesignation;
        public String  CurrentDuration;
        public String  Designation;
        public String  Company;
        public String  Primary_Skills;
        public String  Technical_Skills;
        public String  General_Skills;
        public String  SkillList;
        public Boolean AutoPopulate_Skillset;
        public String  ResumeRich;
        public String  Years_of_Experience;
        public String  Current_CTC;
        public String  Expected_CTC;
        public String  Notice_Period;
        public String  LinkedIn_URL;
        public String  Web_address;
    }
}
```

---

### Step 6 — Call from a Salesforce Flow (no Apex needed)

If you prefer a no-code approach, use **Flow → Action → HTTP Callout** (available in Salesforce Spring '23+):

1. Create an **External Service** pointing to your API's OpenAPI spec (`http://your-api/openapi.json`)
2. Use the auto-generated **Invocable Action** in a Record-Triggered Flow
3. Pass `{!$Record.Id}` as `record_id`
4. Map response fields back to the candidate record

---

### Step 7 — Trigger automatically on resume upload

Use a **Record-Triggered Flow** or **Apex Trigger** on `SCSCHAMPS__Candidate__c`:

```apex
trigger CandidateResumeTrigger on SCSCHAMPS__Candidate__c (after insert, after update) {
    for (SCSCHAMPS__Candidate__c c : Trigger.new) {
        SCSCHAMPS__Candidate__c old = Trigger.oldMap?.get(c.Id);

        // Only parse when attachment ID changes (new resume uploaded)
        Boolean attachmentChanged =
            c.SCSCHAMPS__Resume_Attachment_Id__c != null &&
            c.SCSCHAMPS__Resume_Attachment_Id__c != old?.SCSCHAMPS__Resume_Attachment_Id__c;

        if (attachmentChanged) {
            ResumeParserService.parseCandidate(c.Id);
        }
    }
}
```

---

### Salesforce API limits to keep in mind

| Limit | Value | Impact |
|---|---|---|
| HTTP callout timeout | **120 seconds** max | LLM parsing takes 15–30s — well within limit |
| Callouts per transaction | 100 | Only 1 callout per resume — no issue |
| Response body size | 12 MB | Resume JSON is ~5 KB — no issue |
| `@future` calls per request | 50 | Fine for batch processing |
| Daily API calls | Depends on org edition | Monitor via Setup → System Overview |

---

### Integration flow summary

```
Recruiter uploads PDF to Salesforce candidate record
              ↓
  SCSCHAMPS__Resume_Attachment_Id__c is set
              ↓
  Apex Trigger fires → ResumeParserService.parseCandidate(recordId)
              ↓
  POST /api/v1/salesforce/parse-candidate?record_id=...
  (API fetches PDF from Salesforce via OAuth2 internally)
              ↓
  Ollama LLM extracts all fields
              ↓
  Returns SCSCHAMPS-mapped JSON
              ↓
  Apex updates SCSCHAMPS__Candidate__c with parsed data
              ↓
  Recruiter sees populated fields instantly
```

---

## 11. Error Reference

| HTTP Status | When it occurs | Example detail |
|---|---|---|
| `400 Bad Request` | Unsupported file type | `"Unsupported file type: text/plain. Only PDF and DOCX are accepted."` |
| `401 Unauthorized` | Missing or wrong `X-API-Key` | `"Invalid or missing API key."` |
| `404 Not Found` | Salesforce record/attachment not found | `"Candidate record 'a001XXX' not found."` |
| `413 Payload Too Large` | File exceeds `MAX_FILE_SIZE` | `"File size exceeds maximum allowed size of 10485760 bytes."` |
| `422 Unprocessable Entity` | Text could not be extracted from file | `"Could not extract text from PDF. The file may be image-based or empty."` |
| `422 Unprocessable Entity` | Candidate has no resume attachment/URL | `"Candidate record has no resume attachment or URL set."` |
| `502 Bad Gateway` | Salesforce API returned an error | `"Salesforce authentication failed: ..."` |
| `503 Service Unavailable` | Ollama is not running | `"Ollama service is unavailable. Ensure it is running."` |
| `503 Service Unavailable` | Salesforce credentials not configured | `"Salesforce credentials are not configured. Set SF_CLIENT_ID..."` |
| `504 Gateway Timeout` | LLM took too long | `"LLM processing timed out."` |

---

## 12. CI/CD Pipeline

GitHub Actions runs automatically on every push and pull request.

**Pipeline:** `.github/workflows/ci.yml`

```
push / PR
    │
    ├─► lint        ruff check on app/ and tests/
    │
    ├─► test        pytest tests/test_api.py
    │               (Ollama + Salesforce fully mocked)
    │               Uploads results as artifact
    │
    └─► docker-build  docker buildx build docker/Dockerfile
                      Verifies image builds cleanly
```

**Running tests locally:**

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests (no Ollama or Salesforce needed)
pytest tests/test_api.py -v

# Run a specific test class
pytest tests/test_api.py::TestSalesforceEndpoint -v

# Run with coverage
pip install pytest-cov
pytest tests/test_api.py --cov=app --cov-report=term-missing
```

**Test coverage:**

| Test Class | What it covers |
|---|---|
| `TestHealth` | Health endpoint, Ollama connectivity |
| `TestAuth` | Missing key, wrong key, public paths |
| `TestParseEndpoint` | PDF upload, DOCX upload, wrong type, missing file |
| `TestModelsEndpoint` | Ollama model listing |
| `TestSalesforceEndpoint` | parse-candidate, parse-attachment, parse-url (all mocked) |
| `TestSalesforceMapping` | `map_to_salesforce()` field mapping, city/state extraction |
| `TestScoreComputation` | Score logic with full and empty resume data |

---

## 13. Testing

### Manual integration test (requires running services)

```bash
# Generate a sample PDF first
python tests/generate_sample_pdf.py

# Run integration test
python tests/test_parse_resume.py tests/sample_resume.pdf
```

This script:
1. Hits `/health` and verifies Ollama is connected
2. Lists available models
3. Verifies auth rejection with no key
4. Uploads the PDF to `/api/v1/parse` and prints all extracted fields

### Test the Salesforce endpoint manually (with real credentials)

```bash
# 1. Set your .env with real SF credentials
# 2. Start the API
uvicorn app.main:app --reload

# 3. Test with a real record ID
curl -X POST \
  "http://localhost:8000/api/v1/salesforce/parse-candidate?record_id=YOUR_RECORD_ID_HERE" \
  -H "X-API-Key: changeme"

# 4. Or test with a direct attachment ID
curl -X POST \
  "http://localhost:8000/api/v1/salesforce/parse-attachment?attachment_id=068XXXX" \
  -H "X-API-Key: changeme"
```
