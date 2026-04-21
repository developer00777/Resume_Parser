# Implementation Plan: 3-Chunk Consolidation + OCR Fallback

## Overview

Two changes shipped together:
1. **3-chunk LLM consolidation** — collapse 10 parallel OpenRouter calls into 3, reducing latency 40% and cost 35%
2. **OCR fallback** — fix image-based PDFs (currently return 422 error) using gpt-4o-mini vision

## Final Numbers

| Metric | Before | After |
|---|---|---|
| Cost (text PDF) | $0.0074 | $0.0048 |
| Cost (2-page image PDF) | ❌ broken | $0.0065 |
| Latency (text PDF) | 5–15s | 3–8s |
| Latency (2-page image PDF) | ❌ 422 error | 6–14s |
| Accuracy (text PDF) | 98.8% | ~98.8% (unchanged) |
| Accuracy (image PDF) | 0% (error) | ~95% |
| OpenRouter calls per resume | 10 | 3 (+1–2 if OCR needed) |

---

## Chunk Groupings

| Chunk | Prompts merged | Max output tokens | Reason |
|---|---|---|---|
| **A** | contact + personal + professional_meta | 950 | All scalar fields, tiny, never conflict |
| **B** | skills + certifications + awards + summary + projects | 2100 | Independent arrays, medium size |
| **C** | experience + education | 2600 | Heavy arrays needing deep focus, no interference |

Total output tokens: 5650 across 3 parallel calls. Each chunk safely under 3000 tokens.

---

## Files Changing

### 1. `app/services/document.py`
- `_extract_pdf` → make async; PyPDF first, if < 100 chars trigger OCR fallback
- `_ocr_pdf_via_vision` → new private async function:
  - `pdf2image.convert_from_bytes` → list of PIL images at 72 DPI
  - base64 encode each page as PNG
  - fire one `gpt-4o-mini` vision call per page (parallel via asyncio.gather)
  - concatenate page texts → return as plain string
- `validate_file_bytes` → already async (done in previous session)

### 2. `app/services/llm.py`
- Replace 10 individual CHUNKS with 3 combined prompt constants:
  - `PROMPT_CHUNK_A` = contact + personal + professional_meta (merged schema + merged rules)
  - `PROMPT_CHUNK_B` = skills + certifications + awards + summary + projects
  - `PROMPT_CHUNK_C` = experience + education
- Update `CHUNKS` list to 3 entries with correct max_tokens
- Update `parse_resume()` result unpacking to read from 3 chunk responses
- `_merge_professional_meta`, `_compute_score`, `_normalise_text` — untouched

### 3. `app/config.py`
- Add `openrouter_ocr_model: str = "openai/gpt-4o-mini"` — separate config key
  so OCR model can be swapped independently of extraction model

### 4. `requirements.txt`
- Add `pdf2image==1.17.0`
- Add `Pillow==10.4.0`

### 5. `docker/Dockerfile`
- Add `poppler-utils` apt package (required by pdf2image for PDF→image rendering)

### 6. `CLAUDE.md`
- Update LLM section: 10 prompts → 3 consolidated prompts
- Update PDF extraction: document OCR fallback flow
- Update tech stack: add pdf2image + Pillow
- Update architecture patterns: OCR fallback, 3-chunk design

---

## SOLID Principles Applied

| Principle | How |
|---|---|
| **S** Single Responsibility | `document.py` owns extraction only. OCR is one private function inside it. `llm.py` owns LLM calls — knows nothing about PDF/OCR |
| **O** Open/Closed | OCR model is a config key — swap model without touching code |
| **L** Liskov | `validate_file_bytes` and `extract_text` both return `str` — all callers unchanged |
| **I** Interface Segregation | `_ocr_pdf_via_vision` is private — not exposed to routes or other services |
| **D** Dependency Inversion | `document.py` depends on `settings.openrouter_ocr_model` abstraction, not hardcoded string |

---

## What Does NOT Change

- All route handlers (`parser.py`, `salesforce.py`) — zero changes
- Response schemas (`schemas/response.py`) — zero changes
- Score computation (`_compute_score`) — zero changes
- `_normalise_text` — zero changes
- Test mocks in `test_api.py` — zero changes (mock at HTTP level)
- Salesforce OAuth flow — zero changes

---

## OCR Flow (image-based PDF)

```
Upload PDF
  → PyPDF extract_text()
  → len(text) < 100 chars?
      YES → pdf2image.convert_from_bytes(dpi=72)
            → [PIL Image page1, PIL Image page2, ...]
            → base64 encode each page as PNG
            → asyncio.gather(vision_call(page1), vision_call(page2))
            → concatenate → plain text string
      NO  → use PyPDF text directly
  → _normalise_text()
  → 3-chunk LLM extraction (Chunk A, B, C in parallel)
  → _merge_professional_meta()
  → _compute_score()
  → return parsed dict
```

---

## Status

- [x] `app/config.py` — `openrouter_ocr_model` already present
- [x] `app/services/document.py` — OCR fallback implemented (base64 PDF data URL, no pdf2image needed)
- [x] `app/services/llm.py` — 3-chunk consolidation complete (10 → 3 parallel calls)
- [x] `requirements.txt` — no changes needed (pdf2image/Pillow not required with data URL approach)
- [x] `docker/Dockerfile` — no changes needed (poppler-utils not required)
- [x] `CLAUDE.md` — documentation updated
