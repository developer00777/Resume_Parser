# Async Parsing Queue (Redis + arq)

## 1. Why this exists

### The 504s were self-inflicted

Every 504 in the Railway request log lasted **exactly 1m 50s**. That is not
Railway's edge timing out — it is this line, which used to live in
`app/routes/parser.py`:

```python
_BULK_TIMEOUT = 110.0
...
results = await asyncio.wait_for(asyncio.gather(...), timeout=_BULK_TIMEOUT)
except asyncio.TimeoutError:
    raise HTTPException(status_code=504, ...)   # ← the 504, 110s on the dot
```

Successful requests in the same log ran 22s, 40s, 1m 22s, **1m 42s** — right up
against the 110s wall. The endpoint was a coin flip, and the coin was weighted
by how many image-based PDFs happened to land in the batch.

Worse than the 504 itself: `asyncio.gather` was cancelled wholesale, so a batch
where 13 of 15 resumes had already parsed returned **nothing**. All that
OpenRouter spend, discarded.

### Why it was slow

15 files per request, `_BULK_CONCURRENCY = 5` → 3 sequential waves. Each file
fires 3 parallel LLM chunks (A/B/C) that the logs show taking 8–26s, and any
image-based PDF adds a `gpt-4o-mini` vision OCR pass over the whole document.
Three waves × ~25–35s is already past 110s before anything goes wrong.

### Why the budget cannot simply be raised

Salesforce caps a single callout at **120 seconds**. The 110s budget was chosen
to sit under it. There is no number that both fits Salesforce's ceiling and
finishes a 15-resume batch. The request/response shape is the problem, not the
timeout value.

## 2. What changed

**Two fixes, independent of each other.**

**a) The synchronous endpoints no longer throw work away.** When the wall-clock
budget expires, finished resumes are returned intact and only the unfinished
ones are marked failed, each carrying a pointer to the queue. `POST
/api/v1/parse/salesforce` now answers 200 with partial results instead of 504.
No client change required — this alone stops the data loss.

**b) A Redis-backed queue removes the ceiling entirely.** Submitting returns a
`batch_id` in ~30ms, so the callout can never time out no matter how large the
batch. Workers drain the queue at their own pace.

```
                    ┌──────────────────────────────────────────┐
   Salesforce       │  app  (uvicorn)                          │
      │             │  accepts uploads, never parses inline    │
      │  POST .../jobs                                         │
      ├────────────►│  store bytes + enqueue 1 job per file    │
      │  202 {batch_id}  (~30 ms)                              │
      │             └───────────────┬──────────────────────────┘
      │                             │
      │                       ┌─────▼─────┐
      │                       │   redis   │  queue + batch store
      │                       └─────┬─────┘
      │                             │  BRPOP
      │             ┌───────────────▼──────────────────────────┐
      │             │  worker  (arq)   × N replicas            │
      │             │  extract_text → parse_resume → map       │
      │             │  writes each result row as it lands      │
      │             └──────────────────────────────────────────┘
      │  GET .../jobs/{batch_id}
      └────────────►  status + results so far
```

**One arq job per file, not per batch.** That is the design decision that makes
bulk parsing safe: the unit of retry, timeout and failure is a single resume, so
one corrupt PDF or one slow OCR pass can no longer sink the other fourteen.

## 3. Endpoints

Three submit transports, one batch model. `{mode}` is `jobs` (generic),
`salesforce/jobs`, or `client/jobs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/parse/{mode}` | multipart upload |
| POST | `/api/v1/parse/{mode}/base64` | JSON + base64 — no multipart body to hand-build |
| POST | `/api/v1/parse/{mode}/records` | **record ids only** — the worker downloads the files |
| GET | `/api/v1/parse/jobs/{batch_id}` | Status + results so far |
| DELETE | `/api/v1/parse/jobs/{batch_id}` | Drop a batch early |

**Prefer `/records` from Salesforce.** Apex heap is 6 MB sync / 12 MB async and
a 2 MB PDF costs roughly 8 MB once you hold the Blob, its base64 String and the
request body — so pushing resume *bytes* out of Apex in bulk is impossible
regardless of timeouts. Sending ids makes the payload a few KB whatever the
resumes weigh. See [salesforce/README.md](salesforce/README.md).

All require `X-API-Key`. Output shapes are unchanged — each result's `data` is
exactly the `SalesforceResumeData` / `ClientResumeData` / `ResumeData` object the
synchronous endpoints already return.

### Submit

```bash
curl -X POST https://<host>/api/v1/parse/salesforce/jobs \
  -H "X-API-Key: $API_KEY" \
  -F "files=@resume1.pdf" \
  -F "files=@resume2.pdf" \
  -F "callback_url=https://my.salesforce.com/services/apexrest/resumeCallback"  # optional
  -F "callback_token=$SF_TOKEN"                                                 # optional
```

```json
{
  "batch_id": "470fcaf84a214042a89ecfe7e6ba2b70",
  "status": "queued",
  "mode": "salesforce",
  "total": 2,
  "poll_url": "/api/v1/parse/jobs/470fcaf84a214042a89ecfe7e6ba2b70",
  "callback_url": null
}
```

Returns **202** immediately. Measured: 4 files in 28ms, 30 files in 338ms.

Uploads are validated up front — an unsupported type, an empty file or an
oversized file is rejected at submit (400/413), not discovered later as a failed
row. `application/octet-stream` is accepted when the filename ends in `.pdf` or
`.docx`, which is what Salesforce multipart callouts usually send.

### Poll

```bash
curl https://<host>/api/v1/parse/jobs/$BATCH_ID -H "X-API-Key: $API_KEY"
```

```json
{
  "batch_id": "470fcaf...",
  "status": "processing",
  "mode": "salesforce",
  "total": 15, "completed": 9, "parsed": 8, "failed": 1,
  "remaining": 6, "running": 5,
  "created_at": 1757500412.11, "updated_at": 1757500461.44, "finished_at": null,
  "total_processing_time_ms": 49330.0,
  "results": [
    {"index": 0, "filename": "a.pdf", "success": true,
     "external_id": "a0X1000000CandId",
     "data": { "...": "SCSCHAMPS fields" }, "error": null, "processing_time_ms": 21430.2},
    {"index": 1, "filename": "b.pdf", "success": false,
     "data": null, "error": "Failed to parse PDF: ...", "processing_time_ms": 812.0}
  ]
}
```

`status` is `queued` → `processing` → `completed`. **Results stream in as they
land**, so a caller can consume finished resumes without waiting for the slowest
file in the batch.

### Submit by record id

```bash
curl -X POST https://<host>/api/v1/parse/salesforce/jobs/records \
  -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
  -d '{"records":[
        {"ref":"068XXXXXXXXXXXX","kind":"attachment","external_id":"a0X1000000CandId"},
        {"ref":"a0X2000000Other","kind":"candidate"}
      ]}'
```

`kind` is `attachment` (ContentVersion/Attachment Id, the default), `candidate`
(reads the resume field off the record) or `url`. Requires `SF_CLIENT_ID` /
`SF_CLIENT_SECRET` on both the app and worker services — the same credentials
the existing `/api/v1/salesforce/*` endpoints use.

### Correlating results — `external_id`

Every submit shape accepts an `external_id` per file, echoed back untouched on
the result row (including rows the stale-reaper writes). Set it to the record
you want updated. Matching on `filename` is not safe: duplicates collide, and
for record submissions the parser prefers Salesforce's own filename.

### Callback (instead of polling)

Supply `callback_url` at submit and the finished batch is POSTed there once, with
the same JSON body the poll endpoint returns. `callback_token` is sent as
`Authorization: Bearer <token>`. Delivery is claimed atomically in Redis, so a
retried job can never double-deliver. If the POST fails it is logged and the
results stay readable by polling — the callback is an optimisation, never the
only copy.

Set `JOB_CALLBACK_ALLOWED_HOSTS` in production to stop the queue being pointed at
arbitrary hosts:

```
JOB_CALLBACK_ALLOWED_HOSTS=my.salesforce.com,mycompany.com
```

## 4. Configuration

| Variable | Default | Notes |
|---|---|---|
| `REDIS_URL` | `redis://localhost:6379/0` | Railway's Redis plugin injects this |
| `QUEUE_NAME` | `resume_parse` | Must match between app and worker |
| `QUEUE_MAX_FILES` | `100` | Cap per async batch |
| `WORKER_CONCURRENCY` | `8` | Resumes in flight per worker process |
| `JOB_TIMEOUT` | `300` | Per-file budget inside the worker (s) |
| `JOB_MAX_TRIES` | `3` | Attempts per file, transient errors only |
| `JOB_TTL` | `86400` | How long bytes and results live in Redis (s) |
| `JOB_STALE_SECONDS` | `1800` | Abandoned batch is reaped after this |
| `JOB_CALLBACK_TIMEOUT` | `30` | Callback POST timeout (s) |
| `JOB_CALLBACK_ALLOWED_HOSTS` | *(empty = any)* | Comma-separated host suffixes |

The synchronous path is configurable too — `BULK_MAX_FILES` (15),
`BULK_TIMEOUT` (110), `BULK_CONCURRENCY` (5). Keep `BULK_TIMEOUT` under 120 so it
stays inside the Salesforce callout ceiling.

## 5. Running it

### Local

```bash
docker compose up --build          # redis + app + worker
docker compose up -d --scale worker=3
docker compose logs -f worker
```

Redis runs with `--maxmemory-policy noeviction` on purpose: the queue holds
uploaded resume bytes, and an eviction policy would silently drop pending work
under memory pressure.

### Railway

Railway needs **three** services in the project — this is the part that is easy
to get half-right.

1. **Add the Redis plugin.** It injects `REDIS_URL` into services in the project.
   Reference it in the other two as `${{Redis.REDIS_URL}}`.

2. **The existing `app` service** stays as-is (the Dockerfile's default uvicorn
   command). Add `REDIS_URL`.

3. **Add a second service from the same repo** — the worker. Same image, one
   change:

   ```
   Custom Start Command:  arq app.worker.WorkerSettings
   ```

   Give it the same environment as `app` (OpenRouter key, Redis URL, log level).
   It needs no public domain and no `PORT`.

Scale throughput by raising the worker service's replica count, or
`WORKER_CONCURRENCY` within one replica. Watch OpenRouter rate limits as you do —
each in-flight resume is 3 concurrent chat completions.

> **About the Railway 404 Salesforce is seeing** — *"The train has not arrived at
> the station"* is Railway's edge saying no service is serving that hostname. It
> is not this application returning 404; the request never reached the app. Check
> that the domain is provisioned, that the service has a healthy deployment, and
> that the Named Credential / Remote Site URL in Salesforce matches the current
> Railway domain exactly. Note `app/services/llm_client.py` hardcodes
> `resumeparser-production-45b1.up.railway.app` as its `HTTP-Referer`, which is a
> useful hint as to which domain was in use.

## 6. Salesforce integration

Full reference Apex — Batch submit, chained Queueable poller, result mapper,
callback resource and a callout-mocked test class — lives in
[salesforce/](salesforce/), with the governor-limit reasoning behind it in
[salesforce/README.md](salesforce/README.md). The sketch below is the short
version.

### Option A — submit and poll

```apex
public class ResumeParserQueue {

    private static final String BASE = 'callout:Resume_Parser';  // Named Credential

    // 1. Submit. Returns immediately — no 120s callout risk.
    public static String submit(List<ContentVersion> resumes) {
        String boundary = '----ResumeParser' + String.valueOf(Crypto.getRandomLong());
        String body = MultipartBuilder.build(boundary, resumes);   // your existing helper

        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE + '/api/v1/parse/salesforce/jobs');
        req.setMethod('POST');
        req.setHeader('Content-Type', 'multipart/form-data; boundary=' + boundary);
        req.setHeader('X-API-Key', '{!$Credential.Password}');
        req.setBodyAsBlob(EncodingUtil.base64Decode(body));
        req.setTimeout(30000);                                     // 30s is plenty now

        HttpResponse res = new Http().send(req);
        if (res.getStatusCode() != 202) {
            throw new CalloutException('Submit failed: ' + res.getStatusCode()
                                       + ' ' + res.getBody());
        }
        Map<String, Object> out =
            (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
        return (String) out.get('batch_id');
    }

    // 2. Poll from a Queueable that re-enqueues itself until the batch completes.
    public class PollJob implements Queueable, Database.AllowsCallouts {
        private String batchId;
        private Integer attempt;

        public PollJob(String batchId, Integer attempt) {
            this.batchId = batchId;
            this.attempt = attempt;
        }

        public void execute(QueueableContext ctx) {
            HttpRequest req = new HttpRequest();
            req.setEndpoint(BASE + '/api/v1/parse/jobs/' + batchId);
            req.setMethod('GET');
            req.setHeader('X-API-Key', '{!$Credential.Password}');
            req.setTimeout(30000);

            HttpResponse res = new Http().send(req);
            Map<String, Object> batch =
                (Map<String, Object>) JSON.deserializeUntyped(res.getBody());

            if ('completed'.equals((String) batch.get('status'))) {
                ResumeParserApply.upsertCandidates(
                    (List<Object>) batch.get('results'));       // your mapping logic
                return;
            }
            if (attempt >= 40) {                                // ~20 min ceiling
                System.debug(LoggingLevel.ERROR, 'Batch ' + batchId + ' did not finish');
                return;
            }
            // Re-enqueue with a delay. Async callout limits are per transaction,
            // so each poll is a fresh transaction with a clean budget.
            System.enqueueJob(new PollJob(batchId, attempt + 1), 1);   // 1-minute delay
        }
    }
}
```

Call it as:

```apex
String batchId = ResumeParserQueue.submit(resumes);
System.enqueueJob(new ResumeParserQueue.PollJob(batchId, 0), 1);
```

Because `results` streams, an impatient version can apply the rows it already
has on each poll and track which indexes it has consumed.

### Option B — let the parser call you back

Expose an Apex REST resource and pass its URL as `callback_url` at submit. No
polling, no scheduled jobs:

```apex
@RestResource(urlMapping='/resumeCallback/*')
global with sharing class ResumeParserCallback {
    @HttpPost
    global static void receive() {
        Map<String, Object> batch = (Map<String, Object>)
            JSON.deserializeUntyped(RestContext.request.requestBody.toString());
        ResumeParserApply.upsertCandidates((List<Object>) batch.get('results'));
    }
}
```

Set `callback_token` at submit and verify it in the resource. Keep the batch_id
on the originating record so you can still poll if a callback is ever missed.

## 7. Operational notes

**Nothing hangs forever.** Three layers guarantee a terminal state:

| Failure | Handling |
|---|---|
| One slow resume | `JOB_TIMEOUT` (300s) inside the worker records a failed row |
| Transient upstream error (429/5xx, connect, timeout) | Retried with backoff up to `JOB_MAX_TRIES`, then recorded as failed |
| Bad input (400/413/422) | Recorded as failed immediately — no wasted retries |
| Worker killed mid-job | Batch with nothing running and no progress for `JOB_STALE_SECONDS` is reaped; unfinished files become failed rows and the batch reports `completed` |
| Redis down | Queue endpoints return **503** with a clear message; the synchronous endpoints keep working |

**Memory.** Uploaded bytes live in Redis until their file is parsed, then are
deleted immediately. Worst case is roughly `QUEUE_MAX_FILES × MAX_FILE_SIZE` per
in-flight batch — size the Redis plan accordingly, or lower `QUEUE_MAX_FILES`.
Results (not bytes) persist for `JOB_TTL`.

**Monitoring.** `GET /health` reports queue depth:

```json
{"status":"healthy","openrouter_connected":true,
 "model":"deepseek/deepseek-v4-flash",
 "queue":{"connected":true,"queued_jobs":0,"error":null}}
```

A `queued_jobs` figure that climbs and does not drain means workers are down or
under-provisioned.

**Deprecated.** `POST /api/v1/parse/job` and `GET /api/v1/parse/job/{job_id}`
(singular *job*) keep their state in one process's memory: results vanish on
restart and are invisible to other replicas. They still work, but use the Redis
endpoints (plural *jobs*) for anything new.
