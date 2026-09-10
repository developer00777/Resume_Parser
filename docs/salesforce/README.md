# Salesforce side — bulk parsing that respects the governor limits

Reference Apex for driving the parser's async queue. Deploy it, adapt
`ResumeParserApply` to your fields, and bulk parsing stops fighting the platform.

## Why "just make more callouts" does not work

The 110s you were seeing was the parser's own budget (now removed — see
[../QUEUE.md](../QUEUE.md)). But the Salesforce side has hard ceilings of its
own, and two of them cannot be raised by any amount of chunking *within one
transaction*:

| Limit | Value | What it means here |
|---|---|---|
| **Cumulative callout time** | **120 s per transaction** | 15 sequential callouts at 20 s each die on the sixth — each one legal, the total not |
| **Heap** | 6 MB sync / 12 MB async | A 2 MB PDF costs ~8 MB as Blob + base64 String + request body. One or two per transaction, ever |
| Callouts per transaction | 100 | Rarely the binding constraint |
| CPU time | 10 s sync / 60 s async | Base64-encoding large files eats this fast |

So the fix is not *more* calls — it is **small, fast calls spread across
transactions**. Two changes get you there:

1. **Send record ids, not file bytes.** `POST /api/v1/parse/salesforce/jobs/records`
   takes a list of ContentVersion / Attachment / Candidate ids. The payload is a
   few KB no matter how large the resumes are, the callout returns in ~300 ms,
   and the parser downloads each file server-side with the Connected App
   credentials. Heap stops mattering entirely.
2. **Put each chunk in its own transaction.** Batch Apex gives every `execute()`
   a fresh 120 s callout budget and a fresh heap. Scale comes from the number of
   chunks, not from stretching one request.

Result: 5,000 resumes is 50 chunks × 1 callout × ~0.3 s. Nothing can time out.

## The classes

| Class | Role |
|---|---|
| `ResumeParserService` | Thin client — `submitRecords()` and `pollBatch()` |
| `ResumeParserSubmitBatch` | Batch Apex over every unparsed Candidate; one callout per chunk |
| `ResumeParserPoller` | Queueable that polls outstanding batches and chains itself |
| `ResumeParserApply` | **Edit this** — writes results onto your Candidate fields |
| `ResumeParserCallback` | Optional REST resource, so the parser pushes results and you never poll |
| `ResumeParserServiceTest` | Callout-mocked coverage (Salesforce requires 75% to deploy) |

## Setup

**1. Named Credential** — Setup → Named Credentials → New Legacy:

```
Label:                Resume Parser
Name:                 Resume_Parser
URL:                  https://<your-railway-domain>
Identity Type:        Named Principal
Authentication:       Password Authentication
Username:             resume-parser        (unused; any value)
Password:             <your API_KEY>
Generate Auth Header: unchecked
```

The classes send the key as `X-API-Key: {!$Credential.Password}`, so the key
never appears in source.

> Confirm the URL against the **current** Railway domain. The
> `404 — "The train has not arrived at the station"` you were getting is
> Railway's edge saying no service serves that hostname; the request never
> reached the parser at all.

**2. Connected App on the parser** — the record-id path needs the parser able to
download files from your org. Set `SF_CLIENT_ID` / `SF_CLIENT_SECRET` (and
`SF_LOGIN_URL=https://test.salesforce.com` for a sandbox) on both the `app` and
`worker` services. If you would rather not grant that, use the base64 path
instead (below).

**3. Custom fields** used by the reference `ResumeParserApply` — create these on
`SCSCHAMPS__Candidate__c` or edit the class to match what you already have:

| Field | Type |
|---|---|
| `Resume_Parsed__c` | Checkbox |
| `Resume_Parse_Error__c` | Text(255) |
| `Resume_Parsed_On__c` | Date/Time |

**4. Deploy**

```bash
sf project deploy start --source-dir docs/salesforce --target-org <alias>
sf apex run test --class-names ResumeParserServiceTest --target-org <alias> --wait 10
```

## Running it

```apex
// Everything unparsed. Scope 100 == one parser batch == one callout per chunk.
Database.executeBatch(new ResumeParserSubmitBatch(), 100);
```

Or a single ad-hoc batch with no Batch Apex:

```apex
List<ResumeParserService.RecordRef> refs = new List<ResumeParserService.RecordRef>();
for (SCSCHAMPS__Candidate__c c : candidates) {
    refs.add(new ResumeParserService.RecordRef(
        c.SCSCHAMPS__Resume_Attachment_Id__c, 'attachment', c.Id   // c.Id → external_id
    ));
}
String batchId = ResumeParserService.submitRecords(refs);
System.enqueueJob(new ResumeParserPoller(new List<String>{ batchId }, 0), 1);
```

Skip polling altogether by passing a callback:

```apex
Database.executeBatch(new ResumeParserSubmitBatch(
    'https://<your-domain>/services/apexrest/resumeCallback', 'your-shared-secret'
), 100);
```

## Correlating results back to records

Set `external_id` to the Candidate Id when submitting; the parser echoes it back
on every result row, success or failure, including rows it reaps after a worker
dies. **Do not match on filename** — duplicates collide, and the parser prefers
Salesforce's own filename over whatever you sent.

```json
{"index": 0, "filename": "Jane_Doe_CV.pdf", "external_id": "a0X1000000CandId",
 "success": true, "data": { "FirstName": "Jane", "...": "..." }}
```

## If you cannot grant the parser file access

Use the base64 path — same batch semantics, you send the bytes:

```apex
Blob body = [SELECT VersionData FROM ContentVersion WHERE Id = :cvId].VersionData;
String payload = JSON.serialize(new Map<String, Object>{
    'files' => new List<Object>{ new Map<String, Object>{
        'filename'       => 'cv.pdf',
        'content_base64' => EncodingUtil.base64Encode(body),
        'external_id'    => candidateId
    }}
});
// POST to /api/v1/parse/salesforce/jobs/base64
```

Plain JSON, so no hand-padded multipart boundaries. **Keep the chunk to 1–2
files per transaction** — this path is heap-bound, which is exactly the ceiling
the record-id path avoids.
