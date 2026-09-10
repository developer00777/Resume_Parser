# Salesforce side

**`ResumeParserService.cls`** is the entire integration in one class. Paste it
into a single Apex Class in Setup. The two Queueables are inner classes, so
there is nothing else to create.

Delete the old `ResumeParserQueueable` class — nothing references it any more.

## Before you save it

| Line | Change |
|---|---|
| 53 | `BASE_URL` — the live Railway domain |
| 54 | `API_KEY` — your parser API key |

Confirm `BASE_URL` matches the **current** Railway domain. The
`404 — "The train has not arrived at the station"` is Railway's edge saying no
service serves that hostname; the request never reaches the parser.

Nothing changes in the LWC — `startParsing(fileName, base64Data)` and
`getLatestLogs(resumeNames)` keep their signatures and behaviour.

## What this fixes

### 1. Wrong endpoint — why MaritalStatus and Gender were always blank

The old code called `/api/v1/parse/salesforce` but read **client-mode** keys.
The two modes emit different field names:

| Key the old code read | `salesforce` mode | `client` mode | Result |
|---|---|---|---|
| `MaritalStatus` | *absent* | `MaritalStatus__c` | always null |
| `Nationality__c` | `Nationnality` | `Nationality__c` | always null |
| `Spoken_Language__c` | *absent* | `Spoken_Language__c` | always null |
| `CandidateType` | *absent* | `Type_1__c` | always null |
| `Gender` | `Gender` | `Gender__c` | worked |
| `PhoneNumber` | `PhoneNumber` | `PhoneNumber__c` | worked |

`ClientResumeData` is a 1:1 match for these Contact fields — it was built for
this org. The class now calls `/api/v1/parse/client/jobs/base64` and uses those
names, so every field lands.

### 2. The 504s

The old `parseResume()` held one callout open for the entire parse. Salesforce
caps a callout at 120s and the parser gave up at 110s — that is exactly where
every 504 came from, and the whole batch was discarded with it.

```
LWC ─► startParsing()
         └─► SubmitJob    POST …/jobs/base64   → batch_id in ~15 ms
               └─► PollJob   GET …/jobs/{id}   ← chains itself until complete
                     └─► Contact + Resume_Parser_Log__c + Attachment
```

Each Queueable tick is its own transaction with a fresh 120s cumulative callout
budget, so no amount of bulk can time it out. Ticks 0–1 fire immediately (a fast
resume lands in seconds); after that it polls once a minute, up to 30 times.

### 3. Other fixes folded in

- **Governor limits.** The `RecordType` SOQL and all DML sat inside the results
  loop — one query and three DML statements *per resume*. The record type now
  comes from the describe cache (no SOQL at all) and inserts are bulkified.
- **Debug SOQL removed.** Two `SELECT … WHERE Id = :con.Id` re-queries per row
  existed only to print to the log.
- **Country-code precedence bug.** `length == 10 && startsWith('6') ||
  startsWith('7') || …` parses as `(length==10 && '6') || '7' || '8' || '9'`
  because `&&` binds tighter than `||`, so *any* number starting 7/8/9 was
  tagged India whatever its length. Now parenthesised.
- **Unreachable UAE branch.** The 9-digit UAE test sat after a `>= 9 && <= 10`
  Australia test, so it never ran. Reordered.
- **No more hand-padded multipart.** `safeBase64Concat` and the boundary maths
  are gone; the body is `JSON.serialize()`.

## Two things to decide

**Missing-email records.** The old code logged *"Skipping record due to missing
email"* but had no `continue`, so it inserted the Contact anyway. Current
behaviour is preserved; set `SKIP_WHEN_NO_EMAIL = true` to actually skip.

**Queueable chain depth.** Polling chains one job per tick. Production and
sandboxes have no chain-depth limit, but a **Developer Edition** org caps it at
5 — there, lower `MAX_TICKS`.

## Deploying to production later

Saving this class in a sandbox needs no test class. Deploying to **production**
requires 75% Apex coverage, so a test class will be needed at that point — ask
and it can be added back.

## For true bulk (thousands of resumes)

The base64 path is heap-bound: a 2 MB PDF costs roughly 8 MB as Blob + base64
String + request body, against a 6 MB sync / 12 MB async ceiling. That is fine
for the LWC's one-file-at-a-time flow.

For large back-fills use `POST /api/v1/parse/client/jobs/records` instead — it
takes ContentVersion / Attachment / Candidate **ids**, so the payload is a few KB
however large the resumes are and the parser downloads each file itself. Drive it
from Batch Apex with `scope = 100` (one chunk = one parser batch = one callout),
and set `external_id` to the record you want updated. See
[../QUEUE.md](../QUEUE.md).
