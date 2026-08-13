# Resume Parser → Salesforce: Integration Runbook

How the resume parser gets wired into the Salesforce org, and how a recruiter
ends up with a button that fills in a Candidate record.

**Audience:** whoever holds System Administrator on the org, plus whoever
deploys the parser service. Steps 1–3 and 7–9 are Salesforce-side. Steps 4–6 are
parser-side.

---

## 0. What is actually being installed

Salesforce cannot run this functionality itself. That is not a preference — it is
four hard platform limits:

| Limit | Value | Why it rules Apex out |
|---|---|---|
| Apex CPU time | 10 s sync / 60 s async | Text extraction + 3 LLM round-trips do not fit |
| Heap | 6 MB sync / 12 MB async | The resume text alone strains it |
| Callout time | 120 s single, 120 s cumulative per transaction | One parse consumes the whole transaction |
| Libraries | none addable | Apex has no PDF or DOCX parser, and no way to add one |

So the work is split, and the split is deliberate:

```
SALESFORCE                                PARSER SERVICE (Railway / Docker)
──────────────                            ─────────────────────────────────
Candidate record          ── record ID ▶  downloads the resume from Salesforce
"Parse Resume" button                     extracts text (PDF/DOCX, OCR fallback)
                                          3 parallel LLM calls, with retry
                                          coerces values to org-safe shapes
Resume_Parse_Status__c    ◀── summary ──  PATCHes the fields onto the record
```

Apex sends **one record ID** and reads back a small JSON summary. No file bytes
cross the wire in either direction. That is what makes it fit inside the limits.

**Nothing is installed as a managed package.** You are deploying two Apex
classes, one Lightning web component, one custom field, and one Quick Action.

---

## 1. Create the status field

**Setup → Object Manager → Candidate (`SCSCHAMPS__Candidate__c`) → Fields &
Relationships → New**

| Setting | Value |
|---|---|
| Data Type | Text Area (Long) |
| Field Label | Resume Parse Status |
| Field Name | `Resume_Parse_Status` |
| Length | 1000 |
| Visible to | every profile that will use the button |

This is where the parse outcome lands — how many fields were updated, and
critically **which sections failed to extract**. Without it, a resume whose
experience section was rate-limited looks identical to a candidate with no job
history, and only the parser's logs would know the difference.

---

## 2. Connected App — lets the parser read *into* Salesforce

**Setup → App Manager → New Connected App**

- **Name:** Resume Parser Integration
- **Enable OAuth Settings:** ✓
- **Callback URL:** `https://login.salesforce.com/services/oauth2/success`
  (unused by server flows, but the form requires one)
- **Selected OAuth Scopes:** `Manage user data via APIs (api)`
- **Enable Client Credentials Flow:** ✓

Save, then **Manage → Edit Policies**:

- **Run As:** pick a dedicated integration user (not a person who might leave)
- **Permitted Users:** Admin approved users are pre-authorised
- Assign the Connected App to that user's profile or a permission set

> **The Run-As user's permissions are the parser's permissions.** It needs
> **Read** on Candidate, ContentVersion and Attachment, and **Edit** on every
> field the parser writes. If a field is invisible to that user, the parser
> cannot write it and will report it as unresolved.

> **Do not use the username-password flow.** Salesforce disables it by default
> in modern orgs. Client Credentials is the supported server-to-server path.

Copy the **Consumer Key** and **Consumer Secret** from *Manage Consumer Details*.
They go into the parser's environment in step 4 — put them there directly or via
a password manager. Never in a document, a chat message, or a ticket.

Allow 10 minutes for Salesforce to propagate the app before testing.

---

## 3. Named Credential — lets Salesforce call *out* to the parser

Two objects, in this order.

### 3a. External Credential

**Setup → Named Credentials → External Credentials → New**

| Setting | Value |
|---|---|
| Label / Name | Resume Parser Auth / `Resume_Parser_Auth` |
| Authentication Protocol | Custom |

Save, then under **Principals → New**:

| Setting | Value |
|---|---|
| Parameter Name | `ResumeParserPrincipal` |
| Sequence Number | 1 |
| Identity Type | Named Principal |

Add a **Custom Header** on that principal:

| Name | Value |
|---|---|
| `X-API-Key` | the parser's `API_KEY` value |

### 3b. Named Credential

**Setup → Named Credentials → New**

| Setting | Value |
|---|---|
| Label | Resume Parser API |
| Name | `Resume_Parser_API` |
| URL | `https://<your-parser-host>` |
| External Credential | Resume Parser Auth |
| Generate Authorization Header | **unchecked** |

`Generate Authorization Header` must be off — the parser authenticates on the
`X-API-Key` header, and an extra `Authorization` header will confuse the request.

### 3c. Grant access to the principal — do not skip this

**Setup → Permission Sets → (your recruiter permission set) → External
Credential Principal Access → Edit → add `Resume_Parser_Auth - ResumeParserPrincipal`**

This is the step that gets missed. Without it every callout fails with an
authentication error even though the Named Credential looks correctly configured.

> No **Remote Site Setting** is needed. Named Credentials supersede it.

---

## 4. Configure the parser service

Set these on the parser's host (Railway → Variables, or `.env`):

| Variable | Value |
|---|---|
| `OPENROUTER_API_KEY` | your OpenRouter key |
| `API_KEY` | a strong random string — the same value as the `X-API-Key` header in 3a |
| `SF_CLIENT_ID` | Consumer Key from step 2 |
| `SF_CLIENT_SECRET` | Consumer Secret from step 2 |
| `SF_LOGIN_URL` | `https://test.salesforce.com` for a sandbox, `https://login.salesforce.com` for production |
| `SF_API_VERSION` | `59.0` or later |
| `SF_DESCRIBE_TTL_MINUTES` | `60` |

Leave `SF_USERNAME` / `SF_PASSWORD` **empty** — setting them switches the
service to the password grant, which step 2 deliberately avoided.

Restart the service.

---

## 5. Load the field mapping

The parser does not ship a hardcoded list of Salesforce field names. It reads
the org's actual schema and resolves each field itself — which is what makes it
work against a namespaced managed-package org and a locally-customised org
without editing code.

This now runs automatically at startup. To confirm it worked, or to re-run it
after an admin adds a field:

```bash
curl -X POST https://<parser-host>/api/v1/salesforce/describe/reload \
  -H "X-API-Key: <API_KEY>"
```

Read the response carefully:

| Key | Meaning |
|---|---|
| `fields_resolved` | Fields the parser can write. Should be most of ~90. |
| `unresolved` | The parser expects these; **the org has no such field.** Either create them or accept those values are dropped. |
| `not_updateable` | The field exists but is a formula, rollup or system field. Never writable. |

**`unresolved` is the list to act on.** Every name on it is data the parser
extracts and then throws away.

---

## 6. Dry run before wiring any button

```bash
curl -X POST "https://<parser-host>/api/v1/salesforce/parse-and-update?record_id=<CandidateId>&dry_run=true" \
  -H "X-API-Key: <API_KEY>"
```

This parses a real resume and computes the exact payload **without writing
anything**. Check:

- `fields_written` — what would land on the record, by real Salesforce API name
- `fields_skipped` — every withheld field, with the reason
- `extraction.complete` — `false` means part of the resume was not read

Run this against three or four representative resumes before going further. It
is the only step that shows you the mapping is right while it is still free to
be wrong.

---

## 7. Deploy the Apex

```bash
sf project deploy start -d salesforce/force-app -o <org-alias>
```

Deploy to a **sandbox first**. Production deployment requires 75% Apex test
coverage org-wide.

What gets deployed:

| Component | Purpose |
|---|---|
| `ResumeParserQueueable` | Makes the callout, records the outcome |
| `ResumeParserAction` | `@InvocableMethod` for Flow, `@AuraEnabled` for the button |
| `resumeParser` (LWC) | The button a recruiter clicks |

---

## 8. Add the button

**Setup → Object Manager → Candidate → Buttons, Links and Actions → New Action**

| Setting | Value |
|---|---|
| Action Type | Lightning Web Component |
| Lightning Web Component | `resumeParser` |
| Label | Parse Resume |

Then **Page Layouts → (your layout) → Salesforce Mobile and Lightning
Experience Actions → drag "Parse Resume" into the layout.**

Recruiters now get a **Parse Resume** button on the Candidate page. Clicking it
queues the job and returns immediately; the record updates within a minute or
two, and `Resume Parse Status` shows what happened.

---

## 9. Optional — run it automatically

**Setup → Flow → New → Record-Triggered Flow**

- **Object:** Candidate
- **Trigger:** A record is created or updated
- **Condition:** `SCSCHAMPS__Resume_Attachment_Id__c` *Is Changed* = True
- **Optimize for:** Actions and Related Records
- **Add Element → Action →** search `Parse Resume` (the `@InvocableMethod`)
- Pass the record Id into the `recordId` input

Start with the manual button. Add the Flow only once you trust the dry-run
output, because a Flow will parse every resume that arrives.

---

## What a recruiter sees

1. Opens a Candidate record with a resume attached
2. Clicks **Parse Resume**
3. Gets a toast: *"Parsing started — the record will update shortly"*
4. Within a minute or two, contact details, skills, experience, education and
   scores are populated
5. **Resume Parse Status** reads something like:

   > Parsed 12/08/2026 16:04 — 47 field(s) updated

   or, when something went wrong:

   > Parsed 12/08/2026 16:04 — 22 field(s) updated | NOT extracted (fields left
   > untouched): education, experience

That second line is the point of the whole design. When an LLM call fails, the
affected fields are **withheld, never blanked** — the parser will not overwrite a
real employment history with an empty one just because a call was rate-limited.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `401` from the parser | `X-API-Key` header value does not match the parser's `API_KEY` |
| Callout fails on auth, credential looks fine | Step 3c was skipped — no External Credential Principal Access |
| `409 Field mapping has not been loaded` | Parser cannot reach Salesforce. Check `SF_CLIENT_ID`/`SF_CLIENT_SECRET` and that Client Credentials Flow is enabled with a Run-As user |
| Parse succeeds, no fields change | Everything came back in `fields_skipped`. Read the reasons — usually the Run-As user lacks Edit permission |
| `unresolved` lists many fields | Those fields do not exist in this org. Create them or accept the loss |
| `INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST` | Should not happen — the parser snaps picklist values. If it does, re-run `/describe/reload` |
| Status field never updates | `Resume_Parse_Status__c` missing, or not writable by the running user |
| `422 Candidate record has no resume` | Neither `SCSCHAMPS__Resume_Attachment_Id__c` nor `SCSCHAMPS__Resume_URL__c` is set |

---

## Security notes for the handoff

- The **Consumer Secret** and the parser's **API_KEY** are credentials. Deliver
  them through a password manager, not in this document, an email, or a chat.
- The parser sends resume text to **OpenRouter**, a third-party LLM provider.
  For Indian candidate data this includes Aadhaar, PAN, passport and date of
  birth where the resume states them. Confirm this is acceptable under your
  DPDP Act obligations before running it on real candidates, and get a data
  processing agreement in place.
- Rotate the Connected App's Consumer Secret if it has ever been sent over an
  unencrypted or shared channel.
