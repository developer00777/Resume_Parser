# Deploying the Resume Parser, end to end

Two systems have to find each other, and each needs a URL that only the other
side can produce. That exchange is the part people get stuck on, so this
document is organised around it rather than around the software.

**Who is involved**

| Role | Owns | Section |
|---|---|---|
| **Parser owner** | The service on DigitalOcean | 1, 3, 5 |
| **Salesforce developer/admin** | The org configuration | 2, 4 |
| **Recruiters** | Clicking one button | 6 |

Recruiters do none of the setup. A standard sales profile has no Setup access.

---

## The two URLs

Nothing works until both have crossed:

```
   SALESFORCE                                    DIGITALOCEAN
   ──────────                                    ────────────

   My Domain URL   ──────────────────────────▶   SF_LOGIN_URL
   (SF dev finds it,                             (parser owner sets it)
    sends it over)
        so the parser can log in to Salesforce


   Named Credential  ◀──────────────────────────  App URL
   (SF dev sets it)                               (parser owner sends it)
        so Salesforce can call the parser
```

Neither side can guess the other's value. Section 2 produces the first, section
1 produces the second.

---

## 1. Deploy the parser — parser owner

### Prerequisites

```bash
brew install doctl && doctl auth init      # paste a DigitalOcean API token
open -a Docker                             # Docker Desktop must be running
doctl registry create champ-registry --region blr1
```

> **`blr1`, not `blr`.** DigitalOcean has two region namespaces: Container
> Registry uses compute slugs (`blr1`, `nyc3`), App Platform uses short ones
> (`blr`, `nyc`). Using the wrong one returns a 422 that names no alternative.
> Full list: `doctl registry options available-regions`

### Secrets to have ready

Four values. Only one comes from Salesforce.

| Secret | Where it comes from |
|---|---|
| `OPENROUTER_API_KEY` | openrouter.ai/keys |
| `API_KEY` | **You invent it:** `openssl rand -hex 32` |
| `SF_CLIENT_ID` | Salesforce dev — Connected App Consumer Key |
| `SF_CLIENT_SECRET` | Salesforce dev — Connected App Consumer Secret |

> Use **hex**, not base64, for `API_KEY`. You will paste this exact string into
> a Salesforce Setup field later; base64's `/`, `+` and trailing `=` are easy to
> mis-copy, and the resulting failure is a 401 that looks like a Salesforce
> problem and is not.

### Deploy

```bash
cd ~/Resume_Parser
./deploy/deploy-digitalocean.sh
```

It prompts for the four secrets (never echoed or written to disk), builds a
`linux/amd64` image, pushes it to the registry, creates the app, waits for the
health check, and prints the live URL.

### What you now have

```
https://resume-parser-XXXXX.ondigitalocean.app
```

**Send that URL to the Salesforce developer.** It is what section 4 needs.

Confirm it is alive:

```bash
curl https://resume-parser-XXXXX.ondigitalocean.app/health
# {"status":"healthy","openrouter_connected":true,...}
```

`describe/status` will still report `loaded: false` at this point. That is
expected — it needs section 2.

---

## 2. Find the My Domain URL — Salesforce developer

This is the value the parser needs in order to authenticate into the org, and it
is the single most common thing to get wrong.

**Setup → Company Settings → My Domain → *Current My Domain URL***

Or, faster: log in and append `/lightning/setup/OrgDomain/home` to the URL.

Or, without Setup at all — read it off the address bar and change the suffix:

| Address bar shows | The value you want |
|---|---|
| `acme.lightning.force.com` | `https://acme.my.salesforce.com` |
| `acme--demosbx.sandbox.lightning.force.com` | `https://acme--demosbx.sandbox.my.salesforce.com` |

Keep the subdomain, swap `lightning.force.com` → `my.salesforce.com`, drop the
path.

> ### Why not `login.salesforce.com` / `test.salesforce.com`
>
> The Client Credentials Flow is **rejected** on those hosts with:
>
> ```
> invalid_grant: request not supported on this domain
> ```
>
> They are correct for the older username-password flow, which is why this trips
> almost everyone. Only the My Domain host works.

Verify before sending it — this needs no credentials, and a real Salesforce host
answers with a list of API versions:

```bash
curl -s https://acme--demosbx.sandbox.my.salesforce.com/services/data | head -c 200
```

**Send that URL to the parser owner.**

### While you are in Setup

Two things that otherwise fail confusingly later:

1. **Connected App → Manage → Edit Policies.** Confirm **Client Credentials
   Flow** is enabled and a **Run-As user** is set. That user's permissions
   become the parser's — it needs Read on Candidate, ContentVersion and
   Attachment, and **Edit on every field the parser writes**.
2. **Does that user's profile have Login IP Ranges?** DigitalOcean App Platform
   has no fixed outbound IP, so if it does, tick **Relax IP restrictions** on
   the Connected App — otherwise authentication succeeds and fails at random.
   The alternative is redeploying to a Droplet, which has a static IP
   (`deploy/docker-compose.prod.yml`).

---

## 3. Set `SF_LOGIN_URL` — parser owner

**DigitalOcean → Apps → resume-parser → Settings → `api` component →
Environment Variables → `SF_LOGIN_URL`** → paste the My Domain URL → **Save**.

The app redeploys itself, about a minute.

Then confirm the parser can now read the org's schema:

```bash
curl -s https://resume-parser-XXXXX.ondigitalocean.app/api/v1/salesforce/describe/status \
  -H "X-API-Key: <your API_KEY>" | python3 -m json.tool
```

You want:

```json
{ "loaded": true, "sobject": "SCSCHAMPS__Candidate__c", "fields_resolved": 10, ... }
```

| Result | Meaning |
|---|---|
| `loaded: true` | Authentication and schema read both work. Proceed. |
| `invalid_grant: request not supported on this domain` | `SF_LOGIN_URL` is still a login/test host, not My Domain |
| `invalid_client` | Consumer Key or Secret is wrong |
| `no client credentials user enabled` | No Run-As user on the Connected App |
| `does not exist in this org` | The Run-As user cannot see the Candidate object |

**Read `unresolved`.** Every name on that list is a field the parser extracts
and then discards because the org has no such field. That is not an error — it
is the scope of what will actually be written. If it is longer than expected,
that is a conversation about creating fields, not a bug.

---

## 4. Point Salesforce at the parser — Salesforce developer

You need two things from the parser owner: the **App URL** and the **API key**.

### 4a. Custom field

**Setup → Object Manager → Candidate → Fields & Relationships → New**

Text Area (Long), label `Resume Parse Status`, name `Resume_Parse_Status`,
length 1000. Make it visible to every profile that will use the button.

This is where each parse records what it did — including which sections failed
to extract. Without it, a resume whose experience section was rate-limited looks
identical to a candidate with no job history.

### 4b. External Credential

**Setup → Named Credentials → External Credentials → New**

- Label / Name: `Resume Parser Auth` / `Resume_Parser_Auth`
- Authentication Protocol: **Custom**

Save, then **Principals → New**:

- Parameter Name: `ResumeParserPrincipal`
- Identity Type: **Named Principal**

On that principal, add a **Custom Header**:

| Name | Value |
|---|---|
| `X-API-Key` | the parser's API key, exactly |

### 4c. Named Credential — this is where the parser URL goes

**Setup → Named Credentials → New**

| Setting | Value |
|---|---|
| Label | Resume Parser API |
| Name | `Resume_Parser_API` |
| **URL** | **`https://resume-parser-XXXXX.ondigitalocean.app`** |
| External Credential | Resume Parser Auth |
| Generate Authorization Header | **unchecked** |

The URL is the host only — no path, no trailing slash. The Apex appends
`/api/v1/salesforce/parse-and-update` itself.

`Generate Authorization Header` must be **off**: the parser authenticates on
`X-API-Key`, and an extra `Authorization` header confuses the request.

> No **Remote Site Setting** is needed. Named Credentials supersede it.

### 4d. Grant access to the principal — do not skip

**Setup → Permission Sets → (the recruiters' set) → External Credential
Principal Access → Edit →** add `Resume_Parser_Auth - ResumeParserPrincipal`

Without this every callout fails on authentication while the Named Credential
looks perfectly configured. It is the most commonly missed step in this setup.

### 4e. Deploy the Apex

```bash
sf project deploy start -d salesforce/force-app -o <sandbox-alias>
```

Sandbox first. Production needs 75% Apex coverage org-wide and does not accept
Apex created directly — it must arrive by Change Set or CLI.

**No CLI?** In a sandbox, paste the classes via **Setup → Developer Console →
File → New → Apex Class**. The Lightning web component cannot be created through
any Salesforce UI, so build the button as a **Flow** instead — see §7 of
`SALESFORCE_INTEGRATION.md`. It uses the same `@InvocableMethod` and is entirely
point-and-click.

### 4f. Add the button

**Setup → Object Manager → Candidate → Page Layouts →** your layout **→ Mobile
& Lightning Actions →** drag **Parse Resume** into the layout → Save.

---

## 5. Dry run before anything writes — parser owner

```bash
curl -X POST "https://resume-parser-XXXXX.ondigitalocean.app/api/v1/salesforce/parse-and-update?record_id=<CandidateId>&dry_run=true" \
  -H "X-API-Key: <API_KEY>" | python3 -m json.tool
```

This parses a real resume and computes the exact payload **without writing
anything**. Check:

- `fields_written` — what would land, by real Salesforce API name
- `fields_skipped` — every withheld field, with the reason
- `extraction.complete` — `false` means part of the resume was not read

Do this on three or four representative resumes. It is the last point at which a
wrong mapping is free to be wrong.

---

## 6. What a recruiter does

1. Open a Candidate with a resume attached
2. Click **Parse Resume**
3. Toast: *"Parsing started — this record will update shortly"*
4. A minute or so later the fields are populated
5. **Resume Parse Status** reads:

   > Parsed 13/08/2026 19:41 — 8 field(s) updated

   or, when something failed:

   > Parsed 13/08/2026 19:41 — 3 field(s) updated | NOT extracted (fields left
   > untouched): education, experience

**Train them on that second line.** "NOT extracted" means those fields were left
alone, **not cleared**. The parser will not overwrite a real employment history
with a blank because one LLM call was rate limited — so the right response is to
re-run, not to assume the resume was thin.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `401` from the parser | `X-API-Key` does not match the parser's `API_KEY` |
| Callout fails on auth, credential looks correct | §4d was skipped |
| `409 Field mapping is not loaded` | Parser cannot reach Salesforce — check `SF_LOGIN_URL` is My Domain, and the Connected App's Run-As user |
| `invalid_grant: request not supported on this domain` | `SF_LOGIN_URL` is login/test.salesforce.com |
| Parse succeeds, nothing changes | Everything is in `fields_skipped`. Usually the Run-As user lacks Edit permission |
| Auth works intermittently | Login IP Ranges vs App Platform's rotating egress IP — see §2 |
| `unresolved` is long | Those fields do not exist in this org. Expected; create them or accept the scope |
| Status field never updates | `Resume_Parse_Status__c` missing or not writable by the Run-As user |

---

## Security

- The **Consumer Secret** and the parser's **API key** are credentials. Move
  them through a password manager, never a document, email or chat message. If
  one has been sent that way, rotate it: **Setup → App Manager → Manage Consumer
  Details → Rotate**.
- The parser sends resume text to **OpenRouter**, a third-party LLM provider.
  For Indian candidate data that can include Aadhaar, PAN, passport and date of
  birth where the resume states them. Confirm this is acceptable under your DPDP
  Act obligations, with a data processing agreement in place, before running it
  on real candidates. See item 4 in `REMAINING_WORK.md`.
