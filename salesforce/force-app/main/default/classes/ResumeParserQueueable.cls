/**
 * Sends a Candidate record ID to the Resume Parser API, which parses the
 * attached resume and writes the extracted fields back onto the record itself.
 *
 * WHY THE PARSER OWNS THE WRITE
 * -----------------------------
 * The obvious design — Apex receives ~90 mapped fields and assigns them — runs
 * into four platform walls at once:
 *
 *   1. Apex identifiers are case-insensitive, so a wrapper class cannot hold
 *      both `Resume_Score` and a nested `resume_score`, which the old payload
 *      shipped side by side. JSON.deserialize into a typed class would not
 *      compile at all.
 *   2. Dates arrive as whatever the resume said ("12 May 1990"). Date.valueOf()
 *      throws, and it takes the whole DML with it.
 *   3. Restricted picklists reject free-text values with
 *      INVALID_OR_NULL_FOR_RESTRICTED_PICKLIST — again losing the whole record.
 *   4. Heap is 6 MB sync / 12 MB async, and the payload carries the full resume
 *      text per record.
 *
 * Having the service PATCH the record removes all four: this class sends an ID
 * and reads back a small summary. No wrapper classes, no coercion in Apex, no
 * multipart bodies, no heap pressure.
 *
 * LIMITS THIS RESPECTS
 *   - One callout per execute(). Cumulative callout time per transaction is
 *     capped at 120 s, so a single parse gets the whole budget.
 *   - Queueable rather than @future: @future cannot return state or chain, so a
 *     batch could not be sequenced and a failure could not be recorded.
 *   - One chained enqueue per execute(), which is the async chaining limit.
 *
 * SETUP
 *   Named Credential   : Resume_Parser_API   -> https://<parser-host>
 *   External Credential: custom header `X-API-Key` (Named Principal), so the
 *                        key never appears in Apex source or metadata.
 *
 * USAGE
 *   System.enqueueJob(new ResumeParserQueueable(candidateId));
 *   System.enqueueJob(new ResumeParserQueueable(setOfCandidateIds));  // chains
 */
public with sharing class ResumeParserQueueable implements Queueable, Database.AllowsCallouts {

    private static final String NAMED_CREDENTIAL = 'callout:Resume_Parser_API';

    // 120_000 ms is the platform maximum for a single callout. The parser
    // targets well under this; the ceiling is here so a slow parse fails on our
    // terms rather than being cut off mid-write.
    private static final Integer CALLOUT_TIMEOUT_MS = 120000;

    /** Field the parse outcome is written to. Resolved at runtime, so this
     *  class still compiles and runs in an org that has not created it. */
    @TestVisible
    private static final String STATUS_FIELD = 'Resume_Parse_Status__c';

    private final List<Id> queue;

    public ResumeParserQueueable(Id candidateId) {
        this.queue = new List<Id>{ candidateId };
    }

    public ResumeParserQueueable(List<Id> candidateIds) {
        this.queue = candidateIds == null ? new List<Id>() : candidateIds.clone();
    }

    public ResumeParserQueueable(Set<Id> candidateIds) {
        this(candidateIds == null ? new List<Id>() : new List<Id>(candidateIds));
    }

    public void execute(QueueableContext context) {
        if (queue.isEmpty()) {
            return;
        }

        // One record per transaction: the parse consumes the full 120 s
        // cumulative callout budget, so batching within a transaction is not
        // possible regardless of how many IDs were handed in.
        Id candidateId = queue.remove(0);

        try {
            ParseResult result = callParser(candidateId);
            recordOutcome(candidateId, result);
        } catch (Exception e) {
            // Never let one bad record stop the chain — record it and move on.
            recordFailure(candidateId, e.getMessage());
        }

        // Chain the remainder. Skipped in a test context, where chaining is not
        // permitted.
        if (!queue.isEmpty() && !Test.isRunningTest()) {
            System.enqueueJob(new ResumeParserQueueable(queue));
        }
    }

    private ParseResult callParser(Id candidateId) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(
            NAMED_CREDENTIAL + '/api/v1/salesforce/parse-and-update?record_id=' + candidateId
        );
        req.setMethod('POST');
        req.setTimeout(CALLOUT_TIMEOUT_MS);
        // The record ID travels in the query string and there is no body. Some
        // stacks reject a POST carrying neither.
        req.setHeader('Content-Length', '0');

        HttpResponse res = new Http().send(req);
        Integer status = res.getStatusCode();

        if (status == 200) {
            return (ParseResult) JSON.deserialize(res.getBody(), ParseResult.class);
        }

        // 409 means the parser has no field mapping loaded. It is a
        // configuration state, not a bad record, and retrying this record will
        // not help until Salesforce becomes reachable from the parser.
        if (status == 409) {
            throw new ResumeParserException(
                'Parser has no field mapping loaded — check its Salesforce credentials, ' +
                'then call POST /api/v1/salesforce/describe/reload. ' + truncate(res.getBody(), 300)
            );
        }

        throw new ResumeParserException(
            'Parser returned HTTP ' + status + ': ' + truncate(res.getBody(), 400)
        );
    }

    /**
     * Records what the parser did. The parse status lands on the record so a
     * partial extraction is visible in Salesforce rather than only in the
     * parser's logs — a resume whose experience section failed to extract looks
     * identical to one that had no experience unless this is written down.
     */
    @TestVisible
    private void recordOutcome(Id candidateId, ParseResult result) {
        List<String> notes = new List<String>();
        Integer updated = (result != null && result.fields_written != null)
            ? result.fields_written.size()
            : 0;
        notes.add('Parsed ' + Datetime.now().format() + ' — ' + updated + ' field(s) updated');

        if (result != null && result.extraction != null) {
            if (result.extraction.text_truncated == true) {
                notes.add(
                    'Resume truncated: ' + result.extraction.text_sent + ' of ' +
                    result.extraction.text_length + ' characters were read'
                );
            }
            if (result.extraction.failed_sections != null
                    && !result.extraction.failed_sections.isEmpty()) {
                notes.add(
                    'NOT extracted (fields left untouched): ' +
                    String.join(result.extraction.failed_sections, ', ')
                );
            }
        }

        writeStatus(candidateId, String.join(notes, ' | '));
    }

    @TestVisible
    private void recordFailure(Id candidateId, String message) {
        System.debug(LoggingLevel.ERROR, 'Resume parse failed for ' + candidateId + ': ' + message);
        writeStatus(
            candidateId,
            'Parse failed ' + Datetime.now().format() + ' — ' + truncate(message, 200)
        );
    }

    /**
     * Writes the parse status onto the record.
     *
     * The field is resolved through Describe rather than referenced directly so
     * this class compiles in an org that does not have it — the status is
     * useful but never worth blocking a deployment over.
     */
    private void writeStatus(Id candidateId, String message) {
        SObjectType candidateType = candidateId.getSObjectType();
        Map<String, SObjectField> fields = candidateType.getDescribe().fields.getMap();

        SObjectField statusField = fields.get(STATUS_FIELD);
        if (statusField == null || !statusField.getDescribe().isUpdateable()) {
            System.debug(LoggingLevel.INFO, 'No writable ' + STATUS_FIELD + ' — ' + message);
            return;
        }

        Integer maxLength = statusField.getDescribe().getLength();
        SObject record = candidateType.newSObject(candidateId);
        record.put(STATUS_FIELD, truncate(message, maxLength));

        try {
            update record;
        } catch (DmlException e) {
            // The parse itself succeeded; failing to annotate it must not
            // surface as a failed parse.
            System.debug(LoggingLevel.ERROR, 'Could not write parse status: ' + e.getMessage());
        }
    }

    @TestVisible
    private static String truncate(String value, Integer maxLength) {
        if (value == null) {
            return null;
        }
        return value.length() > maxLength ? value.substring(0, maxLength) : value;
    }

    // ── Response shape ──────────────────────────────────────────────────────
    // Deliberately partial: JSON.deserialize ignores attributes with no matching
    // member, so the parser can add fields without breaking this class. Every
    // member name here is unique case-insensitively — the constraint that made
    // the old `Resume_Score`/`resume_score` pair impossible to model.

    public class ParseResult {
        public Boolean success;
        public String record_id;
        public Boolean dry_run;
        public Map<String, Object> fields_written;
        public Map<String, String> fields_skipped;
        public Extraction extraction;
        public Decimal processing_time_ms;
    }

    public class Extraction {
        public Boolean complete;
        public Integer chunks_total;
        public Integer chunks_ok;
        public List<String> failed_chunks;
        public List<String> failed_sections;
        public Boolean text_truncated;
        public Integer text_length;
        public Integer text_sent;
    }

    public class ResumeParserException extends Exception {}
}
