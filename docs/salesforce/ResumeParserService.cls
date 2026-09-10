/**
 * Thin client for the Resume Parser queue API.
 *
 * Why record ids and not files
 * ----------------------------
 * Two governor limits make it impossible to push resume *bytes* out of Apex in
 * bulk, and neither is a timeout you can raise:
 *
 *   1. Heap — 6 MB sync / 12 MB async. A 2 MB PDF costs roughly 8 MB by the
 *      time you hold the Blob, its base64 String, and the request body. You
 *      cannot fit more than one or two per transaction, ever.
 *   2. Cumulative callout time — 120 s per transaction, no matter how many
 *      callouts you split the work into. Fifteen 20-second callouts blow the
 *      budget on the sixth, even though each one is individually legal.
 *
 * So this class sends *ids*. submitRecords() posts a few KB regardless of how
 * large the resumes are, returns in ~300 ms, and the parser downloads each file
 * server-side with the Connected App credentials. One callout for 100 resumes
 * costs ~0.3 s of the 120 s budget instead of blowing straight through it.
 *
 * Setup: a Named Credential called Resume_Parser pointing at the parser host,
 * with the API key stored as its password.
 */
public with sharing class ResumeParserService {

    // Named Credential; the API key travels as {!$Credential.Password}.
    public static final String BASE = 'callout:Resume_Parser';

    // The parser caps a batch at QUEUE_MAX_FILES (default 100).
    public static final Integer MAX_PER_BATCH = 100;

    private static final Integer CALLOUT_TIMEOUT_MS = 30000;

    public class ResumeParserException extends Exception {}

    /** One resume to parse, identified by id rather than carried as bytes. */
    public class RecordRef {
        public String ref;           // ContentVersion / Attachment / Candidate Id, or URL
        public String kind;          // attachment (default) | candidate | url
        public String filename;      // optional
        public String external_id;   // echoed back on the result — put your Candidate Id here

        public RecordRef(String ref, String kind, String externalId) {
            this.ref = ref;
            this.kind = kind;
            this.external_id = externalId;
        }
    }

    private class SubmitBody {
        public List<RecordRef> records;
        public String callback_url;
        public String callback_token;
    }

    /**
     * Queue a batch of resumes. Returns the batch_id to poll.
     *
     * Costs one callout of a few hundred milliseconds however many records are
     * in the list, so several of these fit comfortably in one transaction.
     */
    public static String submitRecords(List<RecordRef> records) {
        return submitRecords(records, null, null);
    }

    public static String submitRecords(
        List<RecordRef> records, String callbackUrl, String callbackToken
    ) {
        if (records == null || records.isEmpty()) {
            throw new ResumeParserException('submitRecords called with no records.');
        }
        if (records.size() > MAX_PER_BATCH) {
            throw new ResumeParserException(
                'Batch of ' + records.size() + ' exceeds the parser limit of ' + MAX_PER_BATCH + '.'
            );
        }

        SubmitBody payload = new SubmitBody();
        payload.records = records;
        payload.callback_url = callbackUrl;
        payload.callback_token = callbackToken;

        HttpResponse res = send(
            'POST', '/api/v1/parse/salesforce/jobs/records', JSON.serialize(payload, true)
        );

        if (res.getStatusCode() != 202) {
            throw new ResumeParserException(
                'Submit failed (' + res.getStatusCode() + '): ' + res.getBody()
            );
        }
        Map<String, Object> body =
            (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
        return (String) body.get('batch_id');
    }

    /** Fetch a batch's current state, including whatever results have landed. */
    public static Map<String, Object> pollBatch(String batchId) {
        HttpResponse res = send('GET', '/api/v1/parse/jobs/' + batchId, null);

        if (res.getStatusCode() == 404) {
            throw new ResumeParserException('Batch ' + batchId + ' not found (expired?).');
        }
        if (res.getStatusCode() != 200) {
            throw new ResumeParserException(
                'Poll failed (' + res.getStatusCode() + '): ' + res.getBody()
            );
        }
        return (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
    }

    /** True once every file in the batch has a result row. */
    public static Boolean isComplete(Map<String, Object> batch) {
        return 'completed'.equals((String) batch.get('status'));
    }

    private static HttpResponse send(String method, String path, String body) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE + path);
        req.setMethod(method);
        req.setHeader('X-API-Key', '{!$Credential.Password}');
        req.setTimeout(CALLOUT_TIMEOUT_MS);
        if (body != null) {
            req.setHeader('Content-Type', 'application/json');
            req.setBody(body);
        }
        return new Http().send(req);
    }
}
