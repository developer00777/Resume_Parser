/**
 * Batch Apex that queues every unparsed resume, however many there are.
 *
 * This is the piece that makes bulk work. Each execute() is its own
 * transaction with a fresh 120-second cumulative callout budget and a fresh
 * heap, so scale comes from the number of chunks, not from stretching one
 * request. Inside a chunk we make exactly one callout — a few KB of record ids
 * — which costs ~0.3 s of that 120 s.
 *
 * Run it:
 *     Database.executeBatch(new ResumeParserSubmitBatch(), 100);
 *
 * Scope 100 matches the parser's per-batch cap, so one chunk == one parser
 * batch == one callout. 5,000 resumes is 50 chunks and 50 callouts, none of
 * which can time out.
 */
public with sharing class ResumeParserSubmitBatch
    implements Database.Batchable<SObject>, Database.AllowsCallouts, Database.Stateful {

    // Batch ids handed back by the parser, polled once submission finishes.
    private List<String> batchIds = new List<String>();
    private Integer submitted = 0;
    private Integer failedChunks = 0;

    private final String callbackUrl;
    private final String callbackToken;
    private final String query;

    public ResumeParserSubmitBatch() {
        this(null, null);
    }

    /**
     * Pass a callback URL to skip polling entirely — the parser POSTs each
     * finished batch to your Apex REST resource instead.
     */
    public ResumeParserSubmitBatch(String callbackUrl, String callbackToken) {
        this.callbackUrl = callbackUrl;
        this.callbackToken = callbackToken;
        // Adjust to your own object and "needs parsing" criteria.
        this.query =
            'SELECT Id, SCSCHAMPS__Resume_Attachment_Id__c ' +
            'FROM SCSCHAMPS__Candidate__c ' +
            'WHERE SCSCHAMPS__Resume_Attachment_Id__c != null ' +
            '  AND Resume_Parsed__c = false';
    }

    public Database.QueryLocator start(Database.BatchableContext bc) {
        return Database.getQueryLocator(query);
    }

    public void execute(Database.BatchableContext bc, List<SObject> scope) {
        List<ResumeParserService.RecordRef> refs = new List<ResumeParserService.RecordRef>();

        for (SObject row : scope) {
            String attachmentId = (String) row.get('SCSCHAMPS__Resume_Attachment_Id__c');
            if (String.isBlank(attachmentId)) {
                continue;
            }
            // external_id carries the Candidate Id so results come back
            // correlated — filenames collide and ordering is fragile.
            refs.add(new ResumeParserService.RecordRef(
                attachmentId, 'attachment', (String) row.get('Id')
            ));
        }

        if (refs.isEmpty()) {
            return;
        }

        try {
            batchIds.add(
                ResumeParserService.submitRecords(refs, callbackUrl, callbackToken)
            );
            submitted += refs.size();
        } catch (Exception e) {
            // One bad chunk must not abort the run; the rest still submit.
            failedChunks++;
            System.debug(LoggingLevel.ERROR,
                'Resume parser submit chunk failed: ' + e.getMessage());
        }
    }

    public void finish(Database.BatchableContext bc) {
        System.debug(
            'Resume parser: submitted ' + submitted + ' resume(s) across ' +
            batchIds.size() + ' batch(es); ' + failedChunks + ' chunk(s) failed.'
        );

        // With a callback configured the parser pushes results to us — nothing to poll.
        if (String.isNotBlank(callbackUrl) || batchIds.isEmpty()) {
            return;
        }
        System.enqueueJob(new ResumeParserPoller(batchIds, 0), 1);
    }
}
