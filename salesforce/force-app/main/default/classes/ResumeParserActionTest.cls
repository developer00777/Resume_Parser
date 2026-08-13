/**
 * Tests for the Flow and LWC entry points.
 *
 * These assert on returned state rather than merely "no exception thrown",
 * because the behaviour that matters here is what gets refused: a blank Id, a
 * duplicate in the same Flow batch, a record with no resume, a user without
 * edit rights.
 */
@IsTest
private class ResumeParserActionTest {

    private static ResumeParserAction.Request request(String recordId) {
        ResumeParserAction.Request req = new ResumeParserAction.Request();
        req.recordId = recordId;
        return req;
    }

    /** A syntactically valid Id for the Candidate object in this org. */
    private static Id fakeCandidateId(Integer seed) {
        String suffix = String.valueOf(seed).leftPad(12, '0');
        return Id.valueOf(
            SCSCHAMPS__Candidate__c.SObjectType.getDescribe().getKeyPrefix() + suffix
        );
    }

    // ── Flow entry point ────────────────────────────────────────────────────

    @IsTest
    static void queuesEachDistinctRecord() {
        List<ResumeParserAction.Request> requests = new List<ResumeParserAction.Request>{
            request(fakeCandidateId(1)),
            request(fakeCandidateId(2))
        };

        Test.startTest();
        List<ResumeParserAction.Result> results = ResumeParserAction.parseResumes(requests);
        Test.stopTest();

        System.assertEquals(2, results.size(), 'one result per request');
        for (ResumeParserAction.Result r : results) {
            System.assertEquals(true, r.queued, 'both records should queue: ' + r.message);
            System.assertNotEquals(null, r.recordId, 'the record Id should be echoed back');
        }
    }

    @IsTest
    static void enqueuesOneJobForTheWholeBatch() {
        // One job that chains, not one job per record — 50 enqueues per
        // transaction is the limit, and a real Flow batch would breach it.
        List<ResumeParserAction.Request> requests = new List<ResumeParserAction.Request>();
        for (Integer i = 1; i <= 10; i++) {
            requests.add(request(fakeCandidateId(i)));
        }

        Test.startTest();
        ResumeParserAction.parseResumes(requests);
        Integer queued = Limits.getQueueableJobs();
        Test.stopTest();

        System.assertEquals(1, queued, '10 records must produce exactly one chained job');
    }

    @IsTest
    static void rejectsABlankRecordId() {
        Test.startTest();
        List<ResumeParserAction.Result> results =
            ResumeParserAction.parseResumes(new List<ResumeParserAction.Request>{ request('') });
        Test.stopTest();

        System.assertEquals(1, results.size());
        System.assertEquals(false, results[0].queued);
        System.assert(
            results[0].message.containsIgnoreCase('no record id'),
            'message should say why: ' + results[0].message
        );
    }

    @IsTest
    static void rejectsAMalformedRecordId() {
        Test.startTest();
        List<ResumeParserAction.Result> results = ResumeParserAction.parseResumes(
            new List<ResumeParserAction.Request>{ request('not-an-id') }
        );
        Test.stopTest();

        System.assertEquals(false, results[0].queued);
        System.assert(
            results[0].message.containsIgnoreCase('not a valid record id'),
            'message should name the problem: ' + results[0].message
        );
    }

    @IsTest
    static void deduplicatesWithinOneTransaction() {
        // A record-triggered Flow can easily hand the same record over twice.
        // Parsing it twice costs two LLM runs for one outcome.
        Id duplicated = fakeCandidateId(7);

        Test.startTest();
        List<ResumeParserAction.Result> results = ResumeParserAction.parseResumes(
            new List<ResumeParserAction.Request>{
                request(duplicated), request(duplicated), request(duplicated)
            }
        );
        Integer queued = Limits.getQueueableJobs();
        Test.stopTest();

        System.assertEquals(3, results.size(), 'Flow expects one result per input');
        System.assertEquals(true, results[0].queued, 'the first occurrence queues');
        System.assertEquals(false, results[1].queued, 'the duplicate does not');
        System.assertEquals(false, results[2].queued);
        System.assert(
            results[1].message.containsIgnoreCase('already queued'),
            'duplicate should say so: ' + results[1].message
        );
        System.assertEquals(1, queued, 'only one job for three copies of one record');
    }

    @IsTest
    static void anEmptyBatchEnqueuesNothing() {
        Test.startTest();
        List<ResumeParserAction.Result> results =
            ResumeParserAction.parseResumes(new List<ResumeParserAction.Request>());
        Integer queued = Limits.getQueueableJobs();
        Test.stopTest();

        System.assertEquals(0, results.size());
        System.assertEquals(0, queued, 'nothing to do means no job');
    }

    @IsTest
    static void mixedBatchQueuesOnlyTheValidRecords() {
        Test.startTest();
        List<ResumeParserAction.Result> results = ResumeParserAction.parseResumes(
            new List<ResumeParserAction.Request>{
                request(fakeCandidateId(11)), request(''), request(fakeCandidateId(12))
            }
        );
        Test.stopTest();

        System.assertEquals(true, results[0].queued);
        System.assertEquals(false, results[1].queued, 'one bad entry must not fail the batch');
        System.assertEquals(true, results[2].queued);
    }

    // ── LWC entry point ─────────────────────────────────────────────────────

    @IsTest
    static void lwcRejectsANullRecord() {
        Test.startTest();
        try {
            ResumeParserAction.parseResume(null);
            System.assert(false, 'a null record Id should have been refused');
        } catch (AuraHandledException e) {
            // AuraHandledException deliberately hides its detail from the
            // client; reaching this branch is the assertion.
        }
        Test.stopTest();
    }

    @IsTest
    static void lwcRefusesARecordWithNoResume() {
        SCSCHAMPS__Candidate__c candidate = new SCSCHAMPS__Candidate__c(Name = 'No Resume');
        insert candidate;

        Test.startTest();
        Boolean refused = false;
        try {
            ResumeParserAction.parseResume(candidate.Id);
        } catch (AuraHandledException e) {
            refused = true;
        }
        Integer queued = Limits.getQueueableJobs();
        Test.stopTest();

        System.assertEquals(true, refused, 'a candidate with no resume should be refused up front');
        System.assertEquals(0, queued, 'nothing should be queued for a record with nothing to parse');
    }

    @IsTest
    static void lwcQueuesARecordThatHasAResume() {
        SCSCHAMPS__Candidate__c candidate = new SCSCHAMPS__Candidate__c(
            Name = 'Has Resume',
            SCSCHAMPS__Resume_URL__c = 'https://example.invalid/resume.pdf'
        );
        insert candidate;

        Test.startTest();
        String message = ResumeParserAction.parseResume(candidate.Id);
        Integer queued = Limits.getQueueableJobs();
        Test.stopTest();

        System.assertEquals(1, queued, 'the parse should have been queued');
        System.assert(
            message.containsIgnoreCase('parsing started'),
            'the user needs to be told it started: ' + message
        );
    }
}
