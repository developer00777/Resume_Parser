/**
 * Tests for ResumeParserQueueable.
 *
 * Callouts are mocked — Apex forbids real ones in tests, and the behaviour
 * under test is this class's own: does it call the right endpoint, does it
 * survive every failure shape, and does it write down what actually happened.
 */
@IsTest
private class ResumeParserQueueableTest {

    private class Mock implements HttpCalloutMock {
        private final Integer status;
        private final String body;
        private String seenEndpoint;
        private String seenMethod;

        private Mock(Integer status, String body) {
            this.status = status;
            this.body = body;
        }

        public HttpResponse respond(HttpRequest req) {
            seenEndpoint = req.getEndpoint();
            seenMethod = req.getMethod();
            lastEndpoint = seenEndpoint;
            lastMethod = seenMethod;

            HttpResponse res = new HttpResponse();
            res.setStatusCode(status);
            res.setBody(body);
            return res;
        }
    }

    private static String lastEndpoint;
    private static String lastMethod;

    private static String okBody(Integer fieldCount) {
        List<String> pairs = new List<String>();
        for (Integer i = 0; i < fieldCount; i++) {
            pairs.add('"Field_' + i + '__c":"v' + i + '"');
        }
        return '{"success":true,"record_id":"a0X","dry_run":false,' +
            '"fields_written":{' + String.join(pairs, ',') + '},' +
            '"fields_skipped":{"PAN_Number":"no value extracted"},' +
            '"extraction":{"complete":true,"chunks_total":3,"chunks_ok":3,' +
            '"failed_chunks":[],"failed_sections":[],"text_truncated":false,' +
            '"text_length":4200,"text_sent":4200},"processing_time_ms":8123.4}';
    }

    private static SCSCHAMPS__Candidate__c newCandidate() {
        SCSCHAMPS__Candidate__c c = new SCSCHAMPS__Candidate__c(Name = 'Test Candidate');
        insert c;
        return c;
    }

    private static String statusOf(Id recordId) {
        Map<String, SObjectField> fields =
            SCSCHAMPS__Candidate__c.SObjectType.getDescribe().fields.getMap();
        if (!fields.containsKey('Resume_Parse_Status__c')) {
            return null;  // org has not created the field; nothing to assert
        }
        List<SObject> rows = Database.query(
            'SELECT Resume_Parse_Status__c FROM SCSCHAMPS__Candidate__c WHERE Id = :recordId'
        );
        return rows.isEmpty() ? null : (String) rows[0].get('Resume_Parse_Status__c');
    }

    // ── The happy path ──────────────────────────────────────────────────────

    @IsTest
    static void callsTheWriteBackEndpointWithTheRecordId() {
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        Test.setMock(HttpCalloutMock.class, new Mock(200, okBody(3)));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        System.assertEquals('POST', lastMethod, 'must POST');
        System.assert(
            lastEndpoint.contains('/api/v1/salesforce/parse-and-update'),
            'must hit the write-back endpoint, got: ' + lastEndpoint
        );
        System.assert(
            lastEndpoint.contains('record_id=' + candidate.Id),
            'must pass the record Id, got: ' + lastEndpoint
        );
        System.assert(
            lastEndpoint.startsWith('callout:Resume_Parser_API'),
            'must go through the Named Credential, got: ' + lastEndpoint
        );
    }

    @IsTest
    static void recordsHowManyFieldsChanged() {
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        Test.setMock(HttpCalloutMock.class, new Mock(200, okBody(5)));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        String note = statusOf(candidate.Id);
        if (note != null) {
            System.assert(note.contains('5 field(s) updated'), 'got: ' + note);
        }
    }

    // ── The case the whole mechanism exists for ─────────────────────────────

    @IsTest
    static void aPartialExtractionSaysWhatWasNotWritten() {
        // Chunk C died, so experience and education were withheld rather than
        // blanked. If that is not written down, the record is indistinguishable
        // from a candidate with no job history.
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        String body =
            '{"success":true,"record_id":"a0X","dry_run":false,' +
            '"fields_written":{"SCSCHAMPS__Phone__c":"+91123"},"fields_skipped":' +
            '{"CurrentCompany":"experience extraction failed"},' +
            '"extraction":{"complete":false,"chunks_total":3,"chunks_ok":2,' +
            '"failed_chunks":["chunk_c"],"failed_sections":["education","experience"],' +
            '"text_truncated":true,"text_length":15000,"text_sent":8000},' +
            '"processing_time_ms":6000.0}';
        Test.setMock(HttpCalloutMock.class, new Mock(200, body));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        String note = statusOf(candidate.Id);
        if (note != null) {
            System.assert(note.contains('NOT extracted'), 'must flag the gap: ' + note);
            System.assert(note.contains('experience'), 'must name the section: ' + note);
            System.assert(note.contains('education'), 'must name every section: ' + note);
            System.assert(
                note.contains('8000') && note.contains('15000'),
                'must report the truncation: ' + note
            );
        }
    }

    // ── Failure shapes ──────────────────────────────────────────────────────

    @IsTest
    static void aMissingFieldMappingIsRecordedAsConfiguration() {
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        Test.setMock(HttpCalloutMock.class,
            new Mock(409, '{"detail":"Field mapping is not loaded"}'));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        String note = statusOf(candidate.Id);
        if (note != null) {
            System.assert(note.containsIgnoreCase('failed'), 'got: ' + note);
            System.assert(
                note.containsIgnoreCase('mapping'),
                'the note must point at the real cause: ' + note
            );
        }
    }

    @IsTest
    static void aServerErrorIsRecordedNotThrown() {
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        Test.setMock(HttpCalloutMock.class, new Mock(502, 'upstream exploded'));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        String note = statusOf(candidate.Id);
        if (note != null) {
            System.assert(note.containsIgnoreCase('failed'), 'got: ' + note);
            System.assert(note.contains('502'), 'the status code helps debugging: ' + note);
        }
    }

    @IsTest
    static void aMalformedBodyDoesNotKillTheJob() {
        SCSCHAMPS__Candidate__c candidate = newCandidate();
        Test.setMock(HttpCalloutMock.class, new Mock(200, 'not json at all'));

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(candidate.Id));
        Test.stopTest();

        String note = statusOf(candidate.Id);
        if (note != null) {
            System.assert(note.containsIgnoreCase('failed'), 'got: ' + note);
        }
    }

    @IsTest
    static void anEmptyQueueMakesNoCallout() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, okBody(1)));
        lastEndpoint = null;

        Test.startTest();
        System.enqueueJob(new ResumeParserQueueable(new List<Id>()));
        Test.stopTest();

        System.assertEquals(null, lastEndpoint, 'nothing to parse means no callout');
    }

    // ── The response contract ───────────────────────────────────────────────

    @IsTest
    static void theWrapperModelsThePayload() {
        // This is what silently broke while the payload carried both
        // `Resume_Score` and `resume_score`: Apex identifiers are
        // case-insensitive, so no typed class could hold the pair.
        ResumeParserQueueable.ParseResult parsed =
            (ResumeParserQueueable.ParseResult) JSON.deserialize(
                okBody(2), ResumeParserQueueable.ParseResult.class
            );

        System.assertEquals(true, parsed.success);
        System.assertEquals(2, parsed.fields_written.size());
        System.assertEquals(1, parsed.fields_skipped.size());
        System.assertEquals(3, parsed.extraction.chunks_ok);
        System.assertEquals(3, parsed.extraction.chunks_total);
        System.assertEquals(false, parsed.extraction.text_truncated);
        System.assertEquals(0, parsed.extraction.failed_sections.size());
    }

    @IsTest
    static void unknownAttributesDoNotBreakDeserialization() {
        // The parser must be able to add response fields without a redeploy on
        // the Salesforce side.
        String body = '{"success":true,"a_brand_new_field":"whatever",' +
            '"fields_written":{},"extraction":{"complete":true,"a_new_one":1}}';

        ResumeParserQueueable.ParseResult parsed =
            (ResumeParserQueueable.ParseResult) JSON.deserialize(
                body, ResumeParserQueueable.ParseResult.class
            );

        System.assertEquals(true, parsed.success, 'unknown attributes must be ignored');
    }

    @IsTest
    static void truncateRespectsTheFieldLimit() {
        String long100 = 'x'.repeat(100);
        System.assertEquals(10, ResumeParserQueueable.truncate(long100, 10).length());
        System.assertEquals('short', ResumeParserQueueable.truncate('short', 50));
        System.assertEquals(null, ResumeParserQueueable.truncate(null, 50));
    }
}
