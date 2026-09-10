/**
 * Coverage for the parser client. Salesforce needs 75% before these deploy to
 * production, and callouts must be mocked — Test.setMock is mandatory.
 */
@IsTest
private class ResumeParserServiceTest {

    /** Serves a canned response per (method, path fragment). */
    private class Mock implements HttpCalloutMock {
        private final Integer status;
        private final String body;
        public Integer callCount = 0;
        public String lastBody;
        public String lastEndpoint;

        Mock(Integer status, String body) {
            this.status = status;
            this.body = body;
        }

        public HttpResponse respond(HttpRequest req) {
            callCount++;
            lastBody = req.getBody();
            lastEndpoint = req.getEndpoint();
            HttpResponse res = new HttpResponse();
            res.setStatusCode(status);
            res.setBody(body);
            res.setHeader('Content-Type', 'application/json');
            return res;
        }
    }

    private static String submitJson(String batchId, Integer total) {
        return JSON.serialize(new Map<String, Object>{
            'batch_id' => batchId,
            'status'   => 'queued',
            'mode'     => 'salesforce',
            'total'    => total,
            'poll_url' => '/api/v1/parse/jobs/' + batchId
        });
    }

    private static String batchJson(String status, String externalId) {
        return JSON.serialize(new Map<String, Object>{
            'batch_id'  => 'b1',
            'status'    => status,
            'mode'      => 'salesforce',
            'total'     => 1,
            'completed' => status == 'completed' ? 1 : 0,
            'parsed'    => status == 'completed' ? 1 : 0,
            'failed'    => 0,
            'remaining' => status == 'completed' ? 0 : 1,
            'running'   => 0,
            'results'   => status == 'completed'
                ? new List<Object>{ new Map<String, Object>{
                      'index'       => 0,
                      'filename'    => 'cv.pdf',
                      'external_id' => externalId,
                      'success'     => true,
                      'data'        => new Map<String, Object>{
                          'FirstName' => 'Jane',
                          'LastName'  => 'Doe',
                          'Email'     => 'jane@example.com'
                      }
                  }}
                : new List<Object>()
        });
    }

    private static List<ResumeParserService.RecordRef> oneRef() {
        return new List<ResumeParserService.RecordRef>{
            new ResumeParserService.RecordRef('068000000000001', 'attachment', null)
        };
    }

    @IsTest
    static void submitReturnsBatchId() {
        Mock mock = new Mock(202, submitJson('abc123', 1));
        Test.setMock(HttpCalloutMock.class, mock);

        Test.startTest();
        String batchId = ResumeParserService.submitRecords(oneRef());
        Test.stopTest();

        System.assertEquals('abc123', batchId, 'batch_id should come from the response');
        System.assert(
            mock.lastEndpoint.endsWith('/api/v1/parse/salesforce/jobs/records'),
            'Should post to the record-id endpoint, got ' + mock.lastEndpoint
        );
        System.assert(
            mock.lastBody.contains('068000000000001'),
            'Body should carry the record id, not file bytes'
        );
    }

    @IsTest
    static void submitSendsExternalIdForCorrelation() {
        Mock mock = new Mock(202, submitJson('abc123', 1));
        Test.setMock(HttpCalloutMock.class, mock);

        List<ResumeParserService.RecordRef> refs = new List<ResumeParserService.RecordRef>{
            new ResumeParserService.RecordRef('068000000000001', 'attachment', 'a0X000000000001')
        };

        Test.startTest();
        ResumeParserService.submitRecords(refs);
        Test.stopTest();

        System.assert(
            mock.lastBody.contains('a0X000000000001'),
            'external_id must travel so results can be matched back'
        );
    }

    @IsTest
    static void submitRejectsEmptyList() {
        Test.setMock(HttpCalloutMock.class, new Mock(202, submitJson('x', 0)));

        Test.startTest();
        try {
            ResumeParserService.submitRecords(new List<ResumeParserService.RecordRef>());
            System.assert(false, 'Expected an exception for an empty batch');
        } catch (ResumeParserService.ResumeParserException e) {
            System.assert(e.getMessage().contains('no records'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void submitRejectsOversizedBatch() {
        Test.setMock(HttpCalloutMock.class, new Mock(202, submitJson('x', 0)));

        List<ResumeParserService.RecordRef> refs = new List<ResumeParserService.RecordRef>();
        for (Integer i = 0; i <= ResumeParserService.MAX_PER_BATCH; i++) {
            refs.add(new ResumeParserService.RecordRef('068' + i, 'attachment', null));
        }

        Test.startTest();
        try {
            ResumeParserService.submitRecords(refs);
            System.assert(false, 'Expected an exception for an oversized batch');
        } catch (ResumeParserService.ResumeParserException e) {
            System.assert(e.getMessage().contains('exceeds'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void submitSurfacesServerErrors() {
        Test.setMock(HttpCalloutMock.class, new Mock(503, '{"detail":"queue down"}'));

        Test.startTest();
        try {
            ResumeParserService.submitRecords(oneRef());
            System.assert(false, 'Expected an exception on 503');
        } catch (ResumeParserService.ResumeParserException e) {
            System.assert(e.getMessage().contains('503'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void pollReportsProcessingThenComplete() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, batchJson('processing', null)));

        Test.startTest();
        Map<String, Object> running = ResumeParserService.pollBatch('b1');
        Test.stopTest();

        System.assertEquals(false, ResumeParserService.isComplete(running));
        System.assertEquals('processing', (String) running.get('status'));
    }

    @IsTest
    static void pollDetectsCompletion() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, batchJson('completed', 'a0X1')));

        Test.startTest();
        Map<String, Object> done = ResumeParserService.pollBatch('b1');
        Test.stopTest();

        System.assertEquals(true, ResumeParserService.isComplete(done));
        List<Object> results = (List<Object>) done.get('results');
        System.assertEquals(1, results.size());
    }

    @IsTest
    static void pollRaisesOnMissingBatch() {
        Test.setMock(HttpCalloutMock.class, new Mock(404, '{"detail":"not found"}'));

        Test.startTest();
        try {
            ResumeParserService.pollBatch('gone');
            System.assert(false, 'Expected an exception on 404');
        } catch (ResumeParserService.ResumeParserException e) {
            System.assert(e.getMessage().contains('not found'), e.getMessage());
        }
        Test.stopTest();
    }

    @IsTest
    static void pollerStopsOnceEveryBatchCompletes() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, batchJson('completed', null)));

        Test.startTest();
        System.enqueueJob(new ResumeParserPoller(new List<String>{ 'b1' }, 0));
        Test.stopTest();

        // A completed batch must not chain another poll.
        System.assertEquals(
            0,
            [SELECT COUNT() FROM AsyncApexJob
             WHERE JobType = 'Queueable' AND Status IN ('Queued', 'Processing')],
            'Poller should not re-enqueue once the batch is complete'
        );
    }

    @IsTest
    static void pollerSurvivesAFailingBatch() {
        Test.setMock(HttpCalloutMock.class, new Mock(500, '{"detail":"boom"}'));

        Test.startTest();
        // Must not throw — one unhealthy batch cannot break the whole poll tick.
        System.enqueueJob(new ResumeParserPoller(new List<String>{ 'b1' }, 0));
        Test.stopTest();
    }
}
