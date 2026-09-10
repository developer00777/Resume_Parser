/**
 * Coverage for ResumeParserService. Salesforce needs 75% before this deploys to
 * production, and every callout must be mocked.
 *
 * The Queueable chain is suppressed under Test.isRunningTest(), so each step is
 * driven directly rather than relying on chaining (which Apex caps at depth 1
 * inside a test).
 */
@IsTest
private class ResumeParserServiceTest {

    private class Mock implements HttpCalloutMock {
        private final Integer status;
        private final String body;
        public String lastBody;
        public String lastEndpoint;
        public String lastMethod;

        Mock(Integer status, String body) {
            this.status = status;
            this.body = body;
        }

        public HttpResponse respond(HttpRequest req) {
            lastBody = req.getBody();
            lastEndpoint = req.getEndpoint();
            lastMethod = req.getMethod();
            HttpResponse res = new HttpResponse();
            res.setStatusCode(status);
            res.setBody(body);
            res.setHeader('Content-Type', 'application/json');
            return res;
        }
    }

    private static final String B64 = EncodingUtil.base64Encode(Blob.valueOf('%PDF-fake'));

    private static String submitJson(String batchId) {
        return JSON.serialize(new Map<String, Object>{
            'batch_id' => batchId, 'status' => 'queued',
            'mode' => 'client', 'total' => 1,
            'poll_url' => '/api/v1/parse/jobs/' + batchId
        });
    }

    /** A completed batch carrying client-mode field names. */
    private static String completedJson(String fileName) {
        return JSON.serialize(new Map<String, Object>{
            'batch_id' => 'b1', 'status' => 'completed', 'mode' => 'client',
            'total' => 1, 'completed' => 1, 'parsed' => 1, 'failed' => 0,
            'remaining' => 0, 'running' => 0,
            'results' => new List<Object>{ new Map<String, Object>{
                'index' => 0,
                'filename' => fileName,
                'external_id' => fileName,
                'success' => true,
                'data' => new Map<String, Object>{
                    'Name'                      => 'Jane Marie Doe',
                    'Email'                     => 'jane.doe@example.com',
                    'Nationality__c'            => 'Indian',
                    'MaritalStatus__c'          => 'Single',
                    'Gender__c'                 => 'Female',
                    'Date_of_Birth__c'          => '1994-04-12',
                    'Years_of_Experience__c'    => 7.5,
                    'Current_Location__c'       => 'Dubai',
                    'CurrentDesignation__c'     => 'Senior Engineer',
                    'CurrentCompany__c'         => 'Acme Corp',
                    'PhoneNumber__c'            => '+971 50 123 4567',
                    'SCSCHAMPS__PhoneNumber__c' => '+91 98765 43210',
                    'Type_1__c'                 => 'Graduate',
                    'Graduation_Year2__c'       => '2016',
                    'Institution_College__c'    => 'IIT Bombay',
                    'Name__c'                   => 'B.Tech Computer Science',
                    'Year__c'                   => '2016-01-01',
                    'Spoken_Language__c'        => new List<Object>{ 'English', 'Hindi' }
                }
            }}
        });
    }

    private static String failedRowJson(String fileName) {
        return JSON.serialize(new Map<String, Object>{
            'batch_id' => 'b1', 'status' => 'completed', 'mode' => 'client',
            'total' => 1, 'completed' => 1, 'parsed' => 0, 'failed' => 1,
            'remaining' => 0, 'running' => 0,
            'results' => new List<Object>{ new Map<String, Object>{
                'index' => 0, 'filename' => fileName, 'external_id' => fileName,
                'success' => false, 'data' => null,
                'error' => 'Failed to parse PDF: corrupt header'
            }}
        });
    }

    // ── Entry point ──────────────────────────────────────────────────────────

    @IsTest
    static void startParsingEnqueuesAJob() {
        Test.setMock(HttpCalloutMock.class, new Mock(202, submitJson('abc123')));

        Test.startTest();
        String status = ResumeParserService.startParsing('cv.pdf', B64);
        Test.stopTest();

        System.assertEquals('Processing Started', status);
    }

    @IsTest
    static void startParsingRejectsMissingInput() {
        Test.startTest();
        try {
            ResumeParserService.startParsing('', null);
            System.assert(false, 'Expected an AuraHandledException');
        } catch (AuraHandledException e) {
            // expected
        }
        Test.stopTest();
    }

    // ── Submit ───────────────────────────────────────────────────────────────

    @IsTest
    static void submitPostsJsonToTheClientQueue() {
        Mock mock = new Mock(202, submitJson('abc123'));
        Test.setMock(HttpCalloutMock.class, mock);

        Test.startTest();
        System.enqueueJob(new ResumeParserService.SubmitJob('cv.pdf', B64));
        Test.stopTest();

        System.assertEquals('POST', mock.lastMethod);
        System.assert(
            mock.lastEndpoint.endsWith('/api/v1/parse/client/jobs/base64'),
            'Must use client mode — salesforce mode omits MaritalStatus__c etc. Got: '
                + mock.lastEndpoint
        );
        System.assert(mock.lastBody.contains('content_base64'), mock.lastBody);
        System.assert(mock.lastBody.contains('cv.pdf'), mock.lastBody);
    }

    @IsTest
    static void submitFailureIsLoggedNotThrown() {
        Test.setMock(HttpCalloutMock.class, new Mock(503, '{"detail":"queue down"}'));

        Test.startTest();
        System.enqueueJob(new ResumeParserService.SubmitJob('broken.pdf', B64));
        Test.stopTest();

        List<Resume_Parser_Log__c> logs = [
            SELECT Status__c, Error_Message__c
            FROM Resume_Parser_Log__c
            WHERE Resume_Name__c = 'broken.pdf'
        ];
        System.assertEquals(1, logs.size(), 'A failure must still be logged');
        System.assertEquals('Failed', logs[0].Status__c);
        System.assert(logs[0].Error_Message__c.contains('503'), logs[0].Error_Message__c);
    }

    // ── Poll + apply ─────────────────────────────────────────────────────────

    @IsTest
    static void completedBatchCreatesAContactLogAndAttachment() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, completedJson('cv.pdf')));

        Test.startTest();
        System.enqueueJob(new ResumeParserService.PollJob('b1', 'cv.pdf', B64, 0));
        Test.stopTest();

        List<Contact> contacts = [
            SELECT FirstName, LastName, Email, Nationality__c, MaritalStatus__c,
                   Gender__c, Years_of_Experience__c, Current_Location__c,
                   CurrentDesignation__c, CurrentCompany__c, PhoneNumber__c,
                   Type_1__c, Spoken_Language__c, Mobile_Phone_Country_Code__c
            FROM Contact WHERE Email = 'jane.doe@example.com'
        ];
        System.assertEquals(1, contacts.size(), 'One Contact should be created');
        Contact c = contacts[0];

        System.assertEquals('Jane', c.FirstName);
        System.assertEquals('Marie Doe', c.LastName);
        // The fields that were silently blank before the endpoint switch:
        System.assertEquals('Single', c.MaritalStatus__c);
        System.assertEquals('Female', c.Gender__c);
        System.assertEquals('Indian', c.Nationality__c);
        System.assertEquals('Graduate', c.Type_1__c);
        System.assertEquals('English;Hindi', c.Spoken_Language__c);
        System.assertEquals(7.5, c.Years_of_Experience__c);
        System.assertEquals('United Arab Emirates (+971)', c.Mobile_Phone_Country_Code__c);

        System.assertEquals(1, [
            SELECT COUNT() FROM Resume_Parser_Log__c
            WHERE Resume_Name__c = 'cv.pdf' AND Status__c = 'Success'
        ]);
        System.assertEquals(1, [
            SELECT COUNT() FROM Attachment WHERE ParentId = :c.Id
        ]);
    }

    @IsTest
    static void failedRowLogsWithoutCreatingAContact() {
        Test.setMock(HttpCalloutMock.class, new Mock(200, failedRowJson('bad.pdf')));

        Test.startTest();
        System.enqueueJob(new ResumeParserService.PollJob('b1', 'bad.pdf', B64, 0));
        Test.stopTest();

        List<Resume_Parser_Log__c> logs = [
            SELECT Status__c, Error_Message__c, Contact__c
            FROM Resume_Parser_Log__c WHERE Resume_Name__c = 'bad.pdf'
        ];
        System.assertEquals(1, logs.size());
        System.assertEquals('Failed', logs[0].Status__c);
        System.assert(logs[0].Error_Message__c.contains('corrupt header'),
            logs[0].Error_Message__c);
        System.assertEquals(null, logs[0].Contact__c);
    }

    @IsTest
    static void expiredBatchIsLogged() {
        Test.setMock(HttpCalloutMock.class, new Mock(404, '{"detail":"not found"}'));

        Test.startTest();
        System.enqueueJob(new ResumeParserService.PollJob('gone', 'cv.pdf', B64, 0));
        Test.stopTest();

        System.assertEquals(1, [
            SELECT COUNT() FROM Resume_Parser_Log__c
            WHERE Resume_Name__c = 'cv.pdf' AND Status__c = 'Failed'
        ]);
    }

    @IsTest
    static void stillProcessingCreatesNoRecords() {
        String running = JSON.serialize(new Map<String, Object>{
            'batch_id' => 'b1', 'status' => 'processing', 'mode' => 'client',
            'total' => 1, 'completed' => 0, 'parsed' => 0, 'failed' => 0,
            'remaining' => 1, 'running' => 1, 'results' => new List<Object>()
        });
        Test.setMock(HttpCalloutMock.class, new Mock(200, running));

        Test.startTest();
        System.enqueueJob(new ResumeParserService.PollJob('b1', 'cv.pdf', B64, 0));
        Test.stopTest();

        System.assertEquals(0, [SELECT COUNT() FROM Resume_Parser_Log__c]);
    }

    @IsTest
    static void pollGivingUpIsLogged() {
        String running = JSON.serialize(new Map<String, Object>{
            'batch_id' => 'b1', 'status' => 'processing', 'mode' => 'client',
            'total' => 1, 'remaining' => 1, 'running' => 0,
            'results' => new List<Object>()
        });
        Test.setMock(HttpCalloutMock.class, new Mock(200, running));

        Test.startTest();
        // tick 999 is past MAX_TICKS, so this run must give up and log.
        System.enqueueJob(new ResumeParserService.PollJob('b1', 'cv.pdf', B64, 999));
        Test.stopTest();

        System.assertEquals(1, [
            SELECT COUNT() FROM Resume_Parser_Log__c
            WHERE Resume_Name__c = 'cv.pdf' AND Status__c = 'Failed'
        ]);
    }

    // ── Country code ─────────────────────────────────────────────────────────

    @IsTest
    static void diallingPrefixWins() {
        System.assertEquals('United Arab Emirates (+971)',
            ResumeParserService.detectCountryCode('+971 50 123 4567'));
        System.assertEquals('India (+91)',
            ResumeParserService.detectCountryCode('+91-98765-43210'));
        System.assertEquals('USA (+1)',
            ResumeParserService.detectCountryCode('+1 (415) 555 0100'));
        System.assertEquals('Saudi Arabia (+966)',
            ResumeParserService.detectCountryCode('+966501234567'));
    }

    @IsTest
    static void bareNumbersFallBackToShape() {
        // 10 digits starting 9 → India
        System.assertEquals('India (+91)',
            ResumeParserService.detectCountryCode('9876543210'));
        // 8 digits → Singapore
        System.assertEquals('Singapore (+65)',
            ResumeParserService.detectCountryCode('81234567'));
        // 9 digits → UAE. This branch was unreachable before the reorder.
        System.assertEquals('United Arab Emirates (+971)',
            ResumeParserService.detectCountryCode('501234567'));
    }

    @IsTest
    static void shortNumberStartingWithNineIsNotIndia() {
        // The old condition read (len==10 && '6') || '7' || '8' || '9', so any
        // number starting with 9 was tagged India regardless of length.
        System.assertNotEquals('India (+91)',
            ResumeParserService.detectCountryCode('9123'));
    }

    @IsTest
    static void blankPhoneYieldsNothing() {
        System.assertEquals(null, ResumeParserService.detectCountryCode(null));
        System.assertEquals(null, ResumeParserService.detectCountryCode('   '));
    }

    // ── getLatestLogs ────────────────────────────────────────────────────────

    @IsTest
    static void getLatestLogsReturnsTheNewestPerResume() {
        insert new Resume_Parser_Log__c(
            Resume_Name__c = 'cv.pdf', Status__c = 'Failed',
            Uploaded_By__c = UserInfo.getUserId(), Parsed_Date__c = System.now()
        );
        insert new Resume_Parser_Log__c(
            Resume_Name__c = 'cv.pdf', Status__c = 'Success',
            Uploaded_By__c = UserInfo.getUserId(), Parsed_Date__c = System.now()
        );

        Test.startTest();
        List<Resume_Parser_Log__c> logs =
            ResumeParserService.getLatestLogs(new List<String>{ 'cv.pdf' });
        Test.stopTest();

        System.assertEquals(1, logs.size(), 'One row per resume name');
    }

    @IsTest
    static void getLatestLogsHandlesEmptyInput() {
        System.assertEquals(0, ResumeParserService.getLatestLogs(null).size());
        System.assertEquals(0,
            ResumeParserService.getLatestLogs(new List<String>()).size());
    }
}
