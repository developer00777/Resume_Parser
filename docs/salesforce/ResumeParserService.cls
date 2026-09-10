/**
 * ResumeParserService — single-file Apex client for the Resume Parser queue.
 *
 * Everything lives in this one class (the two Queueables are inner classes), so
 * it can be pasted into a single Apex Class in Setup. The old
 * ResumeParserQueueable class is no longer referenced and can be deleted.
 *
 * ─── What changed and why ────────────────────────────────────────────────────
 *
 * 1. WRONG ENDPOINT (this is why MaritalStatus and Gender were always blank).
 *    The old code called /api/v1/parse/salesforce but read client-mode keys.
 *    The two modes emit different field names:
 *
 *        read by old code   salesforce mode   client mode        result
 *        ----------------   ---------------   ----------------   ------
 *        MaritalStatus      (absent)          MaritalStatus__c   always null
 *        Nationality__c     Nationnality      Nationality__c     always null
 *        Spoken_Language__c (absent)          Spoken_Language__c always null
 *        CandidateType      (absent)          Type_1__c          always null
 *        Gender             Gender            Gender__c          worked
 *
 *    Client mode is a 1:1 match for these Contact fields — it was built for
 *    this org — so this class now calls the client endpoint and uses its keys.
 *
 * 2. NO MORE 120-SECOND CALLOUT. The old flow held one callout open for the
 *    whole parse; Salesforce caps a callout at 120s and the parser gave up at
 *    110s, which is where the 504s came from. Now: submit (~15 ms, returns a
 *    batch_id) then poll from a chained Queueable. Each tick is its own
 *    transaction with a fresh 120s cumulative callout budget, so bulk uploads
 *    cannot time out however many resumes are in flight.
 *
 * 3. NO MORE HAND-PADDED MULTIPART. safeBase64Concat and the boundary maths are
 *    gone — the parser now accepts plain JSON, so the body is JSON.serialize().
 *
 * 4. GOVERNOR-LIMIT FIXES. The RecordType SOQL and the DML were inside the
 *    results loop (a query and 3 DML statements per resume); they are now
 *    hoisted and bulkified. The post-insert debug SOQL is removed.
 *
 * 5. COUNTRY-CODE PRECEDENCE BUG. `length == 10 && startsWith('6') ||
 *    startsWith('7') || ...` parses as `(length==10 && '6') || '7' || '8' ||
 *    '9'` because && binds tighter than ||, so ANY number starting 7/8/9 was
 *    tagged India regardless of length. Now parenthesised.
 *
 * ─── Setup ───────────────────────────────────────────────────────────────────
 * Set BASE_URL and API_KEY below. Everything else works as before: the LWC
 * still calls startParsing(fileName, base64Data) and polls getLatestLogs().
 */
public with sharing class ResumeParserService {

    // ═══ CONFIG ══════════════════════════════════════════════════════════════
    // Move these to a Named Credential / Custom Metadata when you get a chance;
    // an API key in source is visible to anyone with read access to the class.
    private static final String  BASE_URL    = 'https://resumeparser-production-45b1.up.railway.app';
    private static final String  API_KEY     = 'changeme';

    // Client mode: field names match this org's Contact fields exactly.
    private static final String  SUBMIT_PATH = '/api/v1/parse/client/jobs/base64';
    private static final String  POLL_PATH   = '/api/v1/parse/jobs/';

    // 30s is generous now that no callout waits for a parse.
    private static final Integer TIMEOUT_MS  = 30000;

    // Poll budget. Ticks 0-1 fire immediately, the rest at one-minute intervals,
    // so a fast resume lands in seconds and a slow one still gets ~30 minutes.
    private static final Integer MAX_TICKS       = 30;
    private static final Integer FAST_TICKS      = 2;

    private static final String  RECORD_TYPE_NAME = 'Talent';

    // The old code logged "Skipping record due to missing email" but had no
    // `continue`, so it inserted anyway. Left as-is to preserve behaviour —
    // set this true if you actually want those records skipped.
    private static final Boolean SKIP_WHEN_NO_EMAIL = false;

    public class ResumeParserException extends Exception {}


    // ═══ LWC ENTRY POINTS (signatures unchanged) ═════════════════════════════

    @AuraEnabled
    public static String startParsing(String fileName, String base64Data) {
        if (String.isBlank(fileName) || String.isBlank(base64Data)) {
            throw new AuraHandledException('File name and file data are both required.');
        }
        System.enqueueJob(new SubmitJob(fileName, base64Data));
        return 'Processing Started';
    }

    @AuraEnabled(cacheable=false)
    public static List<Resume_Parser_Log__c> getLatestLogs(List<String> resumeNames) {
        if (resumeNames == null || resumeNames.isEmpty()) {
            return new List<Resume_Parser_Log__c>();
        }

        Map<String, Resume_Parser_Log__c> latestLogs =
            new Map<String, Resume_Parser_Log__c>();

        for (Resume_Parser_Log__c log : [
            SELECT Id, Resume_Name__c, Status__c, Error_Message__c, CreatedDate
            FROM Resume_Parser_Log__c
            WHERE Uploaded_By__c = :UserInfo.getUserId()
              AND Resume_Name__c IN :resumeNames
            ORDER BY CreatedDate DESC
        ]) {
            if (!latestLogs.containsKey(log.Resume_Name__c)) {
                latestLogs.put(log.Resume_Name__c, log);
            }
        }
        return latestLogs.values();
    }


    // ═══ STEP 1 — SUBMIT (fast; cannot time out) ═════════════════════════════

    public class SubmitJob implements Queueable, Database.AllowsCallouts {
        private final String fileName;
        private final String base64Data;

        public SubmitJob(String fileName, String base64Data) {
            this.fileName = fileName;
            this.base64Data = base64Data;
        }

        public void execute(QueueableContext context) {
            try {
                String batchId = submitBatch(fileName, base64Data);
                System.debug('Resume parser: queued ' + fileName + ' as batch ' + batchId);

                // Poll from a fresh transaction — this one's callout budget is done.
                if (!Test.isRunningTest()) {
                    System.enqueueJob(new PollJob(batchId, fileName, base64Data, 0), 0);
                }
            } catch (Exception e) {
                System.debug(LoggingLevel.ERROR,
                    'Resume parser submit failed for ' + fileName + ': ' + e.getMessage());
                logFailure(fileName, e.getMessage());
            }
        }
    }

    /** POST the file as JSON and return the batch_id. */
    private static String submitBatch(String fileName, String base64Data) {
        Map<String, Object> file = new Map<String, Object>{
            'filename'       => fileName,
            'content_base64' => base64Data,
            'external_id'    => fileName      // echoed back on the result row
        };
        String body = JSON.serialize(new Map<String, Object>{
            'files' => new List<Object>{ file }
        });

        HttpResponse res = send('POST', SUBMIT_PATH, body);

        if (res.getStatusCode() != 202) {
            throw new ResumeParserException(
                'Submit returned HTTP ' + res.getStatusCode() + ': ' + res.getBody()
            );
        }
        Map<String, Object> parsed =
            (Map<String, Object>) JSON.deserializeUntyped(res.getBody());
        String batchId = (String) parsed.get('batch_id');

        if (String.isBlank(batchId)) {
            throw new ResumeParserException('Submit succeeded but returned no batch_id.');
        }
        return batchId;
    }


    // ═══ STEP 2 — POLL (chains itself until the batch completes) ═════════════

    public class PollJob implements Queueable, Database.AllowsCallouts {
        private final String  batchId;
        private final String  fileName;
        private final String  base64Data;
        private final Integer tick;

        public PollJob(String batchId, String fileName, String base64Data, Integer tick) {
            this.batchId = batchId;
            this.fileName = fileName;
            this.base64Data = base64Data;
            this.tick = tick;
        }

        public void execute(QueueableContext context) {
            try {
                HttpResponse res = send('GET', POLL_PATH + batchId, null);

                if (res.getStatusCode() == 404) {
                    logFailure(fileName, 'Batch ' + batchId + ' expired before it was read.');
                    return;
                }
                if (res.getStatusCode() != 200) {
                    throw new ResumeParserException(
                        'Poll returned HTTP ' + res.getStatusCode() + ': ' + res.getBody()
                    );
                }

                Map<String, Object> batch =
                    (Map<String, Object>) JSON.deserializeUntyped(res.getBody());

                if ('completed'.equals(String.valueOf(batch.get('status')))) {
                    List<Object> results = (List<Object>) batch.get('results');
                    if (results == null || results.isEmpty()) {
                        logFailure(fileName, 'Batch completed but returned no results.');
                        return;
                    }
                    applyResults(results, fileName, base64Data);
                    return;
                }

                if (tick >= MAX_TICKS) {
                    logFailure(fileName,
                        'Still not finished after ' + MAX_TICKS + ' polls; giving up.');
                    return;
                }

                // Ticks 0-1 immediately, then once a minute.
                Integer delay = tick < FAST_TICKS ? 0 : 1;
                if (!Test.isRunningTest()) {
                    System.enqueueJob(
                        new PollJob(batchId, fileName, base64Data, tick + 1), delay
                    );
                }
            } catch (Exception e) {
                System.debug(LoggingLevel.ERROR,
                    'Resume parser poll failed for ' + fileName + ': ' + e.getMessage());
                logFailure(fileName, e.getMessage());
            }
        }
    }


    // ═══ STEP 3 — APPLY RESULTS ══════════════════════════════════════════════

    /**
     * Turn result rows into Contacts, logs and attachments.
     *
     * Bulkified: one RecordType query, one insert for the Contacts, one for the
     * logs, one for the attachments — regardless of how many rows arrive. The
     * previous version ran a SOQL query and three DML statements per resume.
     */
    public static void applyResults(
        List<Object> results, String fileName, String base64Data
    ) {
        Id recordTypeId = talentRecordTypeId();

        List<Contact>              contacts    = new List<Contact>();
        List<String>               sourceNames = new List<String>();
        List<Resume_Parser_Log__c> failedLogs  = new List<Resume_Parser_Log__c>();

        for (Object raw : results) {
            Map<String, Object> row = (Map<String, Object>) raw;

            // external_id is echoed back by the parser; fall back to filename.
            String rowName = row.get('external_id') != null
                ? String.valueOf(row.get('external_id'))
                : (row.get('filename') != null
                    ? String.valueOf(row.get('filename'))
                    : fileName);

            if (String.valueOf(row.get('success')) != 'true') {
                String message = row.get('error') != null
                    ? String.valueOf(row.get('error'))
                    : 'Resume Parser API returned failure.';
                failedLogs.add(buildLog(rowName, 'Failed', message, null));
                continue;
            }

            Object rawData = row.get('data');
            if (rawData == null) {
                failedLogs.add(buildLog(rowName, 'Failed', 'Parser returned no data.', null));
                continue;
            }

            try {
                Contact con = buildContact((Map<String, Object>) rawData, recordTypeId);

                if (con == null) {
                    failedLogs.add(buildLog(
                        rowName, 'Failed', 'Skipped: no email address found.', null));
                    continue;
                }
                contacts.add(con);
                sourceNames.add(rowName);
            } catch (Exception e) {
                failedLogs.add(buildLog(rowName, 'Failed', e.getMessage(), null));
            }
        }

        List<Resume_Parser_Log__c> logs = new List<Resume_Parser_Log__c>(failedLogs);

        if (!contacts.isEmpty()) {
            // allOrNone = false: one bad Contact must not roll back the rest.
            List<Database.SaveResult> saved = Database.insert(contacts, false);

            List<Attachment> attachments = new List<Attachment>();

            for (Integer i = 0; i < saved.size(); i++) {
                String rowName = sourceNames[i];

                if (saved[i].isSuccess()) {
                    Id contactId = saved[i].getId();
                    logs.add(buildLog(rowName, 'Success', 'Parsed Successfully', contactId));

                    // We only hold the bytes for the file this job submitted.
                    if (rowName == fileName && String.isNotBlank(base64Data)) {
                        Attachment att = new Attachment();
                        att.ParentId = contactId;
                        att.Name     = fileName;
                        att.Body     = EncodingUtil.base64Decode(base64Data);
                        attachments.add(att);
                    }
                } else {
                    logs.add(buildLog(
                        rowName, 'Failed', saved[i].getErrors()[0].getMessage(), null));
                }
            }

            if (!attachments.isEmpty()) {
                Database.insert(attachments, false);
            }
        }

        if (!logs.isEmpty()) {
            Database.insert(logs, false);
        }
    }


    // ═══ FIELD MAPPING (client-mode keys) ════════════════════════════════════

    /** Build one Contact. Returns null when the row should be skipped. */
    private static Contact buildContact(Map<String, Object> d, Id recordTypeId) {
        Contact con = new Contact();
        con.RecordTypeId = recordTypeId;

        // ===== NAME =====
        String fullName = str(d, 'Name');
        if (String.isBlank(fullName)) {
            fullName = str(d, 'FullName__c');
        }

        if (String.isNotBlank(fullName)) {
            List<String> parts = fullName.trim().split('\\s+');
            if (parts.size() > 1) {
                con.FirstName = parts[0];
                List<String> rest = new List<String>();
                for (Integer i = 1; i < parts.size(); i++) {
                    rest.add(parts[i]);
                }
                con.LastName = String.join(rest, ' ');
            } else {
                con.LastName = fullName;
            }
        } else {
            con.LastName = 'Unknown';
        }

        // ===== EMAIL =====
        con.Email = str(d, 'Email');
        if (SKIP_WHEN_NO_EMAIL && String.isBlank(con.Email)) {
            return null;
        }

        // ===== IDENTITY =====
        con.Nationality__c   = str(d, 'Nationality__c');
        con.MaritalStatus__c = str(d, 'MaritalStatus__c');
        con.Gender__c        = str(d, 'Gender__c');

        // ===== DOB (client mode sends YYYY-MM-DD) =====
        con.Date_of_Birth__c = toDate(str(d, 'Date_of_Birth__c'));

        // ===== EXPERIENCE =====
        con.Years_of_Experience__c = toDecimal(d.get('Years_of_Experience__c'));

        // ===== LOCATION / ROLE / COMPANY =====
        con.Current_Location__c   = str(d, 'Current_Location__c');
        con.CurrentDesignation__c = str(d, 'CurrentDesignation__c');
        con.CurrentCompany__c     = str(d, 'CurrentCompany__c');

        // ===== PHONE =====
        con.PhoneNumber__c            = str(d, 'PhoneNumber__c');
        con.SCSCHAMPS__PhoneNumber__c = str(d, 'SCSCHAMPS__PhoneNumber__c');

        String phoneCode = detectCountryCode(con.PhoneNumber__c);
        if (String.isNotBlank(phoneCode)) {
            con.Mobile_Phone_Country_Code__c            = phoneCode;
            con.Other_Phone_Country_Code__c             = phoneCode;
            con.Emergency_Contact_Phone_Country_Code__c = phoneCode;
        }

        // ===== EDUCATION =====
        con.Graduation_Year2__c    = str(d, 'Graduation_Year2__c');
        con.Institution_College__c = str(d, 'Institution_College__c');
        con.Name__c                = str(d, 'Name__c');          // professional degree
        con.Year__c                = toDate(str(d, 'Year__c'));  // degree completion date

        // ===== TYPE (Graduate / Non Graduate) =====
        con.Type_1__c = str(d, 'Type_1__c');

        // ===== LANGUAGES (multi-select picklist) =====
        Object langs = d.get('Spoken_Language__c');
        if (langs instanceof List<Object>) {
            List<String> langList = new List<String>();
            for (Object o : (List<Object>) langs) {
                if (o != null && String.isNotBlank(String.valueOf(o))) {
                    langList.add(String.valueOf(o).trim());
                }
            }
            if (!langList.isEmpty()) {
                con.Spoken_Language__c = String.join(langList, ';');
            }
        }

        return con;
    }


    // ═══ COUNTRY CODE ════════════════════════════════════════════════════════

    private static Map<String, String> COUNTRY_MAP = new Map<String, String>{
        '+971' => 'United Arab Emirates (+971)',
        '+966' => 'Saudi Arabia (+966)',
        '+974' => 'Qatar (+974)',
        '+968' => 'Oman (+968)',
        '+965' => 'Kuwait (+965)',
        '+880' => 'Bangladesh (+880)',
        '+91'  => 'India (+91)',
        '+92'  => 'Pakistan (+92)',
        '+94'  => 'Sri Lanka (+94)',
        '+65'  => 'Singapore (+65)',
        '+60'  => 'Malaysia (+60)',
        '+61'  => 'Australia (+61)',
        '+44'  => 'UK (+44)',
        '+86'  => 'China (+86)',
        '+1'   => 'USA (+1)',
        '+213' => 'Algeria (+213)',
        '+376' => 'Andorra (+376)',
        '+244' => 'Angola (+244)',
        '+54'  => 'Argentina (+54)',
        '+43'  => 'Austria (+43)',
        '+973' => 'Bahrain (+973)',
        '+32'  => 'Belgium (+32)',
        '+55'  => 'Brazil (+55)',
        '+359' => 'Bulgaria (+359)',
        '+855' => 'Cambodia (+855)',
        '+237' => 'Cameroon (+237)',
        '+56'  => 'Chile (+56)',
        '+57'  => 'Colombia (+57)',
        '+385' => 'Croatia (+385)',
        '+45'  => 'Denmark (+45)',
        '+20'  => 'Egypt (+20)',
        '+251' => 'Ethiopia (+251)',
        '+358' => 'Finland (+358)',
        '+33'  => 'France (+33)',
        '+49'  => 'Germany (+49)',
        '+30'  => 'Greece (+30)',
        '+852' => 'Hong Kong (+852)',
        '+36'  => 'Hungary (+36)',
        '+354' => 'Iceland (+354)',
        '+62'  => 'Indonesia (+62)',
        '+98'  => 'Iran (+98)',
        '+964' => 'Iraq (+964)',
        '+353' => 'Ireland (+353)',
        '+972' => 'Israel (+972)',
        '+39'  => 'Italy (+39)',
        '+81'  => 'Japan (+81)',
        '+962' => 'Jordan (+962)',
        '+254' => 'Kenya (+254)',
        '+82'  => 'Korea South (+82)',
        '+856' => 'Laos (+856)',
        '+961' => 'Lebanon (+961)',
        '+218' => 'Libya (+218)',
        '+370' => 'Lithuania (+370)',
        '+352' => 'Luxembourg (+352)',
        '+960' => 'Maldives (+960)',
        '+52'  => 'Mexico (+52)',
        '+976' => 'Mongolia (+976)',
        '+212' => 'Morocco (+212)',
        '+258' => 'Mozambique (+258)',
        '+95'  => 'Myanmar (+95)',
        '+977' => 'Nepal (+977)',
        '+31'  => 'Netherlands (+31)',
        '+64'  => 'New Zealand (+64)',
        '+234' => 'Nigeria (+234)',
        '+47'  => 'Norway (+47)',
        '+507' => 'Panama (+507)',
        '+595' => 'Paraguay (+595)',
        '+51'  => 'Peru (+51)',
        '+63'  => 'Philippines (+63)',
        '+48'  => 'Poland (+48)',
        '+351' => 'Portugal (+351)',
        '+40'  => 'Romania (+40)',
        '+250' => 'Rwanda (+250)',
        '+221' => 'Senegal (+221)',
        '+381' => 'Serbia (+381)',
        '+248' => 'Seychelles (+248)',
        '+232' => 'Sierra Leone (+232)',
        '+421' => 'Slovak Republic (+421)',
        '+386' => 'Slovenia (+386)',
        '+252' => 'Somalia (+252)',
        '+27'  => 'South Africa (+27)',
        '+34'  => 'Spain (+34)',
        '+46'  => 'Sweden (+46)',
        '+41'  => 'Switzerland (+41)',
        '+963' => 'Syria (+963)',
        '+886' => 'Taiwan (+886)',
        '+66'  => 'Thailand (+66)',
        '+90'  => 'Turkey (+90)',
        '+380' => 'Ukraine (+380)',
        '+598' => 'Uruguay (+598)',
        '+84'  => 'Vietnam (+84)',
        '+260' => 'Zambia (+260)',
        '+263' => 'Zimbabwe (+263)'
    };

    /** Longest dialling prefix wins; falls back to a length/prefix heuristic. */
    @TestVisible
    private static String detectCountryCode(String rawPhone) {
        if (String.isBlank(rawPhone)) {
            return null;
        }

        String phone = rawPhone.replace(' ', '')
                               .replace('-', '')
                               .replace('(', '')
                               .replace(')', '');

        // Match the longest prefix first, so +971 beats +9x.
        for (Integer len = 4; len >= 2; len--) {
            if (phone.length() < len) {
                continue;
            }
            String candidate = phone.substring(0, len);
            if (COUNTRY_MAP.containsKey(candidate)) {
                return COUNTRY_MAP.get(candidate);
            }
        }

        // No dialling code — guess from the national number's shape.
        String digits = phone.replace('+', '');

        // NOTE: && binds tighter than ||, so the original condition tagged any
        // number starting 7/8/9 as India whatever its length. Parenthesised.
        if (digits.length() == 10 &&
            (digits.startsWith('6') || digits.startsWith('7') ||
             digits.startsWith('8') || digits.startsWith('9'))) {
            return 'India (+91)';
        }
        if (digits.length() == 11 && digits.startsWith('03')) {
            return 'Pakistan (+92)';
        }
        if (digits.length() == 8) {
            return 'Singapore (+65)';
        }
        // 9 digits is checked before the 9-10 range; in the original the
        // Australia branch came first and made this one unreachable.
        if (digits.length() == 9) {
            return 'United Arab Emirates (+971)';
        }
        if (digits.length() == 10) {
            return 'Australia (+61)';
        }
        return null;
    }


    // ═══ HELPERS ═════════════════════════════════════════════════════════════

    private static Id talentRecordTypeId() {
        // Describe cache — no SOQL, and no per-row query as in the old loop.
        Schema.RecordTypeInfo info = Schema.SObjectType.Contact
            .getRecordTypeInfosByName()
            .get(RECORD_TYPE_NAME);

        if (info == null) {
            throw new ResumeParserException(
                'Contact record type "' + RECORD_TYPE_NAME + '" not found.');
        }
        return info.getRecordTypeId();
    }

    private static HttpResponse send(String method, String path, String body) {
        HttpRequest req = new HttpRequest();
        req.setEndpoint(BASE_URL + path);
        req.setMethod(method);
        req.setHeader('X-API-Key', API_KEY);
        req.setTimeout(TIMEOUT_MS);
        if (body != null) {
            req.setHeader('Content-Type', 'application/json');
            req.setBody(body);
        }
        return new Http().send(req);
    }

    private static Resume_Parser_Log__c buildLog(
        String resumeName, String status, String message, Id contactId
    ) {
        Resume_Parser_Log__c log = new Resume_Parser_Log__c();
        log.Resume_Name__c  = resumeName;
        log.Status__c       = status;
        log.Error_Message__c = message == null
            ? null
            : (message.length() > 255 ? message.substring(0, 255) : message);
        log.Parsed_Date__c  = System.now();
        log.Uploaded_By__c  = UserInfo.getUserId();
        if (contactId != null) {
            log.Contact__c = contactId;
        }
        return log;
    }

    private static void logFailure(String resumeName, String message) {
        try {
            insert buildLog(resumeName, 'Failed', message, null);
        } catch (Exception e) {
            System.debug(LoggingLevel.ERROR, 'Could not write failure log: ' + e.getMessage());
        }
    }

    private static String str(Map<String, Object> d, String key) {
        Object v = d.get(key);
        if (v == null) {
            return null;
        }
        String s = String.valueOf(v).trim();
        return String.isBlank(s) ? null : s;
    }

    private static Decimal toDecimal(Object v) {
        if (v == null) {
            return null;
        }
        try {
            return Decimal.valueOf(String.valueOf(v));
        } catch (Exception e) {
            System.debug('Not a number: ' + v);
            return null;
        }
    }

    private static Date toDate(String v) {
        if (String.isBlank(v)) {
            return null;
        }
        try {
            return Date.valueOf(v);          // expects YYYY-MM-DD
        } catch (Exception e) {
            System.debug('Not a date: ' + v);
            return null;
        }
    }
}
