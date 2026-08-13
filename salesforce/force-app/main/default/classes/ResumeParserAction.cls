/**
 * Entry points for the resume parser.
 *
 * ResumeParserQueueable does the work; this class is how anything other than
 * Apex reaches it:
 *
 *   @InvocableMethod  — Flow, Process Builder, the REST Actions API
 *   @AuraEnabled      — the resumeParser LWC behind the "Parse Resume" button
 *
 * Both enqueue and return immediately. They deliberately do NOT wait for the
 * parse: it takes tens of seconds, a Flow would block its transaction, and a
 * user staring at a spinner for a minute is a worse experience than a toast
 * plus a status field that fills in.
 */
public with sharing class ResumeParserAction {

    // ── Flow / Process Builder ──────────────────────────────────────────────

    /**
     * Flow hands this a LIST — one entry per record in the batch — so it must
     * be bulk-safe. All the IDs go into a single Queueable that chains through
     * them one per transaction, rather than one job each, which would breach
     * the 50-enqueue-per-transaction limit on any real batch.
     */
    @InvocableMethod(
        label='Parse Resume'
        description='Parses the resume attached to a Candidate and writes the extracted fields back onto the record.'
        category='Recruiting'
    )
    public static List<Result> parseResumes(List<Request> requests) {
        List<Result> results = new List<Result>();
        List<Id> toQueue = new List<Id>();
        Set<Id> seen = new Set<Id>();

        for (Request req : requests) {
            Result result = new Result();

            if (req == null || String.isBlank(req.recordId)) {
                result.queued = false;
                result.message = 'No record Id supplied.';
                results.add(result);
                continue;
            }

            Id recordId;
            try {
                recordId = Id.valueOf(req.recordId);
            } catch (Exception e) {
                result.queued = false;
                result.message = 'Not a valid record Id: ' + req.recordId;
                results.add(result);
                continue;
            }

            // A Flow that fires on every update can easily hand us the same
            // record twice in one transaction; parsing it twice would cost two
            // LLM runs for one outcome.
            if (!seen.add(recordId)) {
                result.queued = false;
                result.recordId = recordId;
                result.message = 'Already queued in this transaction.';
                results.add(result);
                continue;
            }

            toQueue.add(recordId);
            result.queued = true;
            result.recordId = recordId;
            result.message = 'Queued for parsing.';
            results.add(result);
        }

        if (!toQueue.isEmpty()) {
            System.enqueueJob(new ResumeParserQueueable(toQueue));
        }
        return results;
    }

    public class Request {
        @InvocableVariable(
            label='Candidate Record Id'
            description='The Candidate record whose attached resume should be parsed.'
            required=true
        )
        public String recordId;
    }

    public class Result {
        @InvocableVariable(label='Queued' description='True if the record was accepted for parsing.')
        public Boolean queued = false;

        @InvocableVariable(label='Record Id')
        public Id recordId;

        @InvocableVariable(label='Message' description='Why the record was or was not queued.')
        public String message;
    }

    // ── Lightning web component ─────────────────────────────────────────────

    /**
     * Called by the resumeParser LWC when a recruiter clicks "Parse Resume".
     *
     * Returns the message the component shows in its toast. Throws an
     * AuraHandledException on refusal so the component gets a readable reason
     * rather than a raw stack trace.
     *
     * Not cacheable: it mutates state.
     */
    @AuraEnabled
    public static String parseResume(Id recordId) {
        if (recordId == null) {
            throw new AuraHandledException('No record was supplied.');
        }

        SObjectType recordType = recordId.getSObjectType();
        DescribeSObjectResult describe = recordType.getDescribe();

        // The parser writes to this record on the user's behalf, so refuse when
        // the user could not have made the edit themselves. Without this check
        // the button would be a privilege-escalation path: the parser runs as
        // its own integration user, not as the clicker.
        if (!describe.isUpdateable()) {
            throw new AuraHandledException(
                'You do not have permission to edit ' + describe.getLabel() + ' records.'
            );
        }

        if (!hasResume(recordId, recordType)) {
            throw new AuraHandledException(
                'This candidate has no resume attached. Add one, then parse.'
            );
        }

        try {
            System.enqueueJob(new ResumeParserQueueable(recordId));
        } catch (Exception e) {
            // Almost always the 50-jobs-per-transaction limit.
            throw new AuraHandledException('Could not queue the parse: ' + e.getMessage());
        }

        return 'Parsing started — this record will update shortly.';
    }

    /**
     * Whether the record has something to parse.
     *
     * Checked here rather than in the parser so the recruiter gets an immediate,
     * specific answer instead of a job that quietly fails a minute later. Both
     * fields are resolved through Describe: an org that lacks them should not
     * fail to compile, and in that case we let the parse proceed and report
     * whatever it finds.
     */
    private static Boolean hasResume(Id recordId, SObjectType recordType) {
        Map<String, SObjectField> fields = recordType.getDescribe().fields.getMap();

        List<String> candidates = new List<String>();
        for (String api : new List<String>{
            'SCSCHAMPS__Resume_Attachment_Id__c', 'Resume_Attachment_Id__c',
            'SCSCHAMPS__Resume_URL__c', 'Resume_URL__c'
        }) {
            SObjectField f = fields.get(api);
            if (f != null && f.getDescribe().isAccessible()) {
                candidates.add(api);
            }
        }

        if (candidates.isEmpty()) {
            // Neither field exists in this org — not something the recruiter can
            // act on, so do not block them on it.
            return true;
        }

        String soql =
            'SELECT ' + String.join(candidates, ', ') +
            ' FROM ' + recordType.getDescribe().getName() +
            ' WHERE Id = :recordId LIMIT 1';

        List<SObject> rows = Database.query(String.escapeSingleQuotes(soql));
        if (rows.isEmpty()) {
            return false;
        }

        for (String api : candidates) {
            if (rows[0].get(api) != null) {
                return true;
            }
        }
        return false;
    }
}
