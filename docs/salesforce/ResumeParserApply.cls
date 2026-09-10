/**
 * Writes parser results back onto Candidate records.
 *
 * This is the one class you should expect to edit — the field mapping below is
 * a starting point using the names the parser's `salesforce` mode emits. The
 * parser's payload keys are listed in docs/API.md under SalesforceResumeData.
 *
 * Correlation comes from `external_id`, which the submitter set to the
 * Candidate Id and the parser echoed back untouched. Do not match on filename:
 * duplicates collide, and Salesforce may rename the file on download.
 */
public with sharing class ResumeParserApply {

    /** Apply a list of result rows (the `results` array from a batch). */
    public static void applyResults(List<Object> results) {
        Map<Id, SCSCHAMPS__Candidate__c> updates = new Map<Id, SCSCHAMPS__Candidate__c>();
        Integer skipped = 0;

        for (Object raw : results) {
            Map<String, Object> row = (Map<String, Object>) raw;

            if (row.get('success') != true) {
                // Failed rows carry a human-readable `error`. Park it on the
                // record rather than losing it.
                Id failedId = asId((String) row.get('external_id'));
                if (failedId != null) {
                    updates.put(failedId, new SCSCHAMPS__Candidate__c(
                        Id = failedId,
                        Resume_Parse_Error__c = truncate((String) row.get('error'), 255),
                        Resume_Parsed__c = false
                    ));
                }
                skipped++;
                continue;
            }

            Id candidateId = asId((String) row.get('external_id'));
            if (candidateId == null) {
                skipped++;
                continue;
            }

            Map<String, Object> d = (Map<String, Object>) row.get('data');
            if (d == null) {
                skipped++;
                continue;
            }

            updates.put(candidateId, new SCSCHAMPS__Candidate__c(
                Id                                   = candidateId,
                SCSCHAMPS__First_Name__c             = str(d, 'FirstName'),
                SCSCHAMPS__Last_Name__c              = str(d, 'LastName'),
                SCSCHAMPS__Email__c                  = str(d, 'Email'),
                SCSCHAMPS__PhoneNumber__c            = str(d, 'PhoneNumber'),
                SCSCHAMPS__Mobile__c                 = str(d, 'MobilePhone'),
                SCSCHAMPS__Current_Location__c       = str(d, 'Current_Location'),
                SCSCHAMPS__Country__c                = str(d, 'Country'),
                SCSCHAMPS__Current_Designation__c    = str(d, 'CurrentDesignation'),
                SCSCHAMPS__Current_Employer__c       = str(d, 'CurrentCompany'),
                SCSCHAMPS__Total_Experience__c       = dec(d, 'Years_of_Experience'),
                SCSCHAMPS__LinkedIn__c               = str(d, 'LinkedIn_URL'),
                SCSCHAMPS__Nationality__c            = str(d, 'Nationnality'),
                Resume_Parsed__c                     = true,
                Resume_Parse_Error__c                = null,
                Resume_Parsed_On__c                  = System.now()
            ));
        }

        if (updates.isEmpty()) {
            System.debug('Resume parser: nothing to apply (' + skipped + ' skipped).');
            return;
        }

        // allOrNone = false so one bad record cannot roll back the whole batch —
        // the same principle the parser applies on its side.
        List<Database.SaveResult> saveResults =
            Database.update(updates.values(), false);

        Integer failures = 0;
        for (Integer i = 0; i < saveResults.size(); i++) {
            if (!saveResults[i].isSuccess()) {
                failures++;
                System.debug(LoggingLevel.ERROR,
                    'Failed to update ' + updates.values()[i].Id + ': ' +
                    saveResults[i].getErrors()[0].getMessage());
            }
        }
        System.debug(
            'Resume parser: applied ' + (updates.size() - failures) + ' record(s), ' +
            failures + ' failed, ' + skipped + ' skipped.'
        );
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private static String str(Map<String, Object> d, String key) {
        Object v = d.get(key);
        return v == null ? null : String.valueOf(v);
    }

    private static Decimal dec(Map<String, Object> d, String key) {
        Object v = d.get(key);
        if (v == null) {
            return null;
        }
        try {
            return Decimal.valueOf(String.valueOf(v));
        } catch (Exception e) {
            return null;
        }
    }

    private static Id asId(String value) {
        if (String.isBlank(value)) {
            return null;
        }
        try {
            return (Id) value;
        } catch (Exception e) {
            return null;
        }
    }

    private static String truncate(String value, Integer max) {
        if (value == null) {
            return null;
        }
        return value.length() > max ? value.substring(0, max) : value;
    }
}
