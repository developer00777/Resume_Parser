/**
 * Optional: let the parser push results to Salesforce instead of polling.
 *
 * Pass this resource's URL as `callback_url` when submitting and the parser
 * POSTs the finished batch here exactly once (delivery is claimed atomically
 * on its side, so a retried worker cannot double-deliver). Polling remains
 * available as a fallback — the callback is an optimisation, never the only
 * copy of the results.
 *
 *   callback_url:   https://<your-domain>/services/apexrest/resumeCallback
 *   callback_token: any shared secret; arrives as "Authorization: Bearer <token>"
 *
 * The endpoint must be reachable from the parser, so the Site / Connected App
 * guest access has to be configured — otherwise use the polling path instead.
 */
@RestResource(urlMapping='/resumeCallback/*')
global with sharing class ResumeParserCallback {

    // Store the real secret in Custom Metadata or a protected Custom Setting.
    private static final String EXPECTED_TOKEN = 'replace-me';

    @HttpPost
    global static void receive() {
        RestResponse res = RestContext.response;

        String auth = RestContext.request.headers.get('Authorization');
        if (auth != 'Bearer ' + EXPECTED_TOKEN) {
            res.statusCode = 401;
            res.responseBody = Blob.valueOf('{"error":"unauthorized"}');
            return;
        }

        try {
            Map<String, Object> batch = (Map<String, Object>)
                JSON.deserializeUntyped(RestContext.request.requestBody.toString());

            List<Object> results = (List<Object>) batch.get('results');
            if (results != null && !results.isEmpty()) {
                ResumeParserService.applyResults(results, null, null);
            }

            res.statusCode = 200;
            res.responseBody = Blob.valueOf('{"status":"ok"}');
        } catch (Exception e) {
            // Answer 500 so the failure is visible in the parser's logs; the
            // batch stays readable by polling.
            System.debug(LoggingLevel.ERROR, 'Resume callback failed: ' + e.getMessage());
            res.statusCode = 500;
            res.responseBody = Blob.valueOf('{"error":"apply failed"}');
        }
    }
}
