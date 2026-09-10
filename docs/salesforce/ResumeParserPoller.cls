/**
 * Queueable that polls outstanding batches and applies results as they land.
 *
 * Each execution is a separate transaction, so the 120-second cumulative
 * callout budget resets every tick. Polls are ~100 ms each, so even a hundred
 * outstanding batches poll comfortably inside one tick.
 *
 * A Queueable may enqueue only one child job, so this chains itself once per
 * tick with a one-minute delay rather than fanning out.
 */
public with sharing class ResumeParserPoller implements Queueable, Database.AllowsCallouts {

    // ~40 ticks at one minute each: a 40-minute ceiling before we give up.
    private static final Integer MAX_TICKS = 40;
    private static final Integer DELAY_MINUTES = 1;

    private final List<String> batchIds;
    private final Integer tick;

    public ResumeParserPoller(List<String> batchIds, Integer tick) {
        this.batchIds = batchIds;
        this.tick = tick;
    }

    public void execute(QueueableContext ctx) {
        List<String> stillRunning = new List<String>();
        List<Object> readyResults = new List<Object>();

        for (String batchId : batchIds) {
            try {
                Map<String, Object> batch = ResumeParserService.pollBatch(batchId);
                if (ResumeParserService.isComplete(batch)) {
                    readyResults.addAll((List<Object>) batch.get('results'));
                } else {
                    stillRunning.add(batchId);
                }
            } catch (Exception e) {
                // A batch that expired or 404s is dropped rather than retried
                // forever; the others keep going.
                System.debug(LoggingLevel.ERROR,
                    'Poll failed for batch ' + batchId + ': ' + e.getMessage());
            }
        }

        if (!readyResults.isEmpty()) {
            ResumeParserApply.applyResults(readyResults);
        }

        if (stillRunning.isEmpty()) {
            System.debug('Resume parser: all batches complete.');
            return;
        }
        if (tick >= MAX_TICKS) {
            System.debug(LoggingLevel.ERROR,
                'Resume parser: giving up on ' + stillRunning.size() +
                ' batch(es) after ' + MAX_TICKS + ' ticks: ' + stillRunning);
            return;
        }

        // Chain the next tick. Callouts and DML from this transaction are done.
        System.enqueueJob(new ResumeParserPoller(stillRunning, tick + 1), DELAY_MINUTES);
    }
}
