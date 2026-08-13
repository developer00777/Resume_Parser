import { LightningElement, api } from 'lwc';
import { ShowToastEvent } from 'lightning/platformShowToastEvent';
import { CloseActionScreenEvent } from 'lightning/actions';
import parseResume from '@salesforce/apex/ResumeParserAction.parseResume';

/**
 * "Parse Resume" quick action.
 *
 * Queues the parse and closes. It deliberately does not wait for a result: a
 * parse takes tens of seconds, and holding a modal open that long is worse than
 * a toast plus the Resume Parse Status field filling in on the record.
 */
export default class ResumeParser extends LightningElement {
    @api recordId;

    working = false;
    // Guards the double-click. Without it an impatient second click queues a
    // second parse — two LLM runs, billed twice, for one outcome.
    submitted = false;

    get disabled() {
        return this.working || this.submitted;
    }

    async handleParse() {
        if (this.disabled) {
            return;
        }
        this.working = true;

        try {
            const message = await parseResume({ recordId: this.recordId });
            this.submitted = true;
            this.toast('Parse started', message, 'success');
            this.close();
        } catch (error) {
            // AuraHandledException puts the readable reason in body.message.
            const reason =
                error?.body?.message ||
                error?.message ||
                'Something went wrong queueing the parse.';
            this.toast('Could not start parsing', reason, 'error', 'sticky');
        } finally {
            this.working = false;
        }
    }

    handleCancel() {
        this.close();
    }

    toast(title, message, variant, mode = 'dismissable') {
        this.dispatchEvent(new ShowToastEvent({ title, message, variant, mode }));
    }

    close() {
        this.dispatchEvent(new CloseActionScreenEvent());
    }
}
