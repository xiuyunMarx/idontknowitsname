# Big fan-out triage agent with visit by, long-prefix variant: every
# sem <Desk>Agent.resolve_* carries a ~700-token desk-specific operating manual,
# so each of the 13 successor prefixes is expensive to prefill and shares
# nothing with the others.

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=1145,
    program_name="big_fan_out"
);

glob TRIAGE_INTENT: str = "Send the customer's request to the one specialist desk whose duty covers it. Read the walker's request, compare it against each candidate's duty and playbook, and choose exactly one desk. Prefer the desk that owns the customer's actual problem over one that merely shares a word with it.";

# ------------------------------------------------------------------ graph shape

edge ServiceDuty {
    has duty: str;
}

node TriageCenter {
    has queue: str = "front desk";
}

node SubAgent {
    has desk: str;
    has playbook: str;
}

node AccountAgent(SubAgent) {
    def resolve_account(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_account(visitor.request, self.playbook);
    }
}

node BillingAgent(SubAgent) {
    def resolve_billing(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_billing(visitor.request, self.playbook);
    }
}

node TechSupportAgent(SubAgent) {
    def resolve_tech(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_tech(visitor.request, self.playbook);
    }
}

node ProductQAAgent(SubAgent) {
    def resolve_qa(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_qa(visitor.request, self.playbook);
    }
}

node FinanceAgent(SubAgent) {
    def resolve_finance(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_finance(visitor.request, self.playbook);
    }
}

node ShippingAgent(SubAgent) {
    def resolve_shipping(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_shipping(visitor.request, self.playbook);
    }
}

node ReturnsAgent(SubAgent) {
    def resolve_returns(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_returns(visitor.request, self.playbook);
    }
}

node SecurityAgent(SubAgent) {
    def resolve_security(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_security(visitor.request, self.playbook);
    }
}

node SubscriptionAgent(SubAgent) {
    def resolve_subscription(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_subscription(visitor.request, self.playbook);
    }
}

node PrivacyAgent(SubAgent) {
    def resolve_privacy(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_privacy(visitor.request, self.playbook);
    }
}

node IntegrationAgent(SubAgent) {
    def resolve_integration(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_integration(visitor.request, self.playbook);
    }
}

node PartnerAgent(SubAgent) {
    def resolve_partner(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_partner(visitor.request, self.playbook);
    }
}

node EscalationAgent(SubAgent) {
    def resolve_escalation(request: str, playbook: str) -> str by llm(
        temperature=0.0,
        max_tokens=220
    );

    can handle with Agent entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.route = self.desk;
        visitor.answer = self.resolve_escalation(visitor.request, self.playbook);
    }
}


walker Agent {
    has request: str;
    has route: str = "";
    has answer: str = "";
    has summary: str = "";

    def summarize(request: str, desk: str, answer: str) -> str by llm(
        temperature=0.0,
        max_tokens=140
    );

    can serve with TriageCenter entry {
        # The one fan-out point: every sub-agent hanging off this desk is a
        # candidate, and each one leads to a different `by llm()` callsite.
        visit [->:ServiceDuty:->] by llm(select=1, intent=TRIAGE_INTENT, temperature=0.0);
    }

    can wrap with TriageCenter exit {
        time.sleep(0.20 * TOOL_SCALE);
        if not self.route {
            # The router returned no usable handle: nothing ran, so there is
            # nothing to summarize.
            print("ROUTE:   (unrouted)");
        } else {
            self.summary = self.summarize(self.request, self.route, self.answer);
            print("ROUTE:  ", self.route);
            print("ANSWER: ", self.answer);
            print("SUMMARY:", self.summary);
        }
    }
}


# ---------------------------------------------------------------------- driver

def build_desk() -> TriageCenter {
    desk = TriageCenter();
    root ++> desk;
    desk +>: ServiceDuty(duty="password resets, lockouts, profile and identity changes") :+>
        AccountAgent(
            desk="account desk",
            playbook="verify identity first, then reset credentials and clear the lockout"
        );
    desk +>: ServiceDuty(duty="invoices, charges, receipts and disputed amounts") :+>
        BillingAgent(
            desk="billing desk",
            playbook="reconcile the charge against the order before crediting anything"
        );
    desk +>: ServiceDuty(duty="crashes, errors and broken pages in the product") :+>
        TechSupportAgent(
            desk="tech support desk",
            playbook="reproduce with the trace id, check the last release, then roll back"
        );
    desk +>: ServiceDuty(duty="how-to questions about product behaviour and limits") :+>
        ProductQAAgent(
            desk="product answers desk",
            playbook="answer from the shipped documentation and confirm the customer's version"
        );
    desk +>: ServiceDuty(duty="payouts, tax forms, purchase orders and remittance") :+>
        FinanceAgent(
            desk="finance desk",
            playbook="confirm the payout batch and the tax document on file before promising a date"
        );
    desk +>: ServiceDuty(duty="delivery, tracking and packages that never arrived") :+>
        ShippingAgent(
            desk="shipping desk",
            playbook="open a carrier trace before offering a replacement shipment"
        );
    desk +>: ServiceDuty(duty="returns, exchanges and damaged or wrong items") :+>
        ReturnsAgent(
            desk="returns desk",
            playbook="issue the RMA and a prepaid label, refund only after the item scans in"
        );
    desk +>: ServiceDuty(duty="suspicious logins, account takeover and 2FA problems") :+>
        SecurityAgent(
            desk="security desk",
            playbook="revoke sessions immediately, re-enrol 2FA, then audit the login history"
        );
    desk +>: ServiceDuty(duty="plan changes, seats, renewals, trials and cancellations") :+>
        SubscriptionAgent(
            desk="subscription desk",
            playbook="apply plan changes at period end and prorate seat counts"
        );
    desk +>: ServiceDuty(duty="data deletion, exports and privacy-regulation requests") :+>
        PrivacyAgent(
            desk="privacy desk",
            playbook="deliver the export before the erasure, and log the statutory deadline"
        );
    desk +>: ServiceDuty(duty="API keys, webhooks, SDKs, rate limits and auth errors") :+>
        IntegrationAgent(
            desk="integration desk",
            playbook="rotate the key, replay failed deliveries, then review the rate limit"
        );
    desk +>: ServiceDuty(duty="reseller, affiliate and referral-commission matters") :+>
        PartnerAgent(
            desk="partner desk",
            playbook="verify the partner tier before recalculating any commission"
        );
    desk +>: ServiceDuty(duty="complaints, repeat failures and supervisor requests") :+>
        EscalationAgent(
            desk="escalation desk",
            playbook="own the case end to end and commit to a named callback window"
        );
    return desk;
}

glob REQUESTS: list[str] = [
    "I was charged twice for invoice INV-2291 this month and the receipt only lists one order.",
    "The reports tab crashes with a 500 error every single time I open the dashboard.",
    "My api key stopped working and the webhook endpoint returns 401 after the sdk upgrade.",
    "Someone signed in from a country I have never visited and 2fa was disabled without me.",
    "I want to cancel my pro plan before the renewal date and drop the two extra seats.",
    "The tracking has not moved in nine days and the delivery estimate already passed.",
    "Please delete my data under gdpr, but send me the data export before the erasure.",
    "This is the third time i have called about the same fault and i want a supervisor."
];


def chosen_requests() -> list[str] {
    direct = str(os.environ.get("REQUEST") or "");
    if direct {
        return [direct];
    }
    idx_env = os.environ.get("TRIAGE_INDEX");
    if idx_env is not None {
        return [REQUESTS[int(idx_env) % len(REQUESTS)]];
    }
    return REQUESTS;
}


with entry:__main__ {
    desk = build_desk();
    fan_out = len([desk ->:ServiceDuty:->]);
    print(f"FAN-OUT: {fan_out} sub-agents on one routed callsite");
    for request in chosen_requests() {
        print("-" * 78);
        print("REQUEST:", request);
        started = time.perf_counter();
        Agent(request=request) spawn desk;
        print(f"ELAPSED: {time.perf_counter() - started:.2f}s");
    }
}


# --------------------------------------------------------------------- semantics

sem TriageCenter = "the front desk that owns triage and nothing else";
sem TriageCenter.queue = "the intake queue this desk drains";
sem SubAgent = "a specialist desk that can close one class of customer request";
sem SubAgent.desk = "the desk's name";
sem SubAgent.playbook = "the procedure this desk follows before it answers";
sem ServiceDuty = "the class of request this desk is on the hook for";
sem ServiceDuty.duty = "what the desk at the far end of this link handles";

sem AccountAgent = "the desk for account access and identity";
sem AccountAgent.resolve_account = "Resolve a request at the AccountAgent using this desk's playbook, ending with the action to take now. Operating manual for the AccountAgent. Step 1: log the root cause in the shared register so the next desk can find it for the AccountAgent. Step 2: look for a known incident that overlaps the reported window for the AccountAgent. Step 3: summarise the plan in one paragraph before executing any of it for the AccountAgent. Step 4: never leave the case in an unowned state overnight for the AccountAgent. Step 5: list the systems that must be touched and who owns each one for the AccountAgent. Step 6: keep the reply under two hundred words unless the customer asked for detail for the AccountAgent. Step 7: decide whether a temporary workaround is safer than the full fix for the AccountAgent. Step 8: confirm the customer identity against the record on file for the AccountAgent. Step 9: set an internal reminder for the review checkpoint for the AccountAgent. Step 10: estimate the blast radius if the fix is applied wrongly for the AccountAgent. Step 11: offer the least disruptive remedy first and the refund path second for the AccountAgent. Step 12: check whether the same customer has other open cases that this touches for the AccountAgent. Step 13: double-check numbers, dates and reference ids character by character for the AccountAgent. Step 14: escalate immediately if money, safety or legal exposure is involved for the AccountAgent. Step 15: make sure the customer knows exactly what will happen next and when for the AccountAgent. Step 16: close the loop with any internal team that was paged along the way for the AccountAgent. Step 17: re-read the reply once for tone and once for accuracy for the AccountAgent. Step 18: record the exact wording of the customer's ask before interpreting it for the AccountAgent. Step 19: compare the current state with the last known good state for the AccountAgent. Step 20: translate internal jargon into plain language before it reaches the customer for the AccountAgent. Step 21: attach every artifact the customer sent to the case before replying for the AccountAgent. Step 22: write down the acceptance criterion the customer will use to call this closed for the AccountAgent. Step 23: note every timestamp the customer mentions and align it to the audit log for the AccountAgent. Step 24: ask a second pair of eyes to read the plan when the account is enterprise tier for the AccountAgent. Step 25: check whether an open case already exists for this account for the AccountAgent. Step 26: pull the full event history for the last ninety days for the AccountAgent. Step 27: confirm the fix from the customer's side, not only from the dashboard for the AccountAgent. Step 28: check the regional rules that constrain what the desk may promise for the AccountAgent. Step 29: verify the account tier and any contractual commitments for the AccountAgent. Step 30: confirm the customer's preferred channel for the follow-up for the AccountAgent. Rules of the AccountAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the AccountAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem AccountAgent.resolve_account.request = "the customer's request";

sem BillingAgent = "the desk for charges, invoices and disputed amounts";
sem BillingAgent.resolve_billing = "Resolve a request at the BillingAgent using this desk's playbook, ending with the action to take now. Operating manual for the BillingAgent. Step 1: check the regional rules that constrain what the desk may promise for the BillingAgent. Step 2: summarise the plan in one paragraph before executing any of it for the BillingAgent. Step 3: check whether the same customer has other open cases that this touches for the BillingAgent. Step 4: pull the full event history for the last ninety days for the BillingAgent. Step 5: decide whether a temporary workaround is safer than the full fix for the BillingAgent. Step 6: log the root cause in the shared register so the next desk can find it for the BillingAgent. Step 7: translate internal jargon into plain language before it reaches the customer for the BillingAgent. Step 8: set an internal reminder for the review checkpoint for the BillingAgent. Step 9: make sure the customer knows exactly what will happen next and when for the BillingAgent. Step 10: close the loop with any internal team that was paged along the way for the BillingAgent. Step 11: verify the account tier and any contractual commitments for the BillingAgent. Step 12: offer the least disruptive remedy first and the refund path second for the BillingAgent. Step 13: attach every artifact the customer sent to the case before replying for the BillingAgent. Step 14: re-read the reply once for tone and once for accuracy for the BillingAgent. Step 15: check whether an open case already exists for this account for the BillingAgent. Step 16: keep the reply under two hundred words unless the customer asked for detail for the BillingAgent. Step 17: write down the acceptance criterion the customer will use to call this closed for the BillingAgent. Step 18: never leave the case in an unowned state overnight for the BillingAgent. Step 19: ask a second pair of eyes to read the plan when the account is enterprise tier for the BillingAgent. Step 20: note every timestamp the customer mentions and align it to the audit log for the BillingAgent. Step 21: escalate immediately if money, safety or legal exposure is involved for the BillingAgent. Step 22: estimate the blast radius if the fix is applied wrongly for the BillingAgent. Step 23: list the systems that must be touched and who owns each one for the BillingAgent. Step 24: record the exact wording of the customer's ask before interpreting it for the BillingAgent. Step 25: confirm the fix from the customer's side, not only from the dashboard for the BillingAgent. Step 26: confirm the customer identity against the record on file for the BillingAgent. Step 27: look for a known incident that overlaps the reported window for the BillingAgent. Step 28: double-check numbers, dates and reference ids character by character for the BillingAgent. Step 29: confirm the customer's preferred channel for the follow-up for the BillingAgent. Step 30: compare the current state with the last known good state for the BillingAgent. Rules of the BillingAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the BillingAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem BillingAgent.resolve_billing.request = "the customer's request";

sem TechSupportAgent = "the desk for product faults, crashes and errors";
sem TechSupportAgent.resolve_tech = "Resolve a request at the TechSupportAgent using this desk's playbook, ending with the action to take now. Operating manual for the TechSupportAgent. Step 1: re-read the reply once for tone and once for accuracy for the TechSupportAgent. Step 2: translate internal jargon into plain language before it reaches the customer for the TechSupportAgent. Step 3: check the regional rules that constrain what the desk may promise for the TechSupportAgent. Step 4: note every timestamp the customer mentions and align it to the audit log for the TechSupportAgent. Step 5: attach every artifact the customer sent to the case before replying for the TechSupportAgent. Step 6: pull the full event history for the last ninety days for the TechSupportAgent. Step 7: confirm the customer's preferred channel for the follow-up for the TechSupportAgent. Step 8: make sure the customer knows exactly what will happen next and when for the TechSupportAgent. Step 9: summarise the plan in one paragraph before executing any of it for the TechSupportAgent. Step 10: never leave the case in an unowned state overnight for the TechSupportAgent. Step 11: decide whether a temporary workaround is safer than the full fix for the TechSupportAgent. Step 12: escalate immediately if money, safety or legal exposure is involved for the TechSupportAgent. Step 13: compare the current state with the last known good state for the TechSupportAgent. Step 14: estimate the blast radius if the fix is applied wrongly for the TechSupportAgent. Step 15: confirm the customer identity against the record on file for the TechSupportAgent. Step 16: keep the reply under two hundred words unless the customer asked for detail for the TechSupportAgent. Step 17: write down the acceptance criterion the customer will use to call this closed for the TechSupportAgent. Step 18: record the exact wording of the customer's ask before interpreting it for the TechSupportAgent. Step 19: look for a known incident that overlaps the reported window for the TechSupportAgent. Step 20: check whether the same customer has other open cases that this touches for the TechSupportAgent. Step 21: list the systems that must be touched and who owns each one for the TechSupportAgent. Step 22: log the root cause in the shared register so the next desk can find it for the TechSupportAgent. Step 23: set an internal reminder for the review checkpoint for the TechSupportAgent. Step 24: close the loop with any internal team that was paged along the way for the TechSupportAgent. Step 25: offer the least disruptive remedy first and the refund path second for the TechSupportAgent. Step 26: double-check numbers, dates and reference ids character by character for the TechSupportAgent. Step 27: ask a second pair of eyes to read the plan when the account is enterprise tier for the TechSupportAgent. Step 28: check whether an open case already exists for this account for the TechSupportAgent. Step 29: verify the account tier and any contractual commitments for the TechSupportAgent. Step 30: confirm the fix from the customer's side, not only from the dashboard for the TechSupportAgent. Rules of the TechSupportAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the TechSupportAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem TechSupportAgent.resolve_tech.request = "the customer's request";

sem ProductQAAgent = "the desk that answers how-to questions about the product";
sem ProductQAAgent.resolve_qa = "Resolve a request at the ProductQAAgent using this desk's playbook, ending with the action to take now. Operating manual for the ProductQAAgent. Step 1: summarise the plan in one paragraph before executing any of it for the ProductQAAgent. Step 2: compare the current state with the last known good state for the ProductQAAgent. Step 3: never leave the case in an unowned state overnight for the ProductQAAgent. Step 4: confirm the customer's preferred channel for the follow-up for the ProductQAAgent. Step 5: note every timestamp the customer mentions and align it to the audit log for the ProductQAAgent. Step 6: pull the full event history for the last ninety days for the ProductQAAgent. Step 7: close the loop with any internal team that was paged along the way for the ProductQAAgent. Step 8: confirm the fix from the customer's side, not only from the dashboard for the ProductQAAgent. Step 9: double-check numbers, dates and reference ids character by character for the ProductQAAgent. Step 10: re-read the reply once for tone and once for accuracy for the ProductQAAgent. Step 11: verify the account tier and any contractual commitments for the ProductQAAgent. Step 12: check whether an open case already exists for this account for the ProductQAAgent. Step 13: list the systems that must be touched and who owns each one for the ProductQAAgent. Step 14: make sure the customer knows exactly what will happen next and when for the ProductQAAgent. Step 15: attach every artifact the customer sent to the case before replying for the ProductQAAgent. Step 16: write down the acceptance criterion the customer will use to call this closed for the ProductQAAgent. Step 17: ask a second pair of eyes to read the plan when the account is enterprise tier for the ProductQAAgent. Step 18: confirm the customer identity against the record on file for the ProductQAAgent. Step 19: keep the reply under two hundred words unless the customer asked for detail for the ProductQAAgent. Step 20: check whether the same customer has other open cases that this touches for the ProductQAAgent. Step 21: escalate immediately if money, safety or legal exposure is involved for the ProductQAAgent. Step 22: set an internal reminder for the review checkpoint for the ProductQAAgent. Step 23: decide whether a temporary workaround is safer than the full fix for the ProductQAAgent. Step 24: look for a known incident that overlaps the reported window for the ProductQAAgent. Step 25: record the exact wording of the customer's ask before interpreting it for the ProductQAAgent. Step 26: offer the least disruptive remedy first and the refund path second for the ProductQAAgent. Step 27: check the regional rules that constrain what the desk may promise for the ProductQAAgent. Step 28: log the root cause in the shared register so the next desk can find it for the ProductQAAgent. Step 29: translate internal jargon into plain language before it reaches the customer for the ProductQAAgent. Step 30: estimate the blast radius if the fix is applied wrongly for the ProductQAAgent. Rules of the ProductQAAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the ProductQAAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem ProductQAAgent.resolve_qa.request = "the customer's request";

sem FinanceAgent = "the desk for payouts, tax documents and remittance";
sem FinanceAgent.resolve_finance = "Resolve a request at the FinanceAgent using this desk's playbook, ending with the action to take now. Operating manual for the FinanceAgent. Step 1: record the exact wording of the customer's ask before interpreting it for the FinanceAgent. Step 2: attach every artifact the customer sent to the case before replying for the FinanceAgent. Step 3: summarise the plan in one paragraph before executing any of it for the FinanceAgent. Step 4: decide whether a temporary workaround is safer than the full fix for the FinanceAgent. Step 5: verify the account tier and any contractual commitments for the FinanceAgent. Step 6: check the regional rules that constrain what the desk may promise for the FinanceAgent. Step 7: confirm the customer identity against the record on file for the FinanceAgent. Step 8: estimate the blast radius if the fix is applied wrongly for the FinanceAgent. Step 9: keep the reply under two hundred words unless the customer asked for detail for the FinanceAgent. Step 10: make sure the customer knows exactly what will happen next and when for the FinanceAgent. Step 11: ask a second pair of eyes to read the plan when the account is enterprise tier for the FinanceAgent. Step 12: set an internal reminder for the review checkpoint for the FinanceAgent. Step 13: double-check numbers, dates and reference ids character by character for the FinanceAgent. Step 14: look for a known incident that overlaps the reported window for the FinanceAgent. Step 15: escalate immediately if money, safety or legal exposure is involved for the FinanceAgent. Step 16: translate internal jargon into plain language before it reaches the customer for the FinanceAgent. Step 17: never leave the case in an unowned state overnight for the FinanceAgent. Step 18: offer the least disruptive remedy first and the refund path second for the FinanceAgent. Step 19: log the root cause in the shared register so the next desk can find it for the FinanceAgent. Step 20: list the systems that must be touched and who owns each one for the FinanceAgent. Step 21: confirm the fix from the customer's side, not only from the dashboard for the FinanceAgent. Step 22: note every timestamp the customer mentions and align it to the audit log for the FinanceAgent. Step 23: re-read the reply once for tone and once for accuracy for the FinanceAgent. Step 24: close the loop with any internal team that was paged along the way for the FinanceAgent. Step 25: compare the current state with the last known good state for the FinanceAgent. Step 26: write down the acceptance criterion the customer will use to call this closed for the FinanceAgent. Step 27: confirm the customer's preferred channel for the follow-up for the FinanceAgent. Step 28: check whether the same customer has other open cases that this touches for the FinanceAgent. Step 29: pull the full event history for the last ninety days for the FinanceAgent. Step 30: check whether an open case already exists for this account for the FinanceAgent. Rules of the FinanceAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the FinanceAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem FinanceAgent.resolve_finance.request = "the customer's request";

sem ShippingAgent = "the desk for delivery and tracking";
sem ShippingAgent.resolve_shipping = "Resolve a request at the ShippingAgent using this desk's playbook, ending with the action to take now. Operating manual for the ShippingAgent. Step 1: compare the current state with the last known good state for the ShippingAgent. Step 2: set an internal reminder for the review checkpoint for the ShippingAgent. Step 3: attach every artifact the customer sent to the case before replying for the ShippingAgent. Step 4: confirm the customer identity against the record on file for the ShippingAgent. Step 5: never leave the case in an unowned state overnight for the ShippingAgent. Step 6: ask a second pair of eyes to read the plan when the account is enterprise tier for the ShippingAgent. Step 7: decide whether a temporary workaround is safer than the full fix for the ShippingAgent. Step 8: offer the least disruptive remedy first and the refund path second for the ShippingAgent. Step 9: escalate immediately if money, safety or legal exposure is involved for the ShippingAgent. Step 10: check whether an open case already exists for this account for the ShippingAgent. Step 11: re-read the reply once for tone and once for accuracy for the ShippingAgent. Step 12: double-check numbers, dates and reference ids character by character for the ShippingAgent. Step 13: write down the acceptance criterion the customer will use to call this closed for the ShippingAgent. Step 14: confirm the fix from the customer's side, not only from the dashboard for the ShippingAgent. Step 15: list the systems that must be touched and who owns each one for the ShippingAgent. Step 16: look for a known incident that overlaps the reported window for the ShippingAgent. Step 17: check whether the same customer has other open cases that this touches for the ShippingAgent. Step 18: close the loop with any internal team that was paged along the way for the ShippingAgent. Step 19: make sure the customer knows exactly what will happen next and when for the ShippingAgent. Step 20: keep the reply under two hundred words unless the customer asked for detail for the ShippingAgent. Step 21: pull the full event history for the last ninety days for the ShippingAgent. Step 22: log the root cause in the shared register so the next desk can find it for the ShippingAgent. Step 23: translate internal jargon into plain language before it reaches the customer for the ShippingAgent. Step 24: record the exact wording of the customer's ask before interpreting it for the ShippingAgent. Step 25: confirm the customer's preferred channel for the follow-up for the ShippingAgent. Step 26: note every timestamp the customer mentions and align it to the audit log for the ShippingAgent. Step 27: estimate the blast radius if the fix is applied wrongly for the ShippingAgent. Step 28: verify the account tier and any contractual commitments for the ShippingAgent. Step 29: check the regional rules that constrain what the desk may promise for the ShippingAgent. Step 30: summarise the plan in one paragraph before executing any of it for the ShippingAgent. Rules of the ShippingAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the ShippingAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem ShippingAgent.resolve_shipping.request = "the customer's request";

sem ReturnsAgent = "the desk for returns, exchanges and damaged goods";
sem ReturnsAgent.resolve_returns = "Resolve a request at the ReturnsAgent using this desk's playbook, ending with the action to take now. Operating manual for the ReturnsAgent. Step 1: note every timestamp the customer mentions and align it to the audit log for the ReturnsAgent. Step 2: confirm the fix from the customer's side, not only from the dashboard for the ReturnsAgent. Step 3: double-check numbers, dates and reference ids character by character for the ReturnsAgent. Step 4: verify the account tier and any contractual commitments for the ReturnsAgent. Step 5: never leave the case in an unowned state overnight for the ReturnsAgent. Step 6: offer the least disruptive remedy first and the refund path second for the ReturnsAgent. Step 7: log the root cause in the shared register so the next desk can find it for the ReturnsAgent. Step 8: close the loop with any internal team that was paged along the way for the ReturnsAgent. Step 9: estimate the blast radius if the fix is applied wrongly for the ReturnsAgent. Step 10: keep the reply under two hundred words unless the customer asked for detail for the ReturnsAgent. Step 11: check whether the same customer has other open cases that this touches for the ReturnsAgent. Step 12: escalate immediately if money, safety or legal exposure is involved for the ReturnsAgent. Step 13: translate internal jargon into plain language before it reaches the customer for the ReturnsAgent. Step 14: check whether an open case already exists for this account for the ReturnsAgent. Step 15: compare the current state with the last known good state for the ReturnsAgent. Step 16: make sure the customer knows exactly what will happen next and when for the ReturnsAgent. Step 17: confirm the customer's preferred channel for the follow-up for the ReturnsAgent. Step 18: re-read the reply once for tone and once for accuracy for the ReturnsAgent. Step 19: check the regional rules that constrain what the desk may promise for the ReturnsAgent. Step 20: ask a second pair of eyes to read the plan when the account is enterprise tier for the ReturnsAgent. Step 21: list the systems that must be touched and who owns each one for the ReturnsAgent. Step 22: record the exact wording of the customer's ask before interpreting it for the ReturnsAgent. Step 23: attach every artifact the customer sent to the case before replying for the ReturnsAgent. Step 24: decide whether a temporary workaround is safer than the full fix for the ReturnsAgent. Step 25: set an internal reminder for the review checkpoint for the ReturnsAgent. Step 26: summarise the plan in one paragraph before executing any of it for the ReturnsAgent. Step 27: pull the full event history for the last ninety days for the ReturnsAgent. Step 28: look for a known incident that overlaps the reported window for the ReturnsAgent. Step 29: confirm the customer identity against the record on file for the ReturnsAgent. Step 30: write down the acceptance criterion the customer will use to call this closed for the ReturnsAgent. Rules of the ReturnsAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the ReturnsAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem ReturnsAgent.resolve_returns.request = "the customer's request";

sem SecurityAgent = "the desk for account takeover, suspicious access and 2FA";
sem SecurityAgent.resolve_security = "Resolve a request at the SecurityAgent using this desk's playbook, ending with the action to take now. Operating manual for the SecurityAgent. Step 1: translate internal jargon into plain language before it reaches the customer for the SecurityAgent. Step 2: log the root cause in the shared register so the next desk can find it for the SecurityAgent. Step 3: offer the least disruptive remedy first and the refund path second for the SecurityAgent. Step 4: confirm the customer identity against the record on file for the SecurityAgent. Step 5: confirm the fix from the customer's side, not only from the dashboard for the SecurityAgent. Step 6: list the systems that must be touched and who owns each one for the SecurityAgent. Step 7: check whether an open case already exists for this account for the SecurityAgent. Step 8: confirm the customer's preferred channel for the follow-up for the SecurityAgent. Step 9: note every timestamp the customer mentions and align it to the audit log for the SecurityAgent. Step 10: estimate the blast radius if the fix is applied wrongly for the SecurityAgent. Step 11: check the regional rules that constrain what the desk may promise for the SecurityAgent. Step 12: attach every artifact the customer sent to the case before replying for the SecurityAgent. Step 13: close the loop with any internal team that was paged along the way for the SecurityAgent. Step 14: re-read the reply once for tone and once for accuracy for the SecurityAgent. Step 15: look for a known incident that overlaps the reported window for the SecurityAgent. Step 16: check whether the same customer has other open cases that this touches for the SecurityAgent. Step 17: set an internal reminder for the review checkpoint for the SecurityAgent. Step 18: never leave the case in an unowned state overnight for the SecurityAgent. Step 19: double-check numbers, dates and reference ids character by character for the SecurityAgent. Step 20: keep the reply under two hundred words unless the customer asked for detail for the SecurityAgent. Step 21: record the exact wording of the customer's ask before interpreting it for the SecurityAgent. Step 22: decide whether a temporary workaround is safer than the full fix for the SecurityAgent. Step 23: compare the current state with the last known good state for the SecurityAgent. Step 24: pull the full event history for the last ninety days for the SecurityAgent. Step 25: summarise the plan in one paragraph before executing any of it for the SecurityAgent. Step 26: verify the account tier and any contractual commitments for the SecurityAgent. Step 27: make sure the customer knows exactly what will happen next and when for the SecurityAgent. Step 28: write down the acceptance criterion the customer will use to call this closed for the SecurityAgent. Step 29: ask a second pair of eyes to read the plan when the account is enterprise tier for the SecurityAgent. Step 30: escalate immediately if money, safety or legal exposure is involved for the SecurityAgent. Rules of the SecurityAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the SecurityAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem SecurityAgent.resolve_security.request = "the customer's request";

sem SubscriptionAgent = "the desk for plans, seats, renewals and cancellations";
sem SubscriptionAgent.resolve_subscription = "Resolve a request at the SubscriptionAgent using this desk's playbook, ending with the action to take now. Operating manual for the SubscriptionAgent. Step 1: look for a known incident that overlaps the reported window for the SubscriptionAgent. Step 2: log the root cause in the shared register so the next desk can find it for the SubscriptionAgent. Step 3: make sure the customer knows exactly what will happen next and when for the SubscriptionAgent. Step 4: translate internal jargon into plain language before it reaches the customer for the SubscriptionAgent. Step 5: ask a second pair of eyes to read the plan when the account is enterprise tier for the SubscriptionAgent. Step 6: check whether the same customer has other open cases that this touches for the SubscriptionAgent. Step 7: confirm the fix from the customer's side, not only from the dashboard for the SubscriptionAgent. Step 8: verify the account tier and any contractual commitments for the SubscriptionAgent. Step 9: double-check numbers, dates and reference ids character by character for the SubscriptionAgent. Step 10: check whether an open case already exists for this account for the SubscriptionAgent. Step 11: decide whether a temporary workaround is safer than the full fix for the SubscriptionAgent. Step 12: confirm the customer identity against the record on file for the SubscriptionAgent. Step 13: re-read the reply once for tone and once for accuracy for the SubscriptionAgent. Step 14: close the loop with any internal team that was paged along the way for the SubscriptionAgent. Step 15: estimate the blast radius if the fix is applied wrongly for the SubscriptionAgent. Step 16: escalate immediately if money, safety or legal exposure is involved for the SubscriptionAgent. Step 17: check the regional rules that constrain what the desk may promise for the SubscriptionAgent. Step 18: summarise the plan in one paragraph before executing any of it for the SubscriptionAgent. Step 19: keep the reply under two hundred words unless the customer asked for detail for the SubscriptionAgent. Step 20: never leave the case in an unowned state overnight for the SubscriptionAgent. Step 21: confirm the customer's preferred channel for the follow-up for the SubscriptionAgent. Step 22: pull the full event history for the last ninety days for the SubscriptionAgent. Step 23: list the systems that must be touched and who owns each one for the SubscriptionAgent. Step 24: attach every artifact the customer sent to the case before replying for the SubscriptionAgent. Step 25: set an internal reminder for the review checkpoint for the SubscriptionAgent. Step 26: note every timestamp the customer mentions and align it to the audit log for the SubscriptionAgent. Step 27: compare the current state with the last known good state for the SubscriptionAgent. Step 28: offer the least disruptive remedy first and the refund path second for the SubscriptionAgent. Step 29: write down the acceptance criterion the customer will use to call this closed for the SubscriptionAgent. Step 30: record the exact wording of the customer's ask before interpreting it for the SubscriptionAgent. Rules of the SubscriptionAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the SubscriptionAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem SubscriptionAgent.resolve_subscription.request = "the customer's request";

sem PrivacyAgent = "the desk for data deletion, exports and privacy regulation";
sem PrivacyAgent.resolve_privacy = "Resolve a request at the PrivacyAgent using this desk's playbook, ending with the action to take now. Operating manual for the PrivacyAgent. Step 1: confirm the customer's preferred channel for the follow-up for the PrivacyAgent. Step 2: close the loop with any internal team that was paged along the way for the PrivacyAgent. Step 3: log the root cause in the shared register so the next desk can find it for the PrivacyAgent. Step 4: ask a second pair of eyes to read the plan when the account is enterprise tier for the PrivacyAgent. Step 5: decide whether a temporary workaround is safer than the full fix for the PrivacyAgent. Step 6: write down the acceptance criterion the customer will use to call this closed for the PrivacyAgent. Step 7: check the regional rules that constrain what the desk may promise for the PrivacyAgent. Step 8: attach every artifact the customer sent to the case before replying for the PrivacyAgent. Step 9: re-read the reply once for tone and once for accuracy for the PrivacyAgent. Step 10: look for a known incident that overlaps the reported window for the PrivacyAgent. Step 11: pull the full event history for the last ninety days for the PrivacyAgent. Step 12: check whether an open case already exists for this account for the PrivacyAgent. Step 13: summarise the plan in one paragraph before executing any of it for the PrivacyAgent. Step 14: keep the reply under two hundred words unless the customer asked for detail for the PrivacyAgent. Step 15: check whether the same customer has other open cases that this touches for the PrivacyAgent. Step 16: offer the least disruptive remedy first and the refund path second for the PrivacyAgent. Step 17: record the exact wording of the customer's ask before interpreting it for the PrivacyAgent. Step 18: note every timestamp the customer mentions and align it to the audit log for the PrivacyAgent. Step 19: translate internal jargon into plain language before it reaches the customer for the PrivacyAgent. Step 20: verify the account tier and any contractual commitments for the PrivacyAgent. Step 21: make sure the customer knows exactly what will happen next and when for the PrivacyAgent. Step 22: compare the current state with the last known good state for the PrivacyAgent. Step 23: set an internal reminder for the review checkpoint for the PrivacyAgent. Step 24: estimate the blast radius if the fix is applied wrongly for the PrivacyAgent. Step 25: confirm the customer identity against the record on file for the PrivacyAgent. Step 26: never leave the case in an unowned state overnight for the PrivacyAgent. Step 27: list the systems that must be touched and who owns each one for the PrivacyAgent. Step 28: confirm the fix from the customer's side, not only from the dashboard for the PrivacyAgent. Step 29: double-check numbers, dates and reference ids character by character for the PrivacyAgent. Step 30: escalate immediately if money, safety or legal exposure is involved for the PrivacyAgent. Rules of the PrivacyAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the PrivacyAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem PrivacyAgent.resolve_privacy.request = "the customer's request";

sem IntegrationAgent = "the desk for API keys, webhooks, SDKs and rate limits";
sem IntegrationAgent.resolve_integration = "Resolve a request at the IntegrationAgent using this desk's playbook, ending with the action to take now. Operating manual for the IntegrationAgent. Step 1: ask a second pair of eyes to read the plan when the account is enterprise tier for the IntegrationAgent. Step 2: verify the account tier and any contractual commitments for the IntegrationAgent. Step 3: confirm the customer identity against the record on file for the IntegrationAgent. Step 4: check the regional rules that constrain what the desk may promise for the IntegrationAgent. Step 5: re-read the reply once for tone and once for accuracy for the IntegrationAgent. Step 6: escalate immediately if money, safety or legal exposure is involved for the IntegrationAgent. Step 7: close the loop with any internal team that was paged along the way for the IntegrationAgent. Step 8: record the exact wording of the customer's ask before interpreting it for the IntegrationAgent. Step 9: check whether an open case already exists for this account for the IntegrationAgent. Step 10: list the systems that must be touched and who owns each one for the IntegrationAgent. Step 11: estimate the blast radius if the fix is applied wrongly for the IntegrationAgent. Step 12: attach every artifact the customer sent to the case before replying for the IntegrationAgent. Step 13: look for a known incident that overlaps the reported window for the IntegrationAgent. Step 14: translate internal jargon into plain language before it reaches the customer for the IntegrationAgent. Step 15: confirm the customer's preferred channel for the follow-up for the IntegrationAgent. Step 16: decide whether a temporary workaround is safer than the full fix for the IntegrationAgent. Step 17: note every timestamp the customer mentions and align it to the audit log for the IntegrationAgent. Step 18: confirm the fix from the customer's side, not only from the dashboard for the IntegrationAgent. Step 19: set an internal reminder for the review checkpoint for the IntegrationAgent. Step 20: compare the current state with the last known good state for the IntegrationAgent. Step 21: check whether the same customer has other open cases that this touches for the IntegrationAgent. Step 22: summarise the plan in one paragraph before executing any of it for the IntegrationAgent. Step 23: log the root cause in the shared register so the next desk can find it for the IntegrationAgent. Step 24: pull the full event history for the last ninety days for the IntegrationAgent. Step 25: make sure the customer knows exactly what will happen next and when for the IntegrationAgent. Step 26: never leave the case in an unowned state overnight for the IntegrationAgent. Step 27: keep the reply under two hundred words unless the customer asked for detail for the IntegrationAgent. Step 28: write down the acceptance criterion the customer will use to call this closed for the IntegrationAgent. Step 29: double-check numbers, dates and reference ids character by character for the IntegrationAgent. Step 30: offer the least disruptive remedy first and the refund path second for the IntegrationAgent. Rules of the IntegrationAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the IntegrationAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem IntegrationAgent.resolve_integration.request = "the customer's request";

sem PartnerAgent = "the desk for resellers, affiliates and referral commissions";
sem PartnerAgent.resolve_partner = "Resolve a request at the PartnerAgent using this desk's playbook, ending with the action to take now. Operating manual for the PartnerAgent. Step 1: verify the account tier and any contractual commitments for the PartnerAgent. Step 2: decide whether a temporary workaround is safer than the full fix for the PartnerAgent. Step 3: log the root cause in the shared register so the next desk can find it for the PartnerAgent. Step 4: attach every artifact the customer sent to the case before replying for the PartnerAgent. Step 5: keep the reply under two hundred words unless the customer asked for detail for the PartnerAgent. Step 6: estimate the blast radius if the fix is applied wrongly for the PartnerAgent. Step 7: confirm the customer's preferred channel for the follow-up for the PartnerAgent. Step 8: compare the current state with the last known good state for the PartnerAgent. Step 9: confirm the customer identity against the record on file for the PartnerAgent. Step 10: confirm the fix from the customer's side, not only from the dashboard for the PartnerAgent. Step 11: set an internal reminder for the review checkpoint for the PartnerAgent. Step 12: make sure the customer knows exactly what will happen next and when for the PartnerAgent. Step 13: escalate immediately if money, safety or legal exposure is involved for the PartnerAgent. Step 14: offer the least disruptive remedy first and the refund path second for the PartnerAgent. Step 15: look for a known incident that overlaps the reported window for the PartnerAgent. Step 16: record the exact wording of the customer's ask before interpreting it for the PartnerAgent. Step 17: list the systems that must be touched and who owns each one for the PartnerAgent. Step 18: write down the acceptance criterion the customer will use to call this closed for the PartnerAgent. Step 19: check whether an open case already exists for this account for the PartnerAgent. Step 20: ask a second pair of eyes to read the plan when the account is enterprise tier for the PartnerAgent. Step 21: summarise the plan in one paragraph before executing any of it for the PartnerAgent. Step 22: close the loop with any internal team that was paged along the way for the PartnerAgent. Step 23: translate internal jargon into plain language before it reaches the customer for the PartnerAgent. Step 24: re-read the reply once for tone and once for accuracy for the PartnerAgent. Step 25: double-check numbers, dates and reference ids character by character for the PartnerAgent. Step 26: note every timestamp the customer mentions and align it to the audit log for the PartnerAgent. Step 27: check the regional rules that constrain what the desk may promise for the PartnerAgent. Step 28: check whether the same customer has other open cases that this touches for the PartnerAgent. Step 29: pull the full event history for the last ninety days for the PartnerAgent. Step 30: never leave the case in an unowned state overnight for the PartnerAgent. Rules of the PartnerAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the PartnerAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem PartnerAgent.resolve_partner.request = "the customer's request";

sem EscalationAgent = "the desk that owns complaints and supervisor requests";
sem EscalationAgent.resolve_escalation = "Resolve a request at the EscalationAgent using this desk's playbook, ending with the action to take now. Operating manual for the EscalationAgent. Step 1: check whether an open case already exists for this account for the EscalationAgent. Step 2: double-check numbers, dates and reference ids character by character for the EscalationAgent. Step 3: check the regional rules that constrain what the desk may promise for the EscalationAgent. Step 4: confirm the fix from the customer's side, not only from the dashboard for the EscalationAgent. Step 5: log the root cause in the shared register so the next desk can find it for the EscalationAgent. Step 6: confirm the customer's preferred channel for the follow-up for the EscalationAgent. Step 7: offer the least disruptive remedy first and the refund path second for the EscalationAgent. Step 8: close the loop with any internal team that was paged along the way for the EscalationAgent. Step 9: never leave the case in an unowned state overnight for the EscalationAgent. Step 10: compare the current state with the last known good state for the EscalationAgent. Step 11: make sure the customer knows exactly what will happen next and when for the EscalationAgent. Step 12: escalate immediately if money, safety or legal exposure is involved for the EscalationAgent. Step 13: confirm the customer identity against the record on file for the EscalationAgent. Step 14: set an internal reminder for the review checkpoint for the EscalationAgent. Step 15: translate internal jargon into plain language before it reaches the customer for the EscalationAgent. Step 16: decide whether a temporary workaround is safer than the full fix for the EscalationAgent. Step 17: attach every artifact the customer sent to the case before replying for the EscalationAgent. Step 18: record the exact wording of the customer's ask before interpreting it for the EscalationAgent. Step 19: summarise the plan in one paragraph before executing any of it for the EscalationAgent. Step 20: re-read the reply once for tone and once for accuracy for the EscalationAgent. Step 21: pull the full event history for the last ninety days for the EscalationAgent. Step 22: note every timestamp the customer mentions and align it to the audit log for the EscalationAgent. Step 23: ask a second pair of eyes to read the plan when the account is enterprise tier for the EscalationAgent. Step 24: estimate the blast radius if the fix is applied wrongly for the EscalationAgent. Step 25: look for a known incident that overlaps the reported window for the EscalationAgent. Step 26: verify the account tier and any contractual commitments for the EscalationAgent. Step 27: check whether the same customer has other open cases that this touches for the EscalationAgent. Step 28: list the systems that must be touched and who owns each one for the EscalationAgent. Step 29: keep the reply under two hundred words unless the customer asked for detail for the EscalationAgent. Step 30: write down the acceptance criterion the customer will use to call this closed for the EscalationAgent. Rules of the EscalationAgent: never promise a date the upstream team has not confirmed, never reveal another customer's data, always quote the reference number back, and always close with exactly one concrete next action. When the request is ambiguous the EscalationAgent asks one clarifying question at most and otherwise proceeds on the most conservative reading.";
sem EscalationAgent.resolve_escalation.request = "the customer's request";

sem Agent = "the triage walker carrying one customer request through the desks";
sem Agent.request = "the customer's request in their own words";
sem Agent.route = "the desk the request was routed to";
sem Agent.answer = "the desk's resolution";
sem Agent.summarize = "Compress the desk's resolution into a two-sentence note for the ticket log.";
sem Agent.summarize.desk = "the desk that handled the request";
sem Agent.summarize.answer = "the resolution the desk produced";
