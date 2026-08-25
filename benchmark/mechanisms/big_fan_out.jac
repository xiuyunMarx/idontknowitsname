"""Big-fan-out triage agent: one routed callsite, thirteen possible successors.

The mechanism this file isolates is *fan-out at a routing callsite*. A single
`visit [->:ServiceDuty:->] by llm(select=1, ...)` on the triage desk chooses one
of thirteen specialist sub-agents, and every sub-agent carries its own
`by llm()` callsite. So one routing decision has thirteen distinct successors:
the routing call is the only thing standing between the served prefix and
thirteen different next prefixes, which is exactly the branch set a speculating
server has to reason about.

Shape of one request:

    Agent(request) spawn TriageCenter
        TriageCenter.serve  -> visit [->:ServiceDuty:->] by llm(select=1)   # 13 candidates
            <XyzAgent>.handle -> self.resolve_xyz(request, playbook)        # 1 of 13 callsites
        TriageCenter.wrap   -> self.summarize(...)                          # join point

The routing call is the only wide point: it is served once per request and its
answer picks which of the thirteen successor prefixes the walker runs next.

Run:

    python start_server.py --model Qwen/Qwen2.5-7B-Instruct \
        --program big_fan_out:benchmark/mechanisms/big_fan_out.jac:1145

    jac run benchmark/mechanisms/big_fan_out.jac

Inputs: $REQUEST (one request), or $TRIAGE_INDEX into the bundled request list,
or neither, which runs every bundled request in order.
"""

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
sem AccountAgent.resolve_account = "Resolve an account-access request using this desk's playbook, ending with the action to take now.";
sem AccountAgent.resolve_account.request = "the customer's request";

sem BillingAgent = "the desk for charges, invoices and disputed amounts";
sem BillingAgent.resolve_billing = "Resolve a billing request using this desk's playbook, ending with the action to take now.";
sem BillingAgent.resolve_billing.request = "the customer's request";

sem TechSupportAgent = "the desk for product faults, crashes and errors";
sem TechSupportAgent.resolve_tech = "Resolve a product fault using this desk's playbook, ending with the action to take now.";
sem TechSupportAgent.resolve_tech.request = "the customer's request";

sem ProductQAAgent = "the desk that answers how-to questions about the product";
sem ProductQAAgent.resolve_qa = "Answer a product question from the documented behaviour, ending with the action to take now.";
sem ProductQAAgent.resolve_qa.request = "the customer's request";

sem FinanceAgent = "the desk for payouts, tax documents and remittance";
sem FinanceAgent.resolve_finance = "Resolve a finance request using this desk's playbook, ending with the action to take now.";
sem FinanceAgent.resolve_finance.request = "the customer's request";

sem ShippingAgent = "the desk for delivery and tracking";
sem ShippingAgent.resolve_shipping = "Resolve a delivery problem using this desk's playbook, ending with the action to take now.";
sem ShippingAgent.resolve_shipping.request = "the customer's request";

sem ReturnsAgent = "the desk for returns, exchanges and damaged goods";
sem ReturnsAgent.resolve_returns = "Resolve a return or exchange using this desk's playbook, ending with the action to take now.";
sem ReturnsAgent.resolve_returns.request = "the customer's request";

sem SecurityAgent = "the desk for account takeover, suspicious access and 2FA";
sem SecurityAgent.resolve_security = "Resolve a security incident using this desk's playbook, ending with the action to take now.";
sem SecurityAgent.resolve_security.request = "the customer's request";

sem SubscriptionAgent = "the desk for plans, seats, renewals and cancellations";
sem SubscriptionAgent.resolve_subscription = "Resolve a subscription change using this desk's playbook, ending with the action to take now.";
sem SubscriptionAgent.resolve_subscription.request = "the customer's request";

sem PrivacyAgent = "the desk for data deletion, exports and privacy regulation";
sem PrivacyAgent.resolve_privacy = "Resolve a privacy request using this desk's playbook, ending with the action to take now.";
sem PrivacyAgent.resolve_privacy.request = "the customer's request";

sem IntegrationAgent = "the desk for API keys, webhooks, SDKs and rate limits";
sem IntegrationAgent.resolve_integration = "Resolve an integration failure using this desk's playbook, ending with the action to take now.";
sem IntegrationAgent.resolve_integration.request = "the customer's request";

sem PartnerAgent = "the desk for resellers, affiliates and referral commissions";
sem PartnerAgent.resolve_partner = "Resolve a partner request using this desk's playbook, ending with the action to take now.";
sem PartnerAgent.resolve_partner.request = "the customer's request";

sem EscalationAgent = "the desk that owns complaints and supervisor requests";
sem EscalationAgent.resolve_escalation = "Take ownership of an escalated complaint using this desk's playbook, ending with the action to take now.";
sem EscalationAgent.resolve_escalation.request = "the customer's request";

sem Agent = "the triage walker carrying one customer request through the desks";
sem Agent.request = "the customer's request in their own words";
sem Agent.route = "the desk the request was routed to";
sem Agent.answer = "the desk's resolution";
sem Agent.summarize = "Compress the desk's resolution into a two-sentence note for the ticket log.";
sem Agent.summarize.desk = "the desk that handled the request";
sem Agent.summarize.answer = "the resolution the desk produced";
