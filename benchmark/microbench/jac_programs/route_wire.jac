import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8966,
    program_name="route"
);

glob ROUTE_INTENT: str = "send the ticket to the desk that owns this kind of problem";

def lookup_account(ticket: str) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    return "Account is on the annual plan, last invoice 2026-07-01, no open credits.";
}

sem lookup_account = "Look up the billing account behind one ticket.";
sem lookup_account.ticket = "the customer ticket text";

def resolve_billing(ticket: str, policy: str) -> str by llm(
    tools=[lookup_account],
    temperature=0.0,
    max_tokens=320,
    max_react_iterations=4
);

def resolve_tech(ticket: str, runbook: str) -> str by llm(
    temperature=0.0,
    max_tokens=320
);

def resolve_security(ticket: str, escalation: str) -> str by llm(
    temperature=0.0,
    max_tokens=320
);

def summarize(answer: str) -> str by llm(
    temperature=0.0,
    max_tokens=160
);

sem resolve_billing = "Settle a billing ticket. Look the account up first, then finish with the resolution the customer should be told.";
sem resolve_billing.ticket = "the customer ticket text";
sem resolve_billing.policy = "the billing policy this desk is allowed to apply";
sem resolve_tech = "Settle a technical ticket using the desk runbook.";
sem resolve_tech.ticket = "the customer ticket text";
sem resolve_tech.runbook = "the remediation steps this desk owns";
sem resolve_security = "Settle a security ticket by following the escalation path.";
sem resolve_security.ticket = "the customer ticket text";
sem resolve_security.escalation = "the escalation path this desk owns";
sem summarize = "Compress one desk's resolution into a single sentence for the customer.";
sem summarize.answer = "the resolution the desk produced";

node Triage {
    has queue: str = "inbound";
}

node BillingDesk {
    has policy: str = "refunds within 30 days, credits need a manager";

    can handle with support entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_billing(visitor.ticket, self.policy);
    }
}

node TechDesk {
    has runbook: str = "restart the service, then re-run the health check";

    can handle with support entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_tech(visitor.ticket, self.runbook);
    }
}

node SecurityDesk {
    has escalation: str = "page the on-call security lead within 15 minutes";

    can handle with support entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_security(visitor.ticket, self.escalation);
    }
}

walker support {
    has ticket: str = "";
    has answer: str = "";

    can route with Triage entry {
        visit [-->] by llm(select=1, intent=ROUTE_INTENT);
    }

    can wrap with Triage exit {
        time.sleep(0.20 * TOOL_SCALE);
        print("ANSWER:", self.answer);
        print("SUMMARY:", summarize(self.answer));
    }
}

with entry:__main__ {
    ticket = str(os.environ.get("TICKET") or "I was charged twice for last month's invoice.");
    triage = Triage();
    root ++> triage;
    triage ++> BillingDesk();
    triage ++> TechDesk();
    triage ++> SecurityDesk();
    support(ticket=ticket) spawn triage;
}
