import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-7B-Instruct"),
    comm_ip="localhost",
    comm_port=8970,
    program_name="dispatch"
);

glob DISPATCH_INTENT: str = "send the incident to the engineering desk that owns the failing component";

def check_runbook(component: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    return "Runbook says: roll back the last deploy, then replay the failed jobs from the dead-letter queue.";
}

sem check_runbook = "Fetch the runbook entry for one component.";
sem check_runbook.component = "the component the incident points at";

def resolve_network(issue: str, playbook: str) -> str by llm(
    temperature=0.0,
    max_tokens=300
);

def resolve_database(issue: str, playbook: str) -> str by llm(
    tools=[check_runbook],
    temperature=0.0,
    max_tokens=300,
    max_react_iterations=3
);

def resolve_frontend(issue: str, playbook: str) -> str by llm(
    temperature=0.0,
    max_tokens=300
);

def resolve_payments(issue: str, playbook: str) -> str by llm(
    temperature=0.0,
    max_tokens=300
);

def resolve_auth(issue: str, playbook: str) -> str by llm(
    temperature=0.0,
    max_tokens=300
);

def debrief(answer: str) -> str by llm(
    temperature=0.0,
    max_tokens=140
);

sem resolve_network = "Resolve an incident owned by the network desk using its playbook, ending with the mitigation to apply now.";
sem resolve_network.issue = "the incident description";
sem resolve_network.playbook = "the playbook this desk owns";
sem resolve_database = "Resolve an incident owned by the database desk. Consult the runbook first, then end with the mitigation to apply now.";
sem resolve_database.issue = "the incident description";
sem resolve_database.playbook = "the playbook this desk owns";
sem resolve_frontend = "Resolve an incident owned by the frontend desk using its playbook, ending with the mitigation to apply now.";
sem resolve_frontend.issue = "the incident description";
sem resolve_frontend.playbook = "the playbook this desk owns";
sem resolve_payments = "Resolve an incident owned by the payments desk using its playbook, ending with the mitigation to apply now.";
sem resolve_payments.issue = "the incident description";
sem resolve_payments.playbook = "the playbook this desk owns";
sem resolve_auth = "Resolve an incident owned by the auth desk using its playbook, ending with the mitigation to apply now.";
sem resolve_auth.issue = "the incident description";
sem resolve_auth.playbook = "the playbook this desk owns";
sem debrief = "Compress the desk's resolution into a two-sentence incident debrief for the status page.";
sem debrief.answer = "the resolution the desk produced";

node Intake {
    has queue: str = "pager";
}

node NetworkDesk {
    has playbook: str = "check BGP sessions and packet loss between regions before touching firewalls";

    can handle with dispatch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_network(visitor.issue, self.playbook);
    }
}

node DatabaseDesk {
    has playbook: str = "replication lag and failed migrations are handled here, storage pages the vendor";

    can handle with dispatch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_database(visitor.issue, self.playbook);
    }
}

node FrontendDesk {
    has playbook: str = "client-side errors, broken assets, and rendering regressions after a release";

    can handle with dispatch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_frontend(visitor.issue, self.playbook);
    }
}

node PaymentsDesk {
    has playbook: str = "declined charges, double charges, and settlement mismatches with the processor";

    can handle with dispatch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_payments(visitor.issue, self.playbook);
    }
}

node AuthDesk {
    has playbook: str = "login failures, token expiry storms, and SSO certificate rotations";

    can handle with dispatch entry {
        time.sleep(0.20 * TOOL_SCALE);
        visitor.answer = resolve_auth(visitor.issue, self.playbook);
    }
}

walker dispatch {
    has issue: str = "";
    has answer: str = "";

    can route with Intake entry {
        visit [-->] by llm(select=1, intent=DISPATCH_INTENT);
    }

    can wrap with Intake exit {
        time.sleep(0.20 * TOOL_SCALE);
        print("ANSWER:", self.answer);
        print("DEBRIEF:", debrief(self.answer));
    }
}

with entry:__main__ {
    issue = str(os.environ.get("ISSUE") or "Checkout returns 502 for EU users since 14:00, replication lag on the orders database is climbing.");
    intake = Intake();
    root ++> intake;
    intake ++> NetworkDesk();
    intake ++> DatabaseDesk();
    intake ++> FrontendDesk();
    intake ++> PaymentsDesk();
    intake ++> AuthDesk();
    dispatch(issue=issue) spawn intake;
}
