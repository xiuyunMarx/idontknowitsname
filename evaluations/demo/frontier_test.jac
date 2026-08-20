"""Frontier-topology fixture: a visit-by router over two desks whose abilities
contain MULTI-STAGE sequential byllm calls. Correct topology lists only each
ability's FIRST-reachable calls as the visit's direct successors (AlphaDesk.a1;
BetaDesk.b1 branch-contained + BetaDesk.b2 unconditional) — the later stages
hang off the sequential chain (a1->a2->a3), so BFS rings from the route are
real (a2 at distance 2, a3 at 3), not flattened to one layer."""

import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct");

node Hub {}

node AlphaDesk {
    def a1(q: str) -> str by llm(max_tokens=32);
    def a2(x: str) -> str by llm(max_tokens=32);
    def a3(x: str) -> str by llm(max_tokens=32);

    can handle with Desk entry {
        s1 = self.a1(visitor.request);
        s2 = self.a2(s1);
        s3 = self.a3(s2);
        visitor.response = s3;
    }
}

node BetaDesk {
    def b1(q: str) -> str by llm(max_tokens=32);
    def b2(x: str) -> str by llm(max_tokens=32);

    can handle with Desk entry {
        if len(visitor.request) > 3 {
            visitor.note = self.b1(visitor.request);
        }
        visitor.response = self.b2(visitor.request);
    }
}

walker Desk {
    has request: str = "please route me";
    has note: str = "";
    has response: str = "";

    can triage with `root entry {
        hub = Hub();
        here ++> hub;
        hub ++> AlphaDesk();
        hub ++> BetaDesk();
        visit hub;
    }

    can route with Hub entry {
        visit [-->] by llm(select=1, intent="Route the request to the desk that should handle it");
    }
}

with entry:__main__ {
    root spawn Desk();
}
