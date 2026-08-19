"""Topology test: sequential, branch, join, caller-continuation, and walker visit edges."""

import from jaclang.byllm.lib { Model }

glob llm = Model(model_name="gpt-4o");

def step_a(x: str) -> str by llm();
def step_b(x: str) -> str by llm();
def step_c(x: str) -> str by llm();
def step_d(x: str) -> str by llm();
def plan(x: str) -> str by llm();
def answer(x: str) -> str by llm();

def pipeline(x: str) -> str {
    a = step_a(x);
    if len(a) > 10 {
        b = step_b(a);
    } else {
        c = step_c(a);
    }
    return step_d(a);
}

walker w {
    has q: str = "";

    can start with Root entry {
        self.q = plan(self.q);
        visit [-->];
    }
}

node Agent {
    can act with w entry {
        print(answer(visitor.q));
    }
}

with entry:__main__ {
    print(pipeline("hello world"));
    root ++> Agent();
    root spawn w(q="what is a walker?");
}
