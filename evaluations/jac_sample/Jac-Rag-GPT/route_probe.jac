"""Routing-quality probe: send labeled messages through the real router graph and
record which agent `visit [-->] by llm()` picks. Agents' chat abilities trigger on
`interact`, not on this walker, so each case costs exactly one route call."""

import from main { Router, RagChat, CodingChat, DebuggerChat, QAChat, OffTopicChat, ROUTER_INTENT, llm }

glob CASES: list = [("rag", "How are task dependencies represented between Task nodes in this example, and how does the DependsOn edge affect task readiness?", "RagChat"), ("rag", "What is a walker ability and how does `with entry` differ from `with exit`?", "RagChat"), ("code", "Write a Jac walker that counts all nodes in the graph and reports the total.", "CodingChat"), ("code", "Create a node type Person with name and age fields and connect three of them under root.", "CodingChat"), ("debug", "My walker crashes with 'AttributeError: message'. Here is my code: walker w { can go with Root entry { print(self.message); } } — fix it.", "DebuggerChat"), ("debug", "This Jac import line gives a syntax error, please fix: import from os { path };;", "DebuggerChat"), ("qa", "Hello! How are you doing today?", "QAChat"), ("qa", "Thanks so much for the help, goodbye!", "QAChat"), ("offtopic", "What's the best pizza topping in your opinion?", "OffTopicChat"), ("offtopic", "How do I write a for loop in Rust?", "OffTopicChat")];

walker probe {
    has message: str;
    has agent: str = "";

    can setup with Root entry {
        routers = [-->][?:Router];
        if routers {
            visit routers[0];
        } else {
            router = Router();
            here ++> router;
            router ++> RagChat();
            router ++> CodingChat();
            router ++> DebuggerChat();
            router ++> QAChat();
            router ++> OffTopicChat();
            visit router;
        }
    }

    can route with Router entry {
        visit [-->] by llm(select=1, intent=ROUTER_INTENT);
    }

    can record with RagChat | CodingChat | DebuggerChat | QAChat | OffTopicChat entry {
        self.agent = type(here).__name__;
    }
}

with entry {
    correct = 0;
    for (cat, msg, expected) in CASES {
        w = root spawn probe(message=msg);
        ok = w.agent == expected;
        if ok {
            correct += 1;
        }
        print(f"{'OK ' if ok else 'MISS'} [{cat}] expected={expected} got={w.agent}");
    }
    print(f"ROUTING ACCURACY: {correct}/{len(CASES)}");
}
