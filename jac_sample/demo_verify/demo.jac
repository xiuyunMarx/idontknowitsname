"""Typed byllm demo used by verify_takeover.py: no tools, three return shapes."""

import from jaclang.byllm.lib { Model }

glob llm = Model(model_name="gpt-4o");

enum Priority {
    LOW = 1,
    MEDIUM = 2,
    HIGH = 3
}

obj Ticket {
    has title: str;
    has priority: Priority;
    has tags: list[str] = [];
}

sem Priority = "Urgency of a support ticket.";
sem Ticket = "A support ticket routed from a user message.";
sem Ticket.title = "Short summary of the issue.";
sem Ticket.tags = "Lowercase topic keywords.";

def route(text: str) -> Ticket by llm();
sem route = "Turn a raw user complaint into a structured support ticket.";

def count_words(text: str) -> int by llm(temperature=0.0);
sem count_words = "Count the number of words in the text.";

def summarize(text: str) -> str by llm(temperature=0.3, max_tokens=64);
sem summarize = "Summarize the text in one short sentence.";

with entry:__main__ {
    print(route(text="My GPU catches fire every time I run the training job!!"));
    print(count_words(text="the quick brown fox"));
    print(summarize(text="vLLM caches KV blocks by prefix so repeated prompts skip prefill."));
}
