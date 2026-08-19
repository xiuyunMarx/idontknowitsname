"""Deterministic driver: call RagChat.respond directly (no router), so binding-zone
reorder can be exercised without routing variance. CASE_MSG selects the message."""

import os;
import from main { RagChat, llm }

with entry {
    msg = os.environ.get("CASE_MSG", "How are task dependencies represented between Task nodes in this example, and how does the DependsOn edge affect task readiness?");
    hist = [{"role": "user", "content": msg}];
    out = RagChat().respond(message=msg, chat_history=hist);
    print("RESPONSE:", out);
}
