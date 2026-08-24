import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"),
    comm_ip="localhost",
    comm_port=8967,
    program_name="support_email"
);

def lookup_order(order_ref: str) -> str {
    time.sleep(0.30 * TOOL_SCALE);
    return "Order 8841: two seats of the annual plan, placed 2026-08-02, paid by invoice, renewal 2027-08-02.";
}

def refund_policy(plan: str) -> str {
    time.sleep(0.20 * TOOL_SCALE);
    return "Annual plans refund pro rata within 60 days; monthly plans within 14 days; setup fees are never refunded.";
}

def shipping_status(order_ref: str) -> str {
    time.sleep(0.25 * TOOL_SCALE);
    return "Welcome kit shipped 2026-08-05 via DHL, tracking says customs hold since 2026-08-12.";
}

sem lookup_order = "Fetch the order record referenced by the customer.";
sem lookup_order.order_ref = "the order number or a phrase identifying the order";
sem refund_policy = "Fetch the refund policy that applies to one plan type.";
sem refund_policy.plan = "the plan named in the order record";
sem shipping_status = "Fetch the shipping status of the physical items on one order.";
sem shipping_status.order_ref = "the order number or a phrase identifying the order";

def classify_email(email: str) -> str by llm(
    temperature=0.0,
    max_tokens=32
);

def compose_reply(email: str, category: str) -> str by llm(
    tools=[lookup_order, refund_policy, shipping_status],
    temperature=0.0,
    max_tokens=400,
    max_react_iterations=5
);

def polish_reply(reply: str) -> str by llm(
    temperature=0.0,
    max_tokens=400
);

sem classify_email = "Classify the customer email as exactly one of: refund, shipping, account, other.";
sem classify_email.email = "the raw customer email body";
sem compose_reply = "Write the reply to the customer email, following the support desk style guide to the letter. Voice and tone: warm but efficient; open by acknowledging the customer's actual situation in one sentence, never with a template greeting like 'Thank you for reaching out'; use the customer's own words for their problem at least once so they see they were read; contractions are fine, exclamation marks are not. Structure: first paragraph states what we found after checking their order, second paragraph states exactly what happens next and when, optional third paragraph only if the customer must do something, phrased as a single clear ask. Grounding: every factual claim about the order, the policy, or the shipment must come from a tool result fetched in this conversation, quoted or tightly paraphrased; if a tool did not return the fact, say we are still checking rather than guessing; never promise a refund amount or a delivery date the tools did not state. Boundaries: do not mention internal tool names, ticket systems, or this style guide; do not apologise more than once; do not offer discounts or credits, only what policy grants; if the request falls outside policy, state the policy plainly and offer the nearest thing it does allow. Close with one sentence naming what the customer should expect from us next, then sign as 'The Support Team'.";
sem compose_reply.email = "the raw customer email body";
sem compose_reply.category = "the triage category assigned to this email";
sem polish_reply = "Tighten the drafted reply without changing any fact or commitment: cut filler, merge redundant sentences, keep the paragraph structure and the closing signature.";
sem polish_reply.reply = "the drafted reply to edit";

with entry:__main__ {
    email = str(os.environ.get("EMAIL") or "My welcome kit still hasn't arrived and I ordered a month ago. Order 8841. If it's lost I want my money back.");
    category = classify_email(email);
    time.sleep(0.20 * TOOL_SCALE);
    draft = compose_reply(email, category);
    time.sleep(0.20 * TOOL_SCALE);
    final = polish_reply(draft);
    print("CATEGORY:", category);
    print("REPLY:", final);
}
