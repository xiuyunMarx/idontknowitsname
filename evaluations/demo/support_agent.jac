"""Type-3 E2E-TTFT showcase: a customer-support agent whose upstream is TOOLS,
not long decodes — so the strict E2E TTFT (user submits -> first token the user
actually sees) is dominated by the answer call's prefill, the one thing
proactive prefill removes.

Shape and metric:
  intake_request  — internal query normalization, SHORT decode (max_tokens 64),
                    output is a case ticket, never shown to the user.
  fetch_account / fetch_orders — tool lookups (~300 ms each): the idle windows.
  resolve_case    — the ONLY user-visible output; its sem carries the entire
                    support-policy compendium (~2k tokens, compile-time
                    constant), its reply streams to the user.

  E2E TTFT == program start -> resolve_case's first token. Intake decode and
  tool time are fixed costs in BOTH conditions; the baseline additionally pays
  resolve_case's full ~2k-token prefill at the END of the critical path, while
  speculation warms the invariant in intake's decode shadow and feeds the `ask`
  binding (ret of intake, dataflow) in the first tool window. account/orders
  are tool outputs (unknown provenance) — deliberately unfed, the honest cold
  suffix (~150 tokens). The answer is a plain str on purpose: no grammar
  compile in either condition, so the E2E delta is pure prefill.

  A/B recipe: server on this file (--greedy; --no-speculate --no-prov-feed
  --no-calibrate + one dummy /generate for the fair baseline), client
  `cd evaluations/demo && CASE_MSG="..." python3 -m jaclang run
  support_agent.jac`; E2E TTFT from /stats = (call_start[resolve_case] -
  call_start[intake_request]) + intake's client turnaround + ttft[resolve_case]
  — or simplest, first call_start -> resolve_case first token."""

import from os { getenv }
import from time { sleep }
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct");

def intake_request(question: str) -> str by llm(temperature=0.0, max_tokens=64);
sem intake_request = "You are the intake classifier of a customer-support desk. Rewrite the customer's free-text message as one normalized case ticket in at most three sentences: state the product area, the customer's concrete request or complaint, and any identifiers, deadlines or amounts they mentioned. Preserve every factual detail; add nothing. Output only the ticket text.";

def fetch_account() -> str {
    sleep(0.3);
    return "Account A-58213: premium tier since 2024-03, billing address Singapore, payment method corporate card ending 4417, auto-renewal ON, next renewal 2026-09-01, one prior support case (2025-11, shipping delay, resolved with a 10 USD credit), current loyalty balance 2,340 points.";
}

def fetch_orders() -> str {
    sleep(0.3);
    return "Recent orders: #90312 placed 2026-08-02, wireless headphones, 129 USD, delivered 2026-08-06; #90544 placed 2026-08-11, laptop stand, 59 USD, shipped 2026-08-12, carrier shows customs hold since 2026-08-14; #90671 placed 2026-08-16, USB-C dock, 89 USD, payment authorized, not yet shipped.";
}

def resolve_case(ask: str, account: str, orders: str) -> str by llm(temperature=0.0, max_tokens=400);
sem resolve_case = "You are the senior support agent of an online electronics retailer, and you must resolve the normalized case ticket using the account record, the order records, and the company policy compendium that follows; every statement you make to the customer must be consistent with these policies and you must cite the policy area you are relying on in parentheses at the end of the relevant sentence. RETURNS AND REFUNDS: standard items may be returned within 30 days of delivery in original packaging for a full refund to the original payment method processed within 5 business days of the warehouse receiving the item; opened items in working condition are eligible within 14 days for a refund minus a 10 percent restocking fee; defective items are exempt from the restocking fee at any point inside the 12-month warranty and the return label is free; final-sale and clearance items are not returnable but still carry the statutory warranty; premium-tier accounts get the return window extended to 45 days and restocking fees waived once per calendar year. SHIPPING AND DELIVERY: standard shipping is 3 to 5 business days and express is 1 to 2; if a shipment shows no carrier movement for 5 consecutive days the agent must proactively open a carrier investigation, which takes up to 8 business days, and offer the customer the choice between waiting for the investigation, an immediate reshipment for items under 150 USD, or a full refund; customs holds on international shipments are outside carrier control, the agent should explain the hold, may not promise a clearance date, and after 10 calendar days in customs the customer becomes eligible for a full refund or reshipment at company cost; orders that have not shipped yet can be cancelled for an immediate full refund with no fee. WARRANTY AND REPAIRS: all electronics carry a 12-month limited warranty covering manufacturing defects but not physical damage or liquid ingress; warranty service starts with remote diagnostics, then advance replacement for premium accounts or mail-in repair with a 7-to-10-business-day turnaround otherwise; accessories under 30 USD are replaced outright instead of repaired. BILLING AND CREDITS: duplicate charges are reversed within 3 business days once verified; goodwill credits up to 25 USD may be granted by the agent without approval when the company is at fault, and only one goodwill credit may be granted per case; loyalty points are worth one cent each, may be applied to future orders only, and are never converted to cash; auto-renewal charges can be refunded in full if the customer requests within 14 days of the charge and has not used premium benefits in that period. ESCALATION AND CONDUCT: cases involving amounts over 500 USD, suspected fraud, legal threats, or a third failed resolution attempt must be escalated to a human supervisor within one business day and the customer must be told the escalation timeline; the agent must never share internal system identifiers other than order numbers, must never speculate about other customers, and must summarize the agreed next steps and their deadlines in the final paragraph of the reply. PRICE ADJUSTMENTS AND PROMOTIONS: if an item's listed price drops within 14 days of the customer's purchase the agent grants a one-time price adjustment refunding the difference to the original payment method, promotional bundle discounts are never applied retroactively, coupon codes cannot be combined with each other or with loyalty-point redemptions in a single order, and a coupon that failed to apply at checkout through a verified site error entitles the customer to the equivalent amount as a goodwill credit under the billing rules above. EXCHANGES AND STOCK: exchanges are processed as a return plus a new order so all return-window rules apply unchanged, if the replacement item is out of stock the agent may offer the closest equivalent model and must refund any price difference in the customer's favor while never charging extra without explicit consent, backordered items show an estimated restock date that the agent may quote only with the phrase subject to change, and reserving stock for a customer is not possible under any circumstances. GIFT CARDS AND STORE CREDIT: gift cards are non-refundable and never convertible to cash, purchases paid partly with a gift card are refunded first to the gift card up to the amount it covered and only the remainder to the original payment method, store credit issued as goodwill expires after 24 months, and lost physical gift cards can be reissued only with the original purchase receipt. INTERNATIONAL ORDERS: import duties and local taxes are the customer's responsibility unless the checkout page showed a duties-included total, refused-at-customs shipments are refunded minus the outbound shipping cost once the item returns to the warehouse, warranty service for international customers is mail-in only with the company covering inbound shipping for verified defects, and delivery-date promises for international shipments must always be quoted as estimates. DATA PRIVACY: the agent may discuss an order only with the account holder verified by the registered email on the case, must decline to reveal account details to third parties including family members unless a documented delegate is on file, may not copy account data into channels outside the case record, and must direct any data-deletion request to the privacy team with a confirmation that intake happened the same day. Write the reply directly to the customer in a professional, warm tone: acknowledge the concern in one sentence, resolve each request point by point applying the policies above to the specific account and order facts, state concretely what happens next with dates or business-day counts, and close with the next-steps summary paragraph. Do not use markdown, bullet lists, or headers; write flowing prose paragraphs.";
sem resolve_case.ask = "the normalized case ticket produced by intake";
sem resolve_case.account = "the customer's account record as returned by the account service";
sem resolve_case.orders = "the customer's recent order records as returned by the order service";

with entry:__main__ {
    question = getenv("CASE_MSG", "Hi, my laptop stand from order 90544 has been stuck in customs for days now and I need it before a client workshop on Sept 1 - can you reship it or refund me? Also please cancel the USB-C dock I ordered last week, I picked the wrong model.");
    ask = intake_request(question);
    account = fetch_account();
    orders = fetch_orders();
    reply = resolve_case(ask, account, orders);
    print("REPLY:", reply);
}
