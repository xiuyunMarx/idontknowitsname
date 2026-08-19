"""Full-mechanism showcase: a five-stage corporate travel-desk agent whose shape
is chosen to exercise EVERY proactive-prefill tier at once.

SPATIAL — the calls form a deep chain (profile -> scout -> itinerary -> budget
-> brief), so distance-layered idle warming is visible: during the first call's
decode + tool window the drainer walks ring 1, ring 2, ... ring 4 outward, and
the typed final call's grammar is compiled four calls ahead of its execution.

TEMPORAL — every binding is the ret of an upstream byllm call (dataflow
provenance), and every call is followed by a real tool latency (CRM lookup,
fare search, availability check, policy check; simulated ~450 ms). Those tool
windows are exactly the idle windows the scheduler admits speculation into —
including the ret feed of the LAST producer (budget_review -> final_brief),
which is structurally unwinnable in a zero-think-time flow.

Long sems make every invariant expensive (the thing worth saving on a 3B-class
model), and long producer outputs make final_brief's binding zone ~1.4k tokens
(the thing tier-3 feeding saves)."""

import from time { sleep }
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct");

enum Confidence { LOW = "LOW", MEDIUM = "MEDIUM", HIGH = "HIGH" }

obj Brief {
    has destination: str;
    has summary: str;
    has total_cost_usd: float;
    has confidence: Confidence;
}

sem Brief.destination = "IATA city or airport code of the chosen destination";
sem Brief.summary = "at most four sentences naming the chosen flights, the hotel, and how the per-person budget constraint is met";
sem Brief.total_cost_usd = "estimated total trip cost per person in US dollars";
sem Brief.confidence = "confidence that every hard constraint in the traveler profile is satisfied by the briefed trip";

def profile_traveler(request: str) -> str by llm(temperature=0.0, max_tokens=520);
sem profile_traveler = "You are the intake specialist of a corporate travel desk. Read the traveler's free-text request and produce a structured traveler profile in plain prose covering, in this order: party size and traveler roles, origin and destination cities, exact travel dates or date window, purpose of the trip and every fixed appointment, per-person and total budget with currency, cabin and hotel preferences, dietary or accessibility needs, loyalty programs mentioned, visa or passport constraints, and any hard constraints that must never be violated. Write at least 14 sentences, each at least 12 words, numbered 1. through 14., expanding every implicit requirement into an explicit testable statement. Do not invent facts that are not present or clearly implied; where the request is silent, state the default corporate policy assumption instead and mark it as an assumption. Plain factual prose, no markdown, no headers.";

def scout_destinations(profile: str) -> str by llm(temperature=0.0, max_tokens=640);
sem scout_destinations = "You are the destination and routing scout of a corporate travel desk. Given the structured traveler profile, enumerate concrete flight and lodging options that satisfy it. Write at least 16 sentences, each at least 12 words, numbered 1. through 16. Cover: at least two realistic outbound flight options with airline name, flight number style, plausible departure and arrival times and layover structure; at least two return options in the same detail; at least three hotel options with the neighborhood, star class, plausible nightly rate in US dollars, and walking distance to the stated meeting venue; ground transport between airport, hotel and venue with typical duration and fare; and one backup option in case the preferred route sells out. Every option must respect the profile's dates, budget ceiling and stated preferences. Plain factual prose, no markdown, no headers.";

def draft_itinerary(profile: str, options: str) -> str by llm(temperature=0.0, max_tokens=680);
sem draft_itinerary = "You are the itinerary planner of a corporate travel desk. Merge the traveler profile and the scouted options into one concrete day-by-day itinerary for the whole trip. Write at least 18 sentences, each at least 12 words, numbered 1. through 18. For every day cover morning, afternoon and evening: the chosen flight or transport segment with times, hotel check-in and check-out, each fixed appointment with at least thirty minutes of buffer before it, meal plans that respect every stated dietary need, and the one free-time block the profile asks for. Choose exactly one flight and one hotel from the scouted options and use them consistently throughout; never leave a choice open or offer alternatives. State the running day count explicitly at the start of each day's block. Plain factual prose, no markdown, no headers.";

def budget_review(itinerary: str) -> str by llm(temperature=0.0, max_tokens=560);
sem budget_review = "You are the cost controller of a corporate travel desk. Audit the day-by-day itinerary line by line against the traveler's stated budget ceiling. Write at least 14 sentences, each at least 12 words, numbered 1. through 14. Itemize in prose: round-trip airfare per person, hotel total for all nights, ground transport, meals per day, incidentals and taxes; give each line a US-dollar amount, keep an explicit running total, state the final variance against the per-person budget ceiling as a signed dollar amount, name the two largest saving opportunities with their estimated saving, and end with a single clear verdict sentence saying whether the itinerary is within budget as planned. Plain factual prose, no markdown, no headers.";

def final_brief(profile: str, itinerary: str, budget: str) -> Brief by llm(temperature=0.0, max_tokens=220);
sem final_brief = "You are the senior travel consultant signing off the trip. Given the traveler profile, the day-by-day itinerary and the cost controller's budget review, produce the final booking brief for the approval system: pick the destination, summarize the chosen flights, hotel and total cost, state the total per-person cost in US dollars as a plain number taken from the budget review's running total, and rate your confidence that the briefed trip satisfies every hard constraint in the profile.";

def crm_lookup() { sleep(0.45); }
def fare_search() { sleep(0.45); }
def availability_check() { sleep(0.45); }
def policy_check() { sleep(0.45); }

with entry:__main__ {
    request = "Two colleagues and I need a five-day business trip from Singapore to Tokyo in late November: three client meetings in Shinjuku on separate mornings, one free afternoon for the team, a strict corporate budget of 2600 USD per person covering flights and hotel, one traveler is vegetarian, and we prefer a hotel within walking distance of the meeting venue.";
    profile = profile_traveler(request);
    crm_lookup();
    options = scout_destinations(profile);
    fare_search();
    itinerary = draft_itinerary(profile, options);
    availability_check();
    budget = budget_review(itinerary);
    policy_check();
    brief = final_brief(profile, itinerary, budget);
    print("BRIEF:", brief);
}
