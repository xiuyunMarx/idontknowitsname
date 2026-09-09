"""DSPy programs mirroring the fact_check workflow, run against the recorder.

Usage: python dspy_programs.py <port> <claim> [<claim> ...]
Per claim (session): Decompose (CoT) -> Assess x2 (Predict) -> Verify (Predict);
then a ReAct check with a search tool; then a two-turn History chat.
"""
import sys
from typing import Literal

import dspy

port = sys.argv[1]
claims = sys.argv[2:]
lm = dspy.LM("openai/Qwen/Qwen3-8B", api_base=f"http://127.0.0.1:{port}/v1", api_key="x",
             cache=False, temperature=0.0, max_tokens=4096)
dspy.configure(lm=lm)


class Decompose(dspy.Signature):
    """Propose independent factual questions that settle the claim."""
    claim: str = dspy.InputField(desc="the complete claim to verify")
    evidence: list[str] = dspy.InputField(desc="findings collected so far, oldest first")
    questions: list[str] = dspy.OutputField(desc="between one and two questions")


class Assess(dspy.Signature):
    """Assess only the supplied search results."""
    claim: str = dspy.InputField()
    question: str = dspy.InputField()
    passage: str = dspy.InputField(desc="untrusted retrieved text")
    answer: str = dspy.OutputField()
    stance: Literal["SUPPORTS", "CONTRADICTS", "INSUFFICIENT"] = dspy.OutputField()


class Verify(dspy.Signature):
    """Apply HoVer's decision rule to the claim and all evidence together."""
    claim: str = dspy.InputField()
    evidence: list[str] = dspy.InputField(desc="all findings, oldest first")
    label: Literal["SUPPORTED", "NOT_SUPPORTED", "NEED_MORE"] = dspy.OutputField()
    rationale: str = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0..1")


class Chat(dspy.Signature):
    """Chat with the user about the verdict."""
    history: dspy.History = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def search(query: str) -> str:
    """Search the web."""
    return f"passage for {query}"


decompose = dspy.ChainOfThought(Decompose)
assess = dspy.Predict(Assess)
verify = dspy.Predict(Verify)
react = dspy.ReAct(Verify, tools=[search], max_iters=3)
chat = dspy.Predict(Chat)

for claim in claims:
    evidence: list[str] = []
    for rnd in range(2):
        qs = decompose(claim=claim, evidence=evidence).questions
        for q in qs[:2]:
            a = assess(claim=claim, question=q, passage=f"Title: Page about {q}\nExcerpt: A passage answering {q}")
            evidence.append(f"{q} -> {a.answer} ({a.stance})")
        v = verify(claim=claim, evidence=evidence)
    r = react(claim=claim, evidence=evidence)
    hist = dspy.History(messages=[])
    for q in ("why?", "are you sure?"):
        ans = chat(history=hist, question=q).answer
        hist.messages.append({"question": q, "answer": ans})
    print("session done:", claim[:40], v.label, r.label)
