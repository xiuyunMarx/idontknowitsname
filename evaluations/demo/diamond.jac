"""Dataflow-firing fixture: h consumes BOTH f's and g's results. Correct tier-3
behavior feeds h.a the moment f completes (while g is still running) — a purely
topology-adjacent feeder can never bind h.a."""

import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct");

def f(x: str) -> str by llm(temperature=0.0, max_tokens=48);
sem f = "Summarize the customer complaint in one factual sentence without speculation.";

def g(x: str) -> str by llm(temperature=0.0, max_tokens=48);
sem g = "List the product areas mentioned in the report as a short comma-separated line.";

def h(a: str, b: str) -> str by llm(temperature=0.0, max_tokens=64);
sem h = "Combine the complaint summary and the affected product areas into a one-paragraph incident ticket.";

with entry:__main__ {
    a = f("My GPU catches fire every time I run the training job!!");
    b = g("The crash log mentions the CUDA driver, the cooling fans, and the power supply.");
    ans = h(a, b);
    print("TICKET:", ans);
}
