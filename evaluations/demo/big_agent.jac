"""Tier-3 showcase: three independent long-output producers fan into one consumer,
so the consumer's binding zone is ~1.1k tokens — large enough that feeding it
speculatively must show up in TTFT beyond noise. Also the scaled-up diamond:
final_verdict.timeline must be fed the moment report_timeline ends (two calls
before final_verdict runs), which only dataflow firing can do."""

import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen2.5-3B-Instruct");

def report_timeline(q: str) -> str by llm(temperature=0.0, max_tokens=680);
sem report_timeline = "Write a detailed chronological reconstruction (at least 22 sentences, each sentence at least 15 words, numbered 1. through 22.) of the incident described: when the job started, when the first anomaly appeared, escalation points, and when the failure became terminal. Plain factual prose, no speculation about causes.";

def report_hardware(q: str) -> str by llm(temperature=0.0, max_tokens=680);
sem report_hardware = "Write a detailed hardware inspection report (at least 22 sentences, each sentence at least 15 words, numbered 1. through 22.) for the incident described: observed state of the GPU, thermal readings, fan behavior, power delivery, cabling, and any physical damage. Plain factual prose, no speculation about causes.";

def report_environment(q: str) -> str by llm(temperature=0.0, max_tokens=680);
sem report_environment = "Write a detailed operating-environment report (at least 22 sentences, each sentence at least 15 words, numbered 1. through 22.) for the incident described: room temperature, airflow, rack placement, dust conditions, recent maintenance, and concurrent workloads on the machine. Plain factual prose, no speculation about causes.";

def final_verdict(timeline: str, hardware: str, environment: str) -> str by llm(temperature=0.0, max_tokens=96);
sem final_verdict = "Given the three investigator reports, produce a single-paragraph root-cause verdict: name the most likely cause, cite the two strongest supporting observations across the reports, and recommend two corrective actions.";

with entry:__main__ {
    incident = "During a multi-day training run the GPU began emitting a burning smell around hour six, the fans spun to maximum, thermal alarms fired, and the machine shut down after visible smoke appeared from the rear of the chassis.";
    t = report_timeline(incident);
    hw = report_hardware(incident);
    env = report_environment(incident);
    verdict = final_verdict(t, hw, env);
    print("VERDICT:", verdict);
}
