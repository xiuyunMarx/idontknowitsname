import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(
    model_name=str(os.environ.get("MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"),
    comm_ip="localhost",
    comm_port=8965,
    program_name="operations"
);

def fetch_metric(metric: str) -> str {
    time.sleep(0.30);
    metrics = {
        "requests": "The service handled 12,500 requests.",
        "errors": "The service recorded 125 failed requests.",
        "latency": "The service p95 latency was 240 milliseconds."
    };
    return str(metrics.get(metric.lower(), "Metric unavailable."));
}

def check_threshold(metric: str, value: float) -> str {
    time.sleep(0.20);
    return f"Threshold check recorded for {metric}: {value}";
}

sem fetch_metric = "Fetch one current service metric from the operations store.";
sem fetch_metric.metric = "one of requests, errors, or latency";
sem check_threshold = "Record and validate one calculated operational value.";
sem check_threshold.metric = "the name of the calculated metric";
sem check_threshold.value = "the calculated numeric value";

def collect_metrics(request: str) -> str by llm(
    tools=[fetch_metric],
    temperature=0.0,
    max_tokens=320,
    max_react_iterations=4
);

def analyze_metrics(request: str, observations: str) -> str by llm(
    tools=[check_threshold],
    temperature=0.0,
    max_tokens=320,
    max_react_iterations=3
);

def write_report(request: str, analysis: str, audience: str = "SRE lead") -> str by llm(
    temperature=0.0,
    max_tokens=240
);

sem collect_metrics = "Use fetch_metric to collect every service metric required by the request, then finish with the observations and their exact values.";
sem collect_metrics.request = "the operational analysis request";
sem analyze_metrics = "Calculate the requested operational result from the observations, call check_threshold with that result, and finish with a short analysis.";
sem analyze_metrics.request = "the original operational request";
sem analyze_metrics.observations = "the metrics collected from the operations store";
sem write_report = "Write a concise operations report using only the supplied analysis.";
sem write_report.request = "the original operational request";
sem write_report.analysis = "the completed metric analysis";
sem write_report.audience = "the report's intended reader";

with entry:__main__ {
    request = str(os.environ.get("REQUEST") or "Calculate the error rate and report p95 latency.");
    observations = collect_metrics(request);
    time.sleep(0.20);
    analysis = analyze_metrics(request, observations);
    time.sleep(0.20);
    final_report = write_report(request, analysis, "SRE lead");
    print("REPORT:", final_report);
}
