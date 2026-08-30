"""Jac client smoke: one by-llm call with a tool through InterceptorLLM against tests/canned_server.py."""
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="fake-model", comm_ip="localhost", comm_port=8970, program_name="smoke_client");

def get_count() -> int {
    return 7;
}

def describe(topic: str) -> str by llm(tools=[get_count], temperature=0.0, max_tokens=64);

sem get_count = "how many items there are";
sem describe = "Say how many items topic has";
sem describe.topic = "the thing to count";

with entry {
    print("RESULT:", describe("apples"));
}
