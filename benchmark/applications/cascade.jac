"""Small-to-large answer cascade (FrugalGPT style) with a probabilistic escalation.

answer_draft[2.5-1.5B] -> judge_draft[Qwen3-0.6B] -> (accept | expert_answer[Qwen3-4B])."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "0");
glob PORT = 8964;

glob drafter = InterceptorLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="cascade");
glob judge = InterceptorLLM(model_name="Qwen/Qwen3-0.6B", comm_ip="localhost", comm_port=PORT, program_name="cascade");
glob expert = InterceptorLLM(model_name="Qwen/Qwen3-4B", comm_ip="localhost", comm_port=PORT, program_name="cascade");


glob QUESTIONS: list[str] = [
    "What is the capital of Australia?",
    "How many minutes are there in a week?",
    "Which planet in the solar system has the most moons?",
    "Explain why the sky is blue in two sentences.",
    "A train leaves at 9:40 and arrives at 13:05. How long is the journey?",
    "Write a haiku about a cold server room.",
    "If three painters paint three walls in three hours, how long do nine painters need for nine walls?",
    "What is the difference between a process and a thread?",
    "A bat and a ball cost 1.10 in total, and the bat costs 1.00 more than the ball. How much is the ball?",
    "Translate 'the meeting moved to Thursday' into French.",
    "Prove that the sum of two even integers is even.",
    "What year did the Berlin Wall fall, and which two countries reunified the following year?",
    "Give a one-line shell command that counts the lines in every .py file under the current directory.",
    "Is 221 a prime number? Show the factorization if not.",
    "Summarise the plot of Romeo and Juliet in one sentence.",
    "Which is larger, 2^10 or 10^3, and by how much?"
];

node drafter_node{
    def draft_answer(question: str) -> str by drafter();
}

node verify_node{
    def verify_answer(question: str, answer: str) -> bool by judge();
}