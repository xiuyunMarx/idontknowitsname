"""Small-to-large answer cascade (FrugalGPT style) with a probabilistic escalation.

    answer_draft[2.5-1.5B] -> judge_draft[Qwen3-0.6B] -> (accept | expert_answer[Qwen3-4B])

Per-request model sequence: S J, or S J B when the judge escalates. Escalation
is data-dependent, so the third model is a predicted branch with a rate the
scheduler can learn per program. $CASCADE_FORCE=1 escalates every request and
$CASCADE_FORCE=0 never does, for controlled sequences. Inputs: $QUESTION or
$CASCADE_INDEX."""

import os;
import time;
import from jaclang.byllm.llm { InterceptorLLM }

glob TOOL_SCALE = float(os.environ.get("TOOL_DELAY_SCALE") or "1");
glob PORT = int(os.environ.get("SERVE_PORT") or "8964");

glob drafter = InterceptorLLM(model_name="Qwen/Qwen2.5-1.5B-Instruct", comm_ip="localhost", comm_port=PORT, program_name="cascade");
glob judge = InterceptorLLM(model_name="Qwen/Qwen3-0.6B", comm_ip="localhost", comm_port=PORT, program_name="cascade");
glob expert = InterceptorLLM(model_name="Qwen/Qwen3-4B", comm_ip="localhost", comm_port=PORT, program_name="cascade");

glob FORCE: str = str(os.environ.get("CASCADE_FORCE") or "");

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

def pick_question() -> str {
    direct = str(os.environ.get("QUESTION") or "");
    if direct {
        return direct;
    }
    return QUESTIONS[int(os.environ.get("CASCADE_INDEX") or "0") % len(QUESTIONS)];
}

def should_escalate(verdict: str) -> bool {
    if FORCE == "1" {
        return True;
    }
    if FORCE == "0" {
        return False;
    }
    return "ESCALATE" in verdict.upper();
}


def answer_draft(question: str) -> str by drafter(
    temperature=0.0,
    max_tokens=200
);

def judge_draft(question: str, draft: str) -> str by judge(
    temperature=0.0,
    max_tokens=24
);

def expert_answer(question: str, draft: str) -> str by expert(
    temperature=0.0,
    max_tokens=320
);

sem answer_draft = "Answer the question directly and briefly.";
sem answer_draft.question = "the user's question";
sem judge_draft = "Decide whether the draft answers the question correctly and completely. Reply with exactly ACCEPT when it does, or ESCALATE when it is wrong, incomplete or uncertain.";
sem judge_draft.question = "the user's question";
sem judge_draft.draft = "the draft answer under review";
sem expert_answer = "Give the correct, complete answer to the question, fixing whatever the draft got wrong, with the reasoning shown where the question calls for it.";
sem expert_answer.question = "the user's question";
sem expert_answer.draft = "a first draft that was judged insufficient";


walker CascadeAgent {
    has question: str = "";
    has draft: str = "";
    has verdict: str = "";
    has answer: str = "";
    has escalated: bool = False;

    can solve with Root entry {
        self.draft = answer_draft(self.question);
        time.sleep(0.05 * TOOL_SCALE);
        self.verdict = judge_draft(self.question, self.draft);
        if should_escalate(self.verdict) {
            self.escalated = True;
            time.sleep(0.05 * TOOL_SCALE);
            self.answer = expert_answer(self.question, self.draft);
        } else {
            self.answer = self.draft;
        }
    }

    can report with exit {
        print("QUESTION:", self.question);
        print("DRAFT:", self.draft);
        print("VERDICT:", self.verdict, "| escalated:", self.escalated);
        print("ANSWER:", self.answer);
    }
}


with entry:__main__ {
    CascadeAgent(question=pick_question()) spawn root;
}
