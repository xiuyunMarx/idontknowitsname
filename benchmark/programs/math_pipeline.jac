import os;
import sys;
import from jaclang.byllm.llm { InterceptorLLM }
glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B", call_params={"temperature": 0.0});

obj MathDesk{
    def solve(problem: str) -> str by llm();
    def verify(problem: str, solution: str) -> str by llm();
    def format_answer(problem: str, solution: str, verdict: str) -> str by llm();
}
sem MathDesk = "A desk that solves one word problem, checks it, and reports the result.";
sem MathDesk.solve = "Solve the word problem step by step and end with the numeric answer.";
sem MathDesk.verify = "Recompute the problem independently; reply CORRECT, or WRONG followed by the right number.";
sem MathDesk.format_answer = "One sentence with the final number, honoring the verdict over the solution when they disagree.";

with entry {
    problem = os.environ.get("MP_QUESTION")
        or (sys.argv[1] if len(sys.argv) > 1 else
            "A warehouse ships 14 boxes of 12 items each; 27 items are returned. How many items stay shipped?");

    desk = MathDesk();
    solution = desk.solve(problem);
    verdict = desk.verify(problem, solution);
    print(desk.format_answer(problem, solution, verdict));
}
