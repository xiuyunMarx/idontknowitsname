"""Code-generation agent with a real pytest tool gap, as an OSP retry loop on HumanEval.

Graph topology (the call graph of PBKV's SWE-bench workload, plus the loop over
the functions of one module):

    root -> Planner -> Coder -> Tester --pass, more functions--> Planner
                        ^          |   \\--pass, last function--> Committer
                        +-- Analyzer <-- fail (up to CA_MAX_ATTEMPTS per function)

One session implements one module of K HumanEval functions 

Env: CA_TASKS (comma-separated HumanEval ids, or argv[1]; default HumanEval/0..4),
CA_DATA (HumanEval.jsonl.gz next to this file), CA_MAX_ATTEMPTS (3 per function),
CA_OUTPUT_CHARS (4000, cap on the stored pytest output), CA_TEST_TIMEOUT_S (60),
CA_TOOL_DELAY_S (0; floor on every pytest run).
"""

import os;
import sys;
import gzip;
import json;
import time;
import tempfile;
import subprocess;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen3-8B");
glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob DATA: str = os.environ.get("CA_DATA") or os.path.join(HERE, "HumanEval.jsonl.gz");
glob MAX_ATTEMPTS: int = int(os.environ.get("CA_MAX_ATTEMPTS") or "3");
glob OUTPUT_CHARS: int = int(os.environ.get("CA_OUTPUT_CHARS") or "4000");
glob TEST_TIMEOUT_S: float = float(os.environ.get("CA_TEST_TIMEOUT_S") or "60");
glob TOOL_DELAY_S: float = float(os.environ.get("CA_TOOL_DELAY_S") or "0");


obj Plan {
    has approach: str = "";
    has edge_cases: list[str] = [];
}

sem Plan.approach = "The algorithm to implement, in a few sentences.";
sem Plan.edge_cases = "Inputs the implementation must handle explicitly.";


obj Attempt {
    has function: str = "";
    has code: str = "";
    has passed: bool = False;
    has output: str = "";
}

sem Attempt.function = "Name of the function this attempt implements.";
sem Attempt.code = "The function as submitted to the tests.";
sem Attempt.passed = "Whether pytest passed for the whole module including this function.";
sem Attempt.output = "The pytest output, tracebacks included.";


obj Diagnosis {
    has cause: str = "";
    has fix: str = "";
}

sem Diagnosis.cause = "What in the latest attempt makes the tests fail, grounded in the pytest output.";
sem Diagnosis.fix = "Concrete instructions for the next attempt.";


obj CommitMessage {
    has title: str = "";
    has body: str = "";
}

sem CommitMessage.title = "One line, imperative mood, under 72 characters.";
sem CommitMessage.body = "What the module implements and how the failed attempts, if any, were fixed.";


obj Task {
    has task_id: str = "";
    has spec: str = "";
    has entry_point: str = "";
    has tests: str = "";
}


walker CodeTask {
    has tasks: list[Task];
    has idx: int = 0;
    has module: str = "";
    has spec: str = "";
    has plan: Plan | None = None;
    has advice: str = "";
    has attempts: list[Attempt] = [];
    has accepted: list[str] = [];
    has commit: CommitMessage | None = None;
    has errors: list[str] = [];
}

sem CodeTask = "One session implementing a module of several functions.";
sem CodeTask.module = "The accepted code of the module so far.";
sem CodeTask.attempts = "Every attempt so far, oldest first.";


def strip_fences(text: str) -> str {
    parts = text.split("```", 2);
    if len(parts) < 3 {
        return text;
    }
    inner = parts[1];
    if inner.startswith("python") {
        inner = inner[len("python"):];
    }
    return inner.lstrip("\n");
}


node Planner {
    def plan(spec: str, attempts: list[Attempt]) -> Plan by llm(temperature=0.0, max_tokens=4096);

    can start with CodeTask entry {
        visitor.spec = visitor.tasks[visitor.idx].spec;
        visitor.advice = "";
        try {
            visitor.plan = self.plan(visitor.spec, visitor.attempts);
        } except Exception as e {
            visitor.errors.append(f"planning {visitor.tasks[visitor.idx].entry_point} failed: {e}");
            visitor.plan = Plan(approach="Implement the function as specified.");
        }
        visit [self-->][?:Coder];
    }
}

sem Planner = "Reads the next function's spec and decides how to implement it.";
sem Planner.plan = "Describe the algorithm for the function in `spec` and list the edge cases its tests are likely to probe. Do not write code.";
sem Planner.plan.attempts = "Every attempt so far in this module, oldest first.";
sem Planner.plan.spec = "The signature and docstring of the function to implement next.";


node Coder {
    def implement(spec: str, plan: Plan, advice: str, attempts: list[Attempt]) -> str by llm(temperature=0.0, max_tokens=4096);

    can write with CodeTask entry {
        task = visitor.tasks[visitor.idx];
        plan = visitor.plan or Plan();
        try {
            code = strip_fences(self.implement(visitor.spec, plan, visitor.advice, visitor.attempts));
        } except Exception as e {
            visitor.errors.append(f"generating {task.entry_point} failed: {e}");
            code = visitor.spec + "    pass\n";
        }
        # a bare function still needs the spec's imports
        imports = [ln for ln in visitor.spec.split("\n") if ln.startswith("import ") or ln.startswith("from ")];
        if imports and not any(ln in code for ln in imports) {
            code = "\n".join(imports) + "\n\n" + code;
        }
        visitor.attempts.append(Attempt(function=task.entry_point, code=code));
        visit [self-->][?:Tester];
    }
}

sem Coder = "Writes one function of the module.";
sem Coder.implement = "Return the complete function from `spec` as one string: its imports, the full signature with the docstring kept, and the body. Nothing else: no explanation, no tests, no markdown fences, and do not repeat functions from earlier attempts. The attempts marked passed are already in the module and the new function may call them. Follow the plan; if there is advice from a failed attempt, apply it.";
sem Coder.implement.attempts = "Every attempt so far in this module with its pytest output, oldest first.";
sem Coder.implement.spec = "The signature and docstring of the function to implement.";
sem Coder.implement.plan = "The approach and edge cases decided by the planner.";
sem Coder.implement.advice = "Instructions from the analyzer after a failed attempt, or an empty string on the first attempt.";


node Tester {
    def run(visitor: CodeTask, attempt: Attempt) -> None {
        d = tempfile.mkdtemp(prefix="ca-");
        with open(os.path.join(d, "solution.py"), "w") as f {
            f.write(visitor.module + "\n\n" + attempt.code);
        }

        # one test file per function: HumanEval's `check` functions share a name
        for t in visitor.tasks[:visitor.idx + 1] {
            if t.entry_point in visitor.accepted or t.entry_point == attempt.function {
                with open(os.path.join(d, f"test_{t.entry_point}.py"), "w") as f {
                    f.write(t.tests + f"\n\nfrom solution import {t.entry_point}\n\n\ndef test_{t.entry_point}() -> None:\n    check({t.entry_point})\n");
                }
            }
        }
        t0 = time.monotonic();
        try {
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "."],
                cwd=d, capture_output=True, text=True, timeout=TEST_TIMEOUT_S
            );
            attempt.passed = proc.returncode == 0;
            attempt.output = (proc.stdout + proc.stderr)[-OUTPUT_CHARS:];
        } except subprocess.TimeoutExpired {
            attempt.passed = False;
            attempt.output = f"pytest timed out after {TEST_TIMEOUT_S} s";
        }
        remaining = TOOL_DELAY_S - (time.monotonic() - t0);
        if remaining > 0 {
            time.sleep(remaining);
        }
    }
    can test with CodeTask entry {
        attempt = visitor.attempts[-1];
        self.run(visitor, attempt);
        tries = len([a for a in visitor.attempts if a.function == attempt.function]);
        if attempt.passed {
            visitor.module = (visitor.module + "\n\n" + attempt.code).strip("\n") + "\n";
            visitor.accepted.append(attempt.function);
        } elif tries < MAX_ATTEMPTS {
            visit [self-->][?:Analyzer];
            return;
        } else {
            visitor.errors.append(f"{attempt.function}: gave up after {tries} attempts");
        }
        visitor.idx += 1;
        if visitor.idx < len(visitor.tasks) {
            visit [self-->][?:Planner];
        } else {
            visit [self-->][?:Committer];
        }
    }
}

sem Tester = "Runs pytest over the module plus the latest attempt.";


node Analyzer {
    def diagnose(spec: str, attempts: list[Attempt]) -> Diagnosis by llm(temperature=0.0, max_tokens=4096);

    can analyze with CodeTask entry {
        try {
            diag = self.diagnose(visitor.spec, visitor.attempts);
            visitor.advice = diag.fix;
        } except Exception as e {
            visitor.errors.append(f"analysis after attempt {len(visitor.attempts)} failed: {e}");
            visitor.advice = "Read the pytest output and fix the failing assertion.";
        }
        visit [self-->][?:Coder];
    }
}

sem Analyzer = "Explains a failed attempt from its pytest output.";
sem Analyzer.diagnose = "Explain why the latest attempt fails, citing the failing assertion or exception in its pytest output, and give concrete instructions for the next attempt. Do not write the full solution.";
sem Analyzer.diagnose.attempts = "Every attempt so far with its pytest output, oldest first; the last one failed.";
sem Analyzer.diagnose.spec = "The signature and docstring of the function being implemented.";


node Committer {
    def commit(attempts: list[Attempt]) -> CommitMessage by llm(temperature=0.0, max_tokens=4096);

    can finish with CodeTask entry {
        msg: CommitMessage;
        try {
            msg = self.commit(visitor.attempts);
        } except Exception as e {
            visitor.errors.append(f"commit message failed: {e}");
            msg = CommitMessage(title="Implement " + ", ".join(visitor.accepted));
        }
        visitor.commit = msg;
        all_passed = len(visitor.accepted) == len(visitor.tasks);
        print(f"Verdict: {'PASSED' if all_passed else 'FAILED'}");
        print(f"Commit: {msg.title}");
        print(f"Passed: {len(visitor.accepted)}/{len(visitor.tasks)}");
        print(f"Rounds: {len(visitor.attempts)}, attempts: {len(visitor.attempts)}");
        if visitor.errors {
            print("Diagnostics:");
            for error in visitor.errors {
                print(f"- {error}");
            }
        }
    }
}

sem Committer = "Writes the commit message once every function is accepted or given up on.";
sem Committer.commit = "Write a commit message for the module. If some function still fails its tests, say so in the body.";
sem Committer.commit.attempts = "Every attempt with its pytest output, oldest first; the ones marked passed form the module.";


def load_tasks(ids: list[str]) -> list[Task] {
    rows: dict = {};
    with gzip.open(DATA, "rt") as f {
        for line in f {
            row: dict = dict(json.loads(line));
            rows[str(row["task_id"])] = row;
        }
    }
    out: list[Task] = [];
    for i in ids {
        if i not in rows {
            raise KeyError(f"{i} not in {DATA}");
        }
        r: dict = dict(rows[i]);
        out.append(Task(task_id=i, spec=str(r["prompt"]), entry_point=str(r["entry_point"]), tests=str(r["test"])));
    }
    return out;
}


def build_graph() -> Planner {
    existing = [root-->][?:Planner];
    if existing {
        return existing[0];
    }
    planner = root ++> Planner();
    coder = Coder();
    tester = Tester();
    analyzer = Analyzer();
    committer = Committer();
    planner ++> coder;
    coder ++> tester;
    tester ++> planner;                 # next function
    tester ++> committer;
    tester ++> analyzer;
    analyzer ++> coder;                 # the retry loop
    return planner;
}


with entry {
    spec = os.environ.get("CA_TASKS") or (sys.argv[1] if len(sys.argv) > 1 else ",".join([f"HumanEval/{i}" for i in range(5)]));
    ids = [s.strip() for s in spec.split(",") if s.strip()];
    CodeTask(tasks=load_tasks(ids)) spawn build_graph();
}
