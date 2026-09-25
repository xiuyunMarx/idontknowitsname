"""SWE-bench agent: one session resolves one SWE-bench issue inside the instance's
docker image by calling one repository function per turn.

    root -> Engineer --call--> Sandbox --output--> Engineer ... --submit--> patch

Every function the model may call is an object below: its fields are the call's
arguments (each function has its own field names, so a reply parses to exactly
one of them) and its `execute` runs the call in the container. Each turn the
Engineer returns one `Turn` (a thought and one call); the Sandbox runs the call and
appends the exchange to `trace`, which every later turn reads in full. There is no
history processing: outputs are capped once, when they are recorded.

The repository is at /testbed in the image, the working directory of every call.
At the end `git add -A && git diff --cached` is the patch, written in the SWE-bench
predictions format for offline grading with the swebench harness.

Env: SB_TASK (instance id, or argv[1]; default django__django-11099),
SB_DATA (bench_data/SWE-bench/swe_bench_lite.jsonl, one SWE-bench row per line),
SB_IMAGE (image name template for rows without an `image` field, {key} = instance id with
"__" -> "_1776_", lower case), SB_SHELL_INIT (prefix of every shell command; activates the
testbed env the way the rows' eval_script does),
SB_MAX_TURNS (30), SB_OUTPUT_CHARS (6000, cap on one recorded output, head + tail),
SB_COMMAND_TIMEOUT_S (120), SB_TEST_TIMEOUT_S (600), SB_PATCH_DIR (where predictions go).
"""

import os;
import re;
import sys;
import json;
import time;
import shlex;
import hashlib;
import subprocess;
import from jaclang.byllm.llm { InterceptorLLM }

glob llm = InterceptorLLM(model_name="Qwen/Qwen3-14B-AWQ");
glob HERE: str = os.path.dirname(os.path.abspath(__file__));
glob DATA: str = os.environ.get("SB_DATA") or os.path.join(HERE, "bench_data", "SWE-bench", "swe_bench_lite.jsonl");
glob IMAGE: str = os.environ.get("SB_IMAGE") or "swebench/sweb.eval.x86_64.{key}:latest";
glob SHELL_INIT: str = os.environ.get("SB_SHELL_INIT") or "source /opt/miniconda3/bin/activate >/dev/null 2>&1 && conda activate testbed >/dev/null 2>&1; ";
glob MAX_TURNS: int = int(os.environ.get("SB_MAX_TURNS") or "30");
glob OUTPUT_CHARS: int = int(os.environ.get("SB_OUTPUT_CHARS") or "6000");
glob COMMAND_TIMEOUT_S: float = float(os.environ.get("SB_COMMAND_TIMEOUT_S") or "120");
glob TEST_TIMEOUT_S: float = float(os.environ.get("SB_TEST_TIMEOUT_S") or "600");
glob PATCH_DIR: str = os.environ.get("SB_PATCH_DIR") or os.path.join(HERE, "..", "mixed_results", "swe_bench_patches");
glob WORKDIR: str = "/testbed";
glob VIEW_LINES: int = 200;        # longest window view_file shows
glob LISTING_LINES: int = 300;     # longest listing a find or grep call returns
glob EDIT_CONTEXT: int = 4;        # lines shown around an edit


# ------------------------------------------------------------------ container

obj Container {
    has name: str;
    has image: str;

    def start {
        done = subprocess.run(
            ["docker", "run", "-d", "--rm", "--name", self.name, "-w", WORKDIR, self.image, "sleep", "infinity"],
            capture_output=True, text=True
        );
        if done.returncode != 0 {
            raise RuntimeError(f"docker run {self.image} failed: {done.stderr.strip()}");
        }
    }

    def stop {
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True, text=True);
    }

    """Exit code and merged stdout/stderr of a shell command; -1 on timeout."""
    def capture(command: str, timeout: float) -> tuple[int, str] {
        try {
            done = subprocess.run(
                ["docker", "exec", "-w", WORKDIR, self.name, "bash", "-lc", SHELL_INIT + command],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", timeout=timeout
            );
        } except subprocess.TimeoutExpired {
            return (-1, f"[timed out after {int(timeout)} s]");
        }
        return (done.returncode, done.stdout);
    }

    def run(command: str, timeout: float) -> str {
        (code, out) = self.capture(command, timeout);
        return f"(exit code {code})\n{out}" if out.strip() else f"(exit code {code}, no output)";
    }

    """A file's text, or None when it cannot be read. Bytes that are not UTF-8 survive a read and write."""
    def read(path: str) -> str | None {
        done = subprocess.run(
            ["docker", "exec", "-w", WORKDIR, self.name, "cat", path],
            capture_output=True, text=True, encoding="utf-8", errors="surrogateescape"
        );
        return done.stdout if done.returncode == 0 else None;
    }

    def write(path: str, content: str) -> bool {
        target = shlex.quote(path);
        done = subprocess.run(
            ["docker", "exec", "-i", "-w", WORKDIR, self.name, "bash", "-c", f"mkdir -p \"$(dirname {target})\" && cat > {target}"],
            input=content, capture_output=True, text=True, encoding="utf-8", errors="surrogateescape"
        );
        return done.returncode == 0;
    }
}


def numbered(lines: list[str], first: int, last: int) -> str {
    return "\n".join([f"{i:6d}  {lines[i - 1]}" for i in range(max(1, first), min(len(lines), last) + 1)]);
}

def clip(text: str, limit: int) -> str {
    if len(text) <= limit {
        return text;
    }
    head = limit * 2 // 3;
    tail = limit - head;
    return f"{text[:head]}\n[... {len(text) - limit} characters omitted ...]\n{text[-tail:]}";
}


# ------------------------------------------------------------------ functions

obj ListDirectory {
    has directory: str;
    has depth: int;

    def execute(box: Container) -> str {
        depth = min(max(self.depth, 1), 3);
        return box.run(
            f"find {shlex.quote(self.directory or '.')} -maxdepth {depth} -not -path '*/.git' -not -path '*/.git/*' | sort | head -n {LISTING_LINES}",
            COMMAND_TIMEOUT_S
        );
    }
}

sem ListDirectory = "List the files and folders under a directory of the repository.";
sem ListDirectory.directory = "Directory relative to the repository root; '.' for the root.";
sem ListDirectory.depth = "How many levels to descend, 1 to 3.";


obj FindFiles {
    has filename_pattern: str;
    has start_dir: str;

    def execute(box: Container) -> str {
        return box.run(
            f"find {shlex.quote(self.start_dir or '.')} -name {shlex.quote(self.filename_pattern)} -not -path '*/.git/*' | sort | head -n {LISTING_LINES}",
            COMMAND_TIMEOUT_S
        );
    }
}

sem FindFiles = "Find files whose name matches a shell glob.";
sem FindFiles.filename_pattern = "Shell glob on the file name, e.g. '*validators*.py'.";
sem FindFiles.start_dir = "Directory to search under, relative to the repository root.";


obj SearchCode {
    has text_pattern: str;
    has search_dir: str;

    def execute(box: Container) -> str {
        return box.run(
            f"grep -rnIE --exclude-dir=.git -e {shlex.quote(self.text_pattern)} {shlex.quote(self.search_dir or '.')} | head -n {LISTING_LINES}",
            COMMAND_TIMEOUT_S
        );
    }
}

sem SearchCode = "Search file contents for a regular expression; returns file:line:text matches.";
sem SearchCode.text_pattern = "Extended regular expression (grep -E).";
sem SearchCode.search_dir = "Directory or file to search, relative to the repository root.";


obj ViewFile {
    has file: str;
    has first_line: int;
    has last_line: int;

    def execute(box: Container) -> str {
        text = box.read(self.file);
        if text is None {
            return f"{self.file} does not exist or is not readable.";
        }
        lines = text.splitlines();
        if self.first_line > len(lines) {
            return f"{self.file} has only {len(lines)} lines; line {self.first_line} is past its end.";
        }
        first = max(1, self.first_line);
        last = min(max(first, self.last_line), first + VIEW_LINES - 1, len(lines));
        return f"{self.file}: lines {first}-{last} of {len(lines)}\n" + numbered(lines, first, last);
    }
}

sem ViewFile = "Show a window of a file with line numbers.";
sem ViewFile.file = "File path relative to the repository root.";
sem ViewFile.first_line = "First line to show, 1-based.";
sem ViewFile.last_line = "Last line to show; at most 200 lines are shown per call.";


obj ReplaceText {
    has edit_file: str;
    has old_text: str;
    has new_text: str;

    def execute(box: Container) -> str {
        text = box.read(self.edit_file);
        if text is None {
            return f"{self.edit_file} does not exist or is not readable.";
        }
        found = text.count(self.old_text) if self.old_text else 0;
        if found == 0 {
            return f"old_text does not occur in {self.edit_file}; view the file and copy the lines exactly, indentation included.";
        }
        if found > 1 {
            return f"old_text occurs {found} times in {self.edit_file}; include more surrounding lines so it occurs once.";
        }
        at = text.index(self.old_text);
        updated = text[:at] + self.new_text + text[at + len(self.old_text):];
        if not box.write(self.edit_file, updated) {
            return f"{self.edit_file} could not be written.";
        }
        line = text[:at].count("\n") + 1;
        span = self.new_text.count("\n");
        return f"{self.edit_file} edited; it now reads:\n" + numbered(updated.splitlines(), line - EDIT_CONTEXT, line + span + EDIT_CONTEXT);
    }
}

sem ReplaceText = "Replace one exact occurrence of a piece of text in a file.";
sem ReplaceText.edit_file = "File path relative to the repository root.";
sem ReplaceText.old_text = "The exact text to replace, whole lines with their indentation; it must occur exactly once.";
sem ReplaceText.new_text = "The text to put in its place.";


obj InsertLines {
    has insert_file: str;
    has after_line: int;
    has lines: str;

    def execute(box: Container) -> str {
        text = box.read(self.insert_file);
        if text is None {
            return f"{self.insert_file} does not exist or is not readable.";
        }
        rows = text.splitlines(keepends=True);
        at = min(max(self.after_line, 0), len(rows));
        block = self.lines if self.lines.endswith("\n") else self.lines + "\n";
        updated = "".join(rows[:at]) + block + "".join(rows[at:]);
        if not box.write(self.insert_file, updated) {
            return f"{self.insert_file} could not be written.";
        }
        return f"{self.insert_file} edited; it now reads:\n" + numbered(updated.splitlines(), at + 1 - EDIT_CONTEXT, at + block.count("\n") + EDIT_CONTEXT);
    }
}

sem InsertLines = "Insert lines into a file after a given line.";
sem InsertLines.insert_file = "File path relative to the repository root.";
sem InsertLines.after_line = "Insert after this line number; 0 inserts at the top.";
sem InsertLines.lines = "The lines to insert, with their indentation.";


obj CreateFile {
    has new_file: str;
    has content: str;

    def execute(box: Container) -> str {
        if box.read(self.new_file) is not None {
            return f"{self.new_file} already exists; edit it with replace_text or insert_lines.";
        }
        return f"{self.new_file} created." if box.write(self.new_file, self.content) else f"{self.new_file} could not be written.";
    }
}

sem CreateFile = "Create a new file in the repository.";
sem CreateFile.new_file = "Path of the new file, relative to the repository root.";
sem CreateFile.content = "The whole content of the file.";


obj RunPython {
    has script: str;

    def execute(box: Container) -> str {
        path = f"/tmp/run_{hashlib.md5(self.script.encode()).hexdigest()[:12]}.py";
        if not box.write(path, self.script) {
            return "The script could not be written.";
        }
        return box.run(f"python {path}", COMMAND_TIMEOUT_S);
    }
}

sem RunPython = "Run a Python script in the repository's environment, e.g. to reproduce the issue; the script is not added to the repository.";
sem RunPython.script = "The whole script.";


obj RunTests {
    has test_ids: str;

    def execute(box: Container) -> str {
        return box.run(f"python -m pytest -q -p no:cacheprovider {self.test_ids}", TEST_TIMEOUT_S);
    }
}

sem RunTests = "Run tests with pytest. Repositories with their own test runner (Django: tests/runtests.py) need run_command instead.";
sem RunTests.test_ids = "Test files or pytest node ids, separated by spaces.";


obj RunCommand {
    has command: str;

    def execute(box: Container) -> str {
        return box.run(self.command, COMMAND_TIMEOUT_S);
    }
}

sem RunCommand = "Run a shell command at the repository root.";
sem RunCommand.command = "The bash command line.";


obj Submit {
    has summary: str;
}

sem Submit = "Finish: the current state of the repository is the fix.";
sem Submit.summary = "What was changed and why, in one or two sentences.";


# ------------------------------------------------------------------ turns

obj Turn {
    has thought: str;
    has call: ListDirectory | FindFiles | SearchCode | ViewFile | ReplaceText | InsertLines | CreateFile | RunPython | RunTests | RunCommand | Submit;
}

sem Turn.thought = "What the previous outputs showed and what this call is for.";
sem Turn.call = "The one function to call now, with its arguments.";


obj Exchange {
    has thought: str;
    has call: ListDirectory | FindFiles | SearchCode | ViewFile | ReplaceText | InsertLines | CreateFile | RunPython | RunTests | RunCommand | Submit;
    has output: str;
}

sem Exchange = "One earlier turn.";
sem Exchange.thought = "Why the call was made.";
sem Exchange.call = "The function called, with its arguments.";
sem Exchange.output = "What the call returned.";


walker Resolve {
    has instance_id: str;
    has repo: str;
    has issue: str;
    has box: Container;
    has trace: list[Exchange] = [];
    has pending: Turn | None = None;
    has turns: int = 0;
    has submitted: bool = False;
    has summary: str = "";
    has tool_seconds: float = 0.0;
    has errors: list[str] = [];
}

sem Resolve = "One session resolving one SWE-bench issue.";
sem Resolve.issue = "The issue text, verbatim.";
sem Resolve.trace = "Every call made so far with its output, oldest first.";


node Engineer {
    def next_call(issue: str, repo: str, trace: list[Exchange]) -> Turn by llm(temperature=0.0, max_tokens=4096);

    can decide with Resolve entry {
        try {
            turn = self.next_call(visitor.issue, visitor.repo, visitor.trace);
        } except Exception as e {
            visitor.errors.append(f"turn {visitor.turns + 1}: no call was produced: {e}");
            return;
        }
        visitor.turns += 1;
        visitor.pending = turn;
        visit [self-->][?:Sandbox];
    }
}

sem Engineer = "Decides the next function call from the issue and the trace.";
sem Engineer.next_call = "Resolve `issue` in the repository checked out at /testbed by calling one function per turn. Find the code involved, reproduce the problem with run_python, change the source (not the tests), check the change with the reproduction and the relevant tests, then call submit. `trace` holds every earlier call with its output; do not repeat a call whose output is already there. Paths are relative to the repository root. Reply with a single JSON object that matches the response schema exactly, and nothing else: no prose, no markdown fences.";
sem Engineer.next_call.issue = "The GitHub issue to resolve.";
sem Engineer.next_call.repo = "The repository, as owner/name.";
sem Engineer.next_call.trace = "The calls made so far with their outputs, oldest first.";


node Sandbox {
    can run_call with Resolve entry {
        turn = visitor.pending;
        if turn is None {
            return;
        }
        if isinstance(turn.call, Submit) {
            visitor.submitted = True;
            visitor.summary = turn.call.summary;
            return;
        }
        started = time.monotonic();
        try {
            output = turn.call.execute(visitor.box);
        } except Exception as e {
            output = f"The call failed: {e}";
        }
        visitor.tool_seconds += time.monotonic() - started;
        visitor.trace.append(Exchange(thought=turn.thought, call=turn.call, output=clip(output, OUTPUT_CHARS)));
        print(f"[turn {visitor.turns}] {round(time.monotonic() - started, 2)} s, {len(output)} chars: {str(turn.call)[:300]}", flush=True);
        if visitor.turns >= MAX_TURNS {
            visitor.errors.append(f"stopped at the {MAX_TURNS}-turn limit");
            return;
        }
        visit [self-->][?:Engineer];
    }
}

sem Sandbox = "Runs the chosen call in the instance's container and records its output.";


# ------------------------------------------------------------------ session

def load_instance(instance_id: str) -> Resolve {
    with open(DATA) as f {
        for line in f {
            if line.strip() {
                row = dict(json.loads(line));
                if row["instance_id"] == instance_id {
                    key = instance_id.replace("__", "_1776_").lower();
                    name = "sb-" + str(os.getpid()) + "-" + re.sub(r"[^a-zA-Z0-9_.-]", "-", instance_id);
                    return Resolve(
                        instance_id=instance_id, repo=str(row["repo"]), issue=str(row["problem_statement"]),
                        box=Container(name=name, image=str(row.get("image") or IMAGE.format(key=key)))
                    );
                }
            }
        }
    }
    raise KeyError(f"{instance_id} not in {DATA}");
}

"""Write the repository's diff as a SWE-bench prediction and report the session."""
def finish(task: Resolve) {
    (code, patch) = task.box.capture("git add -A && git diff --cached", COMMAND_TIMEOUT_S);
    if code != 0 {
        task.errors.append(f"git diff failed: {patch[:200]}");
        patch = "";
    }
    os.makedirs(PATCH_DIR, exist_ok=True);
    target = os.path.join(PATCH_DIR, f"{task.instance_id}.json");
    with open(target, "w") as f {
        json.dump({"instance_id": task.instance_id, "model_name_or_path": "swe_bench_agent", "model_patch": patch}, f);
    }
    print(f"Verdict: {'SUBMITTED' if task.submitted else 'UNFINISHED'}");
    print(f"Rounds: {task.turns}, steps: {len(task.trace)}");
    print(f"Patch: {target} ({patch.count(chr(10))} lines)");
    print(f"Tool time: {round(task.tool_seconds, 1)} s");
    if task.summary {
        print(f"Summary: {task.summary}");
    }
    if task.errors {
        print("Diagnostics:");
        for error in task.errors {
            print(f"- {error}");
        }
    }
}

def build_graph() -> Engineer {
    existing = [root-->][?:Engineer];
    if existing {
        return existing[0];
    }
    engineer = root ++> Engineer();
    sandbox = Sandbox();
    engineer ++> sandbox;
    sandbox ++> engineer;
    return engineer;
}


with entry {
    instance_id = os.environ.get("SB_TASK") or (sys.argv[1] if len(sys.argv) > 1 else "django__django-11099");
    task = load_instance(instance_id);
    task.box.start();
    try {
        task spawn build_graph();
        finish(task);
    } finally {
        task.box.stop();
    }
}
