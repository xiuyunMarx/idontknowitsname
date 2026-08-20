import os;
import from jaclang.byllm.llm { BaseLLM }
import argparse;

import from jaclang.byllm.lib {
    Model, IterationAction, IterationContext, mark_serialize
}
import from nodes.common { ToolCall, MAX_TOOL_CHARS, reset_tool_log, get_tool_calls }
import from nodes.explore { ExploreCodeBase }
import from nodes.edit { EditCode }
import from nodes.plan { PlanTasks }
import from nodes.verify { VerifyCode }
import from logger.log_usage { register_token_tracking, usage_totals }
import from logger.log_LLM_history { log_LLM_calls }
import from logger.log_stream { render_stream_event }


# The model is a named glob rather than the ambient `llm` so tests can rebind it
# to a MockLLM and run the whole traversal without an API key.
# gpt-5 rejects every temperature but the default: "Unsupported value:
# 'temperature' does not support 0.0 with this model. Only the default (1) value
# is supported." litellm drops the offending 0.0 rather than forwarding it, so
# the old setting did not error -- it silently became 1 at the provider while the
# source still claimed 0. Stating 1 makes the config describe the run.
glob agent_model: BaseLLM = Model(
    model_name=os.environ.get("CODEAGENT_MODEL", "gpt-5"),
    call_params={"temperature": 1.0}
);


def set_model(model: BaseLLM) {
    global agent_model;
    agent_model = model;
}


def active_model_name() -> str {
    return str(agent_model.model_name);
}


glob MAX_REACT_ITERATIONS: int = 25;
# The ceiling on one completion. It is not the size of the answer on a
# reasoning model: gpt-5 bills its hidden reasoning against this same budget,
# and a single "propose the minimal patch" turn measured 3264 reasoning tokens
# against 1459 of visible content. At the 4096 that served gpt-4o the model
# therefore stops mid-token-call rather than mid-sentence -- a write_file whose
# `content` argument is cut in half is an unparseable edit, which is exactly the
# failure family this run is meant to test. Raised for gpt-5, still overridable
# so the gpt-4o baseline can be reproduced with the same source.
glob MAX_OUTPUT_TOKENS: int = int(os.environ.get("CODEAGENT_MAX_TOKENS", "16384"));
# Tied to the tools' own budget rather than set independently, and deliberately
# not set below it. Every tool already truncates its result, and does so in the
# terms the model can act on -- read_file stops on a line boundary and names the
# start_line that resumes it. A smaller cap here would cut that result a second
# time, mid-line and with no continuation hint, and the model has no way to tell
# a silently shortened result from a complete one: the only move a mystery
# truncation leaves is to repeat the identical call.
#
# byLLM consults this on the streaming path only. It bound nothing until
# `stream=True` was set on run_phase below; it now binds, which is what puts
# this side on the same footing as langgraph and openai_sdk.
glob MAX_TOOL_RESULT_CHARS: int = MAX_TOOL_CHARS;
glob REPEAT_ABORT_THRESHOLD: int = 3;
# How much of a tool result the repeat-abort signature compares.
#
# byLLM does not hand `on_iteration` the same last_result on both of its paths:
# the non-streaming ReAct loop truncates it to 500 characters
# (basellm.impl.jac:992), while the streaming loop passes it cut at
# max_tool_result_length instead (:2143) -- 20000 here. Switching run_phase to
# `stream=True` therefore widened this comparison 40-fold on its own, and made
# the brake markedly less eager: three reads of one large file at different
# offsets that share a header collide at 500 characters and abort, but differ
# within 20000 and do not.
#
# Normalising here rather than living with whichever path is active keeps the
# brake identical on both, and keeps it at the sensitivity
# REPEAT_ABORT_THRESHOLD was actually tuned against.
glob SIG_RESULT_CHARS: int = 500;
glob SAME_TOOL_ABORT_THRESHOLD: int = 6;

# The only two tools that change the workspace. Every other call, however many
# of them a phase makes, leaves the tree exactly as it was found.
glob WRITE_TOOLS: list[str] = ["write_file", "replace_in_file"];
# The only tool that executes anything. Its output is the only evidence about
# behaviour a run can hold -- everything else is reading and writing text.
glob RUN_TOOLS: list[str] = ["run_command"];


"""True once this run has actually modified a file in the workspace.

Asked of the tool log rather than of the phase summary, because those are not
the same claim and have been observed to disagree. A phase that reports
"Summary of changes made: modified django/db/models/fields/__init__.py" while
having called only `grep` and `read_file` is not lying about a file it wrote --
it is describing the edit it decided on and never issued. Only the log knows.
"""
def workspace_written() -> bool {
    for call in get_tool_calls() {
        if call.name in WRITE_TOOLS {
            return True;
        }
    }
    return False;
}


"""True once a command has been executed after the run's most recent edit.

The other half of workspace_written's claim: an edit that nothing ran
afterwards is exactly as unverified as no edit at all. Every command that
executed before the last write observed a tree that no longer exists, so a
Verifying phase quoting green output from before the edit is quoting evidence
about the wrong repository. Consulted only once workspace_written holds; the
no-edits case never reaches it.
"""
def ran_since_last_edit() -> bool {
    ran = False;
    for call in get_tool_calls() {
        if call.name in WRITE_TOOLS {
            ran = False;
        } elif call.name in RUN_TOOLS {
            ran = True;
        }
    }
    return ran;
}


# ---------------------------------------------------------------------------
# Runtime: owns the four toolbox nodes and the per-phase handoff to byLLM.
# ---------------------------------------------------------------------------

obj AgentRuntime {
    has repo_root: str = "";
    has phase_tools: list[Callable] = [];
    has phase_ctx: list[dict[str, str]] = [];
    has recent_sigs: list[str] = [];
    has recent_tools: list[str] = [];
    has explorer: ExploreCodeBase by postinit;
    has editor: EditCode by postinit;
    has planner: PlanTasks by postinit;
    has verifier: VerifyCode by postinit;

    def postinit {
        self.retarget(self.repo_root);
    }

    def retarget(path: str) {
        base = os.path.realpath(path or os.getcwd());
        self.repo_root = base;
        self.explorer = ExploreCodeBase(repo_root=base);
        self.editor = EditCode(repo_root=base);
        self.planner = PlanTasks();
        self.verifier = VerifyCode(repo_root=base);
    }
}

glob rt: AgentRuntime = AgentRuntime();


# ---------------------------------------------------------------------------
# The ReAct call. One `by` function serves every phase; the phase's tools and
# conversation are read from `rt` (see AgentRuntime.phase_tools).
# ---------------------------------------------------------------------------

def run_phase(directive: str) -> str by agent_model(
    tools=rt.phase_tools,
    conversation=rt.phase_ctx,
    temperature=1.0,
    stream=True,
    stream_handler=render_stream_event,
    max_tokens=MAX_OUTPUT_TOKENS,
    max_react_iterations=MAX_REACT_ITERATIONS,
    on_iteration=progress_guard,
    max_tool_result_length=MAX_TOOL_RESULT_CHARS
);
sem run_phase = "Carry out the current phase of the coding task using the tools provided. Work until the phase's goal(Directive) is met, then reply with a short summary of what you did and what the next phase needs to know.";
sem run_phase.directive = "What to accomplish in this phase, which is the phase's goal. ";
 

"""True when the last `n` entries of `history` are all the same value."""
def all_alike(history: list[str], n: int) -> bool {
    if len(history) < n {
        return False;
    }
    tail = history[len(history) - n:];
    return all([item == tail[0] for item in tail]);
}


def progress_guard(ctx: IterationContext) -> IterationAction {
    rt.recent_sigs.append(
        f"{ctx.last_tool}|{str(ctx.last_result)[:SIG_RESULT_CHARS]}"
    );
    rt.recent_tools.append(str(ctx.last_tool));
    # No legitimate workflow issues the identical call three times running,
    # and none reaches for the same tool six times with nothing in between.
    if all_alike(rt.recent_sigs, REPEAT_ABORT_THRESHOLD) {
        return IterationAction.ABORT_WITH_SUMMARY;
    }
    if all_alike(rt.recent_tools, SAME_TOOL_ABORT_THRESHOLD) {
        return IterationAction.ABORT_WITH_SUMMARY;
    }
    return IterationAction.CONTINUE;
}


# ---------------------------------------------------------------------------
# Graph: tools, phases, and the edges between them.
# ---------------------------------------------------------------------------

enum ToolCat { Read = "read", Write = "write", Run = "run", Track = "track" }

"""One capability a phase is allowed to call."""
node Tool {
    has fn: Callable;
    has category: str = "read";
}

# Connects a phase to a tool it may use
edge Exposes: Phase --> Tool {}


edge Flow: Phase --> Phase {
    has reason: str = "";
}


node Phase {
    has title: str = "";
    has goal: str = "";
    has effect: str = "";

    has _ctx: list[dict[str, str]] = [];

    def tools -> list[Callable] {
        return [t.fn for t in [self ->:Exposes:->]];
    }

    def tool_names -> list[str] {
        return sorted([t.fn.__name__ for t in [self ->:Exposes:->]]);
    }


    def work(objective: str, ledger: str) -> str {
        system = (
            f"You are a coding agent working in the repository at {rt.repo_root}.\n"
            f"You are in the {self.title} phase: {self.goal}\n\n"
            f"## Objective\n{objective}\n\n"
            f"## Progress so far\n{ledger or '(nothing yet)'}\n\n"
            "Use the tools you have been given. Read before you edit, and copy "
            "text exactly when you use it as an edit anchor. When the phase's "
            "goal is met, stop and summarise."
        );
        self._ctx = [{"role": "system", "content": system}];
        rt.phase_tools = self.tools();
        rt.phase_ctx = self._ctx;
        rt.recent_sigs = [];
        rt.recent_tools = [];
        rt.explorer.set_scope(self.title);
        return str(run_phase(self.goal));
    }
}


node Planning(Phase) {
    has title: str = "Planning";
    has effect: str = "Spends one step, then routes on.";
    has goal: str = "Understand what is being asked and write down the steps you will take. Call set_plan with those steps before you finish.";
    can run with CodeAgent entry;
}

node Exploring(Phase) {
    has title: str = "Exploring";
    has effect: str = "Spends one step reading the repository. Cannot write.";
    has goal: str = "Locate the exact files and lines that must change. Report the paths and what each one needs.";
    can run with CodeAgent entry;
}

node Editing(Phase) {
    has title: str = "Editing";
    has effect: str = "Spends one step and writes to disk. Always routes to Verifying afterwards.";
    has goal: str = "Make the changes. Use write_file for new files and replace_in_file for surgical edits. Actually call the tools -- describing a change is not making it.";
    can run with CodeAgent entry;
}

node Verifying(Phase) {
    has title: str = "Verifying";
    has effect: str = "Spends one step running commands. Routes back to Editing if anything is wrong.";
    has goal: str = "Prove the change works by running the tests that cover it. First find how this repository runs its own suite: pytest or jac test where available, otherwise a runner script the repository ships (a tests/runtests.py, a bin/test) invoked through python. Run the existing test files nearest the code you changed, plus a reproduction of the issue and its edge cases. A hand-written script alone is not verification -- if the repository's own tests cannot be run, say so plainly and treat the objective as unverified. Report exactly what the output showed.";
    can run with CodeAgent entry;
}


"""Terminal node: the objective is met, or the step budget ran out."""
node Finished(Phase) {
    has title: str = "Finished";
    has goal: str = "Stop. The objective is done and verified.";
    has effect: str = "Ends the run. The ledger becomes the final answer; nothing runs after this.";
    has summary: str = "";
    can close with CodeAgent entry;
}


def expose(phase: Phase, fns: list[Callable], category: ToolCat) {
    for fn in fns {
        phase +>: Exposes() :+> Tool(fn=fn, category=str(category.value));
    }
}


with entry {
    for fn in [
        EditCode.write_file, EditCode.replace_in_file, VerifyCode.run_command
    ] {
        mark_serialize(fn);
    }
}


def build_pipeline() -> Planning {
    planning = Planning();
    exploring = Exploring();
    editing = Editing();
    verifying = Verifying();
    finished = Finished();

    planning +>: Flow(reason="the plan is written; go find the code it touches") :+> exploring;
    planning +>: Flow(reason="shortcut: the request was a question the plan already answers") :+> finished;
    exploring +>: Flow(reason="the code to change has been located; go change it") :+> editing;
    exploring +>: Flow(reason="shortcut: the request was a question the code just read answers") :+> finished;
    editing +>: Flow(reason="an edit was made and must be proven by running it") :+> verifying;
    verifying +>: Flow(reason="the repair loop: verification found problems left to fix") :+> editing;
    verifying +>: Flow(reason="verification proved the objective complete and correct") :+> finished;

    read_tools: list[Callable] = [
        rt.explorer.read_file, rt.explorer.ls_repo,
        rt.explorer.find_files, rt.explorer.grep
    ];
    track_tools: list[Callable] = [rt.planner.show_plan, rt.planner.update_task];
    write_tools: list[Callable] = [rt.editor.write_file, rt.editor.replace_in_file];
    run_tools: list[Callable] = [rt.verifier.run_command];

    # Every phase can read and can update the plan;
    for phase in [planning, exploring, editing, verifying] {
        expose(phase, read_tools, ToolCat.Read);
        expose(phase, track_tools, ToolCat.Track);
    }
    expose(planning, [rt.planner.set_plan], ToolCat.Track);
    expose(editing, write_tools, ToolCat.Write);
    expose(editing, run_tools, ToolCat.Run);
    expose(verifying, run_tools, ToolCat.Run);
    return planning;
}


# ---------------------------------------------------------------------------
# The walker.
# ---------------------------------------------------------------------------

walker CodeAgent {
    has objective: str;
    has max_steps: int = 10;
    has steps: int = 0;
    has _ledger: list[str] = [];
    has answer: str = "";
    has edits_made: bool = False;

    can begin with Root entry;
    can plan with Planning entry;
    can explore with Exploring entry;
    can edit with Editing entry;
    can verify with Verifying entry;
    can close with Finished entry;

    def record(phase: str, summary: str) {
        self._ledger.append(f"### {phase}\n{summary}");
    }

    def rendered -> str {
        return "\n\n".join(self._ledger);
    }

    # True when the step budget is spent, so every phase stops advancing.
    def exhausted -> bool {
        return self.steps >= self.max_steps;
    }
}


impl CodeAgent.begin {
    visit build_pipeline();
}


impl Planning.run with CodeAgent entry {
    visitor.steps += 1;
    summary = self.work(visitor.objective, visitor.rendered());
    visitor.record(self.title, summary);
    # The budget is not a judgement call, so it is settled before the model is
    # asked: a spent budget is exactly when another LLM call is the wrong move.
    if visitor.exhausted() {
        visit [->:Flow:->][?:Finished];
        return;
    }
    visit [edge ->:Flow:->] by agent_model(
        select=1,
        intent="Go to Exploring to find the code this task touches. Choose Finished only if this was a question needing no code change and the plan already answers it.",
        incl_info={"Progress so far": visitor.rendered()}
    );
}


impl Exploring.run with CodeAgent entry {
    visitor.steps += 1;
    summary = self.work(visitor.objective, visitor.rendered());
    visitor.record(self.title, summary);
    if visitor.exhausted() {
        visit [->:Flow:->][?:Finished];
        return;
    }
    visit [edge ->:Flow:->] by agent_model(
        select=1,
        intent="Go to Editing to make the change this exploration located. "
               "Choose Finished only if the objective was a question that the "
               "code just read already answers, with nothing left to edit.",
        incl_info={"Progress so far": visitor.rendered()}
    );
}


impl Editing.run with CodeAgent entry {
    visitor.steps += 1;
    summary = self.work(visitor.objective, visitor.rendered());
    visitor.record(self.title, summary);
    # Asked of the tool log, not assumed from having run. This used to be an
    # unconditional True, which recorded "Editing was visited" under a name that
    # reads as "an edit was made" -- and nothing ever read it, so the untruth
    # stayed invisible until a run reported a fix it had never written.
    visitor.edits_made = workspace_written();
    visit [->:Flow:->][?:Verifying];
}

impl Verifying.run with CodeAgent entry {
    visitor.steps += 1;
    summary = self.work(visitor.objective, visitor.rendered());
    visitor.record(self.title, summary);
    if visitor.exhausted() {
        visit [->:Flow:->][?:Finished];
        return;
    }
    # Nothing was written, so nothing was verified -- whatever the run just
    # observed, it observed about the unmodified tree. This is not a judgement
    # the model gets to make, because the evidence that fools it is exactly the
    # evidence it is looking at: a repository's existing suite passes on an
    # unfixed tree, since the test that would catch the bug is the one being
    # held back. Green tests here mean "I changed nothing", and the model reads
    # them as proof.
    if not visitor.edits_made {
        visitor.record(
            self.title,
            "GUARD: no write_file or replace_in_file call has been made in "
            "this run, so the workspace is still exactly as it was found and "
            "the commands above prove nothing about the objective. Any change "
            "described so far was planned, not applied. Go back to Editing and "
            "issue the edit."
        );
        visit [->:Flow:->][?:Editing];
        return;
    }
    # The guard above in mirror image: this run did edit, but nothing has
    # executed since the last write, so every command output quoted so far
    # describes a tree that no longer exists. Also not a judgement the model
    # gets to make, and for the same reason -- pre-edit green output is the
    # exact evidence that fools it.
    if not ran_since_last_edit() {
        visitor.record(
            self.title,
            "GUARD: the workspace has been edited, but no command has been "
            "run since the most recent edit, so nothing observed so far "
            "describes the tree as it now stands. Go back to Editing, confirm "
            "the change is what you intended, and run the repository's tests "
            "against it before finishing."
        );
        visit [->:Flow:->][?:Editing];
        return;
    }
    # The repair loop's hinge, and the judgement the separate objective_met
    # classifier used to make.
    visit [edge ->:Flow:->] by agent_model(
        select=1,
        intent="Choose ```Finished``` only if the repository's own tests covering the changed code were executed after the final edit and passed -- a hand-written reproduction script alone is not verification, and neither is a suite that could not be run. If tests failed, could not be run, were skipped, or the result is ambiguous, go back to Editing.",
        incl_info={"Progress so far": visitor.rendered()}
    );
}


impl Finished.close with CodeAgent entry {
    self.summary = visitor.rendered();
    visitor.answer = self.summary;
    report self.summary;
}


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

"""What one agent run produced. Typed rather than a dict so the eval harness
reads a stable shape on both sides of the comparison."""
obj RunResult {
    has objective: str;
    has answer: str = "";
    has steps: int = 0;
    has tool_calls: list[ToolCall] = [];
    has llm_calls: int = 0;
    has prompt_tokens: int = 0;
    has completion_tokens: int = 0;
    # Part of prompt_tokens, not extra to it -- see LLMCall.cached_tokens.
    has cached_tokens: int = 0;
}


def solve(objective: str, repo_root: str = "", max_steps: int = 10) -> RunResult {
    rt.retarget(repo_root);
    reset_tool_log();
    register_token_tracking();
    session = CodeAgent(objective=objective, max_steps=max_steps);
    root spawn session;
    totals:dict[str, int] = usage_totals();
    return RunResult(
        objective=objective,
        answer=session.answer,
        steps=session.steps,
        tool_calls=get_tool_calls(),
        llm_calls=int(totals["llm_calls"]),
        prompt_tokens=int(totals["prompt_tokens"]),
        completion_tokens=int(totals["completion_tokens"]),
        cached_tokens=int(totals["cached_tokens"])
    );
}

def parse_args(parser: argparse.ArgumentParser) -> dict[str, str] {
    parser.add_argument(
        "--task", required=True,
        help="The task to be solved."
    );
    parser.add_argument(
        "--repo-path", default="",
        help="Path to the repository to work in. Defaults to the current directory."
    );
    ns = parser.parse_args();
    return {"task": str(ns.task), "repo_path": str(ns.repo_path or "")};
}

with entry:__main__ {
    parser = argparse.ArgumentParser(description="Run the byLLM orchestrator.");
    args = parse_args(parser);
    outcome = solve(args["task"], args["repo_path"]);
    print(outcome.answer);
    _ = log_LLM_calls();
}
