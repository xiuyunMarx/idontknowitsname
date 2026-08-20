import os;
import tempfile;
import from jaclang.byllm.lib { MockLLM, MockToolCall, IterationAction, IterationContext }
import from jaclang.byllm.mtir { MTRuntime }
import from orchestrator {
    Phase, Planning, Exposes, Flow, CodeAgent,
    rt, set_model, build_pipeline, progress_guard, SIG_RESULT_CHARS,
    workspace_written, WRITE_TOOLS, ran_since_last_edit, RUN_TOOLS
}
import from nodes.common { log_tool_call, reset_tool_log }

# ---------------------------------------------------------------------------
# Graph shape. These need no model at all: the capability boundary is an edge
# set, so it can be asserted by traversing it.
# ---------------------------------------------------------------------------

def phase_after(src: Phase, kind: str) -> list[Phase] {
    return [n for n in [src ->:Flow:->] if type(n).__name__ == kind];
}

test "build_pipeline starts at Planning" {
    start = build_pipeline();
    assert isinstance(start, Planning);
    assert start.title == "Planning";
}

test "only Editing may write and only Editing and Verifying may run" {
    # Only Editing may write; only Editing and Verifying may execute.
    start = build_pipeline();
    exploring = phase_after(start, "Exploring")[0];
    editing = phase_after(exploring, "Editing")[0];
    verifying = phase_after(editing, "Verifying")[0];

    plan_tools = start.tool_names();
    assert "read_file" in plan_tools;
    assert "grep" in plan_tools;
    assert "set_plan" in plan_tools;
    assert "write_file" not in plan_tools;
    assert "run_command" not in plan_tools;

    assert "write_file" not in exploring.tool_names();
    assert "run_command" not in exploring.tool_names();

    edit_tools = editing.tool_names();
    assert "write_file" in edit_tools;
    assert "replace_in_file" in edit_tools;
    assert "run_command" in edit_tools;
    assert "read_file" in edit_tools;

    verify_tools = verifying.tool_names();
    assert "run_command" in verify_tools;
    assert "write_file" not in verify_tools;
}

test "every stalling phase has an escape edge to Finished" {
    # Every phase that can stall has an escape edge to the terminal node, so a
    # spent step budget always lands on Finished instead of the walker falling
    # off the end of the graph with no answer.
    start = build_pipeline();
    exploring = phase_after(start, "Exploring")[0];
    editing = phase_after(exploring, "Editing")[0];
    verifying = phase_after(editing, "Verifying")[0];

    assert len(phase_after(start, "Finished")) == 1;
    assert len(phase_after(exploring, "Finished")) == 1;
    assert len(phase_after(verifying, "Finished")) == 1;
    # The repair loop back to Editing.
    assert len(phase_after(verifying, "Editing")) == 1;
    # Editing never shortcuts past verification.
    assert len(phase_after(editing, "Finished")) == 0;
    assert len(phase_after(editing, "Verifying")) == 1;
}

test "exposed tools are categorised graph data" {
    # Tools are reachable as graph data, categorised.
    start = build_pipeline();
    cats = sorted(set([t.category for t in [start ->:Exposes:->]]));
    assert "read" in cats;
    assert "track" in cats;
    assert "write" not in cats;
}

# ---------------------------------------------------------------------------
# Progress guard.
# ---------------------------------------------------------------------------

test "the progress guard aborts on three identical calls" {
    rt.recent_sigs = [];
    rt.recent_tools = [];
    ctx = IterationContext(
        iteration=2, messages=[], last_tool="grep",
        last_result="No matches", total_tokens=0
    );
    assert progress_guard(ctx) == IterationAction.CONTINUE;
    assert progress_guard(ctx) == IterationAction.CONTINUE;
    # Third identical call in a row ends the phase rather than burning budget.
    assert progress_guard(ctx) == IterationAction.ABORT_WITH_SUMMARY;
}

test "the progress guard aborts on a run of one tool" {
    # The other way a phase stalls, and the identical-call test above is blind
    # to it: read_file walking a file one line at a time issues a different
    # call every time, so no two signatures match. Five of eight phase laps
    # died this way on a real astropy instance, 75% of the run's tokens.
    rt.recent_sigs = [];
    rt.recent_tools = [];
    for i in range(5) {
        ctx = IterationContext(
            iteration=i + 1, messages=[], last_tool="read_file",
            last_result=f"# core.py - lines {800 + i}-{820 + i} of 4473",
            total_tokens=0
        );
        assert progress_guard(ctx) == IterationAction.CONTINUE;
    }
    sixth = IterationContext(
        iteration=6, messages=[], last_tool="read_file",
        last_result="# core.py - lines 805-825 of 4473", total_tokens=0
    );
    assert progress_guard(sixth) == IterationAction.ABORT_WITH_SUMMARY;
}

test "the repeat signature ignores anything past SIG_RESULT_CHARS" {
    # byLLM hands on_iteration a 500-character last_result on its non-streaming
    # path and a max_tool_result_length-sized one when streaming, so without the
    # normalisation in progress_guard this brake would silently change
    # sensitivity the moment `stream=True` was set. Three results agreeing over
    # the first 500 characters are the same stalled call whichever path ran.
    rt.recent_sigs = [];
    rt.recent_tools = [];
    head = "x" * SIG_RESULT_CHARS;
    for i in range(2) {
        ctx = IterationContext(
            iteration=i + 1, messages=[], last_tool="grep",
            last_result=head + f" divergent tail {i}", total_tokens=0
        );
        assert progress_guard(ctx) == IterationAction.CONTINUE;
    }
    third = IterationContext(
        iteration=3, messages=[], last_tool="grep",
        last_result=head + " divergent tail 2", total_tokens=0
    );
    assert progress_guard(third) == IterationAction.ABORT_WITH_SUMMARY;
}

test "the repeat signature still separates results differing inside the window" {
    # The converse, so the normalisation cannot be mistaken for "compare the
    # tool name only": a difference before the cut is still a different call.
    rt.recent_sigs = [];
    rt.recent_tools = [];
    for i in range(3) {
        ctx = IterationContext(
            iteration=i + 1, messages=[], last_tool="grep",
            last_result=f"hit {i} " + "x" * (SIG_RESULT_CHARS * 2), total_tokens=0
        );
        assert progress_guard(ctx) == IterationAction.CONTINUE;
    }
}

test "the progress guard tolerates a long run that mixes tools" {
    # Read, edit, read, edit is work, not a loop, however long it goes on.
    rt.recent_sigs = [];
    rt.recent_tools = [];
    for i in range(20) {
        tool = "read_file" if i % 2 == 0 else "replace_in_file";
        ctx = IterationContext(
            iteration=i + 1, messages=[], last_tool=tool,
            last_result=f"result {i}", total_tokens=0
        );
        assert progress_guard(ctx) == IterationAction.CONTINUE;
    }
}

test "the progress guard tolerates alternating calls" {
    rt.recent_sigs = [];
    rt.recent_tools = [];
    a = IterationContext(iteration=2, messages=[], last_tool="grep", last_result="x", total_tokens=0);
    b = IterationContext(iteration=3, messages=[], last_tool="read_file", last_result="y", total_tokens=0);
    assert progress_guard(a) == IterationAction.CONTINUE;
    assert progress_guard(b) == IterationAction.CONTINUE;
    assert progress_guard(a) == IterationAction.CONTINUE;
    assert progress_guard(b) == IterationAction.CONTINUE;
}

# ---------------------------------------------------------------------------
# A full traversal driven by MockLLM -- no API key, no network.
# ---------------------------------------------------------------------------

test "a full traversal edits a real file" {
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "greet.jac"), "w") as f {
        f.write("with entry {\n    print(\"hello\");\n}\n");
    }
    rt.retarget(tmp);

    # Each branching phase spends two outputs: the phase's own reply, then the
    # handle `visit ... by` routes on. The handle is the target node's type
    # name. Editing never routes -- it has a single Flow successor -- so it
    # spends one.
    set_model(MockLLM(
        model_name="mockllm",
        config={
            "outputs": [
                # Planning
                MockToolCall(tool=rt.planner.set_plan, args={"steps": ["change the greeting"]}),
                "Plan written: change the greeting.",
                "Exploring",
                # Exploring
                MockToolCall(tool=rt.explorer.grep, args={"pattern": "hello", "path": "."}),
                "greet.jac line 2 holds the greeting.",
                "Editing",
                # Editing
                MockToolCall(
                    tool=rt.editor.replace_in_file,
                    args={"file_path": "greet.jac", "old": "hello", "new": "goodbye"}
                ),
                "Replaced hello with goodbye in greet.jac.",
                # Verifying
                MockToolCall(tool=rt.verifier.run_command, args={"command": "jac run greet.jac"}),
                "jac run printed goodbye; exit code 0.",
                "Finished"
            ]
        }
    ));

    session = CodeAgent(objective="change the greeting to goodbye", max_steps=10);
    root spawn session;

    # The walker reached the terminal node and carried an answer out.
    assert session.answer != "";
    assert session.steps == 4;
    for phase in ["Planning", "Exploring", "Editing", "Verifying"] {
        assert phase in session.answer;
    }
    # The edit really happened on disk -- the tools were called, not described.
    with open(os.path.join(tmp, "greet.jac")) as f {
        assert "goodbye" in f.read();
    }
}

test "the model can route straight from Planning to Finished" {
    # Proves the routing is the model's and not the fallback's. Every `visit
    # ... by` has an `else` that reproduces the fixed successor the phase used
    # before it routed, so a router that returned nothing would still walk
    # Planning -> Exploring and every other test would still pass. Only the
    # shortcut edge distinguishes them: nothing but a model choice reaches
    # Finished from Planning while steps remain.
    tmp = os.path.realpath(tempfile.mkdtemp());
    rt.retarget(tmp);

    set_model(MockLLM(
        model_name="mockllm",
        config={"outputs": ["read_file returns a greeting; no change needed.", "Finished"]}
    ));

    session = CodeAgent(objective="what does greet.jac print?", max_steps=10);
    root spawn session;

    assert session.steps == 1;
    assert "Planning" in session.answer;
    # The phases the fallback would have walked into never ran.
    assert "Exploring" not in session.answer;
    assert "Editing" not in session.answer;
}

# ---------------------------------------------------------------------------
# The phase conversation does not survive a re-entry.
# ---------------------------------------------------------------------------

glob prompt_sizes: list[int] = [];

"""A MockLLM that records how many messages each call was handed."""
obj CountingMock(MockLLM) {
    def invoke(mt_run: MTRuntime) -> object {
        prompt_sizes.append(len(mt_run.messages));
        return super.invoke(mt_run);
    }
}

test "re-entering a phase does not carry the previous lap's conversation" {
    # Editing and Verifying trade the walker back and forth until the step cap
    # bites. Each lap is two system messages and one directive and nothing
    # else, because the model here calls no tools -- so a fourth message is a
    # message the phase inherited from a lap it had already summarised into the
    # ledger. Left in, that is what put four full pytest runs into Editing's
    # fourth prompt on a real astropy instance, 22k of its 28k prompt tokens.
    tmp = os.path.realpath(tempfile.mkdtemp());
    rt.retarget(tmp);
    prompt_sizes.clear();

    outputs: list[object] = ["planning reply", "Exploring", "exploring reply", "Editing"];
    for i in range(40) {
        outputs.append(f"editing reply {i}");
        outputs.append(f"verifying reply {i}");
        outputs.append("Editing");
    }
    set_model(CountingMock(model_name="mockllm", config={"outputs": outputs}));

    session = CodeAgent(objective="never satisfied", max_steps=6);
    root spawn session;

    # There was something to inherit: both phases ran more than once.
    assert session.answer.count("### Editing") >= 2;
    assert session.answer.count("### Verifying") >= 2;
    # Routing calls send two messages, phase calls three. Nothing sends more.
    assert max(prompt_sizes) == 3;
}

test "the step cap bounds a never satisfied run" {
    # The step cap bounds the run even when the model never says it is done.
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "x.txt"), "w") as f {
        f.write("body\n");
    }
    rt.retarget(tmp);

    # Planning and Exploring route forward once each, then the run settles into
    # the Editing <-> Verifying repair loop: Editing replies and advances on its
    # single Flow edge, Verifying replies and routes back to Editing -- the
    # model never once choosing Finished.
    outputs: list[object] = ["planning reply", "Exploring", "exploring reply", "Editing"];
    # Enough laps for the cap to bite well before the outputs run out.
    for i in range(40) {
        outputs.append(f"editing reply {i}");
        outputs.append(f"verifying reply {i}");
        outputs.append("Editing");
    }
    set_model(MockLLM(model_name="mockllm", config={"outputs": outputs}));

    session = CodeAgent(objective="never satisfied", max_steps=6);
    root spawn session;
    assert session.answer != "";
    # It stops at the cap rather than looping Editing <-> Verifying forever.
    assert session.steps <= 8;
}


# ---------------------------------------------------------------------------
# The write guard. A phase summary and the tool log are two different claims,
# and the run that produced this test had them disagree: Editing reported
# "Summary of changes made: modified django/db/models/fields/__init__.py" after
# 33 tool calls of which none was a write, Verifying ran the repository's
# existing suite, saw 310 tests pass on the untouched tree, and finished.
# ---------------------------------------------------------------------------

test "workspace_written is false until a write tool is actually called" {
    reset_tool_log();
    assert not workspace_written();
    log_tool_call("grep", {"pattern": "get_.*_display"});
    log_tool_call("read_file", {"file_path": "django/db/models/fields/__init__.py"});
    log_tool_call("run_command", {"command": "python tests/runtests.py model_fields"});
    # Reading and running are not changing, however many of them there are.
    assert not workspace_written();
    reset_tool_log();
}

test "workspace_written is true after either write tool" {
    for tool in WRITE_TOOLS {
        reset_tool_log();
        log_tool_call("read_file", {"file_path": "a.py"});
        assert not workspace_written();
        log_tool_call(tool, {"file_path": "a.py"});
        assert workspace_written();
    }
    reset_tool_log();
}

test "the write tools are exactly the two that change the tree" {
    # If a third write tool is ever added, this fails until the guard knows
    # about it -- which is the point: an unguarded write tool reopens the hole.
    assert sorted(WRITE_TOOLS) == ["replace_in_file", "write_file"];
    editing = phase_after(phase_after(build_pipeline(), "Exploring")[0], "Editing")[0];
    writers = sorted([n for n in editing.tool_names() if n in WRITE_TOOLS]);
    assert writers == ["replace_in_file", "write_file"];
}

test "Verifying can reach Finished, and can return to Editing" {
    # The guard sends an unwritten run back down the repair edge, so that edge
    # has to exist for the guard to be reachable at all.
    editing = phase_after(phase_after(build_pipeline(), "Exploring")[0], "Editing")[0];
    verifying = phase_after(editing, "Verifying")[0];
    assert len(phase_after(verifying, "Finished")) == 1;
    assert len(phase_after(verifying, "Editing")) == 1;
}


# ---------------------------------------------------------------------------
# The run-after-edit guard, the write guard's mirror image. Commands that
# executed before the last write observed a tree that no longer exists, so a
# run whose edit nothing followed has verified exactly as much as a run that
# never edited: nothing.
# ---------------------------------------------------------------------------

test "ran_since_last_edit is false while the last edit sits unexecuted" {
    reset_tool_log();
    assert not ran_since_last_edit();
    # A command before the edit is evidence about the pre-edit tree.
    log_tool_call("run_command", {"command": "pytest sympy/utilities/tests/test_codegen.py"});
    log_tool_call("replace_in_file", {"file_path": "sympy/utilities/codegen.py"});
    assert not ran_since_last_edit();
    # Reading afterwards changes nothing: still nothing has executed.
    log_tool_call("read_file", {"file_path": "sympy/utilities/codegen.py"});
    assert not ran_since_last_edit();
    reset_tool_log();
}

test "ran_since_last_edit is true once a command follows the last edit" {
    reset_tool_log();
    log_tool_call("replace_in_file", {"file_path": "sympy/utilities/codegen.py"});
    log_tool_call("run_command", {"command": "python bin/test sympy/utilities/tests/test_codegen.py"});
    assert ran_since_last_edit();
    # A later edit re-arms the guard: the tree has changed again since that
    # run, whichever write tool changed it.
    for tool in WRITE_TOOLS {
        log_tool_call(tool, {"file_path": "sympy/utilities/codegen.py"});
        assert not ran_since_last_edit();
        log_tool_call("run_command", {"command": "python bin/test sympy/utilities"});
        assert ran_since_last_edit();
    }
    reset_tool_log();
}

test "the run tools are exactly the one that executes" {
    # Same pinning as WRITE_TOOLS above: a second execution surface added
    # without telling the guard would let post-edit runs go unseen.
    assert RUN_TOOLS == ["run_command"];
    editing = phase_after(phase_after(build_pipeline(), "Exploring")[0], "Editing")[0];
    verifying = phase_after(editing, "Verifying")[0];
    runners = sorted([n for n in verifying.tool_names() if n in RUN_TOOLS]);
    assert runners == ["run_command"];
}

test "a Verifying lap that executes nothing cannot finish" {
    # The wiring, not just the predicate: Editing writes, Verifying summarises
    # without running a single command, and the guard -- not the model -- sends
    # the walker back. Only after something executes does the router get to
    # choose Finished.
    reset_tool_log();
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "greet.jac"), "w") as f {
        f.write("with entry {\n    print(\"hello\");\n}\n");
    }
    rt.retarget(tmp);

    set_model(MockLLM(
        model_name="mockllm",
        config={
            "outputs": [
                # Planning
                MockToolCall(tool=rt.planner.set_plan, args={"steps": ["change the greeting"]}),
                "Plan written: change the greeting.",
                "Exploring",
                # Exploring
                "greet.jac line 2 holds the greeting.",
                "Editing",
                # Editing: writes, runs nothing.
                MockToolCall(
                    tool=rt.editor.replace_in_file,
                    args={"file_path": "greet.jac", "old": "hello", "new": "goodbye"}
                ),
                "Replaced hello with goodbye in greet.jac.",
                # Verifying lap 1: summarises without executing -- the guard
                # must bounce this, so no routing output is consumed.
                "The suite was green earlier, so the change is fine.",
                # Editing lap 2: executes the changed file.
                MockToolCall(tool=rt.verifier.run_command, args={"command": "jac run greet.jac"}),
                "jac run printed goodbye; exit code 0.",
                # Verifying lap 2: now there is post-edit evidence to route on.
                "The command output proves the change.",
                "Finished"
            ]
        }
    ));

    session = CodeAgent(objective="change the greeting to goodbye", max_steps=10);
    root spawn session;

    assert session.answer != "";
    # Two laps of the repair loop: the guard forced the second.
    assert session.steps == 6;
    assert "GUARD: the workspace has been edited" in session.answer;
    with open(os.path.join(tmp, "greet.jac")) as f {
        assert "goodbye" in f.read();
    }
    reset_tool_log();
}
