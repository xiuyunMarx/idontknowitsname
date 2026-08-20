import from nodes.plan { PlanTasks, TaskStatus }

test "an empty board renders a sentinel" {
    board = PlanTasks();
    assert "no plan yet" in board.show_plan();
    assert board.show_plan() != "";
}

test "set_plan renders a numbered checklist" {
    board = PlanTasks();
    out = board.set_plan(["read the file", "fix the bug", "run the tests"]);
    assert out.startswith("OK: plan set with 3 steps.");
    assert "[ ] 1. read the file" in out;
    assert "[ ] 3. run the tests" in out;
    assert "(0/3 done)" in out;
}

test "update_task marks a step done" {
    board = PlanTasks();
    board.set_plan(["one", "two"]);
    out = board.update_task(1, TaskStatus.DONE);
    assert out.startswith("OK: step 1 is now done.");
    assert "[x] 1. one" in out;
    assert "(1/2 done)" in out;
}

test "update_task records doing and blocked with a note" {
    board = PlanTasks();
    board.set_plan(["one", "two"]);
    doing = board.update_task(2, TaskStatus.DOING);
    assert "[~] 2. two" in doing;
    blocked = board.update_task(2, TaskStatus.BLOCKED, "waiting on the schema");
    assert "[!] 2. two" in blocked;
    assert "waiting on the schema" in blocked;
}

test "update_task reports an unknown id with the known ids" {
    board = PlanTasks();
    board.set_plan(["one", "two"]);
    out = board.update_task(99, TaskStatus.DONE);
    assert out.startswith("Error: no step with id 99.");
    assert "Known ids: 1, 2." in out;
}

test "update_task points at set_plan when no plan exists" {
    board = PlanTasks();
    out = board.update_task(1, TaskStatus.DONE);
    assert out.startswith("Error: ");
    assert "set_plan" in out;
}

test "set_plan rejects an empty step list" {
    board = PlanTasks();
    assert board.set_plan([]).startswith("Error: ");
    assert board.set_plan(["   ", ""]).startswith("Error: ");
}

test "set_plan replaces the plan and restarts numbering" {
    # set_plan replaces the whole plan and restarts numbering.
    board = PlanTasks();
    board.set_plan(["a", "b", "c"]);
    board.update_task(1, TaskStatus.DONE);
    out = board.set_plan(["x"]);
    assert "(0/1 done)" in out;
    assert "[ ] 1. x" in out;
    assert "a" not in out.split("\n")[1];
}

test "show_plan refetches the current state" {
    # show_plan re-fetches state after transcript compaction.
    board = PlanTasks();
    board.set_plan(["alpha"]);
    board.update_task(1, TaskStatus.DOING, "in flight");
    shown = board.show_plan();
    assert "[~] 1. alpha" in shown;
    assert "in flight" in shown;
}
