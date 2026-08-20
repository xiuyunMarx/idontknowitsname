"""Task state: the agent's own plan, written down and read back.

The only tool group here with no effect outside the process. Its whole
mechanism is that structured state written by the model is rendered back into
the transcript, so intent survives a long tool-call sequence.
"""

import from nodes.common { log_tool_call }


enum TaskStatus {
    TODO = "todo",
    DOING = "doing",
    DONE = "done",
    BLOCKED = "blocked"
}


obj PlanItem {
    has id: int;
    has title: str;
    # Stored as str, not TaskStatus: an enum-typed field on a persisted
    # archetype is not layout-compatible. The enum lives at the LLM boundary,
    # where it buys a JSON-Schema `enum` constraint on the parameter.
    has status: str = "todo";
    has note: str = "";
}


node PlanTasks {
    has items: list[PlanItem] = [];
    has next_id: int = 1;

    def set_plan(steps: list[str]) -> str {
        log_tool_call("set_plan", {"steps": str(len(steps))});
        cleaned = [s.strip() for s in steps if s.strip()];
        if not cleaned {
            return "Error: 'steps' was empty. Pass at least one step.";
        }
        self.items = [];
        self.next_id = 1;
        for title in cleaned {
            self.items.append(PlanItem(id=self.next_id, title=title));
            self.next_id += 1;
        }
        return f"OK: plan set with {len(self.items)} steps.\n" + self._render();
    }

    def update_task(task_id: int, status: TaskStatus, note: str = "") -> str {
        log_tool_call("update_task", {"task_id": str(task_id), "status": str(status)});
        for item in self.items {
            if item.id == task_id {
                item.status = str(status.value);
                if note {
                    item.note = note;
                }
                return f"OK: step {task_id} is now {item.status}.\n" + self._render();
            }
        }
        if not self.items {
            return (
                f"Error: there is no step {task_id} because no plan has been set. "
                "Call set_plan first."
            );
        }
        known = ", ".join([str(i.id) for i in self.items]);
        return f"Error: no step with id {task_id}. Known ids: {known}.";
    }

    def show_plan() -> str {
        log_tool_call("show_plan", {});
        return self._render();
    }

    def _render() -> str {
        if not self.items {
            return "(no plan yet -- call set_plan with the steps you intend to take)";
        }
        marks: dict[str, str] = {
            "todo": "[ ]", "doing": "[~]", "done": "[x]", "blocked": "[!]"
        };
        lines: list[str] = [];
        done = 0;
        for item in self.items {
            if item.status == "done" {
                done += 1;
            }
            mark = marks.get(item.status, "[ ]");
            line = f"{mark} {item.id}. {item.title}";
            if item.note {
                line += f"  -- {item.note}";
            }
            lines.append(line);
        }
        return "\n".join(lines) + f"\n({done}/{len(self.items)} done)";
    }
}


sem TaskStatus = "Progress state of one step in the plan.";
sem TaskStatus.TODO = "Not started yet.";
sem TaskStatus.DOING = "Currently being worked on.";
sem TaskStatus.DONE = "Finished and verified.";
sem TaskStatus.BLOCKED = "Cannot proceed until something else is resolved.";

sem PlanTasks = "Keeps the agent's plan: an ordered checklist of the steps needed to finish the task.";

sem PlanTasks.set_plan = "Replace the whole plan with a new ordered list of steps, numbered from 1. Call this once you know how you intend to solve the task, and again if the approach changes materially. Returns the rendered plan.";
sem PlanTasks.set_plan.steps = "The steps to take, in order, each a short imperative sentence.";

sem PlanTasks.update_task = "Change the status of one step, optionally attaching a short note. Mark a step 'doing' when you start it and 'done' once you have verified it. Returns the rendered plan.";
sem PlanTasks.update_task.task_id = "The number of the step to update, as shown in the plan.";
sem PlanTasks.update_task.status = "The step's new status.";
sem PlanTasks.update_task.note = "An optional short note, for example why a step is blocked. Pass an empty string to leave the existing note unchanged.";

sem PlanTasks.show_plan = "Show the current plan with each step's status. Use this to re-orient after a long stretch of work.";
