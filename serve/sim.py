"""A small discrete-event simulator of Controller._plan_step, for offline planner tests
and for gen_schedule.py to score candidate bursts.

Round structure mirrors the controller: requests whose model is resident are served
at once (`_dispatch`); otherwise the planner is asked for an order, its first
non-resident model is loaded (evicting by keep_value when the slots are full), and
the round repeats. With `gpu_slots > 1` a load may overlap the batch being served.
Waiting is charged unweighted: every unfinished, arrived instance pays each elapsed
second, which is the sum-of-completion-time objective the planners estimate."""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from serve.planner import Chain, PlanInput, Step, first_nonresident, keep_value, next_use


@dataclass
class Scenario:
    load: Dict[str, Dict[str, float]]                 # model -> {"host": s, "ssd": s}
    chains: List[List[Tuple[str, float]]]             # per instance: [(model, exec_s), ...]
    arrivals: Optional[List[float]] = None            # per instance, default all at 0
    resident: Sequence[str] = ()
    tier: Dict[str, str] = field(default_factory=dict)   # initial tier, default "host"
    gpu_slots: int = 1
    rate: Dict[str, float] = field(default_factory=dict)
    tau: float = 10.0
    gamma: float = 1.0
    # What the planner is told about chain i at step j: [(model, exec estimate), ...] with the
    # head first, as the controller's branch predictor + cost model would say. None = oracle
    # (the true remaining chain with true exec times).
    lookahead: Optional[Callable[[int, int], List[Tuple[str, float]]]] = None
    max_chain: int = 12


@dataclass
class SimResult:
    wait: float
    wall: float
    loads: List[Tuple[str, str, float]]
    actions: List[str]


def simulate(scn: Scenario, planner: Callable[[PlanInput], List[str]]) -> SimResult:
    n = len(scn.chains)
    arrivals = list(scn.arrivals or [0.0] * n)
    prog = [0] * n
    ready = list(arrivals)                       # when each chain reached its current step
    resident = set(scn.resident)
    tier = {m: ("gpu" if m in resident else scn.tier.get(m, "host")) for m in scn.load}
    clock, wait = 0.0, 0.0
    loads: List[Tuple[str, str, float]] = []
    actions: List[str] = []

    def unfinished() -> List[int]:
        return [i for i in range(n) if prog[i] < len(scn.chains[i])]

    def load_s(m: str, t: str) -> float:
        return 0.0 if t == "gpu" else scn.load[m][t]

    def plan_input(visible: List[int], busy: Sequence[str]) -> PlanInput:
        chains = []
        for i in visible:
            if prog[i] >= len(scn.chains[i]):
                continue
            seen = scn.chains[i][prog[i]:] if scn.lookahead is None else scn.lookahead(i, prog[i])
            steps = []
            for k, (m, ex) in enumerate(seen[:scn.max_chain]):
                if k == 0:
                    m = scn.chains[i][prog[i]][0]          # the head is a real request: its model is known
                w = 1.0 + (clock - ready[i]) / scn.tau if k == 0 else scn.gamma ** k
                steps.append(Step(m, ex, w, predicted=k > 0))
            head = "running" if steps[0].model in busy else "waiting"
            chains.append(Chain(i, tuple(steps), head))
        return PlanInput(chains, frozenset(resident), dict(tier), load_s, dict(scn.rate), scn.gpu_slots,
                         busy=frozenset(busy))

    def load(target: str, pi: PlanInput, idle: List[str]) -> float:
        if len(resident) >= scn.gpu_slots:
            nu = next_use(pi)
            victim = min(idle, key=lambda v: (keep_value(pi, v, nu), v))
            resident.discard(victim)
            tier[victim] = "host"
        secs = load_s(target, tier[target])
        loads.append((target, tier[target], secs))
        actions.append(target)
        resident.add(target)
        tier[target] = "gpu"
        return secs

    def charge(elapsed: float, visible: List[int]) -> None:
        nonlocal wait, clock
        wait += len(visible) * elapsed
        for i in unfinished():
            if clock < arrivals[i] <= clock + elapsed:
                wait += clock + elapsed - arrivals[i]
        clock += elapsed

    while unfinished():
        visible = [i for i in unfinished() if arrivals[i] <= clock]
        if not visible:
            clock = min(arrivals[i] for i in unfinished())
            continue
        served = [m for m in {scn.chains[i][prog[i]][0] for i in visible} if m in resident]
        if served:
            pi = plan_input(visible, served)
            elapsed = 0.0
            for m in served:
                batch = [i for i in visible if prog[i] < len(scn.chains[i]) and scn.chains[i][prog[i]][0] == m]
                run = 0.0
                for i in batch:
                    t, j = 0.0, prog[i]
                    while j < len(scn.chains[i]) and scn.chains[i][j][0] == m:
                        t += scn.chains[i][j][1]
                        j += 1
                    prog[i], run = j, max(run, t)
                elapsed = max(elapsed, run)
            if scn.gpu_slots > 1 or len(resident) < scn.gpu_slots:
                target = first_nonresident(planner(pi), resident)
                idle = [v for v in resident if v not in served]
                if target is not None and (len(resident) < scn.gpu_slots or idle):
                    elapsed = max(elapsed, load(target, pi, idle))
            charge(elapsed, visible)
            for i in visible:
                ready[i] = clock
        else:
            pi = plan_input(visible, [])
            order = planner(pi)
            target = first_nonresident(order, resident) or scn.chains[visible[0]][prog[visible[0]]][0]
            charge(load(target, pi, list(resident)), visible)
    return SimResult(wait, clock, loads, actions)
