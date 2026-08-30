"""Ordering: which model goes on the GPU next.

Every live instance contributes a chain of steps: the model it waits for or runs on,
then what branch prediction says follows. Three planners turn the chains into a load
order; `Controller._plan_step` executes the first non-resident model of that order,
then re-plans on the next wake (receding horizon), so only the head of the order is
ever committed.

  fifo_order    arrival order over the chains, one model once (the historical planner)
  greedy_order  Smith's rule over model families: the model with the least
                (load + batched exec) per unit of waiting weight goes first
  dp_order      bounded-depth DP over (resident set, per-chain progress) minimising
                the sum of waiting, with a greedy rollout past the horizon

The planners are pure: `PlanInput` carries every estimate they need, so they run
offline in tests.  Step semantics shared by all of them (and by `evaluate`, which
prices any order under the same model):

  action           a model at the head of at least one unfinished chain
  advance(m)       every chain whose head is m runs through its consecutive m-steps;
                   a chain's run costs the SUM of its steps (sequential turns) plus the
                   client gaps between them (arrives_in_s), the batch costs the MAX over
                   chains (they share the residency)
  load(m)          0 if resident, else load_s(m, tier); a model evicted inside the
                   plan reloads from host (host->ssd trimming is not modelled)
  cost(m)          (load + exec) x total weight of unfinished chains
  victim           lowest keep_value = L_reload / min(next use on a chain, 1/lambda)
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple

INF = float("inf")
STARVE_WEIGHT = 7.0   # a waiting head this heavy (age > 6 tau) is served next, whatever the ratio says


@dataclass(frozen=True)
class Step:
    model: str
    exec_s: float
    weight: float          # waiting head: 1 + age/tau; running/open head: 1; j-th predicted: gamma**j
    predicted: bool
    key: str = ""
    arrives_in_s: float = 0.0  # client-side gap (tool run, compute) after the previous step of the
                               # chain completes before this step's request arrives; 0 for a head
                               # that already arrived, the expected tool gap for an "open" head


@dataclass(frozen=True)
class Chain:
    inst: Any
    steps: Tuple[Step, ...]
    head: str              # "waiting" | "running" | "open" | "predicted"


@dataclass
class PlanInput:
    chains: List[Chain]
    resident: FrozenSet[str]
    tier: Dict[str, str]                        # model -> "gpu" | "host" | "ssd"
    load_s: Callable[[str, str], float]         # (model, tier) -> seconds
    rate: Dict[str, float] = field(default_factory=dict)
    gpu_slots: int = 1
    busy: FrozenSet[str] = frozenset()          # resident models with requests in flight
    pinned: FrozenSet[str] = frozenset()        # models an instance is mid-call on


@dataclass(frozen=True)
class State:
    resident: FrozenSet[str]
    evicted: FrozenSet[str]
    prog: Tuple[int, ...]


def fifo_order(seqs: Sequence[Sequence[str]]) -> List[str]:
    """FIFO over predicted chains: every model of each sequence, in arrival order, once."""
    return dedup(m for s in seqs for m in s)


def dedup(seq) -> List[str]:
    out: List[str] = []
    for m in seq:
        if m not in out:
            out.append(m)
    return out


def first_nonresident(order: Sequence[str], resident) -> Optional[str]:
    return next((m for m in order if m not in resident), None)


# ---- step semantics ---------------------------------------------------------------

def root(pi: PlanInput) -> State:
    return State(frozenset(pi.resident), frozenset(), tuple(0 for _ in pi.chains))


def heads(pi: PlanInput, prog: Tuple[int, ...]) -> Dict[str, List[int]]:
    """model -> chains whose current step is on it, in chain (arrival) order."""
    out: Dict[str, List[int]] = {}
    for c, j in enumerate(prog):
        steps = pi.chains[c].steps
        if j < len(steps):
            out.setdefault(steps[j].model, []).append(c)
    return out


def live_weight(pi: PlanInput, prog: Tuple[int, ...]) -> float:
    return sum(pi.chains[c].steps[j].weight for c, j in enumerate(prog) if j < len(pi.chains[c].steps))


def advance(pi: PlanInput, prog: Tuple[int, ...], m: str) -> Tuple[Tuple[int, ...], float]:
    """Serve every chain headed by `m` through its run of m-steps; returns (progress, batch exec)."""
    nxt, exec_s = list(prog), 0.0
    for c, j in enumerate(prog):
        steps = pi.chains[c].steps
        if j < len(steps) and steps[j].model == m:
            run = 0.0
            while j < len(steps) and steps[j].model == m:
                run += steps[j].arrives_in_s + steps[j].exec_s
                j += 1
            nxt[c] = j
            exec_s = max(exec_s, run)
    return tuple(nxt), exec_s


def next_use(pi: PlanInput, prog: Optional[Tuple[int, ...]] = None) -> Dict[str, float]:
    """Seconds until each model is next needed: chain work plus the client gaps between
    steps (0 for a current waiting/running head), inf if never."""
    prog = prog if prog is not None else tuple(0 for _ in pi.chains)
    out: Dict[str, float] = {}
    for c, j in enumerate(prog):
        t = 0.0
        for s in pi.chains[c].steps[j:]:
            t += s.arrives_in_s
            if t < out.get(s.model, INF):
                out[s.model] = t
            t += s.exec_s
    return out


def keep_value(pi: PlanInput, m: str, nu: Dict[str, float], reload: Optional[float] = None) -> float:
    """What evicting `m` is expected to cost per second: its reload (host->gpu unless given)
    divided by how soon it is wanted, on a chain or by its arrival rate."""
    lam = pi.rate.get(m, 0.0)
    horizon = min(nu.get(m, INF), 1.0 / lam if lam > 0 else INF)
    if horizon == INF:
        return 0.0
    return (pi.load_s(m, "host") if reload is None else reload) / max(0.1, horizon)


def _evictable(pi: PlanInput, st: State, hd: Dict[str, List[int]], m: str) -> List[str]:
    """Resident models that may leave to make room for `m`: no chain is on a real (non-predicted) step of them."""
    out = []
    for v in st.resident:
        if v == m:
            continue
        if any(not pi.chains[c].steps[st.prog[c]].predicted for c in hd.get(v, [])):
            continue
        if st == root(pi) and (v in pi.busy or v in pi.pinned):
            continue
        out.append(v)
    return out


def load_cost(pi: PlanInput, st: State, m: str) -> float:
    if m in st.resident:
        return 0.0
    return pi.load_s(m, "host" if m in st.evicted else pi.tier.get(m, "ssd"))


def step(pi: PlanInput, st: State, m: str, hd: Optional[Dict[str, List[int]]] = None
         ) -> Optional[Tuple[State, float, float]]:
    """Apply action `m`: (next state, waiting cost, wall seconds), or None if `m` cannot be loaded now."""
    hd = hd if hd is not None else heads(pi, st.prog)
    resident, evicted = st.resident, st.evicted
    if m not in resident:
        if len(resident) >= pi.gpu_slots:
            nu = next_use(pi, st.prog)
            cands = _evictable(pi, st, hd, m)
            if not cands:
                return None
            v = min(cands, key=lambda x: (keep_value(pi, x, nu), x))
            resident, evicted = resident - {v}, evicted | {v}
        resident, evicted = resident | {m}, evicted - {m}
    prog, exec_s = advance(pi, st.prog, m)
    dur = load_cost(pi, st, m) + exec_s
    return State(resident, evicted, prog), dur * live_weight(pi, st.prog), dur


# ---- planners ---------------------------------------------------------------------

def _greedy_pick(pi: PlanInput, st: State, hd: Dict[str, List[int]]) -> Optional[Tuple[str, State, float, float]]:
    starving = [(pi.chains[cs[0]].steps[st.prog[cs[0]]].weight, m) for m, cs in hd.items()
                if any(pi.chains[c].head == "waiting" and pi.chains[c].steps[st.prog[c]].weight >= STARVE_WEIGHT
                       for c in cs)]
    if starving:
        m = max(starving)[1]
        r = step(pi, st, m, hd)
        if r is not None:
            return (m, *r)
    best = None
    for m, cs in hd.items():
        r = step(pi, st, m, hd)
        if r is None:
            continue
        served = sum(pi.chains[c].steps[st.prog[c]].weight for c in cs)
        score = r[2] / max(served, 1e-9)
        if best is None or score < best[0]:
            best = (score, m, *r)
    return None if best is None else best[1:]


def greedy_order(pi: PlanInput, max_steps: Optional[int] = None) -> List[str]:
    """Smith's rule over model families, replayed to the end of the chains."""
    st, out = root(pi), []
    while max_steps is None or len(out) < max_steps:
        hd = heads(pi, st.prog)
        if not hd:
            break
        pick = _greedy_pick(pi, st, hd)
        if pick is None:
            break
        out.append(pick[0])
        st = pick[1]
    return dedup(out)


def _rollout(pi: PlanInput, st: State, steps: int) -> Tuple[float, List[str]]:
    cost, acts = 0.0, []
    for _ in range(steps):
        hd = heads(pi, st.prog)
        if not hd:
            break
        pick = _greedy_pick(pi, st, hd)
        if pick is None:
            break
        acts.append(pick[0])
        cost += pick[2]
        st = pick[1]
    return cost, acts


class _Budget(Exception):
    pass


def dp_order(pi: PlanInput, horizon: int = 4, budget: int = 500, rollout: int = 64, window: float = 30.0,
             info: Optional[dict] = None) -> List[str]:
    """Minimise the sum of waiting over the next `horizon` loads"""
    memo: Dict[Tuple[State, int], Tuple[float, List[str]]] = {}
    nodes = [0]

    def value(st: State, d: int) -> Tuple[float, List[str]]:
        if d >= horizon:
            cost, acts = _rollout(pi, st, rollout)
            bonus = window * sum(pi.rate.get(m, 0.0) * pi.load_s(m, "host") for m in st.resident)
            return cost - bonus, acts
        
        hd = heads(pi, st.prog)
        if not hd:
            return 0.0, []
        key = (st, d)
        if key in memo:
            return memo[key]
        nodes[0] += 1
        if nodes[0] > budget:
            raise _Budget()
        best: Optional[Tuple[float, List[str]]] = None
        for m in hd:
            r = step(pi, st, m, hd)
            if r is None:
                continue
            nxt, cost, _ = r
            v, acts = value(nxt, d + 1)
            if best is None or cost + v < best[0]:
                best = (cost + v, [m] + acts)
        best = best or (0.0, [])
        memo[key] = best
        return best

    try:
        cost, acts = value(root(pi), 0)
        fallback = False
    except _Budget:
        acts, cost, fallback = greedy_order(pi), INF, True
    if info is not None:
        info.update(nodes=nodes[0], fallback=fallback, cost=cost)
    return dedup(acts)


def evaluate(pi: PlanInput, order: Sequence[str]) -> Tuple[float, float]:
    """(sum of weighted waiting, wall seconds) of following `order` as a policy: at each
    point the earliest model of `order` that is a current head goes next."""
    st, cost, wall = root(pi), 0.0, 0.0
    while True:
        hd = heads(pi, st.prog)
        if not hd:
            return cost, wall
        m = next((m for m in order if m in hd), None)
        r = step(pi, st, m, hd) if m is not None else None
        if r is None:
            pick = _greedy_pick(pi, st, hd)
            if pick is None:
                return cost, wall
            m, r = pick[0], pick[1:]
        st, c, d = r
        cost += c
        wall += d


def choose(
    pi: PlanInput,
    fifo: Sequence[str],
    candidate: Sequence[str],
    hysteresis: float = 0.10,
    floor: float = 1.0,
) -> Tuple[List[str], bool]:
    """
    Choose between the candidate schedule and FIFO.

    If both schedules would load the same next non-resident model, use the
    candidate directly because it does not change the model-loading order.

    Otherwise, use the candidate only if its estimated cost improves over FIFO
    by both:
      - more than `hysteresis` times the FIFO cost, and
      - more than `floor` weighted seconds.

    This avoids changing the load order for small or noisy estimated gains.

    Returns:
        chosen_order: The schedule to execute.
        reordered: True if we intentionally changed the next model load away
                   from FIFO because the candidate was sufficiently better.
    """
    candidate_next_load = first_nonresident(candidate, pi.resident)
    fifo_next_load = first_nonresident(fifo, pi.resident)

    # Candidate does not change the next model load.
    if candidate_next_load is None or candidate_next_load == fifo_next_load:
        return list(candidate), False

    fifo_cost, _ = evaluate(pi, fifo)
    candidate_cost, _ = evaluate(pi, candidate)

    improvement = fifo_cost - candidate_cost
    required_improvement = max(hysteresis * fifo_cost, floor)

    if improvement > required_improvement:
        return list(candidate), True

    return list(fifo), False