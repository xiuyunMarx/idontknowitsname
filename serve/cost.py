"""Online cost model for the load planner.

Everything the planner prices comes from here: how long a model takes to reach the
GPU from each tier, how long a callsite's request runs, and how often each model is
asked for. Tables start from size-proportional priors (parameter count parsed from
the model name) and are replaced by EMAs of what the controller measures. The model
is knowledge, not a counter: `Controller.control({"reset"})` leaves it alone, and
`learn=False` freezes it so A/B cells run on identical estimates."""
import re
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

_PARAMS = re.compile(r"(\d+(?:\.\d+)?)[bB](?![A-Za-z0-9])")

PRIOR_SECS_PER_B = {"host": 0.6, "ssd": 2.0}   # seconds per billion parameters, before any observation
CREATE_PENALTY_S = 30.0                          # a model with no engine yet must be created and warmed up


def param_count(model: str) -> float:
    """Billions of parameters from the model name (`Qwen/Qwen2.5-0.5B-Instruct` -> 0.5); 1.0 if unparseable."""
    m = _PARAMS.search(model.rsplit("/", 1)[-1])
    return float(m.group(1)) if m else 1.0


class CostModel:
    def __init__(self, alpha: float = 0.3, window: int = 32, learn: bool = True):
        self.alpha, self.window, self.learn = alpha, window, learn
        self.load_ema: Dict[Tuple[str, str], float] = {}   # (model, from_tier) -> seconds
        self.load_n: Dict[Tuple[str, str], int] = {}
        self.exec_ema: Dict[str, float] = {}                # callsite_key -> seconds of one generate() turn
        self.exec_by_model: Dict[str, float] = {}           # model -> seconds, fallback for unseen callsites
        self.turns_ema: Dict[str, float] = {}               # callsite_key -> turns per call (tool loops)
        self._turns: Dict[str, int] = {}                    # callsite_key -> turns of the call in progress
        self.arrivals: Dict[str, Deque[float]] = {}         # model -> recent submit timestamps
        self.secs_per_b: Dict[str, float] = dict(PRIOR_SECS_PER_B)
        self._tier_obs: Dict[str, List[float]] = {}
        self.known: set = set()                             # models that have an engine

    # ---- observations ------------------------------------------------------------

    def register(self, model: str) -> None:
        self.known.add(model)

    def _ema(self, table: dict, key, value: float) -> None:
        table[key] = value if key not in table else (1 - self.alpha) * table[key] + self.alpha * value

    def observe_load(self, model: str, from_tier: str, secs: float) -> None:
        if not self.learn or from_tier == "gpu":
            return
        self._ema(self.load_ema, (model, from_tier), secs)
        self.load_n[(model, from_tier)] = self.load_n.get((model, from_tier), 0) + 1
        obs = self._tier_obs.setdefault(from_tier, [])
        obs.append(secs / param_count(model))
        self.secs_per_b[from_tier] = sum(obs) / len(obs)

    def observe_exec(self, callsite_key: str, model: str, secs: float, first_turn: bool = True) -> None:
        """One engine turn of `callsite_key`; `first_turn` marks the opening turn of a call, so
        the turns of the previous call on this callsite are complete and can be counted."""
        if not self.learn:
            return
        self._ema(self.exec_ema, callsite_key, secs)
        self._ema(self.exec_by_model, model, secs)
        if first_turn:
            if callsite_key in self._turns:
                self._ema(self.turns_ema, callsite_key, float(self._turns[callsite_key]))
            self._turns[callsite_key] = 1
        else:
            self._turns[callsite_key] = self._turns.get(callsite_key, 1) + 1

    def observe_arrival(self, model: str, now: float) -> None:
        if not self.learn:
            return
        self.arrivals.setdefault(model, deque(maxlen=self.window)).append(now)

    # ---- estimates ---------------------------------------------------------------

    def load_s(self, model: str, tier: str) -> float:
        """Seconds to bring `model` to the GPU from `tier`."""
        if tier == "gpu":
            return 0.0
        secs = self.load_ema.get((model, tier))
        if secs is None:
            secs = self.secs_per_b.get(tier, PRIOR_SECS_PER_B["ssd"]) * param_count(model)
        if self.known and model not in self.known:
            secs += CREATE_PENALTY_S
        return secs

    def exec_s(self, callsite_key: str, model: str) -> float:
        """Seconds one call of `callsite_key` occupies the engine: per-turn time times the
        turns a call takes (tool loops), or a size prior for unseen callsites."""
        turns = max(1.0, self.turns_ema.get(callsite_key, float(self._turns.get(callsite_key, 1))))
        if callsite_key in self.exec_ema:
            return self.exec_ema[callsite_key] * turns
        if model in self.exec_by_model:
            return self.exec_by_model[model]
        return 0.5 + 0.4 * param_count(model)

    def rate(self, model: str, now: float) -> float:
        """Arrivals per second for `model`, decaying while it stays unrequested."""
        ts = self.arrivals.get(model)
        if not ts or len(ts) < 2:
            return 0.0
        first, last = ts[0], ts[-1]
        lam = (len(ts) - 1) / (last - first) if last > first else float("inf")
        cap = len(ts) / (now - first) if now > first else float("inf")
        lam = min(lam, cap)
        return 0.0 if lam == float("inf") else lam

    def export(self, now: Optional[float] = None) -> dict:
        out = {"learn": self.learn,
               "secs_per_b": dict(self.secs_per_b),
               "load_s": {f"{m}|{t}": round(v, 3) for (m, t), v in self.load_ema.items()},
               "load_n": {f"{m}|{t}": n for (m, t), n in self.load_n.items()},
               "exec_s": {k: round(v, 3) for k, v in self.exec_ema.items()},
               "turns": {k: round(v, 2) for k, v in self.turns_ema.items()},
               "exec_by_model": {k: round(v, 3) for k, v in self.exec_by_model.items()},
               "arrivals": {m: len(ts) for m, ts in self.arrivals.items()}}
        if now is not None:
            out["rate"] = {m: round(self.rate(m, now), 4) for m in self.arrivals}
        return out
