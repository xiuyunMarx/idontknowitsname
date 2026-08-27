"""Live state of one byllm program instance, as observed over its InterceptorLLM link.

What the client pushes (see jaclang/byllm/observe.py):

    {"type": "state", "pid", "arch", "obj", "attr", "value": repr | null, ["len", "digest"]}
    {"type": "enter", "pid", "ability", "here": {"arch","obj","fields":{attr: view}} | null,
                                        "visitor": {...} | null}

and what the server tells it to observe, inside the `registered` ack:

    {"type": "registered", ..., "watch": watch_set(program)}

`InstanceState` is the table; `bind_params` joins it with the static readiness
(`Program.ready_params`) into the bytes a consumer's `name = value` lines take.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from static_pass.primitives import Program

View = Dict[str, Any]  # {"value": repr} | {"value": None, "len": n, "digest": sha1}


@dataclass
class FieldValue:
    text: Optional[str]  # repr, or None when only the change was reported (too long)
    seq: int  # write sequence: later wins, and lets a binding be checked for staleness

    @property
    def available(self) -> bool:
        return self.text is not None


@dataclass
class Instance:
    arch: str
    obj: str
    fields: Dict[str, FieldValue] = field(default_factory=dict)


@dataclass
class Position:
    """Where the walker is, per client process: set by `enter` frames."""
    walker: Optional[str] = None  # obj id
    node: Optional[str] = None
    ability: str = ""


class InstanceState:
    def __init__(self) -> None:
        self.instances: Dict[Tuple[int, str], Instance] = {}  # (pid, obj) -> Instance
        self.position: Dict[int, Position] = {}
        self.seq = 0

    # --------------------------------------------------------------- frames

    def apply(self, frame: dict) -> bool:
        """Absorb a `state` / `enter` frame; False for any other frame type."""
        kind = frame.get("type")
        pid = int(frame.get("pid", -1))
        if kind == "state":
            self._set(pid, frame["arch"], frame["obj"], frame["attr"], frame)
            return True
        if kind == "enter":
            pos = self.position.setdefault(pid, Position())
            pos.ability = str(frame.get("ability", ""))
            for role in ("here", "visitor"):
                snap = frame.get(role)
                if not isinstance(snap, dict):
                    continue
                for attr, view in (snap.get("fields") or {}).items():
                    self._set(pid, snap["arch"], snap["obj"], attr, view)
                if role == "here":
                    pos.node = snap["obj"]
                else:
                    pos.walker = snap["obj"]
            return True
        return False

    def _set(self, pid: int, arch: str, obj: str, attr: str, view: View) -> None:
        self.seq += 1
        inst = self.instances.setdefault((pid, obj), Instance(arch, obj))
        inst.fields[attr] = FieldValue(view.get("value"), self.seq)

    def drop(self, pid: int) -> None:
        """The client process went away."""
        self.instances = {k: v for k, v in self.instances.items() if k[0] != pid}
        self.position.pop(pid, None)

    # -------------------------------------------------------------- queries

    def get(self, pid: int, arch: str, attr: str, obj: Optional[str] = None,
            binding: str = "") -> Optional[FieldValue]:
        """The field's latest value. `obj` pins an instance; otherwise `binding`
        ("walker" | "node") picks the object the walker is / stands on; otherwise
        the most recently written instance of that arch."""
        if obj is None and binding in ("walker", "node"):
            pos = self.position.get(pid)
            obj = (pos.walker if binding == "walker" else pos.node) if pos else None
        if obj is not None:
            inst = self.instances.get((pid, obj))
            return inst.fields.get(attr) if inst and inst.arch == arch else None
        best: Optional[FieldValue] = None
        for (p, _), inst in self.instances.items():
            if p == pid and inst.arch == arch and attr in inst.fields:
                fv = inst.fields[attr]
                if best is None or fv.seq > best.seq:
                    best = fv
        return best


# ------------------------------------------------------------------ binding

@dataclass
class Binding:
    param: str
    text: str  # the repr bytes of the `name = value` line
    origin: str  # "const" | "default" | "produced" | "observed"
    seq: int = 0  # state seq an observed value was taken at (0 for the static origins)


def watch_set(program: "Program") -> Dict[str, List[str]]:
    """arch -> `has` fields whose live value some byllm argument reads, plus every
    archetype hosting a callsite (empty list: its entry abilities still report
    `enter`, the moment to warm the calls inside). Sent in the `registered` ack."""
    out: Dict[str, List[str]] = {}
    for f in program.byLLMs:
        for src in f.param_source.values():
            if src.arch and src.attr and src.attr not in out.setdefault(src.arch, []):
                out[src.arch].append(src.attr)
    for site in program.callsites.values():
        if site.host:
            out.setdefault(site.host, [])
    return out


def bind_params(program: "Program", consumer_key: str, done: "set[str]", state: InstanceState, pid: int,
                via: Optional[str] = None, produced: Optional[Dict[str, str]] = None) -> Dict[str, Binding]:
    """Everything of `consumer_key` that can be bound now, as bytes: constants and
    untouched defaults from the static pass, produced returns (`produced`: producer
    callsite key -> repr the server itself generated), live field values from the
    observation table. A param needed but not observed yet is absent — the request
    will carry it."""
    from static_pass.primitives import SourceKind  # local: primitives imports this module

    out: Dict[str, Binding] = {}
    produced = produced or {}
    for name, r in program.ready_params(consumer_key, done, via).items():
        if not r.ready:
            continue
        src = r.source
        if r.known:
            out[name] = Binding(name, repr(r.value), "const" if src.kind is SourceKind.CONST else "default")
            continue
        if src.kind in (SourceKind.RET, SourceKind.RET_ANY):
            hit = next((k for k in src.producers if k in produced), None)
            if hit is not None:
                out[name] = Binding(name, produced[hit], "produced")
                continue
        if src.arch and src.attr:
            fv = state.get(pid, src.arch, src.attr, binding=src.binding)
            if fv is not None and fv.available:
                out[name] = Binding(name, fv.text or "", "observed", fv.seq)
    return out


def stale(bindings: Dict[str, Binding], program: "Program", consumer_key: str,
          state: InstanceState, pid: int) -> List[str]:
    """Params whose observed binding has been overwritten since it was taken."""
    consumer = program.callsites.get(consumer_key)
    out: List[str] = []
    for name, b in bindings.items():
        if b.origin != "observed" or consumer is None:
            continue
        src = consumer.param_source.get(name)
        fv = state.get(pid, src.arch, src.attr, binding=src.binding) if src else None
        if fv is None or fv.seq != b.seq:
            out.append(name)
    return out
