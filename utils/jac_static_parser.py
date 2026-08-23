import os

import jaclang  # registers the .jac meta importer — must come first
import jaclang.jac0core.unitree as uni #type: ignore
from jaclang.jac0core.compile_options import CompileOptions #type: ignore
from jaclang.jac0core.program import JacProgram #type: ignore
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.utils import (SYSTEM_PERSONA, TOOL_INSTRUCTION, FIELD_EXPR_RE, LOOP_STMTS, norm, clean_type, json_type_of, literal_of, params_of, extract_llm_call, finish_tool_schema, format_tools_for_prompt, schema_entry, render_bindings)


@dataclass
class ByLLMDecl:
    """One `def f(...) -> T by llm(...)` declaration, fully described from UniIR at compile time."""
    name: str
    qualifier: str = ""  # "RagChat." for methods, "" for module-level defs
    kind: str = "func"  # "func" = byllm function; "visit" = `visit ... by llm()` routing call
    intent: str = ""  # visit only: the static intent= text (resolved through glob literals)
    params: List[Dict[str, Any]] = field(default_factory=list)  # {name, type, sem, required}
    return_type: str = "str"
    return_type_render: str = ""  # byllm's `_type_name` rendering (module-qualified generics); "" = use return_type
    sem: str = ""  # authored sem of the function itself
    owner_sem: str = ""  # sem of the owning archetype; drives the `self` identity zone
    tools: List[Dict[str, Any]] = field(default_factory=list)  # OpenAI tool schemas, finish_tool last
    call_params: Dict[str, Any] = field(default_factory=dict)  # literal kwargs of the llm() call
    extra_system_prompt: str = ""  # literal system_prompt= kwarg, if any
    module: str = ""
    lineno: int = 0
    reponse_format: Optional[Dict[str, Any]] = None  # expanded response_format (None for str returns / tools)
    return_type_obj: Optional[Any] = None  # materialized Python type translated from Jac obj/enum defs
    param_type_objs: Dict[str, Any] = field(default_factory=dict)  # param name -> materialized type (drives the schema zone)

    @property
    def key(self) -> str:
        return self.qualifier + self.name

    def signature(self) -> str:
        args = ", ".join(f"{p['name']}: {p.get('type_render') or p['type']}" if p["type"] else p["name"] for p in self.params)
        sig = f"{self.name}({args})"
        ret = self.return_type_render or self.return_type
        return f"{sig} -> {ret}" if ret else sig


class ByLLMCallsite:
    """One invocation site of a byllm decl. Duty includes:
    1. rendering the invariant system prompt and assembling the full prompt with args.
    2. Which byLLM call site will consumes this site's return (for dataflow)
    3. After this call, which callsite's which params is ready to be filled (for scheduling)."""

    def __init__(self, decl: ByLLMDecl, call: uni.FuncCall, scope: Optional[uni.UniNode]):
        self.decl: ByLLMDecl = decl
        self.call: uni.FuncCall = call
        self.scope = scope  # enclosing Ability or ModuleCode; where bare arg names resolve
        self.callsite_uuid: str = f"{decl.key}@{call.loc.first_line}:{call.loc.col_start}"
        self.consumers: List[Tuple[str, str]] = []  # (consumer decl key, param) fed by this site's return
        self.bound_args: Dict[str, str] = {}  # ready param reprs; insertion order = warm prefix order
        self._invariant_system: Optional[str] = None
        self._invariant_user: Optional[str] = None

    @property
    def key(self) -> str:
        return self.decl.key

    @property
    def invariant_system(self) -> str:
        """[SYSTEM] zone: persona, literal system_prompt extension, tool instruction
        and the text tool protocol — same composition order as byllm's factory +
        inject_tool_hint."""
        if self._invariant_system is None:
            d = self.decl
            system = SYSTEM_PERSONA
            if d.extra_system_prompt:
                system += "\n\n" + d.extra_system_prompt
            if d.tools:
                system += TOOL_INSTRUCTION + "\n\n" + format_tools_for_prompt(d.tools)
            self._invariant_system = system
        return self._invariant_system

    def render_invariant_prompt(self) -> str:
        """Compile-time user prefix: qualified typed signature header (+ authored sem) with the described params' schema rows indented beneath"""
        if self._invariant_user is None:
            d = self.decl
            header = d.qualifier + d.signature()
            if d.sem:
                header = f"{header} --- {d.sem}"
            lines = [header]
            for p in d.params:
                row = schema_entry(p["name"], p.get("type_render") or p["type"], p["sem"])
                for ln in row.split("\n") if row else []:
                    lines.append("      " + ln)  # byllm indents every line of the entry
            self._invariant_user = "\n".join(lines)
        return self._invariant_user

    def incremental_prompt(self, known_args: Dict[str, str]) -> None:
        """Bind ready parameter reprs into this site's warm state. Skip the already binded params"""
        names = {p["name"] for p in self.decl.params}
        for name, view in known_args.items():
            if name in names and name not in self.bound_args:
                self.bound_args[name] = view

    def get_ready_prompt(self) -> List[Dict[str, str]]:
        """[system, user-prefix] of what is known now — invariant plus the bound bindings in bind order. 
        The server prefills the KV cache with exactly this."""
        user = "\n".join([self.render_invariant_prompt()]
                         + [f"{n} = {v}" for n, v in self.bound_args.items()])
        return [
            {"role": "system", "content": self.invariant_system},
            {"role": "user", "content": user},
        ]

    def _render_full_prompt(self, args: Dict[str, str], self_view: Optional[str] = None) -> str:
        """Serve-path user content as an extension of the warmed prefix: bound params
        first, in bind order with their bound bytes (a conflicting request value wins
        in place — correctness first, cache past that point is lost), then the
        request's remaining params in declaration order, then the `self` zone."""
        lines = [self.render_invariant_prompt()]
        for name, bound in self.bound_args.items():
            lines.append(f"{name} = {args.get(name, bound)}")
        for p in self.decl.params:
            name = p["name"]
            if name in args and name not in self.bound_args:
                lines.append(f"{name} = {args[name]}")
        zones = ["\n".join(lines)]
        if self_view is not None and self.decl.owner_sem:
            zones.append(f"self = {self_view} ---- {self.decl.owner_sem}")
        return "\n\n".join(zones)

    def assemble_prompt(self, args: Dict[str, str], self_view: Optional[str] = None) -> List[Dict[str, str]]:
        """[system, user] messages for one call frame; already-bound params keep their
        warmed position and bytes (see _render_full_prompt)."""
        return [
            {"role": "system", "content": self.invariant_system},
            {"role": "user", "content": self._render_full_prompt(args, self_view)},
        ]

    def clear_bindings(self) -> None:
        """Reset warm state after the call is served (or speculation is abandoned)."""
        self.bound_args.clear()

class ProgramTopology:
    def __init__(self, program_name: str, src_path: str):
        self.program_name = program_name
        self.src_path = src_path
        self._options = CompileOptions(type_check=False, no_cgen=True, force_target_program=True)
        self.module = JacProgram().compile(self.src_path, options=self._options)
        self.decls: Dict[str, ByLLMDecl] = {}
        self.callsites: List[ByLLMCallsite] = []
        self.topology: Dict[str, List[str]] = {}  # key -> may-run-next keys
        self.provenance: Dict[str, Dict[str, Dict[str, Any]]] = {}  # key -> param -> source spec
        self.ret_consumers: Dict[str, List[Tuple[str, str]]] = {}  # producer -> [(consumer, param)]
        self._parsed = False

    # ------------------------------------------------------------------ build

    def parse_dependency(self) -> None:
        """Build all static tables: decls, callsites (invariants rendered), topology,
        provenance, ret-consumer index; then attach each site's consumer edges."""
        if self._parsed:
            return
        self.decls = self.get_decls()
        self.callsites = self._collect_callsites()
        self.topology = self._build_topology()
        self.provenance = self._build_provenance()
        self.ret_consumers = self._invert_provenance()
        for site in self.callsites:
            site.consumers = list(self.ret_consumers.get(site.key, []))
        self._parsed = True

    def get_decls(self) -> Dict[str, ByLLMDecl]:
        """Every `def ... by llm()` in the module. Type materialization deferred: the
        interceptor wire protocol carries argument reprs, so serving never needs it."""
        if self.decls:
            return self.decls
        decls: Dict[str, ByLLMDecl] = {}
        for ab in self.module.get_all_sub_nodes(uni.Ability):
            if not ab.is_genai_ability:
                continue
            if not isinstance(ab.signature, uni.FuncSignature):
                continue  # event-signature abilities can't be a byllm def
            name = ab.name_ref.value if isinstance(ab.name_ref, uni.Name) else ab.py_resolve_name()
            owner = ab.method_owner
            qualifier = f"{owner.name.value}." if owner is not None else ""
            tool_names, call_params, extra_sys = extract_llm_call(ab.body)
            ret = clean_type(ab.signature.return_type.unparse()) if ab.signature.return_type else "str"
            tools = [self._tool_schema(t) for t in tool_names]
            if tools:
                tools.append(finish_tool_schema(ret))
            decl = ByLLMDecl(
                name=name,
                qualifier=qualifier,
                params=params_of(ab.signature),
                return_type=ret,
                sem=(ab.semstr or "").strip(),
                owner_sem=(getattr(owner, "semstr", "") or "").strip() if owner is not None else "",
                tools=tools,
                call_params=call_params,
                extra_system_prompt=extra_sys,
                module=ab.loc.mod_path,
                lineno=ab.loc.first_line,
            )
            if decl.key in decls:
                print(f"[jac-static] warning: duplicate byllm decl {decl.key!r} at {decl.module}:{decl.lineno}; keeping the first")
                continue
            decls[decl.key] = decl
        return decls

    def _tool_schema(self, tool_name: str) -> Dict[str, Any]:
        simple = tool_name.split(".")[-1]
        ab = None
        for cand in self.module.get_all_sub_nodes(uni.Ability):
            if cand.is_genai_ability or not isinstance(cand.name_ref, uni.Name):
                continue
            if cand.name_ref.value == simple and isinstance(cand.signature, uni.FuncSignature):
                ab = cand
                break
        if ab is None:
            print(f"[jac-static] warning: tool {tool_name!r} not resolvable statically; name-only schema")
            return {"type": "function", "function": {"name": simple, "description": simple, "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
        sem = (ab.semstr or "").strip()
        props: Dict[str, Any] = {}
        required: List[str] = []
        for p in params_of(ab.signature):
            props[p["name"]] = {"type": json_type_of(p["type"]), "description": p["sem"] or p["type"]}
            if p["required"]:
                required.append(p["name"])
        return {"type": "function", "function": {"name": simple, "description": sem or simple, "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}}

    # -------------------------------------------------------------- callsites

    def _by_simple(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for key, d in self.get_decls().items():
            out.setdefault(d.name, []).append(key)
        return out

    def _resolve_call(self, call: uni.FuncCall) -> List[str]:
        """Decl keys a FuncCall may invoke ([] for non-byllm calls). `self.m()` is
        disambiguated by the enclosing archetype; leftover ambiguity returns all
        candidates (over-approximation, fine for topology; provenance requires 1)."""
        target = norm(call.target.unparse()) if getattr(call, "target", None) is not None else ""
        simple = target.split(".")[-1]
        keys = self._by_simple().get(simple, [])
        if len(keys) <= 1:
            return list(keys)
        if target.startswith("self."):
            encl = call.find_parent_of_type(uni.Ability)
            owner = encl.method_owner if encl is not None else None
            if owner is not None and f"{owner.name.value}.{simple}" in keys:
                return [f"{owner.name.value}.{simple}"]
        return list(keys)

    def _collect_callsites(self) -> List[ByLLMCallsite]:
        sites: List[ByLLMCallsite] = []
        for call in self.module.get_all_sub_nodes(uni.FuncCall):
            keys = self._resolve_call(call)
            if not keys:
                continue
            if len(keys) > 1:
                print(f"[jac-static] warning: ambiguous call {call.target.unparse().strip()!r} at line {call.loc.first_line} resolves to {keys}; site skipped")
                continue
            scope = call.find_parent_of_type(uni.Ability) or call.find_parent_of_type(uni.ModuleCode)
            sites.append(ByLLMCallsite(self.get_decls()[keys[0]], call, scope))
        return sites

    def sites_of(self, key: str) -> List[ByLLMCallsite]:
        self.parse_dependency()
        return [s for s in self.callsites if s.key == key]

    def site_of(self, key: str, site: Optional[str] = None) -> ByLLMCallsite:
        """The callsite a wire request came from."""
        sites = self.sites_of(key)
        if not sites:
            raise KeyError(f"no byllm callsite for key {key!r}")
        if site:
            fname, _, line = site.rpartition(":")
            for s in sites:
                if (line.isdigit() and s.call.loc.first_line == int(line)
                        and os.path.basename(s.call.loc.mod_path) == os.path.basename(fname)):
                    return s
        return sites[0]

    # --------------------------------------------------------------- topology

    def _calls_in(self, node: uni.UniNode) -> List[str]:
        """Decl keys of every byllm call within `node` (itself included)."""
        out: List[str] = []
        calls = ([node] if isinstance(node, uni.FuncCall) else []) + list(node.get_all_sub_nodes(uni.FuncCall))
        for c in calls:
            for k in self._resolve_call(c):
                if k not in out:
                    out.append(k)
        return out

    def _next_keys(self, call: uni.FuncCall) -> Tuple[List[str], Optional[uni.Ability]]:
        """byllm keys reachable after `call` completes, scanning forward in control flow.

        Over-approximation on purpose: every byllm call in any later statement at any
        enclosing level counts (branch precision dropped — a spurious edge only wastes
        a speculative prefill; a missed edge costs a cold TTFT). Loop parents add their
        contained calls as back-edges. Second value = enclosing Ability whose body can
        end, i.e. the caller's continuation decides what runs next."""
        out: List[str] = []
        cur: uni.UniNode = call
        while True:
            p = cur.parent
            if p is None:
                return out, None
            if isinstance(p, uni.FuncCall):
                for k in self._resolve_call(p):  # nested arg: the outer call runs right after
                    if k not in out:
                        out.append(k)
            kids = list(getattr(p, "kid", None) or [])
            if cur in kids:
                for st in kids[kids.index(cur) + 1:]:
                    if isinstance(st, uni.CodeBlockStmt) and not isinstance(st, uni.ElseIf):
                        for k in self._calls_in(st):
                            if k not in out:
                                out.append(k)
            if isinstance(p, LOOP_STMTS):
                for k in self._calls_in(p):
                    if k not in out:
                        out.append(k)
            if isinstance(p, uni.Ability):
                return out, p
            if isinstance(p, (uni.ModuleCode, uni.Module)):
                return out, None
            cur = p

    def _continuation_keys(self, ability: uni.Ability, visited: set) -> List[str]:
        """What can run after `ability`'s body ends: the byllm calls following each of
        its own invocation sites, chased recursively through enclosing callables."""
        owner = ability.method_owner
        kind = owner.arch_type.value if owner is not None and hasattr(getattr(owner, "arch_type", None), "value") else ""
        if kind in ("walker", "node"):
            return []  # traversal-queue continuation (visit routing) not modeled in this rewrite yet
        name = ability.name_ref.value if isinstance(ability.name_ref, uni.Name) else ability.py_resolve_name()
        if name in visited:
            return []
        visited.add(name)
        out: List[str] = []
        for call in self.module.get_all_sub_nodes(uni.FuncCall):
            if getattr(call, "target", None) is None or norm(call.target.unparse()).split(".")[-1] != name:
                continue
            keys, escaped = self._next_keys(call)
            if escaped is not None:
                keys = keys + self._continuation_keys(escaped, visited)
            for k in keys:
                if k not in out:
                    out.append(k)
        return out

    def _build_topology(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {k: [] for k in self.get_decls()}
        for site in self.callsites:
            succ, escaped = self._next_keys(site.call)
            if escaped is not None:
                succ = succ + self._continuation_keys(escaped, set())
            out = graph[site.key]
            for k in succ:
                if k not in out:
                    out.append(k)
        return graph

    # ------------------------------------------------------------- provenance

    def _classify_name(self, name: str, scope: uni.UniNode, seen: Optional[set] = None) -> Dict[str, Any]:
        """Where a local variable's value comes from, within one ability/module scope."""
        seen = seen if seen is not None else set()
        if name in seen:
            return {"kind": "unknown", "expr": name}
        seen.add(name)
        for asg in scope.get_all_sub_nodes(uni.Assignment):
            tgt = asg.target[0] if asg.target else None
            if not (isinstance(tgt, uni.AstSymbolNode) and tgt.sym_name == name and asg.value is not None): #type: ignore
                continue
            if isinstance(asg.value, uni.FuncCall):
                keys = self._resolve_call(asg.value)
                if len(keys) == 1:
                    return {"kind": "ret", "of": keys[0]}
                # A tool or plain function: producer named, value not derivable statically.
                return {"kind": "call", "of": norm(asg.value.target.unparse())}
            m = FIELD_EXPR_RE.match(norm(asg.value.unparse()))
            if m:  # `x = self.request`: the local keeps the field's identity
                return {"kind": "field", "scope": m.group(1), "attr": m.group(2), "slice": m.group(3)}
        for loop in scope.get_all_sub_nodes(uni.InForStmt):
            if getattr(loop.target, "sym_name", None) != name:
                continue
            src = loop.collection
            keys = self._resolve_call(src) if isinstance(src, uni.FuncCall) else []
            key = keys[0] if len(keys) == 1 else None
            if key is None and isinstance(src, uni.Name):
                inner = self._classify_name(src.value, scope, seen)
                key = inner.get("of") if inner.get("kind") == "ret" else None
            if key:
                return {"kind": "ret_item", "of": key}  # loop var over a byllm call's returned list
        return {"kind": "unknown", "expr": name}

    def _classify_arg(self, expr: uni.UniNode, site: ByLLMCallsite) -> Dict[str, Any]:
        """Provenance spec of one argument: const / field / ret / ret_item / call / unknown."""
        ok, v = literal_of(expr)
        if ok:
            return {"kind": "const", "value": v}
        m = FIELD_EXPR_RE.match(norm(expr.unparse()))
        if m:
            return {"kind": "field", "scope": m.group(1), "attr": m.group(2), "slice": m.group(3)}
        if isinstance(expr, uni.FuncCall):
            keys = self._resolve_call(expr)
            if len(keys) == 1:
                return {"kind": "ret", "of": keys[0]}
        if isinstance(expr, uni.Name) and site.scope is not None:
            return self._classify_name(expr.value, site.scope)
        return {"kind": "unknown", "expr": expr.unparse().strip()}

    def _build_provenance(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """key -> param -> source spec, merged over invocation sites (disagreement -> unknown)."""
        prov: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for site in self.callsites:
            pos_args: List[uni.UniNode] = []
            kw_args: Dict[str, uni.UniNode] = {}
            for prm in site.call.params or []:
                if isinstance(prm, uni.KWPair) and prm.key is not None:
                    kw_args[prm.key.unparse().strip()] = prm.value
                elif not isinstance(prm, uni.KWPair):
                    pos_args.append(prm)
            merged = prov.setdefault(site.key, {})
            for i, p in enumerate(site.decl.params):
                expr = kw_args.get(p["name"]) if p["name"] in kw_args else (pos_args[i] if i < len(pos_args) else None)
                if expr is None:
                    continue
                spec = self._classify_arg(expr, site)
                if p["name"] in merged and merged[p["name"]] != spec:
                    merged[p["name"]] = {"kind": "unknown", "expr": "<sites disagree>"}
                else:
                    merged.setdefault(p["name"], spec)
        return prov

    def _invert_provenance(self) -> Dict[str, List[Tuple[str, str]]]:
        """producer key -> [(consumer key, param)]: dataflow, NOT topology adjacency —
        a result reaches consumers any number of calls downstream."""
        out: Dict[str, List[Tuple[str, str]]] = {}
        for ckey, params in self.provenance.items():
            for pname, spec in params.items():
                if spec.get("kind") in ("ret", "ret_item") and spec.get("of"):
                    out.setdefault(spec["of"], []).append((ckey, pname))
        return out

    # ---------------------------------------------------------------- queries

    def next_calls(self, key: str) -> List[str]:
        """1. Which byllm calls may run next after `key`."""
        self.parse_dependency()
        return self.topology.get(key, [])

    def ready_params(self, consumer: str, done: set) -> Dict[str, Dict[str, Any]]:
        """2. Readiness of `consumer`'s params given the completed call keys `done`.
        const: known at compile time. field: derivable now from live walker/node state
        (needs observation). ret/ret_item: ready iff the producer completed.
        call/unknown: never derivable ahead — only observable after a serve."""
        self.parse_dependency()
        out: Dict[str, Dict[str, Any]] = {}
        for pname, spec in self.provenance.get(consumer, {}).items():
            kind = spec["kind"]
            ready = kind in ("const", "field") or (kind in ("ret", "ret_item") and spec.get("of") in done)
            out[pname] = {"ready": ready, "spec": spec}
        return out

    def consumers_of(self, producer: str) -> List[Tuple[str, str]]:
        """3. Which (call, param) the result of `producer` feeds into."""
        self.parse_dependency()
        return self.ret_consumers.get(producer, [])


if __name__ == "__main__":
    import sys

    p = ProgramTopology(program_name="cli", src_path=sys.argv[1])
    p.parse_dependency()
    print("decls:", list(p.decls))
    print("\ncallsites (uuid | consumers):")
    for s in p.callsites:
        print(f"  {s.callsite_uuid} | {s.consumers}")
    print("\ntopology (may-run-next):")
    for k, succ in p.topology.items():
        print(f"  {k} -> {succ}")
    print("\nprovenance (param -> value source):")
    for k, params in p.provenance.items():
        for pname, spec in params.items():
            print(f"  {k}.{pname} <- {spec}")
    if p.callsites:
        s0 = p.callsites[-1]
        demo = {prm["name"]: repr(f"<{prm['name']} value>") for prm in s0.decl.params}
        partial = dict(list(demo.items())[:1])
        print(f"\n--- invariant system of {s0.callsite_uuid} ---\n{s0.invariant_system}")
        print(f"\n--- partial assembly ({list(partial)}) ---\n{s0.assemble_prompt(partial)[1]['content']}")
        print(f"\n--- full assembly ---\n{s0.assemble_prompt(demo)[1]['content']}")
