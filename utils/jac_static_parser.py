import json
import os
import re

import jaclang  # registers the .jac meta importer — must come first
import jaclang.jac0core.unitree as uni #type: ignore
from jaclang.byllm.schema import json_to_instance #type: ignore
from jaclang.jac0core.compile_options import CompileOptions #type: ignore
from jaclang.jac0core.program import JacProgram #type: ignore
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from utils.utils import (SYSTEM_PERSONA, TOOL_INSTRUCTION, FIELD_EXPR_RE, LOOP_STMTS, ROUTE_SYSTEM, ROUTE_ZONE_LABEL, ROUTE_LAYOUT_DEFAULT, ROUTE_LAYOUT_CACHE, norm, clean_type, json_type_of, literal_of, params_of, extract_llm_call, finish_tool_schema, format_tools_for_prompt, schema_entry, render_bindings)

# Statements whose byllm calls are not certain to run: an enclosing body may end without them.
CONDITIONAL_STMTS = LOOP_STMTS + tuple(getattr(uni, n) for n in ("IfStmt", "TryStmt", "MatchStmt") if hasattr(uni, n))

VISIT_PREFIX = "__visit@"

# The builtin names a return-type annotation may eval to. A user obj/enum name
# is a NameError on purpose: its class is never materialized server-side, so its
# instance repr cannot be reproduced here.
_SERVE_TYPES = {"str": str, "int": int, "float": float, "bool": bool, "list": list,
                "dict": dict, "tuple": tuple, "set": set, "None": None, "any": object, "Any": object}


def visit_key(mod_path: str, lineno: int) -> str:
    """Wire identity of one `visit <edges> by llm(...)` site.

    Location-derived on purpose: a routing call has no callable behind it, so the
    interceptor cannot name it. Both sides reach this string from the same two
    facts — source file and line — the static pass from `VisitStmt.loc`, the
    client from the `.jac` stack frame that is executing the visit."""
    return f"{VISIT_PREFIX}{os.path.basename(mod_path)}:{lineno}"


def route_system_prompt(select: Any) -> str:
    """byllm's routing system text for a given `select=`; mirrors route_visit."""
    if isinstance(select, int) and not isinstance(select, bool) and select < 1:
        select = "all"  # route_visit normalizes a nonsense count away before use
    text = ROUTE_SYSTEM
    if select == 1:
        text += " Choose exactly one."
    elif isinstance(select, int) and not isinstance(select, bool):
        text += f" Choose exactly {select}."
    elif isinstance(select, tuple) and len(select) == 2:
        text += f" Choose between {select[0]} and {select[1]} (inclusive)."
    return text


def route_layout() -> Tuple[str, ...]:
    """Runtime zone order route_visit will use. JAC_ROUTE_CACHE_LAYOUT is read in
    the *client* process; the server only mirrors it for diagnostics, since the
    warmable prefix (system + `Goal:`) precedes the zones under either layout."""
    return ROUTE_LAYOUT_CACHE if os.environ.get("JAC_ROUTE_CACHE_LAYOUT") == "1" else ROUTE_LAYOUT_DEFAULT


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
    owner_arch: str = ""  # archetype owning the enclosing ability ("" at module level)
    owner_kind: str = ""  # "walker" | "node" | "obj" | ""
    zones: List[str] = field(default_factory=list)  # visit only: runtime zone order (see route_layout)
    candidates: List[str] = field(default_factory=list)  # visit only: node archetypes the router may pick
    walkers: List[str] = field(default_factory=list)  # visit only: walker archetypes that can be running the visit
    reponse_format: Optional[Dict[str, Any]] = None  # expanded response_format (None for str returns / tools)
    return_type_obj: Optional[Any] = None  # materialized Python type translated from Jac obj/enum defs
    param_type_objs: Dict[str, Any] = field(default_factory=dict)  # param name -> materialized type (drives the schema zone)

    @property
    def key(self) -> str:
        return self.qualifier + self.name

    @property
    def is_visit(self) -> bool:
        return self.kind == "visit"

    def signature(self) -> str:
        args = ", ".join(f"{p['name']}: {p.get('type_render') or p['type']}" if p["type"] else p["name"] for p in self.params)
        sig = f"{self.name}({args})"
        ret = self.return_type_render or self.return_type
        return f"{sig} -> {ret}" if ret else sig

    def parse_response(self, output: Any) -> Tuple[bool, Any]:
        """The typed value the client will build from this decl's `final` frame,
        mirrored byte-for-byte (interceptorLLM.impl.jac json-dumps a non-str wire
        output, MTRuntime.parse_response passes str returns through and feeds the
        rest to json.loads + json_to_instance) — so a warm arg binding of its repr
        equals the client's serve-time arg repr. (False, None) when the value is
        not derivable here: a user obj/enum return, or a payload whose parse fails
        (which the client answers with a reject)."""
        if not isinstance(output, str):
            output = json.dumps(output)
        if self.return_type == "str" or not output.strip():
            return True, output
        try:
            ty = eval(self.return_type, {"__builtins__": {}}, _SERVE_TYPES)
            return True, json_to_instance(json.loads(output), ty)
        except Exception:
            return False, None


class ByLLMCallsite:
    """One invocation site of a byllm decl. Duty includes:
    1. rendering the invariant system prompt and assembling the full prompt with args.
    2. Which byLLM call site will consumes this site's return (for dataflow)
    3. After this call, which callsite's which params is ready to be filled (for scheduling)."""

    def __init__(self, decl: ByLLMDecl, call: uni.FuncCall, scope: Optional[uni.UniNode],
                 stmt: Optional[uni.VisitStmt] = None):
        self.decl: ByLLMDecl = decl
        self.call: uni.FuncCall = call
        self.scope = scope  # enclosing Ability or ModuleCode; where bare arg names resolve
        self.stmt = stmt  # visit routing: the VisitStmt whose location is the wire identity
        # A visit decl is already one-per-location, so its key is the whole identity.
        self.callsite_uuid: str = decl.key if stmt is not None else f"{decl.key}@{call.loc.first_line}:{call.loc.col_start}"
        self.consumers: List[Tuple[str, str]] = []  # (consumer decl key, param) fed by this site's return
        # A callsite is an immutable compile-time template. Ready parameter reprs
        # ("bindings": name -> repr, insertion order = warm prefix order) belong to
        # the running workflow instance and are passed in by the server.
        self._invariant_system: Optional[str] = None
        self._invariant_user: Optional[str] = None

    @property
    def key(self) -> str:
        return self.decl.key

    @property
    def is_visit(self) -> bool:
        return self.decl.is_visit

    @property
    def invariant_system(self) -> str:
        """[SYSTEM] zone: persona, literal system_prompt extension, tool instruction
        and the text tool protocol — same composition order as byllm's factory +
        inject_tool_hint. A visit site instead carries route_visit's routing text,
        which `select=` fixes at compile time."""
        if self._invariant_system is None:
            d = self.decl
            if d.is_visit:
                self._invariant_system = route_system_prompt(d.call_params.get("select", "all"))
                return self._invariant_system
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
            if d.is_visit:
                # Everything below `Goal:` — the walker, the current node, the
                # candidate list — is graph state that only exists once the program
                # runs. route_visit emits `Goal:` first under both layouts, so this
                # is the whole compile-time prefix of a routing prompt.
                self._invariant_user = f"Goal: {d.intent}" if d.intent else ""
                return self._invariant_user
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

    def incremental_prompt(self, bindings: Dict[str, str], known_args: Dict[str, str]) -> None:
        """Bind ready parameter reprs into `bindings`, one instance's warm state for
        this site. A rebind (retried producer, fresher output) overwrites in place —
        dict insertion order survives an overwrite, so the param keeps its
        warm-prefix position.
        A routing site binds zones instead of params — they are the observations
        (walker, current node, candidate list) that stand in for arguments there."""
        names = set(self.decl.zones) if self.is_visit else {p["name"] for p in self.decl.params}
        for name, view in known_args.items():
            if name in names:
                bindings[name] = view

    def _visit_zone_lines(self, values: Dict[str, str]) -> List[str]:
        """Bound routing zones, in the order route_visit will emit them. Zones are
        joined with a blank line there, not the single newline a param binding uses."""
        out: List[str] = []
        for zone in self.decl.zones:
            if zone in values:
                out.append(f"{ROUTE_ZONE_LABEL[zone]}\n{values[zone]}")
        return out

    def get_ready_prompt(self, bindings: Dict[str, str]) -> List[Dict[str, str]]:
        """[system, user-prefix] of what is known now — invariant plus `bindings` in
        bind order. The server prefills the KV cache with exactly this."""
        if self.is_visit:
            parts = [self.render_invariant_prompt()] + self._visit_zone_lines(bindings)
            user = "\n\n".join(part for part in parts if part)
        else:
            user = "\n".join([self.render_invariant_prompt()]
                              + [f"{n} = {v}" for n, v in bindings.items()])
        return [
            {"role": "system", "content": self.invariant_system},
            {"role": "user", "content": user},
        ]

    def _render_full_prompt(self, args: Dict[str, str], self_view: Optional[str],
                            bindings: Dict[str, str]) -> str:
        """Serve-path user content as an extension of the warmed prefix: bound params
        first, in bind order with their bound bytes (a conflicting request value wins
        in place — correctness first, cache past that point is lost), then the
        request's remaining params in declaration order, then the `self` zone.

        A visit site renders route_visit's zone layout instead. The served bytes for
        a routing call come from the client (it alone can describe the live graph);
        this rendering is the parity check that the warm prefix is a real prefix."""
        if self.is_visit:
            parts = [self.render_invariant_prompt()] + self._visit_zone_lines(
                {**bindings, **{k: v for k, v in args.items() if k in self.decl.zones}}
            )
            return "\n\n".join(part for part in parts if part)
        lines = [self.render_invariant_prompt()]
        for name, bound in bindings.items():
            lines.append(f"{name} = {args.get(name, bound)}")
        for p in self.decl.params:
            name = p["name"]
            if name in args and name not in bindings:
                lines.append(f"{name} = {args[name]}")
        zones = ["\n".join(lines)]
        if self_view is not None and self.decl.owner_sem:
            zones.append(f"self = {self_view} ---- {self.decl.owner_sem}")
        return "\n\n".join(zones)

    def assemble_prompt(self, args: Dict[str, str], self_view: Optional[str] = None,
                        bindings: Optional[Dict[str, str]] = None) -> List[Dict[str, str]]:
        """[system, user] messages for one call frame; params already bound in this
        instance's `bindings` keep their warmed position and bytes (see
        _render_full_prompt)."""
        return [
            {"role": "system", "content": self.invariant_system},
            {"role": "user", "content": self._render_full_prompt(args, self_view, bindings or {})},
        ]

class ProgramTopology:
    def __init__(self, program_name: str, src_path: str):
        self.program_name = program_name
        self.src_path = src_path
        self._options = CompileOptions(type_check=False, no_cgen=True, force_target_program=True)
        self.module = JacProgram().compile(self.src_path, options=self._options)
        self.decls: Dict[str, ByLLMDecl] = {}
        self.callsites: List[ByLLMCallsite] = []
        self.topology: Dict[str, List[str]] = {}  # callsite_uuid -> may-run-next keys
        self.provenance: Dict[str, Dict[str, Dict[str, Any]]] = {}  # key -> param -> source spec
        self.ret_consumers: Dict[str, List[Tuple[str, str]]] = {}  # producer -> [(consumer, param)]
        self._parsed = False
        self._glob_cache: Optional[Dict[str, Any]] = None
        self._arch_cache: Optional[Dict[str, uni.Archetype]] = None
        self._writer_cache: Optional[Dict[Tuple[str, str], List[str]]] = None

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
        """Every `def ... by llm()` in the module."""
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
            decl.owner_arch = owner.name.value if owner is not None else ""
            decl.owner_kind = self._arch_kind(owner)
            if decl.key in decls:
                print(f"[jac-static] warning: duplicate byllm decl {decl.key!r} at {decl.module}:{decl.lineno}; keeping the first")
                continue
            decls[decl.key] = decl
        for stmt in self.module.get_all_sub_nodes(uni.VisitStmt):
            decl = self._visit_decl(stmt)
            if decl is not None:
                decls[decl.key] = decl  # location-keyed: a duplicate is impossible
        return decls

    # ----------------------------------------------------- visit-by routing

    @staticmethod
    def _visit_by(stmt: uni.VisitStmt) -> Optional[uni.FuncCall]:
        """The `llm(...)` of `visit <edges> by llm(...)`; None for a plain visit.
        The `by` operator is a BinaryExpr in UniIR — there is no Ability behind a
        routing call, which is why it needs a synthesized decl at all."""
        target = stmt.target
        if (isinstance(target, uni.BinaryExpr)
                and getattr(target.op, "name", "") == "KW_BY"
                and isinstance(target.right, uni.FuncCall)):
            return target.right
        return None

    def _globs(self) -> Dict[str, Any]:
        """Module globals that constant-fold. `intent=` is routinely a glob so the
        text can be shared between routers; without this the invariant is empty."""
        if self._glob_cache is None:
            out: Dict[str, Any] = {}
            for gv in self.module.get_all_sub_nodes(uni.GlobalVars):
                for asg in gv.assignments:
                    if asg.value is None:
                        continue
                    ok, v = literal_of(asg.value)
                    if not ok:
                        continue
                    for tgt in asg.target:
                        out[getattr(tgt, "sym_name", None) or tgt.unparse().strip()] = v
            self._glob_cache = out
        return self._glob_cache

    def _fold(self, expr: uni.UniNode) -> Tuple[bool, Any]:
        """literal_of, plus one hop through a module global."""
        ok, v = literal_of(expr)
        if ok:
            return True, v
        if isinstance(expr, uni.Name) and expr.value in self._globs():
            return True, self._globs()[expr.value]
        return False, None

    @staticmethod
    def _arch_kind(arch: Optional[uni.UniNode]) -> str:
        if arch is None or not hasattr(getattr(arch, "arch_type", None), "value"):
            return ""
        return arch.arch_type.value

    def _archs(self) -> Dict[str, uni.Archetype]:
        if self._arch_cache is None:
            self._arch_cache = {
                a.name.value: a for a in self.module.get_all_sub_nodes(uni.Archetype) if a.name is not None
            }
        return self._arch_cache

    @staticmethod
    def _trigger_names(ability: uni.Ability) -> List[str]:
        """Archetype names in an event signature's `with <T> entry` tag. Populated
        from arch_tag_info because event_trigger_type_names is only filled by a
        later pass than the one this parser stops at."""
        sig = ability.signature
        if not isinstance(sig, uni.EventSignature):
            return []
        names = ability.event_trigger_type_names()
        if names:
            return names
        out: List[str] = []
        tag = sig.arch_tag_info
        for nd in ([tag] + list(tag.get_all_sub_nodes(uni.Name))) if tag is not None else []:
            value = getattr(nd, "sym_name", None) or getattr(nd, "value", None)
            if value and value not in out:
                out.append(value)
        return out

    def _visit_decl(self, stmt: uni.VisitStmt) -> Optional[ByLLMDecl]:
        """Pseudo-decl for one routing call: a byllm invocation with no callable."""
        call = self._visit_by(stmt)
        if call is None:
            return None
        call_params: Dict[str, Any] = {}
        intent = ""
        for kw in call.params or []:
            if not isinstance(kw, uni.KWPair) or kw.key is None:
                continue
            key = kw.key.unparse().strip()
            ok, v = self._fold(kw.value)
            if not ok:
                print(f"[jac-static] warning: dynamic {key}= on the visit at "
                      f"{os.path.basename(stmt.loc.mod_path)}:{stmt.loc.first_line}; invariant excludes it")
                continue
            if key == "intent":
                intent = str(v)
            call_params[key] = v
        ability = stmt.find_parent_of_type(uni.Ability)
        owner = ability.method_owner if ability is not None else None
        return ByLLMDecl(
            name=visit_key(stmt.loc.mod_path, stmt.loc.first_line),
            kind="visit",
            intent=intent,
            call_params=call_params,
            module=stmt.loc.mod_path,
            lineno=stmt.loc.first_line,
            owner_arch=owner.name.value if owner is not None else "",
            owner_kind=self._arch_kind(owner),
            zones=list(route_layout()),
            candidates=self._candidate_nodes(stmt, ability),
            walkers=self._ability_walkers(ability) if ability is not None else [],
        )

    def _candidate_nodes(self, stmt: uni.VisitStmt, ability: Optional[uni.Ability]) -> List[str]:
        """Node archetypes the router may pick, over-approximated from the edge
        expression. A node-type filter names them outright; otherwise every node type
        whose entry ability fires for the walker running this visit is a candidate.
        Which of them the graph actually holds is a runtime fact — the point of the
        over-approximation is that speculation covers the real one."""
        target = stmt.target
        edges = target.left if isinstance(target, uni.BinaryExpr) else target
        archs = self._archs()
        named = [
            n.value for n in edges.get_all_sub_nodes(uni.Name)
            if self._arch_kind(archs.get(n.value)) == "node"
        ]
        if named:
            return sorted(set(named) | {sub for name in named for sub in self._subtypes(name)})
        # The walker running this visit is the ability's owner when the ability is
        # the walker's, and the `with <W> entry` trigger when it is the node's.
        walker_isa: set = set()
        for walker in (self._ability_walkers(ability) if ability is not None else []):
            walker_isa |= {walker} | self._supertypes(walker)
        out: List[str] = []
        for name, arch in archs.items():
            if self._arch_kind(arch) != "node":
                continue
            slots = [m for m in arch.get_methods()
                     if isinstance(m.signature, uni.EventSignature) and m.signature.event.name == "KW_ENTRY"]
            # No entry ability at all is still a legal target — the walker arrives,
            # runs nothing, and moves on. Only a node whose every entry slot is
            # triggered by some other walker is genuinely unreachable from here.
            if slots and walker_isa and not any(
                not self._trigger_names(sl) or (set(self._trigger_names(sl)) & walker_isa) for sl in slots
            ):
                continue
            if name not in out:
                out.append(name)
        return out

    def _subtypes(self, name: str) -> List[str]:
        return [n for n, a in self._archs().items() if name in self._supertypes(n) and n != name]

    def _supertypes(self, name: str) -> set:
        out: set = set()
        work = [name]
        while work:
            arch = self._archs().get(work.pop())
            for base in (arch.base_classes or []) if arch is not None else []:
                head = next((n.value for n in [base] + list(base.get_all_sub_nodes(uni.Name))
                             if getattr(n, "value", None) in self._archs()), None)
                if head and head not in out:
                    out.add(head)
                    work.append(head)
        return out

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
        for stmt in self.module.get_all_sub_nodes(uni.VisitStmt):
            call = self._visit_by(stmt)
            if call is None: # A visit with no `by llm(...)`, skip
                continue
            decl = self.get_decls().get(visit_key(stmt.loc.mod_path, stmt.loc.first_line))
            if decl is None:
                continue
            scope = stmt.find_parent_of_type(uni.Ability) or stmt.find_parent_of_type(uni.ModuleCode)
            sites.append(ByLLMCallsite(decl, call, scope, stmt=stmt))
        return sites

    def sites_of(self, key: str) -> List[ByLLMCallsite]:
        self.parse_dependency()
        return [s for s in self.callsites if s.key == key]

    def _visit_site_at(self, site: Optional[str]) -> List[ByLLMCallsite]:
        """Routing sites whose statement spans `file.jac:line`. The client reports
        the line of the frame executing the visit; that is the statement's first
        line today, but a multi-line visit is exactly where codegen could attribute
        it to an inner line instead, so placement falls back to the span."""
        if not site:
            return []
        fname, _, line = site.rpartition(":")
        if not line.isdigit():
            return []
        out = []
        for s in self.callsites:
            if s.stmt is None or os.path.basename(s.stmt.loc.mod_path) != os.path.basename(fname):
                continue
            if s.stmt.loc.first_line <= int(line) <= s.stmt.loc.last_line:
                out.append(s)
        return out

    def site_of(self, key: str, site: Optional[str] = None) -> ByLLMCallsite:
        """The callsite a wire request came from."""
        sites = self.sites_of(key)
        if not sites and key.startswith(VISIT_PREFIX):
            sites = self._visit_site_at(site or key[len(VISIT_PREFIX):])
        if not sites:
            raise KeyError(f"no byllm callsite for key {key!r}")
        if site:
            fname, _, line = site.rpartition(":")
            for s in sites:
                if not (line.isdigit() and os.path.basename(s.call.loc.mod_path) == os.path.basename(fname)):
                    continue
                loc = s.stmt.loc if s.stmt is not None else s.call.loc
                # A visit spanning several lines reports whichever line the `by`
                # operand sits on, so match the statement's span rather than its head.
                if loc.first_line <= int(line) <= loc.last_line:
                    return s
        return sites[0]

    # --------------------------------------------------------------- topology

    def _calls_in(self, node: uni.UniNode) -> List[str]:
        """Decl keys of every byllm call within `node` (itself included), routing
        calls included — a `visit ... by llm()` is a byllm call with no callee."""
        out: List[str] = []
        calls = ([node] if isinstance(node, uni.FuncCall) else []) + list(node.get_all_sub_nodes(uni.FuncCall))
        for c in calls:
            for k in self._resolve_call(c):
                if k not in out:
                    out.append(k)
        stmts = ([node] if isinstance(node, uni.VisitStmt) else []) + list(node.get_all_sub_nodes(uni.VisitStmt))
        for st in stmts:
            if self._visit_by(st) is None:
                continue
            k = visit_key(st.loc.mod_path, st.loc.first_line)
            if k in self.get_decls() and k not in out:
                out.append(k)
        return out

    def _leading_calls(self, node: uni.UniNode) -> List[str]:
        """byllm calls that run first when `node` (a statement or expression) is
        evaluated: those with no byllm call nested inside their own arguments."""
        out: List[str] = []
        calls = ([node] if isinstance(node, uni.FuncCall) else []) + list(node.get_all_sub_nodes(uni.FuncCall))
        for c in calls:
            keys = self._resolve_call(c)
            if not keys:
                continue
            if any(self._resolve_call(inner) for inner in c.get_all_sub_nodes(uni.FuncCall)):
                continue  # an argument makes a byllm call first
            for k in keys:
                if k not in out:
                    out.append(k)
        return out

    def _first_calls(self, stmts: List[uni.UniNode]) -> Tuple[List[str], bool, bool]:
        """(byllm keys that can run first in the statement sequence `stmts`,
        whether control can fall off the end of the sequence without making any
        byllm call, whether it can leave the enclosing body — `return`,
        `disengage` — without one). Branches contribute their own first calls; a
        call-free branch, a missing `else`, or a loop that may run zero times lets
        the scan continue to the following statement."""
        out: List[str] = []
        exits = False

        def add(keys: List[str]) -> None:
            for k in keys:
                if k not in out:
                    out.append(k)

        for st in stmts:
            keys, falls, ex = self._first_calls_stmt(st)
            add(keys)
            exits = exits or ex
            if not falls:
                return out, False, exits
            if isinstance(st, uni.CtrlStmt):
                return out, True, exits  # break/continue: control leaves the sequence
        return out, True, exits

    def _first_calls_stmt(self, st: uni.UniNode) -> Tuple[List[str], bool, bool]:
        def merge(a: List[str], b: List[str]) -> List[str]:
            return a + [k for k in b if k not in a]

        if isinstance(st, uni.VisitStmt):
            if self._visit_by(st) is not None:
                key = visit_key(st.loc.mod_path, st.loc.first_line)
                known = key in self.get_decls()
                return ([key] if known else []), not known, False
            return [], True, False  # a plain visit only enqueues
        if isinstance(st, (uni.ReturnStmt, uni.DisengageStmt)):
            keys = self._leading_calls(st)
            return keys, False, not keys  # the body ends here (after the call, if any)
        if isinstance(st, uni.IfStmt):  # ElseIf included
            keys, falls, ex = self._first_calls(st.body)
            eb = st.else_body
            if eb is None:
                return keys, True, ex
            k2, f2, e2 = self._first_calls_stmt(eb) if isinstance(eb, uni.IfStmt) else self._first_calls(eb.body)
            return merge(keys, k2), falls or f2, ex or e2
        if isinstance(st, LOOP_STMTS):
            keys, _, ex = self._first_calls(st.body)  # zero iterations are possible
            eb = getattr(st, "else_body", None)
            if eb is not None:
                k2, f2, e2 = self._first_calls(eb.body)
                return merge(keys, k2), f2, ex or e2
            return keys, True, ex
        if isinstance(st, uni.TryStmt):
            keys, falls, ex = self._first_calls(st.body)
            for exc in st.excepts or []:
                k2, f2, e2 = self._first_calls(exc.body)
                keys, falls, ex = merge(keys, k2), falls or f2, ex or e2
            for tail in (getattr(st, "else_body", None), getattr(st, "finally_body", None)):
                if tail is not None and falls:
                    k2, falls, e2 = self._first_calls(tail.body)
                    keys, ex = merge(keys, k2), ex or e2
            return keys, falls, ex
        if isinstance(st, uni.MatchStmt):
            keys: List[str] = []
            ex = False
            for case in st.cases or []:
                k2, _, e2 = self._first_calls(case.body)
                keys, ex = merge(keys, k2), ex or e2
            return keys, True, ex  # no case may match
        keys = self._leading_calls(st)
        return keys, not keys, False

    @staticmethod
    def _sequence_of(st: uni.UniNode) -> Tuple[Optional[List[uni.UniNode]], Optional[uni.UniNode]]:
        """The statement list `st` belongs to and the node owning that list."""
        owner = st.parent
        for attr in ("body", "excepts", "cases"):
            seq = getattr(owner, attr, None)
            if isinstance(seq, list) and any(x is st for x in seq):
                return seq, owner
        return None, owner

    def _next_keys(self, call: uni.UniNode) -> Tuple[List[str], Optional[uni.Ability]]:
        """byllm keys that can run right after `call` completes, following control
        flow within the enclosing body: the rest of the current statement (an
        enclosing byllm call), then the first calls of what follows — through
        branches (each branch's first call; a call-free branch continues past the
        `if`), loops (back edge to the loop body's first call, then the loop's
        exit) and `return`/`disengage`. Second value = the enclosing Ability when
        its body's end is reachable from `call` without another byllm call, i.e.
        when the caller's continuation decides what runs next; None otherwise."""
        out: List[str] = []

        def add(keys: List[str]) -> None:
            for k in keys:
                if k not in out:
                    out.append(k)

        # Inside the statement: an enclosing byllm call runs right after this one.
        st: uni.UniNode = call
        while st.parent is not None and not isinstance(st, uni.CodeBlockStmt):
            st = st.parent
            if isinstance(st, uni.FuncCall) and self._resolve_call(st):
                add(self._resolve_call(st))
                return out, None
        ability = call.find_parent_of_type(uni.Ability)
        exit_seen = False  # a `return`/`disengage` reachable without a call: the body ends there
        while True:
            seq, owner = self._sequence_of(st)
            if seq is not None:
                keys, falls, ex = self._first_calls(seq[next(i for i, x in enumerate(seq) if x is st) + 1:])
                add(keys)
                exit_seen = exit_seen or ex
                if not falls:
                    break
            if owner is None or isinstance(owner, (uni.ModuleCode, uni.Module)):
                return out, None
            if isinstance(owner, uni.Ability):
                return out, owner
            if isinstance(owner, LOOP_STMTS) and seq is not None and any(x is st for x in owner.body):
                # The loop body ended: it may iterate again, or run its else and exit.
                keys, _, ex = self._first_calls(owner.body)
                add(keys)
                exit_seen = exit_seen or ex
                eb = getattr(owner, "else_body", None)
                if eb is not None:
                    keys, falls, ex = self._first_calls(eb.body)
                    add(keys)
                    exit_seen = exit_seen or ex
                    if not falls:
                        break
                st = owner
                continue
            if isinstance(owner, (uni.ElseStmt, uni.ElseIf, uni.Except, uni.FinallyStmt, uni.MatchCase)):
                # A branch ended: control continues after the whole if/try/match.
                top = owner
                while isinstance(top, (uni.ElseStmt, uni.ElseIf, uni.Except, uni.FinallyStmt, uni.MatchCase)):
                    top = top.parent
                st = top
                continue
            st = owner  # IfStmt / TryStmt / MatchStmt / anything else: continue after it
        return out, (ability if exit_seen else None)

    def _continuation_keys(self, ability: uni.Ability, visited: set) -> List[str]:
        """What can run after `ability`'s body ends: the byllm calls following each of
        its own invocation sites, chased recursively through enclosing callables."""
        owner = ability.method_owner
        kind = self._arch_kind(owner)
        if kind in ("walker", "node"):
            return self._traversal_continuation(ability, visited)
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

    def _ability_id(self, ability: uni.Ability) -> str:
        owner = ability.method_owner
        name = ability.name_ref.value if isinstance(ability.name_ref, uni.Name) else ability.py_resolve_name()
        return f"{owner.name.value}.{name}" if owner is not None else name

    def _ordered_calls(self, node: uni.UniNode) -> List[Tuple[Tuple[int, int], Any]]:
        """Every call-ish node under `node` in source order, as (position, node).
        Source order is what makes "the node's *first* byllm call" well defined."""
        out: List[Tuple[Tuple[int, int], Any]] = []
        for c in node.get_all_sub_nodes(uni.FuncCall):
            out.append(((c.loc.first_line, c.loc.col_start), c))
        for st in node.get_all_sub_nodes(uni.VisitStmt):
            out.append(((st.loc.first_line, st.loc.col_start), st))
        return sorted(out, key=lambda item: item[0])

    def _first_call_in(self, ability: uni.Ability, visited: Optional[set] = None) -> List[str]:
        """The byllm key that runs first when `ability` executes, following plain
        helper calls into their bodies. [] when the ability makes no byllm call at
        all — that candidate simply contributes no successor.

        A list rather than a single key: an ambiguous `self.m()` over-approximates
        to every candidate, and both are equally "first"."""
        visited = visited if visited is not None else set()
        ident = self._ability_id(ability)
        if ident in visited:
            return []
        visited.add(ident)
        ordered = self._ordered_calls(ability)
        # Direct byllm calls first: source position orders nested calls by column,
        # so descending into a helper before checking the rest of the body would
        # report a call that actually runs later.
        for _, nd in ordered:
            if isinstance(nd, uni.VisitStmt):
                key = visit_key(nd.loc.mod_path, nd.loc.first_line)
                if self._visit_by(nd) is not None and key in self.get_decls():
                    return [key]
                continue
            keys = self._resolve_call(nd)
            if keys:
                return keys
        for _, nd in ordered:
            callee = self._callee_body(nd) if isinstance(nd, uni.FuncCall) else None
            if callee is not None:
                inner = self._first_call_in(callee, visited)
                if inner:
                    return inner
        return []

    def _callee_body(self, call: uni.FuncCall) -> Optional[uni.Ability]:
        """The module-local plain ability a non-byllm call targets, if resolvable."""
        if getattr(call, "target", None) is None:
            return None
        simple = norm(call.target.unparse()).split(".")[-1]
        for cand in self.module.get_all_sub_nodes(uni.Ability):
            if cand.is_genai_ability or cand.body is None:
                continue
            name = cand.name_ref.value if isinstance(cand.name_ref, uni.Name) else None
            if name == simple:
                return cand
        return None

    def _entry_slots(self, arch_name: str, walker: str) -> List[uni.Ability]:
        """Entry abilities of `arch_name` that fire when `walker` arrives."""
        arch = self._archs().get(arch_name)
        if arch is None:
            return []
        walker_isa = ({walker} | self._supertypes(walker)) if walker else set()
        out: List[uni.Ability] = []
        for slot in arch.get_methods():
            if not isinstance(slot.signature, uni.EventSignature) or slot.signature.event.name != "KW_ENTRY":
                continue
            trigs = self._trigger_names(slot)
            if trigs and walker_isa and not (set(trigs) & walker_isa):
                continue
            out.append(slot)
        return out

    def _ability_walkers(self, ability: uni.Ability) -> List[str]:
        """Walker archetypes that can be running when `ability` executes: the owner of
        a walker ability, the `with <W> entry` triggers of a node ability ([] when a
        node ability names no trigger — any walker)."""
        owner = ability.method_owner
        kind = self._arch_kind(owner)
        if kind == "walker":
            return [owner.name.value]
        if kind == "node":
            archs = self._archs()
            return [t for t in self._trigger_names(ability) if self._arch_kind(archs.get(t)) == "walker"]
        return []

    def _ability_nodes(self, ability: uni.Ability) -> List[str]:
        """Node archetypes `ability` runs on: the owner of a node ability, the
        `with <T> entry` triggers of a walker ability."""
        owner = ability.method_owner
        kind = self._arch_kind(owner)
        if kind == "node":
            return [owner.name.value]
        if kind == "walker":
            return list(self._trigger_names(ability))
        return []

    def _arrival_abilities(self, arch_name: str, walker: str) -> List[uni.Ability]:
        """Abilities that fire when `walker` arrives at a node of type `arch_name`:
        the node's entry abilities the walker triggers (node side) and the walker's
        entry abilities the node type triggers (walker side: `can x with <T> entry`
        declared in the walker, T the node type or one of its supertypes)."""
        out: List[uni.Ability] = list(self._entry_slots(arch_name, walker))
        node_isa = {arch_name} | self._supertypes(arch_name)
        archs = self._archs()
        for wname in (({walker} | self._supertypes(walker)) if walker else set()):
            arch = archs.get(wname)
            if arch is None or self._arch_kind(arch) != "walker":
                continue
            for slot in arch.get_methods():
                sig = slot.signature
                if not isinstance(sig, uni.EventSignature) or sig.event.name != "KW_ENTRY":
                    continue
                trigs = self._trigger_names(slot)
                if trigs and not (set(trigs) & node_isa):
                    continue
                if slot not in out:
                    out.append(slot)
        return out

    def _plain_visit_successors(self, stmt: uni.VisitStmt, exclude: set = frozenset()) -> List[str]:
        """Where a plain `visit <edges>` hands control: the first byllm call of every
        ability that fires when this walker arrives at one of its target node types.
        The visit only enqueues — the walker dequeues once the enclosing body ends —
        so these are continuation edges of the body, not forward-scan edges."""
        ability = stmt.find_parent_of_type(uni.Ability)
        walkers = self._ability_walkers(ability) if ability is not None else []
        out: List[str] = []
        for cand in self._candidate_nodes(stmt, ability):
            if cand in exclude:
                continue
            for walker in walkers or [""]:
                for slot in self._arrival_abilities(cand, walker):
                    for key in self._first_call_in(slot):
                        if key not in out:
                            out.append(key)
        return out

    def _visit_successors(self, decl: ByLLMDecl, exclude: set = frozenset()) -> List[str]:
        """Where a routing call hands control next: the first byllm call inside each
        candidate node's firing arrival abilities (node side and walker side). The
        router's answer picks one of them at runtime, so all of them are may-run-next.
        `exclude` drops candidate node types (the sibling case: the type already
        running does not follow itself)."""
        out: List[str] = []
        for cand in decl.candidates:
            if cand in exclude:
                continue
            for walker in decl.walkers or [decl.owner_arch]:
                for slot in self._arrival_abilities(cand, walker):
                    for key in self._first_call_in(slot):
                        if key not in out:
                            out.append(key)
        return out

    def _visits_in(self, body: uni.UniNode) -> List[uni.VisitStmt]:
        """Every visit under `body`, routing or plain, in source order — the order
        in which the walker's queue receives their targets."""
        return sorted(body.get_all_sub_nodes(uni.VisitStmt),
                      key=lambda st: (st.loc.first_line, st.loc.col_start))

    @staticmethod
    def _ancestors(node: uni.UniNode, stop: uni.UniNode) -> List[uni.UniNode]:
        out: List[uni.UniNode] = []
        cur = node.parent
        while cur is not None and cur is not stop:
            out.append(cur)
            cur = cur.parent
        return out

    def _conditional_within(self, st: uni.UniNode, body: uni.UniNode) -> bool:
        """Whether `st` sits inside a branch or loop of `body` that may not execute."""
        return any(isinstance(a, CONDITIONAL_STMTS) for a in self._ancestors(st, body))

    def _exclusive(self, a: uni.UniNode, b: uni.UniNode, body: uni.UniNode) -> bool:
        """Whether `a` and `b` lie in different branches of the same `if` in `body`:
        one execution takes at most one of them."""
        anc_a = self._ancestors(a, body)
        anc_b = set(self._ancestors(b, body))
        common = next((x for x in anc_a if x in anc_b), None)
        if common is None or not isinstance(common, (uni.IfStmt, uni.ElseIf)):
            return False
        branch_a = next((x for x in [a] + anc_a if x.parent is common), None)
        branch_b = next((x for x in [b] + list(self._ancestors(b, body)) if x.parent is common), None)
        return branch_a is not branch_b

    def _visit_candidates(self, st: uni.VisitStmt) -> List[str]:
        """Node archetypes a visit may enqueue, routing or plain."""
        if self._visit_by(st) is not None:
            decl = self.get_decls().get(visit_key(st.loc.mod_path, st.loc.first_line))
            return list(decl.candidates) if decl is not None else []
        return self._candidate_nodes(st, st.find_parent_of_type(uni.Ability))

    def _visit_targets(self, st: uni.VisitStmt, exclude: set = frozenset()) -> List[str]:
        """First byllm calls of the arrival abilities a visit's targets fire."""
        if self._visit_by(st) is not None:
            decl = self.get_decls().get(visit_key(st.loc.mod_path, st.loc.first_line))
            return self._visit_successors(decl, exclude) if decl is not None else []
        return self._plain_visit_successors(st, exclude)

    def _queue_head(self, visits: List[uni.VisitStmt], body: uni.UniNode) -> List[str]:
        """What the walker dequeues first once `body` ends, given the visits it
        issued in queue order: the first visit's targets, plus each following
        visit's for as long as every earlier one sits in a branch the body may
        have skipped."""
        out: List[str] = []
        for st in visits:
            for key in self._visit_targets(st):
                if key not in out:
                    out.append(key)
            if not self._conditional_within(st, body):
                break
        return out

    def _traversal_continuation(self, ability: uni.Ability, visited: set) -> List[str]:
        """What may run once a walker/node ability's body ends. Control returns to
        the walker's traversal queue, not to a caller. The queue holds, in order:
        what this body itself enqueued (its first visit's targets run next); then,
        if this ability was reached by a visit V of some body B, what V enqueued
        alongside this node (a `select>1` router or a multi-type visit) and what
        the visits after V in B enqueued; and when the queue is empty the
        walker's exit abilities."""
        ident = self._ability_id(ability)
        if ident in visited:
            return []
        visited.add(ident)
        out: List[str] = []

        def add(keys: List[str]) -> None:
            for key in keys:
                if key not in out:
                    out.append(key)

        add(self._queue_head(self._visits_in(ability), ability))
        here = set(self._ability_nodes(ability))
        decls = self.get_decls()
        for st in self._visits_in(self.module):
            if not (here & set(self._visit_candidates(st))):
                continue
            body = st.find_parent_of_type(uni.Ability)
            decl = decls.get(visit_key(st.loc.mod_path, st.loc.first_line)) if self._visit_by(st) is not None else None
            # select=1 queues exactly one node, so no sibling candidate follows it.
            # The type already running is not its own successor: a second node of
            # the same type re-enters this very ability, whose prefix is resident.
            if not (decl is not None and decl.call_params.get("select") == 1):
                add(self._visit_targets(st, exclude=here))
            if body is not None:
                pos = (st.loc.first_line, st.loc.col_start)
                later = [v for v in self._visits_in(body)
                         if (v.loc.first_line, v.loc.col_start) > pos and not self._exclusive(st, v, body)]
                add(self._queue_head(later, body))
        # The traversal may end after this body: the walker's exit abilities run —
        # unless this body is one of them already.
        sig = ability.signature
        if not (isinstance(sig, uni.EventSignature) and sig.event.name == "KW_EXIT"):
            for walker in self._ability_walkers(ability):
                add(self._exit_calls(walker))
        return out

    def _exit_calls(self, walker: str) -> List[str]:
        """byllm calls in a walker's exit abilities — the tail of any traversal."""
        arch = self._archs().get(walker)
        out: List[str] = []
        for slot in arch.get_methods() if arch is not None else []:
            sig = slot.signature
            if not isinstance(sig, uni.EventSignature) or sig.event.name != "KW_EXIT":
                continue
            for key in self._calls_in(slot):
                if key not in out:
                    out.append(key)
        return out

    def _build_topology(self) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        for site in self.callsites:
            anchor = site.stmt if site.stmt is not None else site.call
            succ, escaped = self._next_keys(anchor)
            # `visit` only enqueues: its candidates run once the enclosing body ends,
            # in queue order, which _traversal_continuation models. Only a visit
            # outside any walker/node ability (a helper) needs its candidates here.
            if site.is_visit and escaped is not None and self._arch_kind(escaped.method_owner) not in ("walker", "node"):
                succ = succ + self._visit_successors(site.decl)
            if escaped is not None:
                succ = succ + self._continuation_keys(escaped, set())
            out = graph.setdefault(site.callsite_uuid, [])
            for k in succ:
                if k != site.key and k not in out:
                    out.append(k)
        return graph

    # ------------------------------------------------------------- provenance

    def _field_binding(self, scope_kw: str, scope: Optional[uni.UniNode]) -> Tuple[str, str]:
        """(lifetime, archetype) of a `self.` / `here.` / `visitor.` reference.

        Lifetime is what a traversal step turns on. `walker` state rides along with
        the walker and is the same object before and after a routing call; `node`
        state belongs to whichever node the router picks, so its value does not
        exist until the router has answered. In a walker ability `self` is the
        walker and `here` the node; in a node ability those swap — `self` is the
        node and `visitor` the walker (byllm's own codegen names them that way)."""
        ability = scope if isinstance(scope, uni.Ability) else None
        if ability is None:
            return "", ""
        owner = ability.method_owner
        kind = self._arch_kind(owner)
        owner_name = owner.name.value if owner is not None else ""
        trigger = next(iter(self._trigger_names(ability)), "")
        if kind == "walker":
            if scope_kw == "self":
                return "walker", owner_name
            if scope_kw == "here":
                return "node", trigger
        elif kind == "node":
            if scope_kw == "self":
                return "node", owner_name
            if scope_kw == "visitor":
                return "walker", trigger
        return ("obj", owner_name) if scope_kw == "self" else ("", "")

    def _field_writers(self) -> Dict[Tuple[str, str], List[str]]:
        """(archetype, attribute) -> byllm keys whose return is stored there.

        A local dies with its ability, so an archetype field is the only way a
        byllm result reaches a call in another ability — which, across a routing
        call, is every downstream call. Without this index the dataflow edge from
        `visitor.answer = classify(...)` in one node to `report(self.answer)` in
        the walker is invisible and the consumer never warms."""
        if self._writer_cache is None:
            out: Dict[Tuple[str, str], List[str]] = {}
            for asg in self.module.get_all_sub_nodes(uni.Assignment):
                tgt = asg.target[0] if asg.target else None
                if tgt is None or asg.value is None or not isinstance(asg.value, uni.FuncCall):
                    continue
                m = FIELD_EXPR_RE.match(norm(tgt.unparse()))
                if not m:
                    continue
                keys = self._resolve_call(asg.value)
                if len(keys) != 1:
                    continue
                scope = asg.find_parent_of_type(uni.Ability) or asg.find_parent_of_type(uni.ModuleCode)
                _, arch = self._field_binding(m.group(1), scope)
                if not arch:
                    continue
                slot = out.setdefault((arch, m.group(2)), [])
                if keys[0] not in slot:
                    slot.append(keys[0])
            self._writer_cache = out
        return self._writer_cache

    def _field_spec(self, m: "re.Match", scope: Optional[uni.UniNode], consumer: str) -> Dict[str, Any]:
        """Provenance of one `self.x` / `here.x` / `visitor.x` argument, upgraded to
        a dataflow edge when a byllm call is what writes that field."""
        binding, arch = self._field_binding(m.group(1), scope)
        spec: Dict[str, Any] = {"kind": "field", "scope": m.group(1), "attr": m.group(2),
                                "slice": m.group(3), "binding": binding, "arch": arch}
        writers = [k for k in self._field_writers().get((arch, m.group(2)), []) if k != consumer]
        if len(writers) == 1:
            spec.update(kind="ret", of=writers[0], via_field=f"{arch}.{m.group(2)}")
        elif writers:
            # Several branches write the field; whichever one the router reaches is
            # the producer, so any of them completing makes the consumer bindable.
            spec.update(kind="ret_any", of_any=writers, via_field=f"{arch}.{m.group(2)}")
        return spec

    def _classify_name(self, name: str, scope: uni.UniNode, seen: Optional[set] = None,
                       consumer: str = "") -> Dict[str, Any]:
        """Where a local variable's value comes from, within one ability/module scope.
        Scoped on purpose: a local never crosses an ability boundary, so restricting
        the search to `scope` is also what keeps traversal dataflow honest."""
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
                return self._field_spec(m, scope, consumer)
        for loop in scope.get_all_sub_nodes(uni.InForStmt):
            if getattr(loop.target, "sym_name", None) != name:
                continue
            src = loop.collection
            keys = self._resolve_call(src) if isinstance(src, uni.FuncCall) else []
            key = keys[0] if len(keys) == 1 else None
            if key is None and isinstance(src, uni.Name):
                inner = self._classify_name(src.value, scope, seen, consumer)
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
            return self._field_spec(m, site.scope, site.key)
        if isinstance(expr, uni.FuncCall):
            keys = self._resolve_call(expr)
            if len(keys) == 1:
                return {"kind": "ret", "of": keys[0]}
        if isinstance(expr, uni.Name) and site.scope is not None:
            return self._classify_name(expr.value, site.scope, consumer=site.key)
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
                producers = ([spec["of"]] if spec.get("kind") in ("ret", "ret_item") and spec.get("of")
                             else spec.get("of_any", []) if spec.get("kind") == "ret_any" else [])
                for producer in producers:
                    if (ckey, pname) not in out.setdefault(producer, []):
                        out[producer].append((ckey, pname))
        return out

    # ---------------------------------------------------------------- queries

    def next_calls(self, site: ByLLMCallsite) -> List[str]:
        """1. Which byllm calls may run next after this exact callsite."""
        self.parse_dependency()
        return self.topology.get(site.callsite_uuid, [])

    def ready_params(self, consumer: str, done: set, via: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """2. Readiness of `consumer`'s params given the completed call keys `done`.
        const: known at compile time. field: derivable now from live walker/node state
        (needs observation). ret/ret_item/ret_any: ready iff a producer completed.
        call/unknown: never derivable ahead — only observable after a serve.

        `via` names the call control passes through to reach `consumer`. When that
        is a routing call, every node-scoped source is withheld: `self.policy` in a
        node ability reads the node the router is still choosing, so treating it as
        derivable would warm the prefix with one candidate's state and serve
        another's. Walker-scoped sources ride through the traversal unchanged and
        stay ready."""
        self.parse_dependency()
        via_decl = self.decls.get(via) if via else None
        crossing = via_decl is not None and via_decl.is_visit
        out: Dict[str, Dict[str, Any]] = {}
        for pname, spec in self.provenance.get(consumer, {}).items():
            kind = spec["kind"]
            if kind in ("ret", "ret_item"):
                ready = spec.get("of") in done
            elif kind == "ret_any":
                ready = any(k in done for k in spec.get("of_any", []))
            else:
                ready = kind in ("const", "field")
            if crossing and spec.get("binding") == "node":
                ready = False
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
