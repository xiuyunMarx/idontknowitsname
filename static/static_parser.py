"""Utility layer for the static byllm runtime.

Everything here is engine-free (no vllm/torch): UniIR compilation helpers,
literal/type extraction from the IR, and the prompt-zone renderers that
mirror byllm's runtime construction (mtir.impl.jac + tool_protocol.jac).

Produces ByLLMDecl records; AsyncByLLM consumes them.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import enum as enum_mod
import json
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import jaclang  # noqa: F401  # registers the .jac meta importer (needed first)
import jaclang.jac0core.unitree as uni  # type: ignore
from jaclang.jac0core.compile_options import CompileOptions  # type: ignore
from jaclang.jac0core.program import JacProgram  # type: ignore

TOOL_PROTOCOL_HEADER = "# Calling tools"

_JSON_TYPE = {"str": "string", "int": "integer", "float": "number", "bool": "boolean", "list": "array", "tuple": "array", "set": "array", "dict": "object"}


# ---------------------------------------------------------------------------
# Compile-time record of one byllm call site
# ---------------------------------------------------------------------------
@dataclass
class ByLLMDecl:
    """One `def f(...) -> T by llm(...)` call site, fully described from UniIR at compile time."""

    name: str
    qualifier: str = ""  # "RagChat." for methods, "" for module-level defs
    params: List[Dict[str, Any]] = field(default_factory=list)  # {name, type, sem, required}
    return_type: str = "str"
    sem: str = ""  # authored sem of the function itself
    owner_sem: str = ""  # sem of the owning archetype; drives the `self` identity zone
    tools: List[Dict[str, Any]] = field(default_factory=list)  # OpenAI tool schemas, finish_tool last
    call_params: Dict[str, Any] = field(default_factory=dict)  # literal kwargs of the llm() call
    extra_system_prompt: str = ""  # literal system_prompt= kwarg, if any
    module: str = ""
    lineno: int = 0
    reponse_format: Optional[Dict[str, Any]] = None  # expanded response_format (None for str returns / tools)
    return_type_obj: Optional[Any] = None  # materialized Python type translated from Jac obj/enum defs

    def signature(self) -> str:
        args = ", ".join(f"{p['name']}: {p['type']}" if p["type"] else p["name"] for p in self.params)
        sig = f"{self.name}({args})"
        return f"{sig} -> {self.return_type}" if self.return_type else sig


# ---------------------------------------------------------------------------
# UniIR compilation
# ---------------------------------------------------------------------------
def parse_only(file_path: str) -> uni.Module:
    """Lex + parse only: a UniIR tree with no symbol tables or types."""
    with open(file_path, encoding="utf-8") as f:
        source = f.read()
    return JacProgram().parse_str(source, file_path)


def build_uniir(
    file_path: str, type_check: bool = False
) -> tuple[JacProgram, uni.Module]:
    """Full compile to UniIR: symbol tables bound, imports resolved, sems attached.

    Returns the program (holds every module in ``prog.mod.hub`` plus
    ``errors_had``/``warnings_had``) and the root module of ``file_path``.
    """
    prog = JacProgram()
    mod = prog.compile(
        file_path,
        options=CompileOptions(
            type_check=type_check,   # types on nodes; slower, needs deps importable
            no_cgen=True,            # skip Python bytecode generation
            force_target_program=True,
        ),
    )
    for err in prog.errors_had:
        print(f"[side-runtime] compile error: {err}")
    return prog, mod


# ---------------------------------------------------------------------------
# IR value/type extraction
# ---------------------------------------------------------------------------
def clean_type(type_text: str) -> str:
    """Normalize a type annotation's unparse (': list [ dict [ str , str ] ]') to Python style ('list[dict[str, str]]')."""
    t = type_text.strip()
    if t.startswith(":"):
        t = t[1:].strip()
    t = re.sub(r"\s*\[\s*", "[", t)
    t = re.sub(r"\s*\]", "]", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return t


def literal_of(expr: uni.UniNode) -> Tuple[bool, Any]:
    """Best-effort constant-fold of an expression node. Returns (is_literal, value)."""
    if isinstance(expr, (uni.Int, uni.Float, uni.String, uni.MultiString)):
        return True, expr.lit_value
    if isinstance(expr, uni.Bool):
        return True, expr.lit_value
    if isinstance(expr, uni.ListVal):
        vals = []
        for item in expr.values or []:
            ok, v = literal_of(item)
            if not ok:
                return False, None
            vals.append(v)
        return True, vals
    return False, None


def json_type_of(py_type: str) -> str:
    base = py_type.split("[", 1)[0].strip()
    return _JSON_TYPE.get(base, base or "any")


_TYPE_NAMESPACE: Dict[str, Any] = {"str": str, "int": int, "float": float, "bool": bool, "list": list, "dict": dict, "tuple": tuple, "set": set, "frozenset": frozenset, "bytes": bytes, "None": type(None), "Any": typing.Any, "Optional": typing.Optional, "Union": typing.Union, "List": typing.List, "Dict": typing.Dict, "Tuple": typing.Tuple, "Set": typing.Set}


def resolve_type(type_text: str, extra: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Materialize a type annotation string ('list[dict[str, int]]') into a real typing object.

    `extra` supplies materialized user types (from materialize_type). Returns None
    for names unavailable statically — callers fall back to the raw JSON value.
    """
    if not type_text:
        return None
    ns = dict(_TYPE_NAMESPACE)
    if extra:
        ns.update(extra)
    try:
        return eval(type_text, {"__builtins__": {}}, ns)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Jac obj/enum translation: UniIR type defs -> real Python classes
# ---------------------------------------------------------------------------
def _arch_body(node: uni.UniNode) -> list:
    body = getattr(node, "body", None)
    if isinstance(body, list):
        return body
    inner = getattr(body, "body", None)
    return inner if isinstance(inner, list) else []


def collect_type_defs(program: JacProgram) -> Dict[str, Dict[str, Any]]:
    """Harvest every obj/node/edge/walker archetype and enum definition from the program."""
    defs: Dict[str, Dict[str, Any]] = {}
    for mod_path, mod in program.mod.hub.items():
        if not mod_path.endswith(".jac"):
            continue
        for arch in mod.get_all_sub_nodes(uni.Archetype):
            fields: List[Dict[str, Any]] = []
            for st in _arch_body(arch):
                if isinstance(st, uni.ArchHas):
                    for hv in st.vars:
                        fields.append({
                            "name": hv.name.value,
                            "type": clean_type(hv.type_tag.unparse()) if hv.type_tag else "",
                            "sem": (hv.semstr or "").strip(),
                            "default_src": hv.value.unparse().strip() if hv.value is not None else None,
                        })
            kind = arch.arch_type.value if hasattr(arch.arch_type, "value") else str(arch.arch_type)
            defs[arch.name.value] = {"kind": kind, "sem": (arch.semstr or "").strip(), "fields": fields}
        for en in mod.get_all_sub_nodes(uni.Enum):
            members: List[Tuple[str, Any]] = []
            for st in _arch_body(en):
                if isinstance(st, uni.Assignment) and st.target and isinstance(st.target[0], uni.AstSymbolNode):
                    ok, v = literal_of(st.value) if st.value is not None else (False, None)
                    members.append((st.target[0].sym_name, v if ok else len(members) + 1))
                elif isinstance(st, uni.AstSymbolNode):
                    members.append((st.sym_name, len(members) + 1))
            defs[en.name.value] = {"kind": "enum", "sem": (en.semstr or "").strip(), "members": members}
    return defs


def materialize_type(name: str, defs: Dict[str, Dict[str, Any]], cache: Optional[Dict[str, Any]] = None, _stack: Optional[set] = None) -> Optional[Any]:
    """Translate one Jac obj/enum definition into an equivalent Python class.

    obj/node/edge/walker -> dataclass, enum -> enum.Enum — the same shapes
    jaclang's codegen produces, so byllm-style TypeAdapter coercion applies.
    Cyclic references degrade to Any rather than recursing forever.
    """
    cache = cache if cache is not None else {}
    _stack = _stack if _stack is not None else set()
    if name in cache:
        return cache[name]
    spec = defs.get(name)
    if spec is None or name in _stack:
        return None
    _stack.add(name)
    try:
        if spec["kind"] == "enum":
            cls = enum_mod.Enum(name, spec["members"])
            cls._jac_semstr = spec["sem"]  # type: ignore[attr-defined]
            cache[name] = cls
            return cls
        ns: Dict[str, Any] = {}
        plain: List[Tuple[str, Any]] = []
        defaulted: List[Tuple[str, Any, Any]] = []
        for f in spec["fields"]:
            for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f["type"]):
                if ident in defs and ident not in ns:
                    sub = materialize_type(ident, defs, cache, _stack)
                    if sub is not None:
                        ns[ident] = sub
            fty = resolve_type(f["type"], ns) or Any
            if f["default_src"] is None:
                plain.append((f["name"], fty))
                continue
            try:
                dv = ast.literal_eval(f["default_src"])
            except Exception:
                dv = None
            if isinstance(dv, (list, dict, set)):
                defaulted.append((f["name"], fty, dataclasses.field(default_factory=lambda v=dv: copy.deepcopy(v))))
            else:
                defaulted.append((f["name"], fty, dataclasses.field(default=dv)))
        cls = dataclasses.make_dataclass(name, [*plain, *defaulted])
        cls._jac_semstr = spec["sem"]  # type: ignore[attr-defined]
        cls._jac_semstr_inner = {f["name"]: f["sem"] for f in spec["fields"] if f["sem"]}  # type: ignore[attr-defined]
        cache[name] = cls
        return cls
    finally:
        _stack.discard(name)


def materialized_namespace(type_text: str, defs: Optional[Dict[str, Dict[str, Any]]], cache: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Materialize every user type referenced in a type annotation string."""
    ns: Dict[str, Any] = {}
    if not defs:
        return ns
    cache = cache if cache is not None else {}
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", type_text):
        if ident in defs and ident not in ns:
            cls = materialize_type(ident, defs, cache)
            if cls is not None:
                ns[ident] = cls
    return ns


def json_schema_of(type_text: str, defs: Optional[Dict[str, Dict[str, Any]]] = None, _depth: int = 0) -> Dict[str, Any]:
    """JSON schema for a type annotation, expanding Jac obj fields and enum legends."""
    defs = defs or {}
    t = type_text.strip()
    if not t or _depth > 6:
        return {"type": "object"}
    if t in defs:
        spec = defs[t]
        if spec["kind"] == "enum":
            values = [v for _, v in spec["members"]]
            etype = "integer" if all(isinstance(v, int) for v in values) else "string"
            legend = ", ".join(f"{v}={n}" for n, v in spec["members"])
            desc = (spec["sem"] + " " if spec["sem"] else "") + legend
            return {"type": etype, "enum": values, "description": desc}
        props: Dict[str, Any] = {}
        req: List[str] = []
        for f in spec["fields"]:
            fs = json_schema_of(f["type"], defs, _depth + 1)
            if f["sem"]:
                fs = {**fs, "description": f["sem"]}
            props[f["name"]] = fs
            if f["default_src"] is None:
                req.append(f["name"])
        out: Dict[str, Any] = {"type": "object", "properties": props, "required": req, "additionalProperties": False}
        if spec["sem"]:
            out["description"] = spec["sem"]
        return out
    base = t.split("[", 1)[0].strip()
    if base in ("list", "set", "tuple"):
        if "[" in t:
            inner = t[t.index("[") + 1:t.rindex("]")]
            if base != "tuple" or "," not in inner:
                return {"type": "array", "items": json_schema_of(inner, defs, _depth + 1)}
        return {"type": "array"}
    if base == "dict":
        return {"type": "object"}
    return {"type": json_type_of(base)}


def _balanced_json(text: str, start: int) -> Optional[str]:
    """Return the balanced {...} substring starting at `start`, respecting string literals."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def extract_finish_output(text: str) -> Tuple[bool, Any]:
    """Find a finish_tool call in a text-protocol reply and pull out final_output.

    Handles the advertised `<tool_call>{"name": "finish_tool", "arguments": {...}}</tool_call>`
    shape plus bare-JSON and `{"tool_calls": [...]}` envelopes. Returns (found, value).
    """
    i = 0
    while i < len(text):
        if text[i] != "{":
            i += 1
            continue
        blob = _balanced_json(text, i)
        if blob is None:
            return False, None
        try:
            obj = json.loads(blob)
        except Exception:
            obj = None
        if isinstance(obj, dict):
            cands = obj.get("tool_calls") or obj.get("tool_call") or [obj]
            if isinstance(cands, dict):
                cands = [cands]
            for c in cands:
                if isinstance(c, dict) and c.get("name") == "finish_tool":
                    args = c.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    if isinstance(args, dict) and "final_output" in args:
                        return True, args["final_output"]
        i += len(blob)
    return False, None


# ---------------------------------------------------------------------------
# Prompt-zone rendering (mirrors byllm's tool_protocol.jac / schema.jac)
# ---------------------------------------------------------------------------
def _render_tool(tool: Dict[str, Any]) -> str:
    fn = tool.get("function", {})
    if not isinstance(fn, dict):
        return ""
    name = str(fn.get("name", "") or "")
    if not name:
        return ""
    desc = str(fn.get("description", "") or "")
    params = fn.get("parameters", {}) or {}
    props = params.get("properties", {}) if isinstance(params, dict) else {}
    required = params.get("required", []) if isinstance(params, dict) else []
    lines = [f"- {name}: {desc}"]
    for pname, pinfo in props.items():
        ptype = str(pinfo.get("type", "any") or "any") if isinstance(pinfo, dict) else "any"
        pdesc = str(pinfo.get("description", "") or "") if isinstance(pinfo, dict) else ""
        flag = " (required)" if pname in required else ""
        lines.append(f"    - {pname} [{ptype}]{flag}: {pdesc}")
    return "\n".join(lines)


def format_tools_for_prompt(tools: List[Dict[str, Any]]) -> str:
    rendered = [b for b in (_render_tool(t) for t in tools) if b]
    if not rendered:
        return ""
    return TOOL_PROTOCOL_HEADER + "\n\nYou can call tools. The tools available to you are:\n\n" + "\n".join(rendered) + "\n\nTo call a tool, reply with ONLY a tool call in exactly this form and nothing else:\n\n" + '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>' + "\n\nRules:\n" + "- Call exactly one tool per reply.\n" + '- "arguments" must be a JSON object whose keys match the tool\'s parameters.\n' + "- Emit nothing outside the <tool_call> tags when calling a tool.\n" + "- After a tool result is returned, call the next tool, or call finish_tool with your final answer to end."


def response_format_of(return_type: str, defs: Optional[Dict[str, Dict[str, Any]]] = None) -> Optional[Dict[str, Any]]:
    if return_type in ("", "str", "None", "Any", "object"):
        return None
    schema = json_schema_of(return_type, defs)
    if schema.get("type") != "object":
        schema = {"type": "object", "properties": {"schema_object_wrapper": schema}, "required": ["schema_object_wrapper"], "additionalProperties": False}
    return {"type": "json_schema", "json_schema": {"name": return_type, "schema": schema, "strict": True}}


# ---------------------------------------------------------------------------
# byllm declaration extraction
# ---------------------------------------------------------------------------
def find_byllm_abilities(program: JacProgram) -> Iterator[uni.Ability]:
    """Yield every genai (by llm) function declaration across the program, deduped."""
    seen: set = set()
    for mod_path, mod in program.mod.hub.items():
        if not mod_path.endswith(".jac"):
            continue  # hub also carries py-imported modules; byllm defs live in .jac only
        for ab in mod.get_all_sub_nodes(uni.Ability):
            if not ab.is_genai_ability:
                continue
            if not isinstance(ab.signature, uni.FuncSignature):
                continue  # event-signature abilities can't be a byllm def
            key = (ab.loc.mod_path, ab.loc.first_line, ab.loc.col_start)
            if key in seen:
                continue
            seen.add(key)
            yield ab


def params_of(sig: uni.FuncSignature) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sig.get_parameters():
        out.append({
            "name": p.name.value,
            "type": clean_type(p.type_tag.unparse()) if p.type_tag else "",
            "sem": (p.semstr or "").strip(),
            "required": p.value is None,
        })
    return out


def extract_llm_call(call: uni.UniNode) -> Tuple[List[str], Dict[str, Any], str]:
    """Pull tool names, literal kwargs, and a literal system_prompt out of the `by llm(...)` expression."""
    tool_names: List[str] = []
    call_params: Dict[str, Any] = {}
    extra_sys = ""
    if not isinstance(call, uni.FuncCall):
        return tool_names, call_params, extra_sys
    for kw in call.params or []:
        if not isinstance(kw, uni.KWPair) or kw.key is None:
            continue
        key = kw.key.unparse().strip()
        if key == "tools":
            if isinstance(kw.value, uni.ListVal):
                for item in kw.value.values or []:
                    tool_names.append(re.sub(r"\s*\.\s*", ".", item.unparse().strip()))
            else:
                print(f"[side-runtime] warning: non-literal tools list ({kw.value.unparse()}); tool schemas unavailable statically")
        elif key == "system_prompt":
            ok, v = literal_of(kw.value)
            if ok:
                extra_sys = str(v)
            else:
                print(f"[side-runtime] warning: dynamic system_prompt ({kw.value.unparse()}); invariant excludes it")
        else:
            ok, v = literal_of(kw.value)
            if ok:
                call_params[key] = v
    return tool_names, call_params, extra_sys


def resolve_tool_ability(program: JacProgram, tool_name: str) -> Optional[uni.Ability]:
    simple = tool_name.split(".")[-1]
    for mod_path, mod in program.mod.hub.items():
        if not mod_path.endswith(".jac"):
            continue
        for ab in mod.get_all_sub_nodes(uni.Ability):
            if ab.is_genai_ability or not isinstance(ab.name_ref, uni.Name):
                continue
            if ab.name_ref.value == simple and isinstance(ab.signature, uni.FuncSignature):
                return ab
    return None


def tool_schema(program: JacProgram, tool_name: str) -> Dict[str, Any]:
    """Build the OpenAI tool schema statically, mirroring byllm's tool_to_schema."""
    simple = tool_name.split(".")[-1]
    ab = resolve_tool_ability(program, tool_name)
    if ab is None:
        print(f"[side-runtime] warning: tool '{tool_name}' not resolvable statically (imported Python function?); emitting name-only schema")
        return {"type": "function", "function": {"name": simple, "description": simple, "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}}
    params = params_of(ab.signature)
    sem = (ab.semstr or "").strip()
    desc = f"{simple}: {sem}" if sem else simple
    props: Dict[str, Any] = {}
    required: List[str] = []
    for p in params:
        props[p["name"]] = {"type": json_type_of(p["type"]), "description": p["sem"] or p["type"]}
        if p["required"]:
            required.append(p["name"])
    return {"type": "function", "function": {"name": simple, "description": desc, "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}}


def finish_tool_schema(return_type: str, defs: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    prop = dict(json_schema_of(return_type, defs))
    prev = prop.get("description", "")
    prop["description"] = "The final output of the tool calls." + (f" {prev}" if prev else "")
    return {"type": "function", "function": {"name": "finish_tool", "description": "This tool is used to finish the tool calls and return the final output.", "parameters": {"type": "object", "properties": {"final_output": prop}, "required": ["final_output"], "additionalProperties": False}}}


def build_decl(program: JacProgram, ab: uni.Ability, type_defs: Optional[Dict[str, Dict[str, Any]]] = None) -> ByLLMDecl:
    """Assemble the full compile-time record for one genai ability."""
    if isinstance(ab.name_ref, uni.Name):
        name = ab.name_ref.value
    else:
        name = ab.py_resolve_name()
    owner = ab.method_owner
    qualifier = f"{owner.name.value}." if owner is not None else ""
    owner_sem = (getattr(owner, "semstr", "") or "").strip() if owner is not None else ""
    params = params_of(ab.signature)
    ret = clean_type(ab.signature.return_type.unparse()) if ab.signature.return_type else "str"
    tool_names, call_params, extra_sys = extract_llm_call(ab.body)
    tools = [tool_schema(program, t) for t in tool_names]
    if tools:
        tools.append(finish_tool_schema(ret, type_defs))
    return ByLLMDecl(
        name=name,
        qualifier=qualifier,
        params=params,
        return_type=ret,
        sem=(ab.semstr or "").strip(),
        owner_sem=owner_sem,
        tools=tools,
        call_params=call_params,
        extra_system_prompt=extra_sys,
        module=ab.loc.mod_path,
        lineno=ab.loc.first_line,
        reponse_format=None if tools else response_format_of(ret, type_defs),
        return_type_obj=resolve_type(ret, materialized_namespace(ret, type_defs)),
    )


# ---------------------------------------------------------------------------
# Call-site topology: which byllm call can run next after a given one
# ---------------------------------------------------------------------------
_BRANCHING_STMTS = (uni.IfStmt, uni.WhileStmt, uni.IterForStmt, uni.InForStmt, uni.TryStmt, uni.MatchStmt)
_LOOP_STMTS = (uni.WhileStmt, uni.IterForStmt, uni.InForStmt)


def ability_key(ab: uni.Ability) -> str:
    """Graph key of an ability: same qualifier+name scheme as ByLLMDecl."""
    owner = ab.method_owner
    q = f"{owner.name.value}." if owner is not None else ""
    nm = ab.name_ref.value if isinstance(ab.name_ref, uni.Name) else ab.py_resolve_name()
    return q + nm


def _call_target_name(call: uni.FuncCall) -> str:
    """Rightmost name of a call target: `respond` for both `respond(...)` and `self.respond(...)`."""
    tgt = call.target
    if isinstance(tgt, uni.Name):
        return tgt.value
    if isinstance(tgt, uni.AtomTrailer):
        return re.sub(r"\s*\.\s*", ".", tgt.unparse().strip()).split(".")[-1]
    return ""


def _iter_jac_modules(program: JacProgram) -> Iterator[uni.Module]:
    for mod_path, mod in program.mod.hub.items():
        if mod_path.endswith(".jac"):
            yield mod


def _class_chain(program: JacProgram, class_name: str) -> set:
    """The class and its transitive base classes, by name (MRO approximation)."""
    arch_map: Dict[str, uni.Archetype] = {}
    for mod in _iter_jac_modules(program):
        for arch in mod.get_all_sub_nodes(uni.Archetype):
            arch_map.setdefault(arch.name.value, arch)
    seen = {class_name}
    queue = [class_name]
    while queue:
        arch = arch_map.get(queue.pop(0))
        if arch is None:
            continue
        for b in (arch.base_classes or []):
            bn = re.sub(r"\s*\.\s*", ".", b.unparse().strip()).split(".")[-1]
            if bn not in seen:
                seen.add(bn)
                queue.append(bn)
    return seen


def _call_decl_keys(call: uni.FuncCall, program: JacProgram, by_simple: Dict[str, List[str]]) -> List[str]:
    """Decl keys a call site can target, receiver-aware.

    `self.method(...)` resolves through the enclosing ability's owner class and
    its base chain; bare `method(...)` prefers the module-level decl. Anything
    else (arbitrary receiver expression) keeps every same-name candidate — the
    over-approximation side. Dynamic aliasing (f = self.m; f()) is out of scope.
    """
    cands = by_simple.get(_call_target_name(call), [])
    if not cands:
        return []
    tgt = call.target
    if isinstance(tgt, uni.Name):
        bare = [k for k in cands if "." not in k]
        return bare or cands
    if isinstance(tgt, uni.AtomTrailer):
        parts = re.sub(r"\s*\.\s*", ".", tgt.unparse().strip()).split(".")
        if len(parts) == 2 and parts[0] == "self":
            enc = call.find_parent_of_type(uni.Ability)
            owner = enc.method_owner if enc is not None else None
            if owner is not None:
                chain = _class_chain(program, owner.name.value)
                exact = [k for k in cands if "." in k and k.split(".")[0] in chain]
                return exact or cands  # fall back to superset rather than drop edges
    return cands


def _byllm_calls_in(node: uni.UniNode, program: JacProgram, by_simple: Dict[str, List[str]]) -> List[str]:
    """Decl keys of every byllm invocation inside a subtree, in source order."""
    out: List[str] = []
    for call in node.get_all_sub_nodes(uni.FuncCall):
        for key in _call_decl_keys(call, program, by_simple):
            if key not in out:
                out.append(key)
    return out


def _invocation_sites(program: JacProgram, simple_name: str) -> List[uni.FuncCall]:
    """Every FuncCall in the program whose target resolves (by name) to `simple_name`."""
    sites: List[uni.FuncCall] = []
    for mod in _iter_jac_modules(program):
        for call in mod.get_all_sub_nodes(uni.FuncCall):
            if _call_target_name(call) == simple_name:
                sites.append(call)
    return sites


def _sites_of_decl(program: JacProgram, key: str, by_simple: Dict[str, List[str]]) -> List[uni.FuncCall]:
    """Invocation sites that can actually target this decl key (receiver-aware)."""
    simple = key.split(".")[-1]
    return [c for c in _invocation_sites(program, simple) if key in _call_decl_keys(c, program, by_simple)]


def _trigger_names(ab: uni.Ability) -> List[str]:
    """Type names in an ability's `with X entry` clause (event_triggers when the
    pass filled them, else parsed off the EventSignature's arch tag)."""
    names = ab.event_trigger_type_names() or []
    if names:
        return [n.split(".")[-1] for n in names]
    tag = getattr(ab.signature, "arch_tag_info", None)
    if tag is None:
        return []
    txt = clean_type(tag.unparse())
    return [part.strip().split(".")[-1] for part in txt.split("|") if part.strip()]


def _enclosing_walker_names(node: uni.UniNode) -> List[str]:
    """Walker type(s) whose traversal a `visit` inside this node extends.

    Inside a walker's own ability that is the walker itself; inside a node
    ability (`can x with W entry`) it is the triggering walker type(s).
    """
    enc = node.find_parent_of_type(uni.Ability)
    if enc is None:
        return []
    owner = enc.method_owner
    if owner is None:
        return []
    kind = owner.arch_type.value if hasattr(owner.arch_type, "value") else ""
    if kind == "walker":
        return [owner.name.value]
    return _trigger_names(enc)


def _visit_successors(program: JacProgram, walker_name: str, by_simple: Dict[str, List[str]]) -> List[str]:
    """byllm calls inside node abilities this walker can trigger (`can x with W entry`)."""
    out: List[str] = []
    for mod in _iter_jac_modules(program):
        for nab in mod.get_all_sub_nodes(uni.Ability):
            owner = nab.method_owner
            if owner is None or nab.is_genai_ability:
                continue
            kind = owner.arch_type.value if hasattr(owner.arch_type, "value") else ""
            if kind != "node":
                continue
            trigs = set(_trigger_names(nab))
            if walker_name not in trigs:
                continue
            for key in _byllm_calls_in(nab, by_simple):
                if key not in out:
                    out.append(key)
    return out


def _next_after(site: uni.UniNode, program: JacProgram, by_simple: Dict[str, List[str]]) -> Tuple[List[str], bool]:
    """Successor decl keys reachable after `site` completes, scanning forward in control flow.

    Climbs enclosing statement lists via the ordered `kid` children. An
    unconditional byllm call at some level ends the climb (execution surely
    reaches it first); branching statements contribute their contained calls as
    possible successors but the scan continues. Returns (keys, escaped) where
    escaped means the enclosing callable's body was exhausted — the caller's
    continuation decides what runs next.
    """
    out: List[str] = []
    cur: uni.UniNode = site
    while True:
        p = cur.parent
        if p is None:
            return out, False
        kids = list(getattr(p, "kid", None) or [])
        following: List[uni.UniNode] = []
        if cur in kids:
            after = kids[kids.index(cur) + 1:]
            following = [k for k in after if isinstance(k, uni.CodeBlockStmt) and not isinstance(k, uni.ElseIf)]
        stopped = False
        for st in following:
            visit_nodes = [st] if isinstance(st, uni.VisitStmt) else list(st.get_all_sub_nodes(uni.VisitStmt))
            for vs in visit_nodes:
                for wname in _enclosing_walker_names(vs):
                    for key in _visit_successors(program, wname, by_simple):
                        if key not in out:
                            out.append(key)
            calls = _byllm_calls_in(st, by_simple)
            for key in calls:
                if key not in out:
                    out.append(key)
            if calls and not isinstance(st, _BRANCHING_STMTS):
                stopped = True
                break
        if stopped:
            return out, False
        if isinstance(p, _LOOP_STMTS):
            # Falling off a loop-body iteration can re-enter the loop from its top.
            for key in _byllm_calls_in(p, by_simple):
                if key not in out:
                    out.append(key)
        if isinstance(p, uni.Ability):
            return out, True
        if isinstance(p, (uni.ModuleCode, uni.Module)):
            return out, False
        cur = p


def _successors_of_callable(program: JacProgram, simple_name: str, by_simple: Dict[str, List[str]], visited: set) -> List[str]:
    """Union of what can run next after any invocation of `simple_name` returns."""
    out: List[str] = []
    for site in _invocation_sites(program, simple_name):
        keys, escaped = _next_after(site, program, by_simple)
        for key in keys:
            if key not in out:
                out.append(key)
        if not escaped:
            continue
        encl = site.find_parent_of_type(uni.Ability)
        if encl is None:
            continue
        owner = encl.method_owner
        kind = owner.arch_type.value if owner is not None and hasattr(owner.arch_type, "value") else ""
        if kind in ("walker", "node"):
            continue  # ability end hands control to graph traversal, handled via visit edges
        ename = encl.name_ref.value if isinstance(encl.name_ref, uni.Name) else encl.py_resolve_name()
        if ename in visited:
            continue
        visited.add(ename)
        for key in _successors_of_callable(program, ename, by_simple, visited):
            if key not in out:
                out.append(key)
    return out


def parse_callsite_topology(program: JacProgram, ab: uni.Ability, funcDecls: Dict[str, ByLLMDecl]) -> List[str]:
    """Decl keys of the byllm calls that can run next after `ab` completes.

    Static may-happen-next over-approximation: sequential flow, branches, loops
    (self/repeat edges), caller continuations, and walker `visit` edges into
    node abilities. Over-approximating costs only wasted speculative prefill;
    missing an edge costs a cold TTFT — so edges err toward inclusion.
    """
    by_simple: Dict[str, List[str]] = {}
    for key, d in funcDecls.items():
        by_simple.setdefault(d.name, []).append(key)
    simple = ab.name_ref.value if isinstance(ab.name_ref, uni.Name) else ab.py_resolve_name()
    return _successors_of_callable(program, simple, by_simple, visited={simple})


def build_callsite_graph(program: JacProgram, funcDecls: Dict[str, ByLLMDecl]) -> Dict[str, List[str]]:
    """Adjacency over byllm call sites: key -> decl keys that can be invoked next."""
    graph: Dict[str, List[str]] = {}
    for ab in find_byllm_abilities(program):
        graph[ability_key(ab)] = parse_callsite_topology(program, ab, funcDecls)
    return graph