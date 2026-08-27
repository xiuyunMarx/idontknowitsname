"""Static mirror of byllm's schema generation (jaclang/byllm/impl/schema.impl.jac).

byllm builds the `response_format` a typed `by llm()` return is decoded against
from the *runtime* Python type plus MTIR info. Here the same bytes are produced
from the Jac declarations alone (UniIR: `obj`/`node`/`walker` fields, `enum`
members, `sem` strings), so the server can hold a callsite's schema before the
client's first frame arrives — the client still sends it, and the two must agree.

Three renderers, all keyed on a Jac type annotation text (`list[Point] | None`):

  render_response_format(text, registry)  -> the `{"type": "json_schema", ...}` dict
                                             (None for `str`, the schema-less default)
  schema_to_hint_text(rf)                 -> byllm's `Schema requirements:` block, the
                                             prompt-side hint it appends to the last
                                             user message for backends that treat
                                             response_format as a grammar only
  describe_type(text, registry)           -> (head, body) of byllm's schema-zone entry
                                             for a param of that type: the obj/enum
                                             member rows with their nested semstrings

Rules were checked byte-for-byte against byllm on ~40 return types (bool/int/float/
None, enums with/without values and sems, objs with inheritance/defaults/postinit,
list/dict/tuple/union nesting). The registry is built from UniIR by
`parsing.build_type_registry`; this module never touches the compiler.

The quirks below are byllm's, reproduced on purpose:

  * A schema-less obj describes itself with the dataclass auto-docstring
    (`Point(x: 'int', y: 'int' = 3)`, `<factory>` for non-constant defaults).
  * Titles are `str.title()`d (`WithDefault` -> `Withdefault`, `a_b` -> `A B`).
  * MTIR info (field sems, enum member `Details:`) survives through obj fields and one
    level of list/tuple items for objs; it is dropped inside dicts, unions, list-of-
    list, and for enums inside any container. Inherited fields carry no sems.
  * Unsupported at the top level, raising SchemaRenderError exactly where byllm raises
    ConfigurationError (or crashes): unions, `set[T]`, `tuple[A, B]`, bare containers,
    `any`, enums of mixed value types, `static has` fields, unknown names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_OBJECT_WRAPPER = "schema_object_wrapper"
SCHEMA_DICT_WRAPPER = "schema_dict_wrapper"

_PRIMITIVES = {"bool": "boolean", "int": "integer", "float": "number", "str": "string"}
_NONE_NAMES = {"None", "NoneType"}


class SchemaRenderError(Exception):
    """byllm would raise ConfigurationError (or crash) generating this schema."""


class _Factory:
    """Sentinel for a `has` default byllm's auto-docstring prints as `<factory>`."""

    def __repr__(self) -> str:
        return "<factory>"


FACTORY = _Factory()
NO_DEFAULT = object()


# --------------------------------------------------------------------------- decls

@dataclass
class FieldDecl:
    name: str
    type_text: str
    semstr: str = ""
    default: Any = NO_DEFAULT  # a constant, FACTORY, or NO_DEFAULT
    init: bool = True  # False for `has x: T by postinit` (dataclass init=False)
    is_static: bool = False  # `static has`: a ClassVar byllm cannot schema


@dataclass
class ObjDecl:
    name: str
    semstr: str = ""  # `sem Name = "..."` -> `_jac_semstr`
    doc: str = ""  # docstring -> `__doc__` (else the dataclass auto-doc)
    bases: List[str] = field(default_factory=list)
    fields: List[FieldDecl] = field(default_factory=list)  # own fields, declaration order
    kind: str = "obj"


@dataclass
class EnumMember:
    name: str
    value: Any = None  # None: auto-numbered like Python's enum (1, 2, ...)
    value_kind: Optional[str] = None  # MTIR type_info of the literal: "int" | "str" | other
    semstr: str = ""


@dataclass
class EnumDecl:
    name: str
    semstr: str = ""
    doc: str = ""
    members: List[EnumMember] = field(default_factory=list)


@dataclass
class TypeRegistry:
    """Every user type a return annotation may name, from one module's UniIR."""
    objs: Dict[str, ObjDecl] = field(default_factory=dict)
    enums: Dict[str, EnumDecl] = field(default_factory=dict)

    # ------------------------------------------------------------------ MRO

    def mro(self, name: str) -> List[str]:
        """C3 linearization over the objs we know; unknown bases are skipped."""
        decl = self.objs.get(name)
        if decl is None:
            return [name]
        seqs = [self.mro(b) for b in decl.bases if b in self.objs]
        seqs.append([b for b in decl.bases if b in self.objs])
        out = [name]
        while any(seqs):
            for seq in seqs:
                head = seq[0] if seq else None
                if head is not None and not any(head in s[1:] for s in seqs):
                    break
            else:
                raise SchemaRenderError(f"inconsistent MRO for {name}")
            out.append(head)
            for seq in seqs:
                if seq and seq[0] == head:
                    seq.pop(0)
        return out

    def all_fields(self, name: str) -> List[Tuple[FieldDecl, bool]]:
        """(field, is_own) in `typing.get_type_hints` order: bases first (reversed
        MRO), a redeclared name keeping its first position."""
        seen: Dict[str, Tuple[FieldDecl, bool]] = {}
        for cls_name in reversed(self.mro(name)):
            decl = self.objs.get(cls_name)
            for f in decl.fields if decl else []:
                seen[f.name] = (f, cls_name == name)
        return list(seen.values())


# ------------------------------------------------------------------ type text

@dataclass
class TypeExpr:
    name: str  # base name; "|" for a union, "..." for Ellipsis
    args: List["TypeExpr"] = field(default_factory=list)

    @property
    def is_union(self) -> bool:
        return self.name == "|"


_TOKEN = re.compile(r"\s*(\.\.\.|[A-Za-z_][A-Za-z0-9_.]*|\[|\]|,|\|)")


def parse_type(text: str) -> TypeExpr:
    """`list[dict[str, Point]] | None` -> TypeExpr tree."""
    text = _clean_type(text)
    pos = 0
    toks: List[str] = []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise SchemaRenderError(f"cannot parse type {text!r}")
        toks.append(m.group(1))
        pos = m.end()

    def union(i: int) -> Tuple[TypeExpr, int]:
        opts = []
        node, i = atom(i)
        opts.append(node)
        while i < len(toks) and toks[i] == "|":
            node, i = atom(i + 1)
            opts.append(node)
        return (opts[0] if len(opts) == 1 else TypeExpr("|", opts)), i

    def atom(i: int) -> Tuple[TypeExpr, int]:
        if i >= len(toks):
            raise SchemaRenderError(f"unexpected end of type {text!r}")
        tok = toks[i]
        if tok in ("[", "]", ",", "|"):
            raise SchemaRenderError(f"unexpected {tok!r} in type {text!r}")
        node = TypeExpr(tok)
        i += 1
        if i < len(toks) and toks[i] == "[":
            i += 1
            while True:
                arg, i = union(i)
                node.args.append(arg)
                if i < len(toks) and toks[i] == ",":
                    i += 1
                    continue
                if i < len(toks) and toks[i] == "]":
                    return node, i + 1
                raise SchemaRenderError(f"unbalanced brackets in type {text!r}")
        return node, i

    node, i = union(0)
    if i != len(toks):
        raise SchemaRenderError(f"trailing tokens in type {text!r}")
    return node


# ------------------------------------------------------------ response_format

def render_response_format(return_type: str, registry: TypeRegistry,
                           has_tools: bool = False) -> Optional[Dict[str, Any]]:
    """byllm `MTRuntime.get_output_schema()` for a decl returning `return_type`.

    None when byllm sends no schema: a `str` return, or a call with tools (the
    answer then arrives through finish_tool). A missing annotation is NoneType."""
    if has_tools:
        return None
    text = _clean_type(return_type) or "None"
    if text == "str":
        return None
    expr = parse_type(text)
    type_name = _name_of_type(expr)
    schema = _wrap_to_object(_type_to_schema(expr, registry, type_name, info=True))
    return {"type": "json_schema", "json_schema": {"name": type_name, "schema": schema, "strict": True}}


def _name_of_type(expr: TypeExpr) -> str:
    if expr.is_union:
        # byllm's `_name_of_type` recursion passes no `info` and crashes.
        raise SchemaRenderError("byllm cannot generate a schema for a top-level union return")
    return "NoneType" if expr.name in _NONE_NAMES else expr.name.split(".")[-1]


def _title(title: str) -> str:
    return title.replace("_", " ").title()


def _type_to_schema(expr: TypeExpr, reg: TypeRegistry, title: str = "", desc: str = "",
                    info: bool = False, list_depth: int = 0) -> Dict[str, Any]:
    """`_type_to_schema` of schema.impl.jac. `info` says whether byllm would hold
    MTIR info for this position (field sems, enum member details)."""
    title = _title(title)
    context: Dict[str, Any] = {}
    if title:
        context["title"] = title
    if desc:
        context["description"] = desc
    name = expr.name.split(".")[-1]

    if not expr.is_union and not expr.args and name in ("list", "dict", "set", "tuple"):
        raise SchemaRenderError(f"Untyped {name} is not supported for schema generation. Use {name}[T, ...] instead.")
    if name in _NONE_NAMES and not expr.args:
        return _merge({"type": "null"}, context)
    if name in _PRIMITIVES and not expr.args:
        return _merge({"type": _PRIMITIVES[name]}, context)
    if expr.is_union:
        return _merge({"anyOf": [_type_to_schema(a, reg) for a in expr.args], "title": title}, context)
    if name == "list" and expr.args:
        item = expr.args[0]
        keep = info and list_depth == 0 and item.name.split(".")[-1] in reg.objs
        return _merge({"type": "array", "items": _type_to_schema(item, reg, info=keep, list_depth=list_depth + 1)}, context)
    if name in ("tuple", "set"):
        if name == "tuple" and len(expr.args) == 2 and expr.args[1].name == "...":
            item = expr.args[0]
            keep = info and list_depth == 0 and item.name.split(".")[-1] in reg.objs
            return _merge({"type": "array", "items": _type_to_schema(item, reg, info=keep, list_depth=list_depth + 1)}, context)
        raise SchemaRenderError(f"Unsupported {name} type for schema generation: {_unparse(expr)}. Only {name} of the form {name}[T, ...] are supported.")
    if name == "dict":
        if len(expr.args) != 2:
            raise SchemaRenderError(f"Expected a dictionary type, got {_unparse(expr)}.")
        key, value = expr.args
        return _merge({
            "type": "object",
            "title": SCHEMA_DICT_WRAPPER,
            "properties": {SCHEMA_DICT_WRAPPER: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"key": _type_to_schema(key, reg), "value": _type_to_schema(value, reg)},
                    "required": ["key", "value"],
                    "additionalProperties": False,
                },
            }},
            "additionalProperties": False,
            "required": [SCHEMA_DICT_WRAPPER],
        }, context)
    if name in reg.objs:
        return _obj_schema(reg.objs[name], reg, title, info)
    if name in reg.enums:
        return _enum_schema(reg.enums[name], info)
    raise SchemaRenderError(f"Unsupported type for schema generation: {_unparse(expr)}. Only primitive types, dataclasses, and Union types are supported.")


def _obj_schema(decl: ObjDecl, reg: TypeRegistry, title: str, info: bool) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    for f, own in reg.all_fields(decl.name):
        if f.name.startswith("_"):
            continue
        if f.is_static:
            raise SchemaRenderError(f"Unsupported type for schema generation: typing.ClassVar[{f.type_text}]. Only primitive types, dataclasses, and Union types are supported.")
        has_info = info and own  # FieldInfo exists for own fields only
        properties[f.name] = _type_to_schema(
            parse_type(f.type_text or "None"), reg, f.name, f.semstr if has_info else "", info=has_info)
    return {
        "title": title or decl.name,
        "description": decl.semstr or decl.doc or _auto_doc(decl, reg),
        "type": "object",
        "properties": properties,
        "required": list(properties.keys()),
        "additionalProperties": False,
    }


def _auto_doc(decl: ObjDecl, reg: TypeRegistry) -> str:
    """dataclass's generated `__doc__`: the class name + `inspect.signature(cls)`,
    annotations shown as the strings Jac's codegen emits."""
    parts = []
    for f, _ in reg.all_fields(decl.name):
        if not f.init or f.is_static:
            continue
        p = f"{f.name}: {f.type_text!r}" if f.type_text else f.name
        if f.default is not NO_DEFAULT:
            p += f" = {f.default!r}"
        parts.append(p)
    return f"{decl.name}({', '.join(parts)})"


def _enum_values(decl: EnumDecl) -> Tuple[List[str], List[Any], str]:
    """(names, runtime values, "int"|"str"). Bare members auto-number like enum.auto."""
    names, values = [], []
    last_int = 0
    for m in decl.members:
        v = m.value
        if v is None:
            v = last_int + 1
        if isinstance(v, int) and not isinstance(v, bool):
            last_int = v
        names.append(m.name)
        values.append(v)
    kinds = {(m.value_kind or "int") for m in decl.members}
    if len(kinds) > 1:
        raise SchemaRenderError(f"Enum {decl.name} has mixed types. Not supported for schema generation.")
    kind = kinds.pop() if kinds else "int"
    if kind not in ("int", "str"):
        raise SchemaRenderError(f"Enum {decl.name} has unsupported type {kind}. Only int and str enums are supported for schema generation.")
    return names, values, kind


def _enum_schema(decl: EnumDecl, info: bool) -> Dict[str, Any]:
    names, values, kind = _enum_values(decl)
    semstr = decl.semstr or decl.doc or ""
    enum_desc = f"\nThe value *should* be one in this list: {values} where the names are [{', '.join(names)}]."
    if info:
        details = [f"- {m.name}: {m.semstr}" for m in decl.members if m.semstr]
        if details:
            enum_desc += "\nDetails:\n" + "\n".join(details)
        full = f"{decl.semstr or semstr}{enum_desc}"
    else:
        full = semstr + enum_desc
    out: Dict[str, Any] = {"description": full, "type": "integer" if kind == "int" else "string"}
    if values:
        out["enum"] = values
    return out


def _wrap_to_object(schema: Dict[str, Any]) -> Dict[str, Any]:
    if schema.get("type") == "object":
        return schema
    return {
        "type": "object",
        "title": SCHEMA_OBJECT_WRAPPER,
        "properties": {SCHEMA_OBJECT_WRAPPER: schema},
        "required": [SCHEMA_OBJECT_WRAPPER],
        "additionalProperties": False,
    }


def _merge(base: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Python's `base | context`: base key order, overrides in place, new keys last."""
    out = dict(base)
    out.update(context)
    return out


# ------------------------------------------------------------------ hint text
# Port of schema.impl.jac's `_hints_from_schema` / `schema_to_hint_text`.

def _format_enum_hint(values: list, names: list) -> str:
    if not values:
        return ""
    return ", ".join(f"{v}={names[i]}" if i < len(names) else f"{v}" for i, v in enumerate(values))


def _hints_from_schema(name: str, prop: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(prop, dict):
        return out
    desc = prop.get("description")
    enum_vals = prop.get("enum")
    ty = prop.get("type", "")
    label = name or "value"
    if isinstance(enum_vals, list) and enum_vals:
        names: List[str] = []
        if isinstance(desc, str) and "names are [" in desc:
            chunk = desc.split("names are [", 1)[1].split("]", 1)[0]
            names = [n.strip() for n in chunk.split(",") if n.strip()]
        hint = _format_enum_hint(enum_vals, names)
        out.append(f"- {label} must be one of: {hint}" if hint else f"- {label} must be one of: {enum_vals}")
    elif ty:
        suffix = " -- " + desc if isinstance(desc, str) and desc else ""
        out.append(f"- {label} ({ty}){suffix}")
    if prop.get("type") == "object":
        nested = prop.get("properties")
        if isinstance(nested, dict):
            for sub, sub_prop in nested.items():
                if sub != name:
                    out.extend(_hints_from_schema(sub, sub_prop))
    return out


def schema_to_hint_text(rf: Optional[Dict[str, Any]]) -> str:
    """byllm's prompt-side `Schema requirements:` block for a response_format (either
    the `json_schema` or the `json_object` shape); "" when nothing is worth saying."""
    if not isinstance(rf, dict):
        return ""
    schema = None
    if rf.get("type") == "json_schema":
        js = rf.get("json_schema")
        if isinstance(js, dict) and isinstance(js.get("schema"), dict):
            schema = js["schema"]
    elif rf.get("type") == "json_object":
        if isinstance(rf.get("schema"), dict):
            schema = rf["schema"]
    if schema is None:
        return ""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return ""
    lines: List[str] = []
    for name, prop in props.items():
        if isinstance(prop, dict):
            lines.extend(_hints_from_schema(name, prop))
    return "Schema requirements:\n" + "\n".join(lines) if lines else ""


def attach_schema_hint(user_content: str, rf: Optional[Dict[str, Any]]) -> str:
    """byllm `attach_schema_hint` for one user message body (idempotent)."""
    hint = schema_to_hint_text(rf)
    if not hint or hint in user_content:
        return user_content
    return user_content.rstrip() + "\n\n" + hint


# --------------------------------------------------------- schema-zone rows
# Port of mtir.impl.jac `_describe_input_type` / `_type_desc_parts` for a declared
# type. byllm walks the *value* (so an empty list or None expands to nothing); the
# static form expands the declared type, looking through containers the way byllm
# looks through a non-empty one.

def describe_type(type_text: str, registry: TypeRegistry, _depth: int = 0) -> str:
    expr = parse_type(type_text) if type_text else None
    return _describe_expr(expr, registry, _depth) if expr is not None else ""


def _describe_expr(expr: TypeExpr, reg: TypeRegistry, depth: int) -> str:
    name = expr.name.split(".")[-1]
    if expr.is_union or name in ("list", "tuple", "set", "dict"):
        for arg in expr.args:
            if arg.name in _NONE_NAMES:
                continue
            inner = _describe_expr(arg, reg, depth)
            if inner:
                return inner
        return ""
    pad = "    " * (depth + 1)
    if name in reg.enums:
        decl = reg.enums[name]
        names, values, _ = _enum_values(decl)
        rows = [f"{pad}{decl.name}" + (f" -- {decl.semstr}" if decl.semstr else "")]
        for m, v in zip(decl.members, values):
            rows.append(f"{pad}  - {m.name} = {v!r}" + (f" ---- {m.semstr}" if m.semstr else ""))
        return "\n".join(rows)
    if name in reg.objs:
        decl = reg.objs[name]
        rows = [f"{pad}{decl.name}" + (f" -- {decl.semstr}" if decl.semstr else "")]
        for f, own in reg.all_fields(decl.name):
            if f.name.startswith("_") or f.is_static:
                continue
            sem = f.semstr if own else ""  # `_jac_semstr_inner` holds own fields only
            rows.append(f"{pad}  - {f.name}: {f.type_text or 'None'}" + (f" ---- {sem}" if sem else ""))
            nested = _describe_expr(parse_type(f.type_text), reg, depth + 2) if f.type_text else ""
            if nested:
                rows.append(nested)
        return "\n".join(rows)
    return ""


def type_desc_parts(type_text: str, registry: TypeRegistry) -> Tuple[str, str]:
    """(head, body) of byllm `_type_desc_parts`: the type's own line and the member
    rows shifted left by two to sit under the param row."""
    desc = describe_type(type_text, registry)
    if not desc:
        return "", ""
    lines = desc.split("\n")
    head = lines[0].strip()
    body = "\n".join(line[2:] if line.startswith("  ") else line for line in lines[1:])
    return head, body


def schema_entry(pname: str, type_text: str, psem: str, registry: TypeRegistry) -> str:
    """byllm `_schema_entry`: the param's row (`name: Type -- tsem ---- psem`) plus the
    expanded member rows; "" for an undescribed primitive."""
    head, body = type_desc_parts(type_text, registry)
    if not psem and not body:
        return ""
    if not head:
        head = type_text or "None"
    rows = [f"{pname}: {head}" + (f" ---- {psem}" if psem else "")]
    if body:
        rows.append(body)
    return "\n".join(rows)


# ------------------------------------------------------------------ helpers

def _clean_type(text: str) -> str:
    t = (text or "").strip()
    if t.startswith(":"):
        t = t[1:].strip()
    t = re.sub(r"\s*\[\s*", "[", t)
    t = re.sub(r"\s*\]", "]", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s*\|\s*", " | ", t)
    t = re.sub(r"\s*\.\s*", ".", t)
    return t


def _unparse(expr: TypeExpr) -> str:
    if expr.is_union:
        return " | ".join(_unparse(a) for a in expr.args)
    if expr.args:
        return f"{expr.name}[{', '.join(_unparse(a) for a in expr.args)}]"
    return expr.name
