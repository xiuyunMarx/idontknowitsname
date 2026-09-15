"""Static analyzer: one Jac agent program -> its Program of call sites.

    jac run analyzer.jac <agent.jac> [-o program.json]

What it derives, and from where:

  call sites      every call expression whose callee is a `by llm` ability
                  (`FuncCall.callee_decl.is_genai_ability`), keyed by CallSiteID
                  (qualified function name, line of the call, file).
  constant bytes  system prompt, header + schema rows, self member rows, hint,
                  response_format, tool schema: rendered by byllm's own MTIR
                  factory on the compiled module, so they match the runtime
                  byte for byte. The module is compiled with every untagged
                  `with entry` block guarded, so the agent does not run.
  bindings        one per parameter, classified by reaching definitions of the
                  walker field it carries over the visit graph: CONST (literal),
                  COPY (earlier site carries the same bytes), EXTEND (append-only),
                  RESP (an earlier site's reply), VOLATILE (fresh or opaque).
  graph           edges between call sites: the next `by llm` call reachable
                  through the walker's visit statements, abilities without LLM
                  calls being passed through. Entry: the node `root ++> X()`
                  connects. Exits: sites after which no visit follows.
"""

import os;
import re;
import sys;
import json;
import ast as pyast;
import marshal;
import types;
import dataclasses;
import enum;
import from typing { get_type_hints, get_origin, get_args }
import from jaclang { JacRuntime as Jac }
import jaclang.jac0core.unitree as uni;
import from jaclang.jac0core.mtp { fetch_mtir }
import from jaclang.byllm.mtir { MTIR }
import from jaclang.byllm.schema { schema_to_hint_text }
import from primitives {
    CallSiteID, Binding, BindingKind, Heterogeneity, PromptTemplate, Program, program_to_dict
}

glob ENTRY_GUARD = "__static_analysis__";
glob EXTEND_METHODS = {"append", "extend", "insert", "update", "add"};
glob RESET_METHODS = {"clear", "pop", "remove", "sort", "reverse", "popitem", "discard"};
glob GROWABLE_TYPES = ("list", "str", "dict", "set");


# ----------------------------------------------------------------- helpers

"""Source text of a node with all whitespace removed, for cheap matching."""
def text(n: object) -> str {
    if n is None {
        return "";
    }
    return re.sub(r"\s+", "", n.unparse());
}

"""Strip the control-flow wrapper the compiler puts around operands."""
def unwrap(e: object) -> object {
    while isinstance(e, uni.CfgExpr) {
        e = e.expr;
    }
    return e;
}

"""`visitor.f` -> "f"; anything else -> ""."""
def walker_field(e: object) -> str {
    e = unwrap(e);
    if isinstance(e, uni.AtomTrailer) and e.is_attr
        and isinstance(e.target, uni.Name) and e.target.value == "visitor"
        and isinstance(e.right, uni.Name) {
        return e.right.value;
    }
    return "";
}

"""The `by llm(...)` call expression of a genai ability, or None."""
def by_call(ab: uni.Ability) -> uni.FuncCall | None {
    body = ab.body;
    if isinstance(body, uni.FuncCall) {
        return body;
    }
    for fc in ab.get_all_sub_nodes(uni.FuncCall) {
        return fc;
    }
    return None;
}

"""Python literal value of a literal node, else raises."""
def literal_value(e: object) -> object {
    e = unwrap(e);
    if isinstance(e, uni.Null) {
        return None;
    }
    if isinstance(e, (uni.String, uni.MultiString, uni.Int, uni.Float, uni.Bool)) {
        return pyast.literal_eval(e.unparse());
    }
    if isinstance(e, uni.ListVal) and not e.values {
        return [];
    }
    if isinstance(e, uni.DictVal) and not e.kv_pairs {
        return {};
    }
    raise ValueError("not a literal");
}

"""Whether an expression resets a container: `[]`, `{}`, `""`, `set()`."""
def is_empty_literal(e: object) -> bool {
    return text(e) in ("[]", "{}", '""', "''", "set()");
}

"""Placeholder argument of a type: only its shape matters, the rendered bytes
no longer depend on argument values."""
def placeholder(t: object) -> object {
    if t is str {
        return "";
    }
    if t is int {
        return 0;
    }
    if t is float {
        return 0.0;
    }
    if t is bool {
        return False;
    }
    origin = get_origin(t);
    if origin in (list, tuple, set, frozenset) {
        return [];
    }
    if origin is dict {
        return {};
    }
    if origin is not None {   # Optional[X], X | Y
        return None;
    }
    if isinstance(t, type) {
        if issubclass(t, enum.Enum) {
            for m in t {
                return m;
            }
        }
        if dataclasses.is_dataclass(t) {
            try {
                return t();
            } except Exception {
                return None;
            }
        }
    }
    return None;
}

"""An instance of an archetype: defaults where they exist, placeholders for
the required fields. Only the repr's shape matters."""
def instantiate(cls: type) -> object {
    import inspect;
    try {
        hints = get_type_hints(cls);
    } except Exception {
        hints = {};
    }
    kwargs: dict = {};
    for (pname, p) in inspect.signature(cls).parameters.items() {
        if p.default is inspect.Parameter.empty and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) {
            kwargs[pname] = placeholder(hints.get(pname, p.annotation));
        }
    }
    return cls(**kwargs);
}

"""Whether every path through a statement list reaches a visit: a visit at
this level, an if/elif/else chain whose branches all do, or a try whose body does."""
def covered(stmts: list) -> bool {
    for st in stmts {
        if isinstance(st, uni.VisitStmt) {
            return True;
        }
        if isinstance(st, uni.IfStmt) and if_covered(st) {
            return True;
        }
        if isinstance(st, uni.TryStmt) and covered(list(st.body)) {
            return True;
        }
    }
    return False;
}

def if_covered(st: uni.IfStmt) -> bool {
    if not covered(list(st.body)) {
        return False;
    }
    eb = st.else_body;
    if eb is None {
        return False;
    }
    if isinstance(eb, uni.IfStmt) {
        return if_covered(eb);
    }
    return covered(list(eb.body));
}

"""Whether an assignment to walker field `f` sits directly in `if not visitor.f {...}`:
the idiom that substitutes a default when the real value came back empty. Like an
except handler, it is the rare path and does not describe the field."""
def is_empty_guard(a: uni.Assignment, f: str) -> bool {
    parent = a.find_parent_of_type(uni.IfStmt);
    if parent is None {
        return False;
    }
    cond = text(parent.condition);
    return cond in (f"notvisitor.{f}", f"visitor.{f}==[]", f"len(visitor.{f})==0");
}

"""`Owner.name` of a method, `name` of a free function."""
def qualname(ab: uni.Ability) -> str {
    own = ab.method_owner;
    name = ab.name_ref.sym_name;
    return f"{own.name.value}.{name}" if own is not None else name;
}

def owner_name(ab: uni.Ability) -> str {
    own = ab.method_owner;
    return own.name.value if own is not None else "";
}


# ----------------------------------------------------------------- compile

"""Compile the agent with every untagged `with entry` guarded (same line
count, so call-site lines are the file's), execute the bytecode into a fresh
module, and return (AST, module). The AST is type-checked so `callee_decl`
resolves."""
def compile_guarded(path: str) -> tuple {
    with open(path) as f {
        src = f.read();
    }
    guarded = re.sub(r"(\bwith\s+entry)\s*\{", r"\1:" + ENTRY_GUARD + " {", src);
    if guarded.count("\n") != src.count("\n") {
        raise RuntimeError("entry guard changed the line count");
    }
    prog = Jac.get_program();
    mod = prog.compile(path, use_str=guarded, type_check=True);
    if not mod.gen.py_bytecode {
        raise RuntimeError(f"{path}: no bytecode (compile errors?)");
    }
    name = os.path.splitext(os.path.basename(path))[0];
    module = types.ModuleType(name);
    module.__file__ = path;
    sys.modules[name] = module;
    Jac.load_module(name, module);
    exec(marshal.loads(mod.gen.py_bytecode), module.__dict__);
    return (mod, module);
}


# ----------------------------------------------------------------- model

obj Def {
    """One write to a walker field."""
    has ability: str,        # qualified ability, e.g. "Planner.start"
        lineno: int,
        kind: str,           # "extend" (append / += / x = x + ...) or "assign"
        rhs: object,         # the assigned expression for "assign", else None
        in_except: bool;
}

obj Site {
    """One `by llm` call expression inside a walker ability."""
    has key: CallSiteID,
        ability: str,
        owner: str,          # node type the ability runs on
        call: uni.FuncCall,
        callee: uni.Ability,
        args: dict,          # param name -> argument expression node
        fields: dict;        # param name -> walker field it carries ("" if none)
}

obj Analysis {
    has path: str,
        ast: uni.Module,
        module: object,
        walker: str = "",
        abilities: dict = {},      # qualified ability name -> uni.Ability (the walker's entry abilities)
        by_owner: dict = {},       # node type -> qualified ability name
        visits: dict = {},         # ability -> list of node types it visits
        may_end: dict = {},        # ability -> True when some path through it visits nothing
        field_types: dict = {},    # walker field -> type text
        defs: dict = {},           # walker field -> list[Def]
        sites: list = [],          # list[Site], in line order
        entry_owner: str = "";

    # -- program shape ------------------------------------------------------

    def pick_walker -> None {
        walkers = [a for a in self.ast.get_all_sub_nodes(uni.Archetype) if a.arch_type.value == "walker"];
        if not walkers {
            raise RuntimeError("no walker in the program");
        }
        spawned = "";
        for b in self.ast.get_all_sub_nodes(uni.BinaryExpr) {
            op = b.op;
            if isinstance(op, uni.Token) and op.name == "KW_SPAWN" {
                for side in (b.left, b.right) {
                    if isinstance(side, uni.FuncCall) and isinstance(side.target, uni.Name) {
                        if side.target.value in [w.name.value for w in walkers] {
                            spawned = side.target.value;
                        }
                    }
                }
            }
        }
        self.walker = spawned or walkers[0].name.value;
        for w in walkers {
            if w.name.value == self.walker {
                for h in w.get_has_vars() {
                    self.field_types[h.name.value] = text(h.type_tag.tag) if h.type_tag else "";
                }
            }
        }
    }

    def collect_abilities -> None {
        for ab in self.ast.get_all_sub_nodes(uni.Ability) {
            sig = ab.signature;
            if not isinstance(sig, uni.EventSignature) or sig.event.value != "entry" {
                continue;
            }
            tag = text(sig.arch_tag_info) if sig.arch_tag_info else "";
            if tag != self.walker {
                continue;
            }
            name = qualname(ab);
            self.abilities[name] = ab;
            self.by_owner[owner_name(ab)] = name;
            targets: list = [];
            for vs in ab.get_all_sub_nodes(uni.VisitStmt) {
                found = self.visit_targets(vs.target, ab, vs.loc.first_line, 0);
                targets.extend(found if found else ["*"]);   # unfiltered: any connected node type
            }
            self.visits[name] = targets;
            self.may_end[name] = not covered(list(ab.body)) if isinstance(ab.body, (list, tuple)) else True;
        }
        # entry: the node type `root ++> X()` connects
        for b in self.ast.get_all_sub_nodes(uni.BinaryExpr) {
            if isinstance(b.op, uni.ConnectOp) and text(b.left) == "root"
                and isinstance(b.right, uni.FuncCall) and isinstance(b.right.target, uni.Name) {
                self.entry_owner = b.right.target.value;
            }
        }
        if not self.entry_owner and self.by_owner {
            self.entry_owner = list(self.by_owner.keys())[0];
        }
    }

    """Node types a visit target expression names: `[self-->][?:T]` filters,
    or a local bound to a comprehension over such a filter."""
    def visit_targets(e: object, ab: uni.Ability, before: int, depth: int) -> list {
        e = unwrap(e);
        if e is None or depth > 3 {
            return [];
        }
        out = [text(fc.f_type) for fc in e.get_all_sub_nodes(uni.FilterCompr) if fc.f_type is not None];
        if isinstance(e, uni.FilterCompr) and e.f_type is not None {
            out.append(text(e.f_type));
        }
        if out {
            return out;
        }
        if isinstance(e, uni.Name) {
            last = None;
            for a in ab.get_all_sub_nodes(uni.Assignment) {
                if a.loc.first_line < before and a.aug_op is None
                    and any([isinstance(t, uni.Name) and t.value == e.value for t in a.target]) {
                    if last is None or a.loc.first_line > last.loc.first_line {
                        last = a;
                    }
                }
            }
            if last is not None {
                return self.visit_targets(last.value, ab, last.loc.first_line, depth + 1);
            }
        }
        return [];
    }

    """Successor abilities of `name` along its visit statements."""
    def succ(name: str) -> list {
        out: list = [];
        for t in self.visits.get(name, []) {
            if t == "*" {
                for (own, ab) in self.by_owner.items() {
                    if ab not in out {
                        out.append(ab);
                    }
                }
            } elif t in self.by_owner and self.by_owner[t] not in out {
                out.append(self.by_owner[t]);
            }
        }
        return out;
    }

    def preds(name: str) -> list {
        return [a for a in self.abilities.keys() if name in self.succ(a)];
    }

    # -- walker field writes ------------------------------------------------

    def collect_defs -> None {
        for f in self.field_types.keys() {
            self.defs[f] = [];
        }
        for (name, ab) in self.abilities.items() {
            for a in ab.get_all_sub_nodes(uni.Assignment) {
                for tgt in a.target {
                    f = walker_field(tgt);
                    if not f {
                        continue;
                    }
                    growable = self.field_types.get(f, "").startswith(GROWABLE_TYPES);
                    in_exc = a.find_parent_of_type(uni.Except) is not None or is_empty_guard(a, f);
                    if a.aug_op is not None {
                        kind = "extend" if growable else "assign";
                        self.defs.setdefault(f, []).append(Def(name, a.loc.first_line, kind, None, in_exc));
                    } else {
                        self_ref = any([walker_field(x) == f for x in a.value.get_all_sub_nodes(uni.AtomTrailer)]) if a.value is not None else False;
                        kind = "extend" if (growable and self_ref) else ("reset" if (growable and a.value is not None and is_empty_literal(a.value)) else "assign");
                        self.defs.setdefault(f, []).append(Def(name, a.loc.first_line, kind, a.value, in_exc));
                    }
                }
            }
            for fc in ab.get_all_sub_nodes(uni.FuncCall) {
                t = fc.target;
                if isinstance(t, uni.AtomTrailer) and t.is_attr and isinstance(t.right, uni.Name) {
                    f = walker_field(t.target);
                    if not f {
                        continue;
                    }
                    m = t.right.value;
                    in_exc = fc.find_parent_of_type(uni.Except) is not None;
                    if m in EXTEND_METHODS {
                        self.defs.setdefault(f, []).append(Def(name, fc.loc.first_line, "extend", None, in_exc));
                    } elif m in RESET_METHODS {
                        self.defs.setdefault(f, []).append(Def(name, fc.loc.first_line, "assign", None, in_exc));
                    }
                }
            }
        }
    }

    # -- call sites ---------------------------------------------------------

    def collect_sites -> None {
        found: list = [];
        for (name, ab) in self.abilities.items() {
            for fc in ab.get_all_sub_nodes(uni.FuncCall) {
                cd = fc.callee_decl;
                if cd is None or not cd.is_genai_ability {
                    continue;
                }
                params = [p.name.value for p in cd.signature.get_parameters()];
                args: dict = {};
                positional = 0;
                for p in (fc.params or []) {
                    if isinstance(p, uni.KWPair) and p.key is not None {
                        args[text(p.key)] = p.value;
                    } elif positional < len(params) {
                        args[params[positional]] = p;
                        positional += 1;
                    }
                }
                key = CallSiteID(qualname(cd), fc.loc.first_line, self.path);
                fields = {pn: self.field_of(e, ab, fc.loc.first_line, 0) for (pn, e) in args.items()};
                found.append((fc.loc.first_line, len(found), Site(key, name, owner_name(ab), fc, cd, args, fields)));
            }
        }
        self.sites = [s for (ln, i, s) in sorted(found, key=self._by_line)];
    }

    static def _by_line(item: tuple) -> tuple {
        return (item[0], item[1]);
    }

    """The walker field an argument expression carries, seen through a local
    bound earlier in the ability and through `x or default`."""
    def field_of(e: object, ab: uni.Ability, before: int, depth: int) -> str {
        e = unwrap(e);
        if e is None or depth > 3 {
            return "";
        }
        f = walker_field(e);
        if f {
            return f;
        }
        # `visitor.f[i]`: one element of the field, written "f[]"
        if isinstance(e, uni.AtomTrailer) and not e.is_attr and isinstance(unwrap(e.right), uni.IndexSlice) {
            base = walker_field(e.target);
            if base and not unwrap(e.right).is_range {
                return base + "[]";
            }
        }
        if isinstance(e, uni.BoolExpr) and e.op.value == "or" and e.values {
            return self.field_of(e.values[0], ab, before, depth + 1);
        }
        if isinstance(e, uni.Name) {
            last = None;
            for a in ab.get_all_sub_nodes(uni.Assignment) {
                if a.loc.first_line < before and a.aug_op is None
                    and any([isinstance(t, uni.Name) and t.value == e.value for t in a.target]) {
                    if last is None or a.loc.first_line > last.loc.first_line {
                        last = a;
                    }
                }
            }
            if last is not None {
                return self.field_of(last.value, ab, last.loc.first_line, depth + 1);
            }
        }
        return "";
    }

    def local_assigned(name: str, ability: str, before: int) -> bool {
        ab = self.abilities[ability];
        for a in ab.get_all_sub_nodes(uni.Assignment) {
            if a.loc.first_line < before and any([isinstance(t, uni.Name) and t.value == name for t in a.target]) {
                return True;
            }
        }
        return False;
    }

    def sites_in(ability: str) -> list {
        return [s for s in self.sites if s.ability == ability];
    }

    """Sites that carry walker field `f` (under any parameter name)."""
    def carries(s: Site, f: str) -> str {
        for (pn, fld) in s.fields.items() {
            if fld == f {
                return pn;
            }
        }
        return "";
    }

    # -- call-site graph ----------------------------------------------------

    """The first call sites reachable from `ability` after line `after`,
    passing through abilities without LLM calls. The bool says whether a path
    ends (an ability with neither a further site nor a visit)."""
    def first_sites(ability: str, after: int, visited: set) -> tuple {
        later = [s for s in self.sites_in(ability) if s.call.loc.first_line > after];
        if later {
            return ([later[0]], False);
        }
        out: list = [];
        ends = self.may_end.get(ability, False);
        nxt = self.succ(ability);
        if not nxt {
            return ([], True);
        }
        for n in nxt {
            if n in visited {
                continue;
            }
            (ss, e) = self.first_sites(n, 0, visited | {n});
            for s in ss {
                if s not in out {
                    out.append(s);
                }
            }
            ends = ends or e;
        }
        return (out, ends);
    }

    # -- binding classification -------------------------------------------

    """Reaching definitions of field `f` at line `line` of `ability`."""
    def reaching(f: str, ability: str, line: int) -> list {
        all_defs = self.defs.get(f, []);
        local = [d for d in all_defs if d.ability == ability and d.lineno < line];
        if local {
            return self._prefer_main(local);
        }
        out: list = [];
        seen: set = {ability};
        frontier = self.preds(ability);
        while frontier {
            nxt: list = [];
            for p in frontier {
                if p in seen {
                    continue;
                }
                seen.add(p);
                here = [d for d in all_defs if d.ability == p];
                if here {
                    out.extend(self._prefer_main(here));
                } else {
                    nxt.extend(self.preds(p));
                }
            }
            frontier = nxt;
        }
        return out;
    }

    """Definitions outside `except` handlers win over the fallbacks inside them."""
    static def _def_line(d: Def) -> int {
        return d.lineno;
    }

    static def _prefer_main(ds: list) -> list {
        main = [d for d in ds if not d.in_except];
        return main if main else ds;
    }

    """The first site after definition `d` (before `upto` when in the same
    ability) that carries field `f`, searching forward through the visit graph
    and stopping at abilities that redefine `f`."""
    def carrier_after(d: Def, f: str, upto: Site) -> Site | None {
        same = [s for s in self.sites_in(d.ability) if s.call.loc.first_line > d.lineno
                and not (s.ability == upto.ability and s.call.loc.first_line >= upto.call.loc.first_line)];
        for s in same {
            if self.carries(s, f) and s is not upto {
                return s;
            }
        }
        if any([s.ability == upto.ability and s is upto for s in same]) or d.ability == upto.ability {
            return None;
        }
        seen: set = {d.ability};
        frontier = self.succ(d.ability);
        while frontier {
            nxt: list = [];
            for a in frontier {
                if a in seen {
                    continue;
                }
                seen.add(a);
                for s in self.sites_in(a) {
                    if s is not upto and self.carries(s, f) {
                        return s;
                    }
                    if s is upto {
                        break;
                    }
                }
                redefines = any([x.ability == a for x in self.defs.get(f, [])]);
                if a != upto.ability and not redefines {
                    nxt.extend(self.succ(a));
                }
            }
            frontier = nxt;
        }
        return None;
    }

    """Nearest upstream site other than `s` carrying `f` (session constants and
    append-only fields: the bytes an earlier prompt already holds)."""
    def upstream_carrier(s: Site, f: str) -> Site | None {
        earlier = [x for x in self.sites_in(s.ability) if x.call.loc.first_line < s.call.loc.first_line and self.carries(x, f)];
        if earlier {
            return earlier[-1];
        }
        seen: set = {s.ability};
        frontier = self.preds(s.ability);
        while frontier {
            nxt: list = [];
            for p in frontier {
                if p in seen {
                    continue;
                }
                seen.add(p);
                for x in self.sites_in(p) {
                    if x is not s and self.carries(x, f) {
                        return x;
                    }
                }
                nxt.extend(self.preds(p));
            }
            frontier = nxt;
        }
        return None;
    }

    """The site whose reply an expression is, as (site, json path), or None:
    `self.plan(...)`, a local bound to such a call, or `local.attr`."""
    def response_source(e: object, ability: str, before: int) -> tuple | None {
        e = unwrap(e);
        # `[f(q) for q in proposed if ...]`, `proposed[:n]`: still the reply's list
        if isinstance(e, uni.ListCompr) and e.compr {
            return self.response_source(e.compr[0].collection, ability, before);
        }
        if isinstance(e, uni.AtomTrailer) and not e.is_attr and isinstance(unwrap(e.right), uni.IndexSlice)
            and unwrap(e.right).is_range {
            return self.response_source(e.target, ability, before);
        }
        if isinstance(e, uni.FuncCall) {
            for s in self.sites_in(ability) {
                if s.call is e {
                    return (s, "");
                }
            }
            return None;
        }
        if isinstance(e, uni.AtomTrailer) and e.is_attr and isinstance(e.target, uni.Name) and isinstance(e.right, uni.Name) {
            inner = self.response_source(e.target, ability, before);
            if inner is not None and inner[1] == "" {
                return (inner[0], e.right.value);
            }
            return None;
        }
        if isinstance(e, uni.Name) {
            ab = self.abilities[ability];
            last = None;
            for a in ab.get_all_sub_nodes(uni.Assignment) {
                if a.loc.first_line <= before and a.aug_op is None
                    and any([isinstance(t, uni.Name) and t.value == e.value for t in a.target]) {
                    if last is None or a.loc.first_line > last.loc.first_line {
                        last = a;
                    }
                }
            }
            if last is not None and last.value is not None {
                return self.response_source(last.value, ability, last.loc.first_line);
            }
        }
        return None;
    }

    """Classify one parameter of a site: (heterogeneity, source, literal, scope)."""
    def classify(s: Site, pname: str) -> tuple {
        e = s.args.get(pname);
        f = s.fields.get(pname, "");
        if not f {
            try {
                return (Heterogeneity.CONST, None, repr(literal_value(e)), "");
            } except Exception {}
            g = unwrap(e);
            if isinstance(g, uni.Name) and self.field_of(g, self.abilities[s.ability], s.call.loc.first_line, 0) == ""
                and not self.local_assigned(g.value, s.ability, s.call.loc.first_line) {
                # a module-level name: its value at analysis time (the launcher analyzes in the run's environment)
                v = getattr(self.module, g.value, None);
                if isinstance(v, (int, float, str, bool)) {
                    return (Heterogeneity.CONST, None, repr(v), "");
                }
            }
            src = self.response_source(e, s.ability, s.call.loc.first_line);
            if src is not None {
                return (Heterogeneity.RESP, (src[0].key, src[1]), None, "");
            }
            return (Heterogeneity.VOLATILE, None, None, "");
        }
        if f.endswith("[]") {
            # one element of a field: the reply's list element when the field is a
            # reply, else fresh; the field's scope either way
            base = f[:-2];
            ds = self.defs.get(base, []);
            scope = "|".join(sorted({d.ability for d in ds if d.kind == "assign"}));
            srcs: list = [];
            for d in self.reaching(base, s.ability, s.call.loc.first_line) {
                src = self.response_source(d.rhs, d.ability, d.lineno) if d.rhs is not None else None;
                srcs.append((src[0].key, src[1] + "[]") if src is not None else None);
            }
            if srcs and all([x is not None and x == srcs[0] for x in srcs]) {
                return (Heterogeneity.RESP, srcs[0], None, scope);
            }
            return (Heterogeneity.VOLATILE, None, None, scope);
        }
        ds = self.defs.get(f, []);
        if not ds {   # never written: a session constant
            up = self.upstream_carrier(s, f);
            return (Heterogeneity.COPY, (up.key, self.carries(up, f)) if up is not None else None, None, "");
        }
        if all([d.kind == "extend" for d in ds]) {
            up = self.upstream_carrier(s, f);
            return (Heterogeneity.EXTEND, (up.key, self.carries(up, f)) if up is not None else None, None, "");
        }
        scope = "|".join(sorted({d.ability for d in ds if d.kind != "extend"}));
        local = [d for d in ds if d.ability == s.ability and d.lineno < s.call.loc.first_line];
        if local {
            last = max(local, key=self._def_line);
            if last.kind == "reset" {   # emptied just before the call: the empty literal itself
                return (Heterogeneity.CONST, None, repr(literal_value(last.rhs)), scope);
            }
        }
        if all([d.kind in ("extend", "reset") for d in ds]) {
            # append-only between resets: a history that restarts each scope
            up = self.upstream_carrier(s, f);
            return (Heterogeneity.EXTEND, (up.key, self.carries(up, f)) if up is not None else None, None, scope);
        }
        results: list = [];
        for d in self.reaching(f, s.ability, s.call.loc.first_line) {
            if d.kind == "extend" {
                up = self.upstream_carrier(s, f);
                results.append((Heterogeneity.EXTEND, (up.key, self.carries(up, f)) if up is not None else None, None, scope));
                continue;
            }
            c = self.carrier_after(d, f, s);
            if c is not None {
                results.append((Heterogeneity.COPY, (c.key, self.carries(c, f)), None, scope));
                continue;
            }
            lit = None;
            try {
                lit = repr(literal_value(d.rhs));
            } except Exception {}
            if lit is not None {
                results.append((Heterogeneity.CONST, None, lit, scope));
                continue;
            }
            src = self.response_source(d.rhs, d.ability, d.lineno) if d.rhs is not None else None;
            if src is not None {
                results.append((Heterogeneity.RESP, (src[0].key, src[1]), None, scope));
                continue;
            }
            results.append((Heterogeneity.VOLATILE, None, None, scope));
        }
        if not results {
            return (Heterogeneity.VOLATILE, None, None, scope);
        }
        first = results[0];
        if all([r == first for r in results]) {
            return first;
        }
        if all([r[0] == Heterogeneity.COPY for r in results]) {
            return first;
        }
        return (Heterogeneity.VOLATILE, None, None, scope);
    }

    # -- rendering ----------------------------------------------------------

    """The constant bytes of a site's prompt, from byllm's own factory."""
    def render(s: Site) -> PromptTemplate {
        cd = s.callee;
        name = cd.name_ref.sym_name;
        if s.owner {
            cls = getattr(self.module, s.owner);
            try {
                inst = instantiate(cls);
            } except Exception as e {
                raise RuntimeError(f"cannot instantiate {s.owner} to render {s.key}: {e}");
            }
            caller = getattr(inst, name);
        } else {
            inst = None;
            caller = getattr(self.module, name);
        }
        try {
            hints = get_type_hints(caller);
        } except Exception {
            hints = getattr(caller, "__annotations__", {}) or {};
        }
        params = [p.name.value for p in cd.signature.get_parameters()];
        args = {p: placeholder(hints.get(p)) for p in params};
        call_params: dict = {};
        model = "";
        extra_body: dict = {};
        bc = by_call(cd);
        if bc is not None {
            for p in (bc.params or []) {
                if isinstance(p, uni.KWPair) and p.key is not None {
                    try {
                        call_params[text(p.key)] = literal_value(p.value);
                    } except Exception {
                        try {
                            call_params[text(p.key)] = pyast.literal_eval(p.value.unparse());
                        } except Exception {}
                    }
                }
            }
            llm_obj = getattr(self.module, text(bc.target), None) if isinstance(bc.target, uni.Name) else None;
            model = getattr(llm_obj, "model_name", "") or "";
            extra_body = dict(getattr(llm_obj, "extra_body", {}) or {});
        }
        if isinstance(call_params.get("extra_body"), dict) {
            extra_body.update(call_params["extra_body"]);
        }
        mtir = MTIR(caller, args, dict(call_params), fetch_mtir(caller));
        rt = mtir.runtime;
        msgs = rt.messages;
        system_prompt = msgs[0].content if msgs and msgs[0].role.value == "system" else "";
        if not isinstance(system_prompt, str) {
            system_prompt = "".join([getattr(p, "text", "") for p in system_prompt]);
        }
        user = msgs[-1].content;
        as_part = isinstance(user, list);
        user_text = user[0].text if as_part else user;
        lines = user_text.split("\n");
        first_binding = f"{params[0]} = {args[params[0]]!r}" if params else None;
        head_end = len(lines);
        for (i, ln) in enumerate(lines) {
            if first_binding is not None and ln == first_binding {
                head_end = i;
                break;
            }
        }
        header = "\n".join(lines[:head_end]);
        t = PromptTemplate(
            key=s.key, model=model, call_params=call_params, system_prompt=system_prompt,
            tool_schema=(rt.get_tool_list() or None), response_format=rt.get_output_schema(),
            header=header, hint=schema_to_hint_text(rt.get_output_schema()) or "", hint_as_part=as_part,
            no_header_last=bool(extra_body.get("no_header_last", False))
        );
        for p in params {
            (het, src, lit, scope) = self.classify(s, p);
            t.bindings.append(Binding(name=p, heterogeneity=het, kind=BindingKind.PARAM,
                                      source=src, literal=lit, field=s.fields.get(p, ""), scope=scope,
                                      label=f"{p} = "));
        }
        if inst is not None {
            self_repr = repr(inst);
            for (i, ln) in enumerate(lines) {
                if ln.startswith("self = " + self_repr) {
                    tail = ln[len("self = " + self_repr):] + ("\n" + "\n".join(lines[i + 1:]) if i + 1 < len(lines) else "");
                    has_fields = bool([fl for fl in dataclasses.fields(type(inst))]) if dataclasses.is_dataclass(inst) else False;
                    if has_fields {
                        t.bindings.append(Binding(name="self", heterogeneity=Heterogeneity.COPY, kind=BindingKind.SELF,
                                                  label="self = ", tail=tail));
                    } else {
                        t.bindings.append(Binding(name="self", heterogeneity=Heterogeneity.CONST, kind=BindingKind.SELF,
                                                  literal=self_repr, label="self = ", tail=tail));
                    }
                    break;
                }
            }
        }
        return t;
    }

    # -- driver -------------------------------------------------------------

    def run -> Program {
        self.pick_walker();
        self.collect_abilities();
        self.collect_defs();
        self.collect_sites();
        if not self.sites {
            raise RuntimeError(f"no `by llm` call sites in walker {self.walker}'s abilities");
        }
        entry_ab = self.by_owner.get(self.entry_owner, "");
        (entry_sites, _) = self.first_sites(entry_ab, 0, {entry_ab}) if entry_ab else ([], False);
        entry = entry_sites[0].key if entry_sites else self.sites[0].key;
        program = Program(entry=entry);
        for s in self.sites {
            program.add_site(self.render(s));
        }
        exits: set = set();
        for s in self.sites {
            (nxt, ends) = self.first_sites(s.ability, s.call.loc.first_line, set());
            for n in nxt {
                program.add_edge(s.key, n.key);
            }
            if ends {
                exits.add(s.key);
            }
        }
        program.exits = frozenset(exits);
        return program;
    }
}


def analyze(path: str) -> Program {
    path = os.path.abspath(path);
    (ast, module) = compile_guarded(path);
    return Analysis(path=path, ast=ast, module=module).run();
}

def describe(p: Program) -> str {
    out = [f"entry: {p.entry!r}   exits: {sorted([repr(k) for k in p.exits])}"];
    for (k, t) in p.sites.items() {
        out.append(f"{k!r}  model={t.model}  header={len(t.header)}B  hint={len(t.hint)}B  system={len(t.system_prompt)}B");
        for b in t.bindings {
            src = f" <- {b.source[0]!r}.{b.source[1]}" if b.source else "";
            lit = f" = {b.literal}" if b.literal is not None else "";
            sc = f"  [scope {b.scope}]" if b.scope else "";
            out.append(f"    {b.kind.name:5} {b.name:16} {b.heterogeneity.name:8}{src}{lit}{sc}");
        }
        out.append("    -> " + ", ".join([repr(e.dst) for e in p.successors(k)]));
    }
    return "\n".join(out);
}

with entry:__main__ {
    argv = sys.argv[1:];
    if not argv {
        print("usage: jac run analyzer.jac <agent.jac> [-o program.json]", file=sys.stderr);
        sys.exit(2);
    }
    out_path = None;
    if "-o" in argv {
        i = argv.index("-o");
        out_path = argv[i + 1];
        argv = argv[:i] + argv[i + 2:];
    }
    program = analyze(argv[0]);
    print(describe(program), file=sys.stderr);
    payload = json.dumps(program_to_dict(program), indent=2);
    if out_path {
        with open(out_path, "w") as f {
            f.write(payload);
        }
        print(f"wrote {out_path}", file=sys.stderr);
    } else {
        print(payload);
    }
}
