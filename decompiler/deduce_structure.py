"""Deduce a callsite's PromptTemplate from the call messages it was observed to send.

No wire format is hard-wired. Three things every lowering shares are learned from the
traffic of one callsite (a key) and, for labels, across callsites:

  skeleton  lines that recur verbatim in every message of the key (the longest common
            line subsequence). Everything between skeleton lines is a variable region.
  labels    inside a region, a constant string with one identifier hole that marks each
            value: byllm's `name = `, DSPy's `[[ ## name ## ]]` line. A label family
            (prefix, suffix) is accepted when its identifiers are the same set in every
            message, and once two identifiers have been seen for it anywhere, it also
            applies to a key whose region shows a single one.
  tail      per-request bytes after the message that vanish when the same message comes
            back as history (byllm's schema hint, DSPy's output reminder). Learned from
            a request pair (detect_tail); the first line of a learned tail is a marker
            that identifies the same tail on every other callsite.

Not learned: a template needs two distinct messages of a callsite (min_obs), and
whether a permutation changes the model's output is an empirical question the
caller still has to settle.

A template is accepted only when it round-trips every stored message of the key
(render(split(text)) == text). Usage:

    d = StructureDeducer()
    for body in turn0_requests:              # continuation turns are the caller's business
        tpl = d.observe(body)                # None until min_obs messages of the key
    d.observe_pair(prev_body, body)          # learns the hopping tail when there is one
    tpl, values = d.decompose(body)          # split with the current template

Regions never empty across the key's messages are "joined" (value = its lines joined by
newline, so a one-line value carries no newline). Regions empty in some messages are
"blocks" (value = each line plus its newline), which is what keeps an empty DSPy
trajectory and a grown one under the same static bytes.
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .structure_template import SLOT, STATIC, PromptTemplate, Segment

TOOL_RESPONSE_TAG = "<tool_response>"
_LABEL_LINE = re.compile(r"^(?P<pre>[^A-Za-z_]*)(?P<id>[A-Za-z_]\w*)(?P<rest>.*)$")
_OPENERS = "'\"[{("      # a label suffix never ends with these: they open the value's own literal
_CLOSERS = "'\"]})"

Family = Tuple[str, str, bool]   # (prefix, suffix, whole_line): label = prefix + id + suffix [+ "\n"]


# ----------------------------------------------------------------------------- request bodies
def _content(msg: Dict[str, Any]) -> str:
    c = msg.get("content")
    if isinstance(c, list):
        return "\n".join(str(p.get("text", "")) for p in c if isinstance(p, dict) and p.get("type") == "text")
    return c or ""


def system_text(body: Dict[str, Any]) -> str:
    msgs = body.get("messages") or []
    return _content(msgs[0]) if msgs and msgs[0].get("role") == "system" else ""


def call_message(body: Dict[str, Any]) -> str:
    """The message that carries the call: the last user message that is not a tool
    response (byllm: the only one; DSPy: the one after demos and history)."""
    for m in reversed(body.get("messages") or []):
        if m.get("role") == "user" and not _content(m).startswith(TOOL_RESPONSE_TAG):
            return _content(m)
    return ""


def key_of(body: Dict[str, Any]) -> str:
    """Coarse identity: model, system prompt and the first line of the call message.
    byllm's identity is in that first line (its header), DSPy's in the system prompt."""
    system = system_text(body)
    first = call_message(body).split("\n", 1)[0]
    digest = hashlib.sha1((system + "\x00" + first).encode("utf-8", "replace")).hexdigest()[:10]
    return f"{body.get('model', '')}|{digest}|{first[:48]}"


def detect_tail(prev_body: Dict[str, Any], body: Dict[str, Any]) -> Optional[str]:
    """The suffix the previous call message lost when it came back in `body` as a
    non-final user message (a typed retry, the next History turn)."""
    prev = call_message(prev_body)
    if not prev:
        return None
    msgs = body.get("messages") or []
    for m in msgs[:-1]:
        if m.get("role") != "user":
            continue
        cur = _content(m)
        if cur and cur != prev and prev.startswith(cur):
            i = len(cur)
            while i > 0 and prev[i - 1] == "\n":
                i -= 1
            return prev[i:]
    return None


# ----------------------------------------------------------------------------- line algebra
def _lcs(a: List[str], b: List[str]) -> List[str]:
    """One longest common subsequence of two line lists."""
    n, m = len(a), len(b)
    L = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai, Li, Ln = a[i], L[i], L[i + 1]
        for j in range(m - 1, -1, -1):
            Li[j] = Ln[j + 1] + 1 if ai == b[j] else (Ln[j] if Ln[j] >= Li[j + 1] else Li[j + 1])
    out: List[str] = []
    i = j = 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif L[i + 1][j] >= L[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _align(skel: List[str], lines: List[str]) -> Optional[List[int]]:
    """Positions of the skeleton lines in `lines`, each as late as possible before the
    next one: a blank line that is a joiner sits right before the label it precedes,
    and a blank inside a value stays in the value."""
    pos: List[int] = []
    p = 0
    for s in skel:
        try:
            j = lines.index(s, p)
        except ValueError:
            return None
        pos.append(j)
        p = j + 1
    hi = len(lines)
    for k in range(len(skel) - 1, -1, -1):
        for j in range(hi - 1, pos[k] - 1, -1):
            if lines[j] == skel[k]:
                pos[k] = j
                break
        hi = pos[k]
    return pos


def _weak(line: str) -> bool:
    """A line with no letter or digit (blank, `}`, `]`) is skeleton only when it is
    sandwiched between skeleton lines (or a message edge) in every message: a joiner
    blank between two labels qualifies, the `}` closing a value does not."""
    return not any(ch.isalnum() for ch in line)


def _skeleton(docs: List[List[str]], exclude: Optional[Set[str]] = None) -> Tuple[List[str], List[List[int]]]:
    """Common line skeleton of the documents and its position in each of them.
    Lines in `exclude` (demoted constant bindings) never count as skeleton."""
    skel = list(docs[0])
    for d in docs[1:]:
        skel = _lcs(skel, d)
    if exclude:
        skel = [l for l in skel if l not in exclude]      # blank lines are never excluded by name:
    while True:
        poss = [_align(skel, d) for d in docs]
        if any(p is None for p in poss):
            return [], [[] for _ in docs]                        # cannot happen for a subsequence; be safe
        keep: List[bool] = []
        for k, line in enumerate(skel):
            if not _weak(line):
                keep.append(True)
                continue
            sandwiched = True
            for p, d in zip(poss, docs):
                assert p is not None
                before = p[k] == 0 or (k > 0 and p[k - 1] == p[k] - 1)
                after = p[k] == len(d) - 1 or (k + 1 < len(skel) and p[k + 1] == p[k] + 1)
                if not (before and after):
                    sandwiched = False
                    break
            keep.append(sandwiched)
        if all(keep):
            return skel, poss  # type: ignore[return-value]
        skel = [l for l, k in zip(skel, keep) if k]


def _regions(skel: List[str], pos: List[int], lines: List[str]) -> List[List[str]]:
    """Lines between consecutive skeleton lines: len(skel)+1 regions, the first before
    skel[0] and the last after skel[-1]."""
    bounds = [-1] + pos + [len(lines)]
    return [lines[bounds[k] + 1:bounds[k + 1]] for k in range(len(skel) + 1)]


def _common_suffix(xs: Sequence[str]) -> str:
    s = xs[0]
    for x in xs[1:]:
        i = 0
        while i < min(len(s), len(x)) and s[-1 - i] == x[-1 - i]:
            i += 1
        s = s[len(s) - i:]
    return s


def _common_prefix(xs: Sequence[str]) -> str:
    s = xs[0]
    for x in xs[1:]:
        i = 0
        while i < min(len(s), len(x)) and s[i] == x[i]:
            i += 1
        s = s[:i]
    return s


def _slug(line: str, fallback: str) -> str:
    text = re.sub(r"\W+", "_", re.sub(r"\(.*?\)", "", line).strip().rstrip(":")).strip("_").lower()
    return text[:30] if text and len(line) <= 40 else fallback


# ----------------------------------------------------------------------------- region cutting
@dataclass
class _Cut:
    """A joined region cut into labelled slots: parallel lists over the slots."""
    family: Family
    labels: List[str]
    names: List[str]
    joiners: List[str]                     # static bytes after slot i (i < last): "\n" + common blank lines
    suffixes: List[str]                    # static bytes glued after slot i's value (" ---- sem")
    values: List[List[str]]                # per observation, the slot values
    lead: Optional[List[str]] = None       # per observation, unlabeled text before the first label
    trail: str = ""                        # blank lines closing the last slot in every observation


def _label_parts(line: str) -> Optional[Tuple[str, str, str]]:
    m = _LABEL_LINE.match(line)
    return (m["pre"], m["id"], m["rest"]) if m else None


def _pick_family(obs: List[List[str]], known: Dict[Family, Set[str]]) -> Optional[Tuple[Family, List[str]]]:
    """The label family that marks this region's values and its identifiers in
    first-seen order, or None. whole_line: the label is the entire line and the
    value starts on the next one."""
    per_pre: Dict[str, List[Dict[str, List[str]]]] = defaultdict(lambda: [defaultdict(list) for _ in obs])
    for i, lines in enumerate(obs):
        for line in lines:
            parts = _label_parts(line)
            if parts:
                per_pre[parts[0]][i][parts[1]].append(parts[2])
    best: Optional[Tuple[Family, List[str]]] = None
    for pre, per_obs in per_pre.items():
        ids = [k for k in per_obs[0] if all(len(o.get(k, ())) == 1 for o in per_obs)]
        if not ids:
            continue
        rests = [o[k][0] for o in per_obs for k in ids]
        suf = _common_prefix(rests).rstrip(_OPENERS)
        whole = all(r == suf for r in rests)
        fam: Family = (pre, suf, whole)
        if len(ids) < 2:
            # one identifier: only a family already established elsewhere is trusted,
            # and its suffix wins over whatever the values happen to share
            fam = next((f for f, fids in known.items() if len(fids) >= 2 and f[0] == pre and f[2] == whole
                        and all(r.startswith(f[1]) and (not whole or r == f[1]) for r in rests)), None)  # type: ignore[arg-type]
            if fam is None:
                continue
        elif not suf and not whole:
            continue
        if best is None or len(ids) > len(best[1]) or (len(ids) == len(best[1]) and len(pre) > len(best[0][0])):
            best = (fam, ids)
    return best


def _cut_region(obs: List[List[str]], known: Dict[Family, Set[str]], skeleton: Sequence[str] = ()) -> Optional[_Cut]:
    picked = _pick_family(obs, known)
    if picked is None:
        return None
    fam, ids = picked
    pre, suf, whole = fam
    heads = {i: pre + i + suf for i in ids}
    order: Optional[List[str]] = None
    values: List[List[str]] = []
    leads: List[List[str]] = []
    for lines in obs:
        seq: List[str] = []
        vals: Dict[str, List[str]] = {}
        lead: List[str] = []
        cur: Optional[str] = None
        for line in lines:
            parts = _label_parts(line)
            ident = parts[1] if parts else None
            if ident is not None and ident in heads and line.startswith(heads[ident]) and ident not in vals \
                    and (not whole or line == heads[ident]):
                cur = ident
                seq.append(cur)
                vals[cur] = [] if whole else [line[len(heads[cur]):]]
            elif parts and parts[0] == pre and cur is not None and ident not in heads \
                    and len(_common_prefix([parts[2], suf])) >= min(3, len(suf)) and (suf or whole):
                return None                         # the same shape with a varying identifier: a list, not labels
            elif cur is None:
                lead.append(line)                   # unlabeled text before the first label
            else:
                vals[cur].append(line)
        if order is None:
            order = seq
        elif seq != order:
            return None                             # the lowering itself varies the order: leave it alone
        values.append(["\n".join(vals[n]) for n in order])
        leads.append(lead)
    assert order is not None
    if any(leads) and not all(leads):
        return None                                 # a sometimes-present lead has no static home
    n = len(order)
    joiners = ["\n"] * (n - 1)
    suffixes = [""] * n
    trail = ""
    for i in range(n):
        col = [v[i] for v in values]
        # trailing blank lines present in every message belong to the static after the slot
        k = min(len(v) - len(v.rstrip("\n")) for v in col)
        if k:
            if i < n - 1:
                joiners[i] = "\n" * (k + 1)
            else:
                trail = "\n" * k
            for v in values:
                v[i] = v[i][:-k]
            col = [v[i] for v in values]
        # a common suffix is static decoration (byllm's " ---- sem") only when it starts at
        # whitespace, never covers a whole value, and opens with a token the skeleton
        # itself uses (" ---- " in the schema rows); values that merely end alike
        # (docstrings closing with the same quotes) stay whole
        s = _common_suffix(col).lstrip(_CLOSERS)
        if s and not s[0].isspace():
            cut = s.find(" ")
            s = s[cut:] if cut >= 0 else ""
        token = s.split()[0] if s.strip() else ""
        if not token or "\n" in s or not any(f" {token} " in line for line in skeleton):
            s = ""
        if s and all(len(v) > len(s) for v in col):
            suffixes[i] = s
            for v in values:
                v[i] = v[i][:-len(s)]
    labels = [heads[i] + ("\n" if whole else "") for i in order]
    return _Cut(fam, labels, list(order), joiners, suffixes, values,
                ["\n".join(l) for l in leads] if any(leads) else None, trail)


# ----------------------------------------------------------------------------- template assembly
def _unique(name: str, used: Set[str]) -> str:
    base, k = name, 2
    while name in used:
        name = f"{base}_{k}"
        k += 1
    used.add(name)
    return name


def induce(key: str, texts: List[str], tail: str = "", static_pre: str = "",
           known: Optional[Dict[Family, Set[str]]] = None,
           cut_labels: bool = True) -> Optional[PromptTemplate]:
    """A template that round-trips every text (+ tail), or None. `known` collects the
    label families seen so far (updated in place). A binding whose value never
    changed looks like skeleton on the first pass; once its label family is known
    the line is demoted and the pass repeats."""
    known = known if known is not None else {}
    docs = [t.split("\n") for t in texts]
    exclude: Set[str] = set()
    segs: List[Segment] = []
    for _ in range(4):
        segs, demote = _assemble(docs, known, cut_labels, exclude)
        if not demote or not cut_labels:
            break
        exclude |= demote
    tpl = PromptTemplate.build(key, segs, tail=tail, static_pre=static_pre)
    for text in texts:
        full = text + tail
        vals = tpl.split(full)
        if vals is None or tpl.render(vals) != full:
            if cut_labels:
                return induce(key, texts, tail, static_pre, known, cut_labels=False)
            return None
    return tpl


def _assemble(docs: List[List[str]], known: Dict[Family, Set[str]], cut_labels: bool,
              exclude: Set[str]) -> Tuple[List[Segment], Set[str]]:
    """Segments for the documents, plus skeleton lines found to be constant
    bindings of a label family (to exclude on the next pass)."""
    skel, poss = _skeleton(docs, exclude)
    regs = [_regions(skel, p, d) for p, d in zip(poss, docs)]
    n_reg = len(skel) + 1
    # region kinds: None (empty everywhere), "joined" (never empty), "block" (sometimes empty)
    kinds: List[Optional[str]] = []
    for k in range(n_reg):
        col = [r[k] for r in regs]
        kinds.append(None if not any(col) else ("joined" if all(col) else "block"))
    # a region whose preceding skeleton line is `pre + id + suf` (DSPy's [[ ## id ## ]])
    # is that label's value: two such lines with distinct ids, or a known family
    pre_label: Dict[int, Tuple[str, str]] = {}
    if cut_labels:
        cands: Dict[Tuple[str, str], List[Tuple[int, str, str]]] = defaultdict(list)
        for k in range(1, n_reg):
            if kinds[k] is None:
                continue
            parts = _label_parts(skel[k - 1])
            if parts and parts[2]:
                cands[(parts[0], parts[2])].append((k, skel[k - 1], parts[1]))
        for (pre, rest), rows in cands.items():
            ids = {ident for _, _, ident in rows}
            fam: Family = (pre, rest, True)
            if len(ids) != len(rows) or len(known.get(fam, set()) | ids) < 2:
                continue
            known.setdefault(fam, set()).update(ids)
            for k, line, ident in rows:
                pre_label[k] = (line, ident)
    tokens: List[Tuple[str, Any]] = []       # message order: ("line", k) | ("region", k)
    for k in range(n_reg):
        if kinds[k] is not None:
            tokens.append(("region", k))
        if k < len(skel):
            tokens.append(("line", k))
    segs: List[Segment] = []
    used: Set[str] = set()
    cut_by: Dict[int, Family] = {}           # region k -> family it was cut with
    for t, (kind, k) in enumerate(tokens):
        last = t == len(tokens) - 1
        nxt = tokens[t + 1] if not last else None
        # a block region that ends the message carries "\n" + line per line, so the
        # static (or label) before it must not end with a line break
        end_block = nxt is not None and nxt[0] == "region" and kinds[nxt[1]] == "block" and t + 1 == len(tokens) - 1
        if kind == "line":
            if nxt is not None and nxt[0] == "region" and nxt[1] in pre_label:
                continue                                       # the line is the next slot's label
            segs.append(Segment(STATIC, skel[k] + ("" if last or end_block else "\n")))
            continue
        col = [r[k] for r in regs]
        cut = None
        if k in pre_label:
            line, ident = pre_label[k]
            prev_tok = tokens[t - 1]
            label_break = "" if (kinds[k] == "block" and last) else "\n"
            segs.append(Segment(SLOT, line + label_break, _unique(ident, used), keyed=True))
        else:
            cut = _cut_region(col, known, skel) if kinds[k] == "joined" and cut_labels else None
            if cut is not None:
                cut_by[k] = cut.family
                known.setdefault(cut.family, set()).update(cut.names)
                if cut.lead is not None:            # e.g. a schema row that varies with the values' shape
                    prev_line = skel[k - 1] if k > 0 else ""
                    segs.append(Segment(SLOT, "", _unique(_slug(prev_line, f"slot{k}"), used)))
                    segs.append(Segment(STATIC, "\n"))
                for i, (label, name) in enumerate(zip(cut.labels, cut.names)):
                    segs.append(Segment(SLOT, label, _unique(name, used), keyed=True))
                    if cut.suffixes[i]:
                        segs.append(Segment(STATIC, cut.suffixes[i]))
                    if i < len(cut.names) - 1:
                        segs.append(Segment(STATIC, cut.joiners[i]))
                if cut.trail:
                    segs.append(Segment(STATIC, cut.trail))
            else:
                prev_line = skel[k - 1] if k > 0 else ""
                segs.append(Segment(SLOT, "", _unique(_slug(prev_line, f"slot{k}"), used)))
        if kinds[k] == "joined" and not last:
            # blank lines closing the region in every message are static, not value
            blanks = 0 if cut is not None else min(
                len("\n".join(r)) - len("\n".join(r).rstrip("\n")) for r in col)
            segs.append(Segment(STATIC, "\n" * (blanks + 1)))
    # constant bindings: skeleton lines shaped like a label of a family that cut an
    # adjacent region (byllm `n = 2`), or a known whole-line label followed by its
    # constant value lines (DSPy) -- demote them so the next pass cuts them as slots
    demote: Set[str] = set()
    if cut_labels:
        for k, line in enumerate(skel):
            parts = _label_parts(line)
            if not parts:
                continue
            pre, ident, rest = parts
            fam_adjacent = {f for kk, f in cut_by.items() if kk in (k, k + 1)}
            for (fpre, fsuf, whole), ids in known.items():
                if len(ids) < 2 or fpre != pre:
                    continue
                if not whole and rest.startswith(fsuf) and (fpre, fsuf, whole) in fam_adjacent:
                    demote.add(line)
                elif whole and rest == fsuf and k + 1 < len(skel) and kinds[k + 1] is None:
                    def _is_label(line: str) -> bool:
                        lp = _label_parts(line)
                        return bool(lp) and lp[0] == fpre and lp[2] == fsuf
                    j = k + 1
                    if _is_label(skel[j]):
                        # a label directly followed by another label of the family (no blank
                        # line between) is a container: DSPy's trajectory holds its own
                        # [[ ## thought_k ## ]] blocks. Its value is every block that follows,
                        # up to the first skeleton line that is not such a block.
                        while j < len(skel) and _is_label(skel[j]):
                            j += 1
                            while j < len(skel) and skel[j] and not _is_label(skel[j]):
                                j += 1
                            while j < len(skel) and skel[j] == "":
                                j += 1
                        demote.update(l for l in skel[k + 1:j] if not _weak(l))
                    else:
                        # a known label line whose value never changed: the plain lines after
                        # it (up to the next blank or label line) are that constant value
                        while j < len(skel) and skel[j] and not _is_label(skel[j]):
                            j += 1
                        if j > k + 1:
                            demote.add(line)
                            demote.update(skel[k + 1:j])
    return segs, demote


# ----------------------------------------------------------------------------- online deducer
class StructureDeducer:
    def __init__(self, min_obs: int = 2, max_obs: int = 24) -> None:
        self.min_obs = min_obs
        self.max_obs = max_obs
        self.obs: Dict[str, List[str]] = defaultdict(list)
        self.tails: Dict[str, str] = {}
        self.systems: Dict[str, str] = {}
        self.templates: Dict[str, PromptTemplate] = {}
        self.families: Dict[Family, Set[str]] = {}
        self.tail_markers: Set[str] = set()      # first line of every learned tail: byllm's "\n\nSchema requirements:"

    # -- observation
    def observe(self, body: Dict[str, Any]) -> Optional[PromptTemplate]:
        return self.observe_text(key_of(body), call_message(body), system_text(body))

    def observe_pair(self, prev_body: Dict[str, Any], body: Dict[str, Any]) -> Optional[str]:
        """Learn the hopping tail of prev_body's key from its reappearance in `body`."""
        tail = detect_tail(prev_body, body)
        if tail:
            self.note_tail(key_of(prev_body), tail)
        return tail

    def note_tail(self, key: str, tail: str) -> None:
        if self.tails.get(key) == tail:
            return
        self.tails[key] = tail
        marker = tail.split("\n", 1)[0] if tail.strip("\n").find("\n") < 0 else tail[:len(tail) - len(tail.lstrip("\n"))] + tail.strip("\n").split("\n", 1)[0]
        if len(marker.strip()) >= 8:
            self.tail_markers.add(marker)
        self.obs[key] = [t[:-len(tail)] if t.endswith(tail) else t for t in self.obs[key]]
        self.templates.pop(key, None)
        self._refresh(key)

    def _infer_tail(self, key: str) -> None:
        """A site with no retry of its own still carries the framework's tail when its
        messages all end with a suffix that opens with a learned marker."""
        store = self.obs[key]
        if key in self.tails or not store:
            return
        for marker in self.tail_markers:
            i = store[0].rfind(marker)
            if i < 0:
                continue
            tail = store[0][i:]
            if all(t.endswith(tail) for t in store):
                self.tails[key] = tail
                self.obs[key] = [t[:-len(tail)] for t in store]
                return

    def observe_text(self, key: str, text: str, system: str = "") -> Optional[PromptTemplate]:
        tail = self.tails.get(key, "")
        if tail and text.endswith(tail):
            text = text[:-len(tail)]
        self.systems.setdefault(key, system)
        store = self.obs[key]
        if text not in store:
            store.append(text)
            if len(store) > self.max_obs:
                del store[1:2]                     # keep the first, drop the oldest after it
        tpl = self.templates.get(key)
        n = len(store)
        if tpl is not None and (n & (n - 1)) != 0:   # not a power of two: keep a template that still fits
            vals = tpl.split(text + tail)
            if vals is not None and tpl.render(vals) == text + tail:
                return tpl
        return self._refresh(key)

    def _refresh(self, key: str) -> Optional[PromptTemplate]:
        self._infer_tail(key)
        store = self.obs[key]
        if len(store) < self.min_obs:
            return None
        tpl = induce(key, store, self.tails.get(key, ""), self.systems.get(key, ""), self.families)
        if tpl is not None:
            self.templates[key] = tpl
        else:
            self.templates.pop(key, None)
        return tpl

    # -- use
    def template(self, key: str) -> Optional[PromptTemplate]:
        return self.templates.get(key)

    def decompose(self, body: Dict[str, Any]) -> Optional[Tuple[PromptTemplate, Dict[str, str]]]:
        tpl = self.templates.get(key_of(body))
        if tpl is None:
            return None
        vals = tpl.split(call_message(body))
        return (tpl, vals) if vals is not None else None
