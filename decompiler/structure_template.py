"""The structure of one callsite's call message: static bytes and named slots.

A lowering (byllm's mtir, DSPy's ChatAdapter, ...) renders a call as a fixed skeleton
with the argument values dropped into slots. A PromptTemplate is that skeleton, and it
is the only thing the rest of the server needs to know about a wire format:

    values = tpl.split(text)                     # the message -> its slot values, byte exact
    tpl.render(values, order, statics_last)      # the inverse, under a chosen layout
    tpl.relayout(text, order, statics_last)      # split + render; the input unchanged when it does not fit
    tpl.fixed_head(order, statics_last)          # bytes before the first value under that layout
    tpl.movable_names()                          # the slots a layout may permute

The invariant everything rests on: render(split(text)) == text for every message the
template was built from. deduce_structure.py builds templates from observed traffic;
a hand-written parser may build them from its own rules. Neither is special here.

Reorder rules are derived from the template, not from the framework:
  * a slot may move only inside a *run*: consecutive keyed slots (label carries the
    slot's own name) separated by identical static bytes, the run's joiner;
  * the static text before the first run may trail the message (byllm's header-last)
    only when it is the message's leading segment and ends at a line break. A
    lowering whose skeleton lives in the system prompt (DSPy) has no such segment,
    so its user section is permutable and nothing else is.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

STATIC = "static"
SLOT = "slot"


@dataclass
class Segment:
    kind: str                # STATIC or SLOT
    text: str = ""           # static: literal bytes. slot: the label written before the value ("" when unlabeled)
    name: str = ""           # slot only
    keyed: bool = False      # slot only: the label embeds `name` ("name = ", "[[ ## name ## ]]\n" style)
    movable: bool = False    # slot only: member of a run (set by PromptTemplate.build)

    @property
    def is_slot(self) -> bool:
        return self.kind == SLOT


@dataclass
class Run:
    """A maximal sequence of keyed slots joined by identical static bytes."""
    start: int               # index of the first slot segment
    stop: int                # index past the last slot segment
    joiner: str              # the static bytes between consecutive slots

    def slot_indices(self) -> range:
        return range(self.start, self.stop, 2)


@dataclass
class PromptTemplate:
    key: str
    segments: List[Segment]
    tail: str = ""           # per-request bytes after the message (byllm's schema hint, DSPy's output reminder)
    static_pre: str = ""     # the system prompt: bytes before the call message, never re-laid-out
    runs: List[Run] = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def build(cls, key: str, segments: Sequence[Segment], tail: str = "", static_pre: str = "") -> "PromptTemplate":
        """Normalise (merge adjacent statics, drop empty ones) and find the runs."""
        segs: List[Segment] = []
        for s in segments:
            if s.kind == STATIC:
                if not s.text:
                    continue
                if segs and segs[-1].kind == STATIC:
                    segs[-1] = Segment(STATIC, segs[-1].text + s.text)
                    continue
            segs.append(Segment(s.kind, s.text, s.name, s.keyed, False))
        tpl = cls(key=key, segments=segs, tail=tail, static_pre=static_pre)
        tpl.runs = tpl._find_runs()
        for r in tpl.runs:
            for i in r.slot_indices():
                tpl.segments[i].movable = True
        return tpl

    def _eligible(self, i: int) -> bool:
        """A keyed slot whose value ends the line: a slot decorated with same-line
        static text (byllm's `self = X ---- sem`) is pinned."""
        segs = self.segments
        s = segs[i]
        if not (s.is_slot and s.keyed):
            return False
        return i + 1 == len(segs) or segs[i + 1].kind != STATIC or segs[i + 1].text.startswith("\n")

    def _find_runs(self) -> List[Run]:
        runs: List[Run] = []
        segs = self.segments
        n = len(segs)
        i = 0
        while i < n:
            if not self._eligible(i):
                i += 1
                continue
            j = i
            joiner: Optional[str] = None
            while (j + 2 < n and segs[j + 1].kind == STATIC and self._eligible(j + 2)
                   and (joiner is None or segs[j + 1].text == joiner)):
                joiner = segs[j + 1].text
                j += 2
            if j > i:
                runs.append(Run(i, j + 1, joiner or ""))
                i = j + 1
            else:
                i += 1
        return runs

    # ------------------------------------------------------------------ queries
    def slots(self) -> List[Segment]:
        return [s for s in self.segments if s.is_slot]

    def slot_names(self) -> List[str]:
        return [s.name for s in self.slots()]

    def movable_names(self) -> List[str]:
        return [s.name for s in self.segments if s.is_slot and s.movable]

    def prefix_end(self) -> int:
        """Index of the first run's first slot: the segments before it are the
        message's leading block (byllm's header and schema rows)."""
        return self.runs[0].start if self.runs else 0

    def statics_can_trail(self) -> bool:
        """The leading block may be moved behind the message: it starts with a
        static, ends with a static at a line break, holds no keyed slot (schema rows
        that vary with the values' shape are unkeyed slots and travel with it), and a
        run follows it."""
        n = self.prefix_end()
        segs = self.segments
        return (n >= 1 and segs[0].kind == STATIC and segs[n - 1].kind == STATIC and segs[n - 1].text.endswith("\n")
                and not any(s.is_slot and s.keyed for s in segs[:n]))

    def _run_at(self, i: int) -> Optional[Run]:
        for r in self.runs:
            if r.start == i:
                return r
        return None

    # ------------------------------------------------------------------ split
    def split(self, text: str) -> Optional[Dict[str, str]]:
        """Slot values of `text`; None when the text does not fit. Accepts any
        permutation inside a run and the trailing-static layout, so a re-laid-out
        message splits back to the same values."""
        if self.tail:
            if not text.endswith(self.tail):
                return None
            text = text[:-len(self.tail)]
        vals = self._match(text, 0, len(self.segments))
        if vals is None and self.statics_can_trail():
            n = self.prefix_end()
            first = self.segments[0].text
            if n == 1:                                    # a single static: its trailing form is a suffix
                trail = "\n" + first[:-1]
                if text.endswith(trail):
                    vals = self._match(text[:-len(trail)], 1, len(self.segments))
            else:
                j = text.rfind("\n" + first)
                if j >= 0:
                    head = self._match(text[j + 1:], 0, n, trim_last=True)
                    body = self._match(text[:j], n, len(self.segments))
                    if head is not None and body is not None:
                        vals = {**head, **body}
        return vals

    def _anchors(self, i: int, stop: Optional[int] = None) -> List[str]:
        """Byte strings any of which ends the value of the slot at segment `i`: the
        following static plus the label of the slot after it; when that slot opens a
        run, one alternative per run member (the run may arrive permuted). Empty =
        the value runs to the end."""
        segs = self.segments
        end = len(segs) if stop is None else stop
        if i + 1 >= end:
            return []
        nxt = segs[i + 1]
        a = nxt.text if nxt.kind == STATIC else ""
        j = i + 1 if nxt.is_slot else i + 2
        if j < end and segs[j].is_slot:
            run = self._run_at(j)
            if run is not None:
                return [a + segs[k].text for k in run.slot_indices()]
            return [a + segs[j].text]
        return [a]

    @staticmethod
    def _first(text: str, anchors: List[str], pos: int) -> int:
        """Earliest position of any anchor at or after `pos`; -1 when none occurs."""
        hits = [j for j in (text.find(a, pos) for a in anchors) if j >= 0]
        return min(hits) if hits else -1

    def _match(self, text: str, start: int, stop: int, trim_last: bool = False) -> Optional[Dict[str, str]]:
        """Match segments[start:stop] against the whole of `text`; with trim_last the
        final static is matched without its closing line break (a trailing block)."""
        segs = self.segments
        vals: Dict[str, str] = {}
        pos = 0
        i = start
        while i < stop:
            s = segs[i]
            if s.kind == STATIC:
                lit = s.text[:-1] if trim_last and i == stop - 1 else s.text
                if not text.startswith(lit, pos):
                    return None
                pos += len(lit)
                i += 1
                continue
            run = self._run_at(i)
            if run is not None:
                members = {segs[k].name: segs[k] for k in run.slot_indices()}
                pos = self._match_run(text, pos, members, run.joiner, self._anchors(run.stop - 1, stop), vals)
                if pos < 0:
                    return None
                i = run.stop
                continue
            if not text.startswith(s.text, pos):
                return None
            pos += len(s.text)
            anchors = self._anchors(i, stop)
            if anchors and trim_last and i + 1 == stop - 1 and segs[stop - 1].kind == STATIC:
                anchors = [a[:-1] for a in anchors]   # the trailing block's last static lost its line break
                if anchors == [""]:
                    anchors = []                      # it was only the line break: the value runs to the end
            if not anchors:
                vals[s.name] = text[pos:]
                pos = len(text)
            elif anchors == [""]:
                return None                       # two unlabeled slots touching: ambiguous
            else:
                j = self._first(text, anchors, pos)
                if j < 0:
                    return None
                vals[s.name] = text[pos:j]
                pos = j
            i += 1
        return vals if pos == len(text) else None

    @classmethod
    def _match_run(cls, text: str, pos: int, members: Dict[str, Segment], joiner: str,
                   after: List[str], vals: Dict[str, str]) -> int:
        """Match the run's slots in whatever order they appear. Returns the position
        past the run, -1 on failure."""
        left = dict(members)
        first = True
        while left:
            if not first:
                if not text.startswith(joiner, pos):
                    return -1
                pos += len(joiner)
            first = False
            cand = [s for s in left.values() if text.startswith(s.text, pos)]
            if not cand:
                return -1
            s = max(cand, key=lambda x: len(x.text))   # labels may share a prefix: longest wins
            pos += len(s.text)
            del left[s.name]
            if left:
                stops = [j for o in left.values() for j in [text.find(joiner + o.text, pos)] if j >= 0]
                if not stops:
                    return -1
                j = min(stops)
            elif not after:
                j = len(text)
            elif after == [""]:
                return -1
            else:
                j = cls._first(text, after, pos)
                if j < 0:
                    return -1
            vals[s.name] = text[pos:j]
            pos = j
        return pos

    # ------------------------------------------------------------------ render
    @staticmethod
    def _ordered(names: List[str], order: Optional[Sequence[str]]) -> List[str]:
        """`names` stable-sorted by rank in `order`; unlisted names keep their relative
        order after the listed ones (relayout_user's convention)."""
        if not order:
            return list(names)
        rank = {n: i for i, n in enumerate(order)}
        return sorted(names, key=lambda n: rank.get(n, len(rank)))

    def render(self, values: Dict[str, str], order: Optional[Sequence[str]] = None,
               statics_last: bool = False) -> str:
        """The message for `values` under the layout; KeyError when a value is missing."""
        segs = self.segments
        trailing = ""
        i = 0
        if statics_last and self.statics_can_trail():
            n = self.prefix_end()
            head = "".join(s.text if s.kind == STATIC else s.text + values[s.name] for s in segs[:n])
            trailing = "\n" + head[:-1]
            i = n
        out: List[str] = []
        while i < len(segs):
            s = segs[i]
            run = self._run_at(i)
            if run is not None:
                by_name = {segs[k].name: segs[k] for k in run.slot_indices()}
                names = self._ordered(list(by_name), order)
                out.append(run.joiner.join(by_name[n].text + values[n] for n in names))
                i = run.stop
            elif s.kind == STATIC:
                out.append(s.text)
                i += 1
            else:
                out.append(s.text + values[s.name])
                i += 1
        return "".join(out) + trailing + self.tail

    def relayout(self, text: str, order: Optional[Sequence[str]] = None, statics_last: bool = False) -> str:
        vals = self.split(text)
        if vals is None:
            return text
        return self.render(vals, order, statics_last)

    def fixed_head(self, order: Optional[Sequence[str]] = None, statics_last: bool = False) -> str:
        """Bytes before the first value under the layout: leading statics plus the first
        slot's label. Only static_pre is fixed when the leading static trails."""
        segs = self.segments
        i = self.prefix_end() if statics_last and self.statics_can_trail() else 0
        head: List[str] = []
        while i < len(segs):
            s = segs[i]
            if s.kind == STATIC:
                head.append(s.text)
                i += 1
                continue
            run = self._run_at(i)
            if run is not None:
                by_name = {segs[k].name: segs[k] for k in run.slot_indices()}
                head.append(by_name[self._ordered(list(by_name), order)[0]].text)
            else:
                head.append(s.text)
            break
        return "".join(head)

    def describe(self) -> str:
        """One row per segment, for tests and logs."""
        rows = []
        for s in self.segments:
            if s.kind == STATIC:
                rows.append(f"  static {s.text!r}")
            else:
                flag = ("*" if s.movable else "") + ("k" if s.keyed else "")
                rows.append(f"  slot   {s.name}{flag:>3} label={s.text!r}")
        if self.tail:
            rows.append(f"  tail   {self.tail!r}")
        return "\n".join(rows)
