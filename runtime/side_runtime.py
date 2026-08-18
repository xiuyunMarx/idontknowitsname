from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `static` importable

from static.async_byllm import AsyncByLLM
from static.static_parser import ByLLMDecl, ability_key, build_callsite_graph, build_decl, build_uniir, collect_type_defs, find_byllm_abilities


class SideRuntime:
    """Compile-time companion of a Jac program: one AsyncByLLM per byllm call site,
    plus the call-site topology so the runtime knows which byllm call can come next
    (the speculative-prefill target) after each one completes."""

    def __init__(self, repo_path: str, type_check: bool = True):
        self.program, self.mod = build_uniir(repo_path, type_check=type_check)
        self.type_defs = collect_type_defs(self.program)  # Jac obj/enum defs, for return-type translation
        self.func_decls: Dict[str, ByLLMDecl] = {}
        self.byllm_callsites: Dict[str, AsyncByLLM] = {}
        self.callsites_topo: Dict[str, List[str]] = {}  # key -> byllm calls that can run next
        self._parse_byLLM_decl()
        self._parse_topology()

    def _parse_byLLM_decl(self) -> None:
        """Parse the byllm function declarations and build one AsyncByLLM per call site."""
        for ab in find_byllm_abilities(self.program):
            decl = build_decl(self.program, ab, self.type_defs)
            key = ability_key(ab)
            self.func_decls[key] = decl
            self.byllm_callsites[key] = AsyncByLLM(decl)

    def _parse_topology(self) -> None:
        """Static may-happen-next graph over the byllm call sites."""
        self.callsites_topo = build_callsite_graph(self.program, self.func_decls)

    def next_callsites(self, key: str) -> List[AsyncByLLM]:
        """The AsyncByLLM objects worth speculatively prefilling after `key` completes."""
        return [self.byllm_callsites[k] for k in self.callsites_topo.get(key, []) if k in self.byllm_callsites]

    def bind_engine(self, engine) -> None:
        for fn in self.byllm_callsites.values():
            fn.bind_engine(engine)


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump the byllm call-site topology of a Jac program")
    ap.add_argument("file", help="path to the program's entry .jac file")
    ap.add_argument("--no-type-check", action="store_true")
    args = ap.parse_args()
    rt = SideRuntime(args.file, type_check=not args.no_type_check)
    print(f"[side-runtime] {len(rt.byllm_callsites)} call site(s)")
    for key, nxt in rt.callsites_topo.items():
        print(f"  {key} -> {nxt if nxt else '(end)'}")


if __name__ == "__main__":
    main()
