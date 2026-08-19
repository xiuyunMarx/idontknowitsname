from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `static` importable

from static.async_byllm import AsyncByLLM
from static.static_parser import ByLLMDecl, ability_key, build_callsite_graph, build_decl, build_provenance, build_uniir, build_visit_decl, collect_type_defs, find_byllm_abilities, find_genai_visits, invert_provenance, visit_key


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
        """Parse every LLM call site — byllm functions and `visit ... by llm()` routing — build one AsyncByLLM per site."""
        for ab in find_byllm_abilities(self.program):
            decl = build_decl(self.program, ab, self.type_defs)
            key = ability_key(ab)
            self.func_decls[key] = decl
            self.byllm_callsites[key] = AsyncByLLM(decl)
        for vs in find_genai_visits(self.program):
            decl = build_visit_decl(self.program, vs)
            key = visit_key(vs)
            self.func_decls[key] = decl
            self.byllm_callsites[key] = AsyncByLLM(decl)

    def _parse_topology(self) -> None:
        """Static may-happen-next graph over the byllm call sites, plus the binding
        provenance table (param -> value-source spec) driving speculative feeds."""
        self.callsites_topo = build_callsite_graph(self.program, self.func_decls)
        self.provenance = build_provenance(self.program, self.func_decls)
        self.ret_consumers = invert_provenance(self.provenance)  # producer -> [(consumer, param)]

    def next_callsites(self, key: str) -> List[AsyncByLLM]:
        """The AsyncByLLM objects worth speculatively prefilling after `key` completes."""
        return [self.byllm_callsites[k] for k in self.callsites_topo.get(key, []) if k in self.byllm_callsites]

    def prewarm_callsite(self, key: str) -> None:
        if key not in self.byllm_callsites:
            raise ValueError(f"Unknown byllm call site key: {key}")
        self.byllm_callsites[key].warm_invariant()
        
        
    def bind_engine(self, engine) -> None:
        for fn in self.byllm_callsites.values():
            fn.bind_engine(engine)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dump the byllm call-site topology of a Jac program")
    ap.add_argument("file", help="path to the program's entry .jac file")
    ap.add_argument("--no-type-check", action="store_true")
    args = ap.parse_args()
    rt = SideRuntime(args.file, type_check=not args.no_type_check)
    print(f"[side-runtime] {len(rt.byllm_callsites)} call site(s)")
    for key, nxt in rt.callsites_topo.items():
        print(f"  {key} -> {nxt if nxt else '(end)'}")
    print("[side-runtime] provenance:")
    for key, params in rt.provenance.items():
        for name, spec in params.items():
            print(f"  {key}.{name} <- {spec}")

