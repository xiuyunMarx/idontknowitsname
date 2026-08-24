from __future__ import annotations

import re
import uuid

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from runtime.engine import ModelEngine
from utils.jac_static_parser import ByLLMCallsite, ProgramTopology
from utils.utils import ROUTE_ZONE_LABEL

if TYPE_CHECKING:
    from runtime.server import _CallState


_CANDIDATE_LINE = re.compile(
    r"^(?P<handle>[A-Za-z_]\w*)\)\s+(?P<description>.+)$"
)

# (handle, description, statically resolved node archetype)
CandidateNode = Dict[str, str]
VisitParseResult = Tuple[
    str,
    List[CandidateNode],
    Dict[str, List[ByLLMCallsite]],
]


class RoutingSpeculate:
    def __init__(self, engine: ModelEngine) -> None:
        self.engine = engine

    @staticmethod
    def _candidate_archetype(handle: str, archetypes: List[str]) -> str:
        """Resolve a route handle to its statically known node archetype."""
        matches = [
            name
            for name in archetypes
            if handle == name or handle.startswith(name + "_")
        ]
        return max(matches, key=len) if matches else ""

    @staticmethod
    def _first_callsites(
        archetype: str,
        state: _CallState,
        program: ProgramTopology,
    ) -> List[ByLLMCallsite]:
        """First reachable byLLM callsite in every firing entry ability."""
        if not archetype:
            return []

        out: List[ByLLMCallsite] = []
        seen: set[str] = set()
        walker = state.site.decl.owner_arch

        for ability in program._entry_slots(archetype, walker):
            for key in program._first_call_in(ability):
                sites = program.sites_of(key)

                # Prefer the exact site in this entry ability. If the first call
                # lives in a plain helper, retain all sites for its resolved key.
                scoped = [site for site in sites if site.scope is ability]
                for site in scoped or sites:
                    if site.callsite_uuid not in seen:
                        seen.add(site.callsite_uuid)
                        out.append(site)

        return out

    async def sort_candidate_calls(
        self, state: _CallState, program: ProgramTopology, rank: bool = True
    ) -> List[ByLLMCallsite]:
        """Every candidate's first byLLM callsites, most likely route first.
        `rank=False` skips the probe and keeps the candidates' listed order."""
        parsed = self._parse_visit(state, program)
        if parsed is None:
            return []
        _, candidate_nodes, first_callsites_by_handle = parsed

        if rank and len(candidate_nodes) > 1:
            # enable_thinking=False 
            tokenizer = self.engine.engine.get_tokenizer()
            probe_prompt = tokenizer.apply_chat_template(
                state.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False #type: ignore
            )
            top = await self.engine.probe(
                probe_prompt, f"probe-{state.site.callsite_uuid}-{uuid.uuid4().hex}" #type: ignore
            )
            scores = {node["handle"]: float("-inf") for node in candidate_nodes}
            for token_id, info in top.items():
                text = (info.decoded_token or tokenizer.decode([token_id])).strip()
                for handle in scores:
                    if text and (handle.startswith(text) or text.startswith(handle)):
                        scores[handle] = max(scores[handle], info.logprob)
            candidate_nodes = sorted(candidate_nodes, key=lambda n: -scores[n["handle"]])

        out: List[ByLLMCallsite] = []
        seen: set[str] = set()
        for node in candidate_nodes:
            for site in first_callsites_by_handle[node["handle"]]:
                if site.callsite_uuid not in seen:
                    seen.add(site.callsite_uuid)
                    out.append(site)
        return out


    def _parse_visit(
        self,
        state: _CallState,
        program: ProgramTopology,
    ) -> Optional[VisitParseResult]:
        """Return full prompt, live candidate nodes, and their first callsites."""
        if state.request.type != "generate" or not state.site.is_visit:
            return None

        # This is byte-for-byte the operation GuardServer._complete performs
        # before passing the text to ModelEngine.generate(). Do not reconstruct a
        # visit prompt statically: only the client knows the live graph state.
        full_prompt = self.engine.render(state.messages)

        # The newest user message carrying the zone: a typed-retry appends a
        # feedback user turn after it, and a retry state is still speculable.
        marker = ROUTE_ZONE_LABEL["candidates"] + "\n"
        user_content = next(
            (
                str(message.get("content") or "")
                for message in reversed(state.messages)
                if message.get("role") == "user"
                and marker in str(message.get("content") or "")
            ),
            None,
        )
        if user_content is None:
            return None
        _, _, remainder = user_content.partition(marker)

        # A blank line terminates this zone in both route layouts. Candidate
        # descriptions themselves are emitted one per line.
        candidate_block = remainder.split("\n\n", 1)[0]
        static_archetypes = list(state.site.decl.candidates)
        candidate_nodes: List[CandidateNode] = []
        first_callsites_by_handle: Dict[str, List[ByLLMCallsite]] = {}

        for line in candidate_block.splitlines():
            match = _CANDIDATE_LINE.fullmatch(line.strip())
            if match is None:
                continue
            node = match.groupdict()
            node["archetype"] = self._candidate_archetype(
                node["handle"], static_archetypes
            )
            candidate_nodes.append(node)
            first_callsites_by_handle[node["handle"]] = self._first_callsites(
                node["archetype"], state, program
            )

        return full_prompt, candidate_nodes, first_callsites_by_handle
