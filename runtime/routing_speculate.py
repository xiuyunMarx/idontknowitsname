from __future__ import annotations

import json
import re
import uuid

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from runtime.engine import ModelEngine
from utils.jac_static_parser import ByLLMCallsite, ProgramTopology
from utils.utils import ROUTE_ZONE_LABEL
from console_helper.debug_output import console_debug

if TYPE_CHECKING:
    from runtime.server import _CallState


_CANDIDATE_LINE = re.compile(
    r"^(?P<handle>[A-Za-z_]\w*)\)\s+(?P<description>.+)$"
)

# byllm wraps a non-object return type in {"schema_object_wrapper": <schema>}
# (jaclang/byllm/impl/schema.impl.jac) and unwraps it when parsing the answer.
_WRAPPER = "schema_object_wrapper"

# (handle, description, statically resolved node archetype)
CandidateNode = Dict[str, str]
VisitParseResult = Tuple[
    List[CandidateNode],
    Dict[str, List[ByLLMCallsite]],
]


def inner_json_schema(schema: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The JSON schema inside a client `response_format`: OpenAI's
    {"type": "json_schema", "json_schema": {"schema": {...}}}, or a bare schema."""
    if not isinstance(schema, dict):
        return None
    if schema.get("type") == "json_schema":
        inner = (schema.get("json_schema") or {}).get("schema")
        return inner if isinstance(inner, dict) else None
    if "properties" in schema or "enum" in schema or schema.get("type") in (
        "object", "array", "string", "integer", "number", "boolean",
    ):
        return schema
    return None


def answer_prefix(schema: Optional[Dict[str, Any]]) -> str:
    """The bytes a grammar-constrained answer to `schema` must start with, up to
    the first character the model actually chooses. For byllm's routing schema
    {"schema_object_wrapper": [<enum handle>, ...]} that is `{"schema_object_wrapper": ["`,
    so a probe placed after it reads the handle distribution directly."""
    inner = inner_json_schema(schema)
    if inner is None:
        return ""
    return _prefix_of(inner)


def _prefix_of(schema: Dict[str, Any]) -> str:
    kind = schema.get("type")
    if kind == "object":
        props = schema.get("properties") or {}
        required = schema.get("required") or list(props)
        if len(props) == 1 and len(required) == 1 and required[0] in props:
            key = required[0]
            return '{"' + key + '": ' + _prefix_of(props[key])
        return "{"
    if kind == "array":
        items = schema.get("items") or {}
        return "[" + _prefix_of(items) if isinstance(items, dict) else "["
    if kind == "string" or (kind is None and "enum" in schema):
        return '"'
    return ""


def parse_answer_handles(text: str) -> List[str]:
    """Handles named by a routing answer: the JSON list (wrapped or bare), a
    single JSON string, or bare `A, B` text when the answer is not JSON."""
    raw = text.strip()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [h for h in re.split(r"[,\s\[\]\"']+", raw) if h]
    if isinstance(value, dict):
        value = value.get(_WRAPPER, value)
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value not in (None, "") else []


class RoutingSpeculate:
    def __init__(self, engine: ModelEngine) -> None:
        self.engine = engine
        self._pieces: Dict[Tuple[str, str], str] = {}  # (prefix, handle) -> first token text

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
        abilities: list = []
        for walker in state.site.decl.walkers or [state.site.decl.owner_arch]:
            for ability in program._arrival_abilities(archetype, walker):
                if ability not in abilities:
                    abilities.append(ability)

        for ability in abilities:
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

    def _first_piece(self, tokenizer: Any, prefix: str, handle: str) -> str:
        """Text of the first token the model emits for `handle` right after
        `prefix` — the token the probe's top-k is matched against. Tokenized in
        context: BPE merges differ between `WebSearchAgent` alone and after `["`."""
        key = (prefix, handle)
        if key not in self._pieces:
            pre = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
            ids = tokenizer.encode(prefix + handle, add_special_tokens=False)
            if pre and ids[: len(pre)] == pre and len(ids) > len(pre):
                piece = tokenizer.decode([ids[len(pre)]])
            else:
                own = tokenizer.encode(handle, add_special_tokens=False)
                piece = tokenizer.decode(own[:1]) if own else handle
            self._pieces[key] = piece.strip()
        return self._pieces[key]

    async def sort_candidate_calls(
        self, state: _CallState, program: ProgramTopology, rank: bool = False
    ) -> Optional[List[ByLLMCallsite]]:
        """Every candidate's first byLLM callsites, most likely route first.
        `rank=False` skips the probe and keeps the candidates' listed order.
        None when the probe was killed by a real admission.

        The probe rides the router's own prompt (already prefilled for the real
        request) extended by the answer prefix the response schema fixes, so its
        one generated token is the first handle token; the top-k logprobs at that
        position rank the candidates. Without the prefix the first token of a
        JSON answer is `{"` or `["` and carries no routing signal."""
        parsed = self._parse_visit(state, program)
        if parsed is None:
            return []
        candidate_nodes, first_callsites_by_handle = parsed

        if rank and len(candidate_nodes) > 1:
            tokenizer = self.engine.engine.get_tokenizer()
            prefix = answer_prefix(state.request.schema)
            probe_prompt = self.engine.render(state.messages) + prefix
            top = await self.engine.probe(
                probe_prompt, f"probe-{state.site.callsite_uuid}-{uuid.uuid4().hex}",
                cache_salt=getattr(state, "cache_salt", None),
            )
            if top is None:
                return None
            pieces = {
                node["handle"]: self._first_piece(tokenizer, prefix, node["handle"])
                for node in candidate_nodes
            }
            scores = {node["handle"]: float("-inf") for node in candidate_nodes}
            seen_tokens: List[Tuple[str, float]] = []
            for token_id, info in top.items():
                text = (info.decoded_token or tokenizer.decode([token_id])).strip()
                seen_tokens.append((text, info.logprob))
                if not text:
                    continue
                for handle, piece in pieces.items():
                    exact = text == piece
                    # A shorter top token that the handle continues (`We` for
                    # `WebSearchAgent`) still names it; single characters do not.
                    partial = len(text) >= 2 and piece.startswith(text)
                    if exact or partial:
                        scores[handle] = max(scores[handle], info.logprob)
            seen_tokens.sort(key=lambda t: -t[1])
            tag = f"{state.request.program_name}:{state.site.key}:s{state.session.session_id}c{state.request.id}"
            if all(s == float("-inf") for s in scores.values()):
                console_debug(
                    f"[probe] {tag} no-signal prefix={prefix!r} top={seen_tokens[:5]}"
                )
                state.route_probe = []
            else:
                # Stable sort: unscored candidates keep their listed order after the scored ones.
                candidate_nodes = sorted(candidate_nodes, key=lambda n: -scores[n["handle"]])
                state.route_probe = [
                    (n["handle"], scores[n["handle"]]) for n in candidate_nodes
                ]
                console_debug(
                    f"[probe] {tag} prefix={prefix!r} top={seen_tokens[:5]} "
                    f"order={[n['handle'] for n in candidate_nodes]}"
                )

        return self._plan(candidate_nodes, first_callsites_by_handle)

    def plan_from_answer(
        self, state: _CallState, program: ProgramTopology, handles: List[str]
    ) -> Optional[List[ByLLMCallsite]]:
        """The warm plan once the router has answered: the chosen candidates'
        first callsites in answer order, then the rest in listed order. The idle
        window after the answer warms what will run, not what might."""
        parsed = self._parse_visit(state, program)
        if parsed is None:
            return None
        candidate_nodes, first_callsites_by_handle = parsed
        chosen = [n for h in handles for n in candidate_nodes if n["handle"] == h]
        rest = [n for n in candidate_nodes if n not in chosen]
        return self._plan(chosen + rest, first_callsites_by_handle)

    @staticmethod
    def _plan(
        candidate_nodes: List[CandidateNode],
        first_callsites_by_handle: Dict[str, List[ByLLMCallsite]],
    ) -> List[ByLLMCallsite]:
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
        """Live candidate nodes of a routing call and their first callsites.
        Parsed from the client's prompt: only the client knows the live graph."""
        if state.request.type != "generate" or not state.site.is_visit:
            return None

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

        return candidate_nodes, first_callsites_by_handle
