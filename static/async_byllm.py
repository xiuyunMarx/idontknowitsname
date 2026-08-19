import json

import vllm
import torch
from typing import Any, Dict, List, Optional

from pydantic import TypeAdapter
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

try:
    from static_parser import ByLLMDecl, extract_finish_output, format_tools_for_prompt, resolve_type, response_format_of
except ImportError:  # imported as part of the `static` package (e.g. from runtime/)
    from static.static_parser import ByLLMDecl, extract_finish_output, format_tools_for_prompt, resolve_type, response_format_of


class OutputConversionError(ValueError):
    """LLM output could not be converted to the declared return type (mirrors byllm)."""

    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output

# Byte-exact prompt constants, imported from byllm itself: Jac triple-quoted
# strings do NOT dedent, so the real values carry leading indent and newlines —
# a hand-mirrored copy diverges in the first KV block and kills every cache hit.
try:
    from jaclang.byllm.mtir import INSTRUCTION_TOOL as TOOL_INSTRUCTION, SYSTEM_PERSONA #type: ignore[import]
except Exception:  # byllm unavailable: keep the (approximate) fallback
    SYSTEM_PERSONA = "This is a task you must complete by returning only the output. The task will be expressed in the form of function call with arguments. Do not include explanations, code, or extra text—only the result."
    TOOL_INSTRUCTION = " Use the tools provided to reach the goal. Call one tool at a time with proper args—no explanations, no narration. Think step by step, invoking tools as needed. When done, always call finish_tool(output) to return the final output. Only use tools."


class AsyncByLLM(torch.nn.Module):
    """One byllm call site with its invariant prompt prefix prebuilt from the compiled IR.

    The engine is a SHARED handle injected by the caller (one vllm.LLM per process,
    not per call site — N engines would load N copies of the model). warm_invariant()
    is a passive interface: it warms the engine's prefix cache with the invariant
    tokens but is never triggered by this class on its own.
    """

    def __init__(self, decl: ByLLMDecl,
                 engine: Optional["vllm.LLM"] = None,
                 sampling_params: Optional[SamplingParams] = None):
        super().__init__()
        self.decl = decl
        self.model = engine
        self.sampler: Optional[SamplingParams] = sampling_params
        self.system_persona = SYSTEM_PERSONA
        self.tools = decl.tools
        self.tool_schema = ""
        self.reponse_format: Optional[Dict[str, Any]] = None
        self.invariant_system = ""
        self.invariant_user_prefix = ""
        self.last_output: Optional[Any] = None  # raw vllm RequestOutput of the last forward (TTFT via .metrics)
        self._build_invariant()

    def bind_engine(self, engine: "vllm.LLM") -> None:
        self.model = engine

    def _build_invariant(self) -> None:
        """[SYSTEM] [TOOLS] [TOOL_SCHEMA] [RESPONSE_FORMAT] are fixed; prebuild them.

        Renders the same zones byllm's runtime builds in MTRuntime.factory +
        inject_tool_hint, so the token prefix matches what byllm would send.
        """
        d = self.decl
        if d.kind == "visit":
            self._build_visit_invariant()
            return
        system = self.system_persona
        if d.extra_system_prompt:
            system += "\n\n" + d.extra_system_prompt
        if d.tools:
            # byllm's local (non-native-tool) route: INSTRUCTION_TOOL on the persona,
            # then the text tool protocol appended by inject_tool_hint.
            system += TOOL_INSTRUCTION
            self.tool_schema = format_tools_for_prompt(d.tools)
            system = system.rstrip() + "\n\n" + self.tool_schema
        else:
            self.reponse_format = d.reponse_format or response_format_of(d.return_type)
        self.invariant_system = system
        header = d.qualifier + d.signature()
        if d.sem:
            header = f"{header} --- {d.sem}"
        head_lines = [header]
        for p in d.params:
            if p.get("sem"):
                head_lines.append("      " + f"{p['name']}: {p['type']} ---- {p['sem']}")
        self.invariant_user_prefix = "\n".join(head_lines)
        # Sampling params are compile-time constants too
        if self.sampler is None:
            kwargs: Dict[str, Any] = {"max_tokens": int(d.call_params.get("max_tokens", 512))}
            if "temperature" in d.call_params:
                kwargs["temperature"] = float(d.call_params["temperature"])
            if d.tools:
                kwargs["stop"] = ["</tool_call>"]
            so = self._structured_outputs()
            if so is not None:
                kwargs["structured_outputs"] = so
            self.sampler = SamplingParams(**kwargs)

    def _build_visit_invariant(self) -> None:
        """Invariant of a `visit ... by llm()` routing call (mirrors byllm route_visit): the routing system prompt with its select suffix, and the static Goal line."""
        d = self.decl
        system = "You are routing a graph walker. Choose which candidate node(s) the walker should visit next, by handle. Return only valid handles."
        select = d.call_params.get("select")
        if select == 1:
            system += " Choose exactly one."
        elif isinstance(select, int) and not isinstance(select, bool):
            system += f" Choose exactly {select}."
        self.invariant_system = system
        self.invariant_user_prefix = f"Goal: {d.intent}" if d.intent else ""
        if self.sampler is None:
            kwargs: Dict[str, Any] = {"max_tokens": int(d.call_params.get("max_tokens", 64))}
            if "temperature" in d.call_params:
                kwargs["temperature"] = float(d.call_params["temperature"])
            self.sampler = SamplingParams(**kwargs)

    def visit_stable_prefix(self, here: str = "", candidates: str = "") -> str:
        """The cross-request-stable user prefix of a visit-by site under the
        cache-aware layout: Goal (compile-time) + Current node + Candidates
        (graph-dependent, stable until the graph changes). The per-request
        Walker zone comes after — outside this prefix by design."""
        parts = [self.invariant_user_prefix] if self.invariant_user_prefix else []
        if here:
            parts.append(f"Current node:\n{here}")
        if candidates:
            parts.append(f"Candidates (choose by handle):\n{candidates}")
        return "\n\n".join(parts)

    def build_full_prompt(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        """Append the runtime bindings zone (and `self` identity zone) to the invariant prefix."""
        if self.decl.kind == "visit":
            # Cache-aware zone order (route_visit with JAC_ROUTE_CACHE_LAYOUT=1):
            # stable zones first, per-request Walker last.
            user = self.visit_stable_prefix(str(params.get("here") or ""), str(params.get("candidates") or ""))
            if params.get("walker") is not None:
                user = user + f"\n\nWalker:\n{params['walker']}" if user else f"Walker:\n{params['walker']}"
            return [{"role": "system", "content": self.invariant_system}, {"role": "user", "content": user}]
        lines = [self.invariant_user_prefix]
        for p in self.decl.params:
            if p["name"] in params:
                lines.append(f"{p['name']} = {params[p['name']]!r}")
        zones = ["\n".join(lines)]
        if params.get("self") is not None and self.decl.owner_sem:
            zones.append(f"self = {params['self']!r} ---- {self.decl.owner_sem}")
        user = "\n\n".join(zones)
        return [{"role": "system", "content": self.invariant_system}, {"role": "user", "content": user}]

    def warm_invariant(self) -> Any:
        """Warm the engine's prefix cache with the invariant tokens. Caller-triggered only."""
        if self.model is None:
            raise RuntimeError(f"AsyncByLLM({self.decl.qualifier}{self.decl.name}): no engine bound; call bind_engine() first")
        messages = [{"role": "system", "content": self.invariant_system}, {"role": "user", "content": self.invariant_user_prefix}]
        sp = vllm.SamplingParams(max_tokens=1, temperature=0.0)
        return self.model.chat(messages, sampling_params=sp, use_tqdm=False)  # type: ignore[arg-type]

    def _structured_outputs(self) -> Optional["StructuredOutputsParams"]:
        """Decoding constraint from the static response format (None for str returns or when tools carry the contract)."""
        if self.reponse_format is None:
            return None
        schema = self.reponse_format.get("json_schema", {}).get("schema")
        if not schema:
            return None
        return StructuredOutputsParams(json=schema)

    def _coerce(self, obj: Any) -> Any:
        """Coerce a decoded JSON value to the declared return type (byllm's json_to_instance path).

        Targets the materialized translation of the Jac obj/enum (decl.return_type_obj)
        so results come back as real typed instances, not dicts.
        """
        if isinstance(obj, dict) and set(obj.keys()) == {"schema_object_wrapper"}:
            obj = obj["schema_object_wrapper"]
        ty = self.decl.return_type_obj or resolve_type(self.decl.return_type)
        if ty is None or ty is str:
            return obj
        try:
            return TypeAdapter(ty).validate_python(obj)
        except Exception as e:
            raise OutputConversionError(f"Failed to convert LLM output to '{self.decl.return_type}': {e}", raw_output=repr(obj))

    def parse_response(self, text: str) -> Any:
        """Parse the raw completion into the declared return type (mirrors MTRuntime.parse_response)."""
        if self.decl.kind == "visit":
            return text  # handle names; resolution to node instances is runtime-side
        rt = self.decl.return_type
        if self.decl.tools:
            found, val = extract_finish_output(text)
            if not found:
                return text  # no finish_tool in the reply (mid-ReAct or free text): hand back raw
            return self._coerce(val)
        if rt in ("", "str") or not text.strip():
            return text
        try:
            obj = json.loads(text)
        except Exception as e:
            raise OutputConversionError(f"Failed to convert LLM output to '{rt}': {e}", raw_output=text)
        return self._coerce(obj)

    def forward(self, params: Dict[str, Any]) -> Any:
        """Fill in the real argument values, generate, and return the PARSED value like a byllm call.

        The raw vllm RequestOutput (TTFT via .metrics) is kept in self.last_output.
        """
        if self.model is None:
            raise RuntimeError(f"AsyncByLLM({self.decl.qualifier}{self.decl.name}): no engine bound; call bind_engine() first")
        messages = self.build_full_prompt(params)
        outputs = self.model.chat(messages, sampling_params=self.sampler, use_tqdm=False)  # type: ignore[arg-type]
        self.last_output = outputs[0]
        return self.parse_response(outputs[0].outputs[0].text)
