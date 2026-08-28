import uuid
from dataclasses import dataclass
from typing import Optional

from vllm import SamplingParams

from instance_engine.model import Engine
from static_pass.primitives import VisitByLLM


@dataclass
class RouteSpeculate:
    """Speculative routing for `visit [...] by llm(...)` sites, by skipping thinking"""

    # Empty reasoning block (Qwen3 style)
    no_think: str = "<think>\n\n</think>\n\n"
    # Prefix a grammar-constrained byllm routing answer
    answer_prefix: str = '{"schema_object_wrapper": ["'

    async def probe(self, engine: Engine, prompt: str, num_candidate_tokens: int,
                    cache_salt: Optional[str] = None) -> str:
        """Greedy decode of at most `num_candidate_tokens` tokens with thinking suppressed. """
        if not prompt.rstrip().endswith("</think>"):
            prompt = prompt + self.no_think
        prompt = prompt + self.answer_prefix
        cost = len(engine.tokenize(prompt))
        if engine.spec_allowance() < cost:
            return ""  # a real prefill owns the step, or in-flight speculation holds the budget
        sp = SamplingParams(max_tokens=max(1, int(num_candidate_tokens)), temperature=0.0,
                            stop=['"'], include_stop_str_in_output=False)
        output = await engine._speculative(
            engine._with_salt(prompt, cache_salt), sp, f"route-probe-{uuid.uuid4().hex}", cost)
        if output is None or not output.outputs:
            return ""
        return output.outputs[0].text.strip()

    async def speculate(self, engine: Engine, site: VisitByLLM, num_candidate_tokens: int = 16,
                        cache_salt: Optional[str] = None) -> str:
        """Probe the router's answer for `site` before the real call returns.
        Raises ValueError (from site.full_prompt) while any zone is unbound."""
        full_prompt = engine.render(site.full_prompt())
        return await self.probe(engine, full_prompt, num_candidate_tokens, cache_salt)

        
