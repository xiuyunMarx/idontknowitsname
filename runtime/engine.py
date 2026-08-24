import asyncio
import json
import random
import statistics
import time
import uuid
from typing import Dict, List

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.logprobs import Logprob
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

from console_helper.debug_output import console_debug, console_log, console_warn, console_error

class ModelEngine:
    def __init__(self, model_name: str, gpu_memory_utilization: float = 0.9, max_model_len: int = 8192):
        self.model_name = model_name
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enable_prefix_caching=True,
                scheduling_policy="priority",
            )
        )
        self.sp = SamplingParams(temperature=0.7)
        self.max_prefill_tokens: Dict[int, int] = {}  # decode concurrency -> safe in-flight prefill tokens
        self._decode_tasks = 0
        self._prefill_tokens = 0  # estimated tokens of all in-flight prefills, real and speculative

    def tokenize(self, prompt: str) -> List[int]:
        return self.engine.get_tokenizer().encode(prompt)  # type: ignore[union-attr]

    def count_tokens(self, prompt: str) -> int:
        return len(self.tokenize(prompt))

    def has_prefill_room(self, tokens: int) -> bool:
        """Admit a prefill of `tokens` when the engine is free, or when the profiled
        token budget for the current decode concurrency still has room for it."""
        return (
            not self.engine.output_processor.has_unfinished_requests()
            or self._prefill_tokens + tokens <= self.max_prefill_tokens.get(self._decode_tasks, 0)
        )

    def save_profile(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"model": self.model_name, "max_prefill_tokens": self.max_prefill_tokens}, f)

    def load_profile(self, path: str) -> bool:
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return False
        if data.get("model") != self.model_name:
            return False
        self.max_prefill_tokens = {int(k): int(v) for k, v in data["max_prefill_tokens"].items()}
        return True

    def render(self, messages: List[Dict[str, str]]) -> str:
        return self.engine.get_tokenizer().apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #type: ignore

    async def generate(
        self,
        prompt: str,
        request_id: str,
        sampling_params: SamplingParams | None = None,
    ) -> str:
        text = ""
        started = time.perf_counter()
        first_token_at = started
        first_token = True
        out_tokens = 0
        cost = self.count_tokens(prompt)
        self._prefill_tokens += cost  # A real call prefills until its first token, then decodes.
        try:
            async for output in self.engine.generate(
                prompt, sampling_params or self.sp, request_id
            ):
                if output.outputs:
                    if first_token:
                        self._prefill_tokens -= cost
                        self._decode_tasks += 1
                        first_token = False
                        first_token_at = time.perf_counter()
                        console_debug(f"[serve] {request_id} duration_ms={(first_token_at - started) * 1000:.2f} cached_tokens={output.num_cached_tokens or 0} prompt_tokens={len(output.prompt_token_ids or [])}")
                    text = output.outputs[0].text
                    out_tokens = len(output.outputs[0].token_ids)
        finally:
            # Exactly one decrement per increment, whichever phase we ended in.
            if first_token:
                self._prefill_tokens -= cost
            else:
                self._decode_tasks -= 1
                console_debug(f"[decode] {request_id} decode_ms={(time.perf_counter() - first_token_at) * 1000:.2f} output_tokens={out_tokens}")
        return text

    async def prefill(self, prefill_prompt: str | List[int], request_id: str, cost: int | None = None) -> None:
        """`cost` is the caller's estimate of uncached tokens; defaults to full length.
        A token-id list warms exactly that prefix of a longer prompt's tokens —
        how chunked speculation bounds each request to a small uncached slice."""
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        started = time.perf_counter()
        if isinstance(prefill_prompt, list):
            cost = len(prefill_prompt) if cost is None else cost
            prefill_prompt = TokensPrompt(prompt_token_ids=prefill_prompt)
        else:
            cost = self.count_tokens(prefill_prompt) if cost is None else cost
        self._prefill_tokens += cost
        try:
            async for _ in self.engine.generate(
                prefill_prompt, sampling_params, request_id, priority=1
            ):
                pass
        finally:
            self._prefill_tokens -= cost
        print(
            f"[prefill] {request_id} cost_tokens={cost} duration_ms={(time.perf_counter() - started) * 1000:.2f}",
            flush=True,
        )

    async def probe(self, prompt: str, request_id: str, cost: int = 64) -> Dict[int, Logprob]:
        """Greedy one-token probe; returns the top logprobs at the first position.
        The prompt rides the router's own just-computed prefix, so its uncached
        cost is a few think-block tokens — billed as a small nominal `cost`."""
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        self._prefill_tokens += cost
        try:
            async for output in self.engine.generate(prompt, sampling_params, request_id, priority=1):
                if output.outputs and output.outputs[0].logprobs:
                    return output.outputs[0].logprobs[0]
        finally:
            self._prefill_tokens -= cost
        return {}

    @staticmethod
    def _random_prompt(num_words: int) -> str:
        """Create an effectively uncached prompt for a profiling request."""
        return " ".join(f"{random.getrandbits(32):08x}" for _ in range(num_words))

    async def _measure_tbt(
        self,
        num_decode_tasks: int,
        num_prefill_tasks: int,
        *,
        decode_tokens: int,
        prefill_words: int,
        warmup_tokens: int,
    ) -> float:
        """Return median per-stream TBT under sustained prefill pressure."""
        if num_decode_tasks < 1 or num_prefill_tasks < 0:
            raise ValueError("num_decode_tasks must be >= 1 and num_prefill_tasks >= 0")

        stop_prefill = asyncio.Event()
        decode_started = asyncio.Event()
        started_count = 0
        started_lock = asyncio.Lock()

        async def decode_worker(index: int) -> List[float]:
            nonlocal started_count
            sampling = SamplingParams(
                max_tokens=decode_tokens, temperature=0.0, ignore_eos=True
            )
            timestamps: List[float] = []
            request_id = f"profile-decode-{index}-{uuid.uuid4().hex}"
            async for _ in self.engine.generate(
                self._random_prompt(24), sampling, request_id
            ):
                timestamps.append(time.perf_counter())
                if len(timestamps) == 1:
                    async with started_lock:
                        started_count += 1
                        if started_count == num_decode_tasks:
                            decode_started.set()
            return [
                timestamps[i] - timestamps[i - 1]
                for i in range(warmup_tokens + 1, len(timestamps))
            ]

        async def prefill_worker(index: int) -> None:
            await decode_started.wait()
            if stop_prefill.is_set():
                return
            sampling = SamplingParams(
                max_tokens=1, temperature=0.0, ignore_eos=True
            )
            while not stop_prefill.is_set():
                request_id = f"profile-prefill-{index}-{uuid.uuid4().hex}"
                async for _ in self.engine.generate(
                    self._random_prompt(prefill_words), sampling, request_id
                ):
                    pass

        decoders = [
            asyncio.create_task(decode_worker(i)) for i in range(num_decode_tasks)
        ]
        prefills = [
            asyncio.create_task(prefill_worker(i)) for i in range(num_prefill_tasks)
        ]
        try:
            gaps_by_stream = await asyncio.gather(*decoders)
        finally:
            stop_prefill.set()
            decode_started.set()  # release workers if a decoder failed at startup
            if prefills:
                await asyncio.gather(*prefills, return_exceptions=True)

        # Give each decode stream equal weight.
        stream_tbts = [statistics.median(gaps) for gaps in gaps_by_stream if gaps]
        if len(stream_tbts) != num_decode_tasks:
            raise RuntimeError(
                "not enough generated tokens to measure TBT; increase decode_tokens"
            )
        return statistics.median(stream_tbts)

    async def _profile(
        self,
        *,
        tbt_slack: float = 0.10,
        max_decode_tasks: int = 16,
        max_prefill_tasks: int = 8,
        decode_tokens: int = 128,
        prefill_words: int = 512,
        warmup_tokens: int = 8,
    ) -> None:
        """Find the safe in-flight prefill token budget for each decode concurrency.

        For each number of concurrent decode requests, measure unloaded TBT and
        then add sustained prefill workers one at a time. A count is safe when
        median TBT remains within ``tbt_slack`` of baseline; the budget is that
        count times the worker's token length. When not even one worker is safe,
        retry a single worker with halved prompts — token granularity often
        admits a small budget where task granularity says zero.
        """
        if tbt_slack < 0:
            raise ValueError("tbt_slack must be >= 0")
        if max_decode_tasks < 1 or max_prefill_tasks < 1:
            raise ValueError("max_decode_tasks and max_prefill_tasks must be >= 1")
        if decode_tokens <= warmup_tokens + 2:
            raise ValueError("decode_tokens must exceed warmup_tokens by at least 3")

        self.max_prefill_tokens.clear()
        chunk_tokens = self.count_tokens(self._random_prompt(prefill_words))

        # Exclude model/CUDA first-request initialization from the baseline.
        await self._measure_tbt(
            1,
            0,
            decode_tokens=max(16, warmup_tokens + 3),
            prefill_words=prefill_words,
            warmup_tokens=min(warmup_tokens, 4),
        )

        for num_decodes in range(1, max_decode_tasks + 1):
            baseline = await self._measure_tbt(
                num_decodes,
                0,
                decode_tokens=decode_tokens,
                prefill_words=prefill_words,
                warmup_tokens=warmup_tokens,
            )
            tbt_limit = baseline * (1.0 + tbt_slack)
            safe_prefills = 0

            for num_prefills in range(1, max_prefill_tasks + 1):
                measured = await self._measure_tbt(
                    num_decodes,
                    num_prefills,
                    decode_tokens=decode_tokens,
                    prefill_words=prefill_words,
                    warmup_tokens=warmup_tokens,
                )
                if measured > tbt_limit:
                    break
                safe_prefills = num_prefills

            budget = safe_prefills * chunk_tokens
            if safe_prefills == 0:
                words = prefill_words // 2
                while words >= 32:
                    measured = await self._measure_tbt(
                        num_decodes,
                        1,
                        decode_tokens=decode_tokens,
                        prefill_words=words,
                        warmup_tokens=warmup_tokens,
                    )
                    if measured <= tbt_limit:
                        budget = self.count_tokens(self._random_prompt(words))
                        break
                    words //= 2

            self.max_prefill_tokens[num_decodes] = budget
            print(
                f"[profile] decode_tasks={num_decodes} "
                f"baseline_tbt={baseline * 1_000:.2f}ms "
                f"budget_tokens={budget}"
            )
            if budget == 0:
                break
