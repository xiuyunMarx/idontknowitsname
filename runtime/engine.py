import asyncio
import random
import statistics
import time
import uuid
from typing import Dict, List

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


class ModelEngine:
    def __init__(self, model_name: str):
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                enable_prefix_caching=True,
                scheduling_policy="priority",
            )
        )
        self.sp = SamplingParams(temperature=0.7)
        self.max_parallelizable_prefill: Dict[int, int] = {}
        self._decode_tasks = 0
        self._prefill_tasks = 0

    async def is_engine_idle(self) -> bool:
        return (
            not self.engine.output_processor.has_unfinished_requests()
            or self._prefill_tasks < self.max_parallelizable_prefill.get(self._decode_tasks, 0)
        )

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
        first_token = True
        self._decode_tasks += 1
        try:
            async for output in self.engine.generate(
                prompt, sampling_params or self.sp, request_id
            ):
                if output.outputs:
                    if first_token:
                        print(
                            f"[serve] {request_id} "
                            f"ttft_ms={(time.perf_counter() - started) * 1000:.2f} "
                            f"cached_tokens={output.num_cached_tokens or 0} "
                            f"prompt_tokens={len(output.prompt_token_ids or [])}",
                            flush=True,
                        )
                        first_token = False
                    text = output.outputs[0].text
        finally:
            self._decode_tasks -= 1
        return text

    async def prefill(self, prefill_prompt: str, request_id: str) -> None:
        sampling_params = SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True)
        started = time.perf_counter()
        self._prefill_tasks += 1
        try:
            async for _ in self.engine.generate(
                prefill_prompt, sampling_params, request_id, priority=1
            ):
                pass
        finally:
            self._prefill_tasks -= 1
        print(
            f"[prefill] {request_id} duration_ms={(time.perf_counter() - started) * 1000:.2f}",
            flush=True,
        )

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
        """Find safe prefill concurrency for each decode concurrency.

        For each number of concurrent decode requests, measure unloaded TBT and
        then add sustained prefill workers one at a time. A count is safe when
        median TBT remains within ``tbt_slack`` of baseline. Stop as soon as
        even one concurrent prefill is unsafe.
        """
        if tbt_slack < 0:
            raise ValueError("tbt_slack must be >= 0")
        if max_decode_tasks < 1 or max_prefill_tasks < 1:
            raise ValueError("max_decode_tasks and max_prefill_tasks must be >= 1")
        if decode_tokens <= warmup_tokens + 2:
            raise ValueError("decode_tokens must exceed warmup_tokens by at least 3")

        self.max_parallelizable_prefill.clear()

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

            self.max_parallelizable_prefill[num_decodes] = safe_prefills
            print(
                f"[profile] decode_tasks={num_decodes} "
                f"baseline_tbt={baseline * 1_000:.2f}ms "
                f"safe_prefill_tasks={safe_prefills}"
            )
            if safe_prefills == 0:
                break
