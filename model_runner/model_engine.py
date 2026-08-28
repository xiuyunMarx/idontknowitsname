"""AsyncLLM with host-offloadable weights that stream back layer by layer
while prefill requests already run.

    engine = await InstanceEngine.create("Qwen/Qwen3-8B")
    await engine.offload()
    await engine.load(prefill=[system_prompt])   # KV lands in the prefix cache
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from typing import Any, List, Optional

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKER_EXT = "model_runner.layerwise.LayerwiseWorkerExtension"


class InstanceEngine(AsyncLLM):
    @classmethod
    async def create(cls, model: str, **engine_kwargs: Any) -> "InstanceEngine":
        # Worker processes must import model_runner.layerwise.
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        os.environ["PYTHONPATH"] = _REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

        engine_kwargs.setdefault("enable_prefix_caching", True)
        # Prefill must be a single chunk to overlap fully with the load.
        if engine_kwargs.get("max_num_batched_tokens") is None:
            engine_kwargs["max_num_batched_tokens"] = 8192
        # Gates are Python; CUDA-graph replay would skip them.
        engine_kwargs.setdefault("enforce_eager", True)
        args = AsyncEngineArgs(model=model, worker_extension_cls=_WORKER_EXT, **engine_kwargs) #type: ignore
        engine = cls.from_engine_args(args)
        engine._lw_info = await engine._rpc("lw_install") #type: ignore
        engine._lw_resident = True #type: ignore
        return engine #type: ignore

    async def offload(self) -> dict:
        """Weights -> pinned host memory; weight and KV cache GPU memory freed.
        No in-flight requests allowed."""
        if self.output_processor.has_unfinished_requests():
            raise RuntimeError("cannot offload with unfinished requests")
        res = await self._rpc("lw_offload")
        await self.reset_prefix_cache()
        self._lw_resident = False
        return res

    async def load(self, prefill: Optional[List[str]] = None) -> dict:
        """Stream weights back to the GPU. Parallel with prefill requests if provided"""
        await self.collective_rpc("lw_load")
        prefill_results = []
        if prefill:
            prefill_results = await asyncio.gather(*(self._prefill_one(p) for p in prefill))
        res = await self._rpc("lw_wait_loaded")
        self._lw_resident = True
        res["prefilled"] = prefill_results
        return res

    async def status(self) -> dict:
        return await self._rpc("lw_status")

    @property
    def resident(self) -> bool:
        return self._lw_resident

    async def _rpc(self, method: str) -> dict:
        res = await self.collective_rpc(method)
        return res[0] if isinstance(res, list) else res  # rank 0

    async def _prefill_one(self, prompt: str) -> dict:
        rid = f"prefill-{uuid.uuid4().hex}"
        n_prompt = 0
        async for out in self.generate(prompt, SamplingParams(max_tokens=1, temperature=0.0), rid):
            n_prompt = len(out.prompt_token_ids or ())
        return {"request_id": rid, "prompt_tokens": n_prompt}
