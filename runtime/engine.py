import asyncio
import uuid

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from typing import Dict, Any

class ModelEngine:
    def __init__(self, model_name: str):
        self.engine = AsyncLLM.from_engine_args(
            AsyncEngineArgs(
                model=model_name,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                enable_prefix_caching=True,   # on by default in V1
            )
        )
        
        self.sp = SamplingParams(
            temperature=0.7
        )   
        self.max_parallelizable_prefill: Dict[int, int] = {} # For n parallel decode requests, the max number of prefill requests that can be parallelized without hurting TBT

    async def is_engine_idle(self) -> bool:
        return not self.engine.output_processor.has_unfinished_requests()
    
    def _profile(self):
        