from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM
import asyncio
from typing import List

class InstanceEngine:
    def __init__(self, model_name:str) -> None:
        self.args = AsyncEngineArgs(served_model_name=model_name)
        self.engine = AsyncLLM.from_engine_args(self.args)
        
        