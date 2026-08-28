from static_pass.primitives import VisitByLLM, ByLLMFunc, Program
from instance_engine.model import Engine
from instance_engine.route_speculate import RouteSpeculate
from typing import Optional, List, Dict, Any, Union, Tuple
import vllm
import os
class Controller:
    def __init__(self):
        self.engine_pool:Dict[str, Engine] = {}
        self.program_tamplate: Dict[str, Program] = {}
        
    async def add_engine(self, model_name: str):
        if model_name in self.engine_pool:
            print(f"[DEBUG] model {model_name} already exists.")
        self.engine_pool[model_name] = await Engine.create(model_name=model_name, max_num_batched_tokens=8192)
        
    def register_program(self, path: str):
        x = path.split("/")
        self.program_tamplate[x[-1]] = Program(name=x[-1])
        self.program_tamplate[x[-1]].build_program(path)
    