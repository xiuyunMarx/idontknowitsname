import torch
import vllm
from utils.jac_static_parser import ProgramTopology, ByLLMCallsite, ByLLMDecl
from utils.interceptor_receiver import InterceptorLLMBackend, ByLLMRequest
from typing import List, Dict, Any, Tuple
import asyncio

class GuardServer:
    def __init__(self, num_workers:int = 4):
        self._num_workers = num_workers
        self._programs:Dict[str, ProgramTopology] = {} 
        self._llm_backend:Dict[str, InterceptorLLMBackend] = {}
        self._task_queue:asyncio.Queue = asyncio.Queue()
        
    def add_program(self,program_name:str,  src_path: str, port: int):
        self._programs[program_name] = ProgramTopology(program_name, src_path)
        self._programs[program_name].parse_dependency()
        self._llm_backend[program_name] = InterceptorLLMBackend(program_name, queue=self._task_queue, comm_port=port)
        
    async def _llm_backend_monitor(self):
        while True:
            backend, request = await self._task_queue.get() 
            assert isinstance(request, ByLLMRequest) and isinstance(backend, InterceptorLLMBackend)
            
    
    async def _process_request(self, request:ByLLMRequest):
        program:ProgramTopology = self._programs.get(request.program_name, None) #type: ignore
        callsites = program.sites_of(request.key) #type: ignore
        
        
        raise NotImplementedError
    
    def show_info(self, program_name: str):
        if program_name not in self._programs:
            raise ValueError(f"Program {program_name} not found.")
        p = self._programs[program_name]
        print("decls:", list(p.decls))
        print("\ncallsites (uuid | consumers):")
        for s in p.callsites:
            print(f"  {s.callsite_uuid} | {s.consumers}")
        print("\ntopology (may-run-next):")
        for k, succ in p.topology.items():
            print(f"  {k} -> {succ}")
        print("\nprovenance (param -> value source):")
        for k, params in p.provenance.items():
            for pname, spec in params.items():
                print(f"  {k}.{pname} <- {spec}")
        if p.callsites:
            s0 = p.callsites[-1]
            demo = {prm["name"]: repr(f"<{prm['name']} value>") for prm in s0.decl.params}
            partial = dict(list(demo.items())[:1])
            print(f"\n--- invariant system of {s0.callsite_uuid} ---\n{s0.invariant_system}")
            print(f"\n--- partial assembly ({list(partial)}) ---\n{s0.assemble_prompt(partial)[1]['content']}")
            print(f"\n--- full assembly ---\n{s0.assemble_prompt(demo)[1]['content']}")
