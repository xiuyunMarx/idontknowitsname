from engine import ModelEngine
from utils.jac_static_parser import ByLLMCallsite, ProgramTopology
from runtime.server import _CallState

class RoutingSpeculate:
    def __init__(self, engine: ModelEngine) -> None:
        self.engine = engine
        
    def _parse_visit(self,state:_CallState, program: ProgramTopology):
        