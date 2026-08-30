"""GPU-free stand-ins for the controller tests."""
import time

from trace_extractor.primitives import ByLLMCallsite, CallsiteType, Program


class FakeEngine:
    def __init__(self, name, tier="ssd", reply="ok"):
        self.name, self.tier, self.transitioning, self.serving, self.busy = name, tier, False, False, False
        self.killed = self.offloaded = self.loaded = 0
        self.reply = reply
        self.last = {"ttft_ms": 1.0, "cached_tokens": 0, "prompt_tokens": 0}
        self.prompts = []
        self.rendered = []

    @property
    def resident(self):
        return self.tier == "gpu"

    @property
    def speculable(self):
        return self.resident and not self.transitioning

    async def offload(self):
        self.tier, self.offloaded = "host", self.offloaded + 1

    async def kill(self):
        self.tier, self.killed = "ssd", self.killed + 1

    async def load(self, prefill=None):
        self.tier, self.loaded = "gpu", self.loaded + 1
        self.prompts.extend(prefill or [])

    def render(self, messages):
        return "".join(f"<{m['role']}>{m.get('content') or ''}" for m in messages) + "<assistant>"

    def tokenize(self, prompt):
        return prompt.split()

    @staticmethod
    def _with_salt(prompt, salt):
        return (prompt, salt)

    async def generate(self, prompt, request_id, sp, cache_salt, progress):
        self.rendered.append(prompt)
        self.last = {"ttft_ms": 1.0, "cached_tokens": 0, "prompt_tokens": len(str(prompt).split())}
        progress(time.perf_counter(), 3)
        return self.reply(prompt) if callable(self.reply) else self.reply


def line_program(name, chain):
    """A Program that has seen `chain` = [(key, model), ...] once, as one straight session."""
    p = Program(name)
    for key, model in chain:
        p.callsites[key] = ByLLMCallsite(key=key, kind=CallsiteType.BYLLM, label=key, model=model, system_prompt="s",
                                         context_desc=f"{key}()")
    for i, (key, _) in enumerate(chain):
        nxt = chain[i + 1][0] if i + 1 < len(chain) else "END"
        p.succ[key][nxt] += 1
    if chain:
        p.entry[chain[0][0]] += 1
    return p
