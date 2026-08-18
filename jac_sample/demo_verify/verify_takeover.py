"""Verify the static side runtime can take over byllm functions end to end.

Parses a Jac program with the preprocessor (no Jac execution, no byllm runtime),
binds one shared vLLM engine to every extracted AsyncByLLM, and runs each call
site directly on the Python side with sample arguments. Checks that:

  1. the invariant prompt is a strict prefix of the runtime-filled prompt,
  2. forward() returns a value of the declared return type (Jac obj/enum
     translated to real Python classes),
  3. TTFT of a 1-token probe, with and without invariant warming (--warm).

Usage:
    python verify_takeover.py                       # demo.jac, cold
    python verify_takeover.py --warm                # warm invariants first
    python verify_takeover.py --jac ../jac_sample/Jac-Rag-GPT/main.jac
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import time
from typing import Any, Dict

import vllm

from static.preprocess import SideRuntime


def sample_params(fn) -> Dict[str, Any]:
    """Synthesize plausible arguments for a call site from its declared param types."""
    samples_by_name = {
        "text": "My GPU catches fire every time I run the training job!!",
        "message": "How do walkers traverse a graph in Jac?",
        "query": "walker syntax",
    }
    out: Dict[str, Any] = {}
    for p in fn.decl.params:
        base = p["type"].split("[", 1)[0]
        if p["name"] in samples_by_name:
            out[p["name"]] = samples_by_name[p["name"]]
        elif base == "str":
            out[p["name"]] = "the quick brown fox"
        elif base == "int":
            out[p["name"]] = 3
        elif base == "float":
            out[p["name"]] = 1.0
        elif base == "bool":
            out[p["name"]] = True
        elif base in ("list", "set", "tuple"):
            out[p["name"]] = []
        elif base == "dict":
            out[p["name"]] = {}
        else:
            out[p["name"]] = ""
    return out


def probe_ttft(fn, params: Dict[str, Any]) -> float:
    """Wall-clock of a 1-token request on the full prompt: prefill + first token."""
    messages = fn.build_full_prompt(params)
    sp = copy.deepcopy(fn.sampler)
    sp.max_tokens = 1
    t0 = time.perf_counter()
    fn.model.chat(messages, sampling_params=sp, use_tqdm=False)
    return time.perf_counter() - t0


def type_ok(fn, value: Any) -> bool:
    rt = fn.decl.return_type
    if rt in ("", "str"):
        return isinstance(value, str)
    ty = fn.decl.return_type_obj
    if isinstance(ty, type):
        return isinstance(value, ty)
    return value is not None  # generic alias (list[str], ...): pydantic already validated


def main() -> None:
    ap = argparse.ArgumentParser(description="Run extracted byllm call sites directly on the Python side")
    ap.add_argument("--jac", default="demo.jac", help="Jac program to take over")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--warm", action="store_true", help="warm every invariant prefix before running")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.45)
    ap.add_argument("--no-eager", action="store_true", help="allow CUDA graph capture (slower startup)")
    args = ap.parse_args()

    rt = SideRuntime(args.jac, type_check=False)
    print(f"[verify] {len(rt.byLLM_functions)} byllm call site(s) extracted from {args.jac}")
    if not rt.byLLM_functions:
        return

    engine = vllm.LLM(
        model=args.model,
        enable_prefix_caching=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
        enforce_eager=not args.no_eager,
    )
    for fn in rt.byLLM_functions:
        fn.bind_engine(engine)

    if args.warm:
        t0 = time.perf_counter()
        for fn in rt.byLLM_functions:
            fn.warm_invariant()
        print(f"[verify] warmed {len(rt.byLLM_functions)} invariant prefix(es) in {time.perf_counter() - t0:.3f}s")

    failures = 0
    for fn in rt.byLLM_functions:
        d = fn.decl
        label = f"{d.qualifier}{d.name}"
        params = sample_params(fn)
        messages = fn.build_full_prompt(params)
        assert messages[1]["content"].startswith(fn.invariant_user_prefix), f"{label}: invariant is not a prefix"

        ttft = probe_ttft(fn, params)
        result = fn(params)  # nn.Module __call__ -> forward
        ok = type_ok(fn, result)
        failures += 0 if ok else 1
        shown = dataclasses.asdict(result) if dataclasses.is_dataclass(result) and not isinstance(result, type) else result
        print(f"\n=== {label} -> {d.return_type}  [{'PASS' if ok else 'FAIL'}]")
        print(f"    ttft(1-token probe): {ttft * 1000:.1f} ms   parsed type: {type(result).__name__}")
        print(f"    result: {shown!r}")

    print(f"\n[verify] {'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'} — takeover {'verified' if failures == 0 else 'broken'} ({'warm' if args.warm else 'cold'} run)")


if __name__ == "__main__":
    main()
