import argparse
import asyncio
import faulthandler
import os
import signal
os.environ["LD_PRELOAD"] = "/home/xiaoyu/miniconda3/envs/jaseci/lib/libstdc++.so.6"
faulthandler.register(signal.SIGUSR1)  # kill -USR1 <pid> dumps all thread stacks

from runtime.engine import ModelEngine
from runtime.server import GuardServer


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--program",
        action="append",
        required=True,
        metavar="NAME:SOURCE:PORT",
    )
    parser.add_argument("--profile", action="store_true", help="run the TBT interference sweep and save it")
    parser.add_argument("--no-prefill", action="store_true")
    parser.add_argument("--no-budget", action="store_true", help="ignore any saved profile; speculate only when the engine is empty")
    parser.add_argument("--spec-policy", choices=["global", "newest"], default="global")
    parser.add_argument("--spec-chunk", type=int, default=128,
                        help="max uncached tokens per speculative request when the engine is idle; under load the profiled per-step allowance caps it further")
    parser.add_argument("--spec-features", default="const,ret,toolturn,probe",
                        help="comma list of compile-time facts speculation may use; 'none' warms successors' invariants only")
    parser.add_argument("--spec-order", choices=["static", "random"], default="static",
                        help="random: shuffle successor order (predictor ablation)")
    parser.add_argument("--profile-slack", type=float, default=0.10, help="TBT slack for --profile")
    parser.add_argument("--profile-stat", choices=["tbt_mean_ms", "tbt_p95_ms"], default="tbt_mean_ms")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    engine = ModelEngine(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    # The profile lives next to this file, whatever the working directory is.
    profile_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                f"profile-{args.model.rsplit('/', 1)[-1]}.json")
    if args.profile:
        await engine._profile(tbt_slack=args.profile_slack, stat=args.profile_stat)
        engine.save_profile(profile_path)
        print(f"[profile] saved {profile_path}: {engine.spec_tokens_per_step}", flush=True)
    elif not args.no_budget:
        if engine.load_profile(profile_path):
            print(f"[profile] loaded {profile_path}: {engine.spec_tokens_per_step}", flush=True)
        else:
            print(f"[profile] no usable {profile_path}; speculating only when the engine is idle", flush=True)

    server = GuardServer(
        engine,
        num_workers=args.workers,
        proactive_prefill=not args.no_prefill,
        spec_policy=args.spec_policy,
        spec_chunk=args.spec_chunk,
        spec_features=set() if args.spec_features == "none"
        else {f.strip() for f in args.spec_features.split(",") if f.strip()},
        spec_order=args.spec_order,
    )
    print(f"[guard] spec_features={sorted(server._spec_features)} spec_order={args.spec_order}", flush=True)
    for program in args.program:
        name, source, port = program.rsplit(":", 2)
        server.add_program(name, source, int(port))
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
