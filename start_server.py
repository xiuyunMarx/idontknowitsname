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
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-mem", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=8192)
    args = parser.parse_args()

    engine = ModelEngine(args.model, gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len)
    profile_path = f"profile-{args.model.rsplit('/', 1)[-1]}.json"
    if args.profile:
        await engine._profile()
        engine.save_profile(profile_path)
    elif not args.no_budget and engine.load_profile(profile_path):
        print(f"[profile] loaded {profile_path}: {engine.max_prefill_tokens}", flush=True)

    server = GuardServer(
        engine,
        num_workers=args.workers,
        proactive_prefill=not args.no_prefill,
        spec_policy=args.spec_policy,
    )
    for program in args.program:
        name, source, port = program.rsplit(":", 2)
        server.add_program(name, source, int(port))
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
