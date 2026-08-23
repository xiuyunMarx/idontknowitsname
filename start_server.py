import argparse
import asyncio
import os
os.environ["LD_PRELOAD"] = "/home/xiaoyu/miniconda3/envs/jaseci/lib/libstdc++.so.6"

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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-prefill", action="store_true")
    args = parser.parse_args()

    engine = ModelEngine(args.model)
    if args.profile:
        await engine._profile()

    server = GuardServer(engine, proactive_prefill=not args.no_prefill)
    for program in args.program:
        name, source, port = program.rsplit(":", 2)
        server.add_program(name, source, int(port))
    await server.serve()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
