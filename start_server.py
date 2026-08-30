import argparse
import asyncio
import os
import signal
import time

from serve.controller import Controller

PORT = 8964          # fixed; benchmark/applications/*.jac hardcode the same (OpenAI-compatible HTTP)
CONTROL_PORT = 8965  # runtime switches: see Controller.control / send_requests.py --set

model_list = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
]


async def main(args) -> None:
    server = Controller(gpu_slots=args.gpu_slots, host_slots=args.host_slots, speculate=not args.no_speculate,
                        planner=args.planner)
    server.session_idle_s = args.session_idle
    if args.history:
        if os.path.exists(args.history):
            n = server.load_history(args.history)
            print(f"[server] loaded {n} program(s) from {args.history}", flush=True)
        else:
            server.history_path = args.history
    for model in model_list:
        await server.add_engine(model_name=model, gpu_memory_utilization=0.8, max_model_len=4096)
    trace = args.trace or f"results/trace-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    os.makedirs(os.path.dirname(trace) or ".", exist_ok=True)
    await server.listen("localhost", PORT, trace_path=trace)
    await server.control_listen("localhost", CONTROL_PORT)
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    loop.add_signal_handler(signal.SIGUSR1, lambda: print("[dump]\n" + server.dump(), flush=True))
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    print(f"[server] gpu_slots={args.gpu_slots} host_slots={args.host_slots} listening on {PORT} "
          f"trace={trace} history={args.history}", flush=True)
    await stop.wait()
    for user in list(server.sessions):
        server.close_session(user)
    if server.save_history():
        print(f"[server] history saved to {server.history_path}", flush=True)
    for eng in server.engine_pool.values():   # stop the vLLM engine processes, or the parent lingers
        try:
            eng.engine.shutdown()
        except Exception as e:
            print(f"[server] shutdown {eng.name}: {e!r}", flush=True)
    print("[server] exiting", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--host-slots", type=int, default=3)
    ap.add_argument("--no-speculate", action="store_true")
    ap.add_argument("--planner", default="fifo", choices=["fifo", "greedy", "dp"])
    ap.add_argument("--history", default=None, help="workflow JSON to load if present and to save on exit")
    ap.add_argument("--trace", default=None, help="per-request JSONL trace (default results/trace-<ts>.jsonl)")
    ap.add_argument("--session-idle", type=float, default=30.0, help="seconds after the last reply to close a session")
    asyncio.run(main(ap.parse_args()))
