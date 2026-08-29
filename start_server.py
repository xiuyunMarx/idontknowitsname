import argparse
import asyncio
import signal

from serve.controller import Controller

PORT = 8964          # fixed; benchmark/applications/*.jac hardcode the same
CONTROL_PORT = 8965  # runtime switches: see Controller.control / send_requests.py --set

programs = [
    "benchmark/applications/cascade.jac",
    "benchmark/applications/deep_research.jac",
    "benchmark/applications/hover.jac",
    "benchmark/applications/rag_qa.jac",
    "benchmark/applications/text2sql.jac",
    "benchmark/applications/triage.jac",
]

model_list = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-4B",
]


async def main(args) -> None:
    server = Controller(gpu_slots=args.gpu_slots, host_slots=args.host_slots, speculate=not args.no_speculate)
    for program in programs:
        server.register_program(path=program)
    for model in model_list:
        await server.add_engine(model_name=model, gpu_memory_utilization=0.8, max_model_len=4096)
    await server.listen("localhost", PORT)
    await server.control_listen("localhost", CONTROL_PORT)
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, lambda: print("[dump]\n" + server.dump(), flush=True))
    print(f"[server] gpu_slots={args.gpu_slots} host_slots={args.host_slots} listening on {PORT}", flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-slots", type=int, default=1)
    ap.add_argument("--host-slots", type=int, default=3)
    ap.add_argument("--no-speculate", action="store_true")
    asyncio.run(main(ap.parse_args()))
