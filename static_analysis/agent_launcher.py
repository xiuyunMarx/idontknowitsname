"""Agent launcher: analyze, register, run.

    python -m static_analysis.agent_launcher [--server HOST:PORT] [--out program.json]
                                             [--no-register] [--dry-run] <agent.jac> [agent args...]

1. Static analysis of the agent (analyzer.jac): call sites, constant prompt
   bytes, binding heterogeneity and lifetimes, the call-site graph.
2. Registration with the serving side: POST /v1/programs/register with the
   Program's wire form, keyed by the agent's absolute path. Every request the
   agent then sends carries {"callsite": {signature, lineno, file}}, which the
   server resolves against this registration.
3. Launch: `jac run <agent.jac> [args...]` as a child process, in the same
   environment the analysis ran in (module-level constants read from the
   environment are therefore the ones the agent will see).

The analysis is repeated on every launch: it is cheap, and it is the only way
the registration is guaranteed to match the source that runs.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SERVER = "localhost:8964"
REGISTER_PATH = "/v1/programs/register"


def analyze_program(agent: str) -> Dict[str, Any]:
    """Run analyzer.jac in-process and return the registration payload."""
    from jaclang import JacRuntime as Jac #type: ignore[import-not-found]
    (mod,) = Jac.jac_import("analyzer", base_path=HERE)
    sys.path.insert(0, HERE)
    from primitives import program_to_dict  # type: ignore[import-not-found]
    program = mod.analyze(agent)
    print(mod.describe(program), file=sys.stderr)
    return {"file": os.path.abspath(agent), "program": program_to_dict(program)}


def register(server: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST the program to the server. Raises on any failure: an unregistered
    program would make the server serve the agent as opaque text."""
    url = f"http://{server}{REGISTER_PATH}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}") from None
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} unreachable: {e.reason}") from None


def launch(agent: str, args: List[str]) -> int:
    cmd = ["jac", "run", agent, *args]
    print("[launcher] " + " ".join(cmd), file=sys.stderr, flush=True)
    return subprocess.run(cmd, cwd=os.path.dirname(agent) or None).returncode


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description="analyze, register, run a Jac agent")
    ap.add_argument("--server", default=DEFAULT_SERVER, metavar="HOST:PORT")
    ap.add_argument("--out", default="", metavar="FILE", help="also write the registration payload here")
    ap.add_argument("--no-register", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="analyze and register, do not start the agent")
    ap.add_argument("agent")
    ap.add_argument("agent_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    agent = os.path.abspath(a.agent)
    if not os.path.isfile(agent):
        print(f"no such file: {agent}", file=sys.stderr)
        return 2

    payload = analyze_program(agent)
    n = len(payload["program"]["sites"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[launcher] wrote {a.out}", file=sys.stderr)

    if a.no_register:
        print("[launcher] registration skipped (--no-register)", file=sys.stderr)
    else:
        reply = register(a.server, payload)
        print(f"[launcher] registered {n} call sites with {a.server}: {reply}", file=sys.stderr)

    if a.dry_run:
        print("[launcher] dry run: agent not started", file=sys.stderr)
        return 0
    return launch(agent, a.agent_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
