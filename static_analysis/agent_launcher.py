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


CACHE_DIR = os.path.join(HERE, ".cache")


def _cache_key(agent: str) -> str:
    """The analysis is a pure function of the agent's source and the analyzer's
    own code: hash both, so an edit to either invalidates the cached result."""
    import hashlib
    h = hashlib.sha256()
    for path in (agent, os.path.join(HERE, "analyzer.jac"), os.path.join(HERE, "primitives.py")):
        with open(path, "rb") as f:
            h.update(f.read())
        h.update(b"\0")
    h.update(os.path.abspath(agent).encode())
    return h.hexdigest()[:16]


def analyze_program(agent: str, use_cache: bool = True) -> Dict[str, Any]:
    """The registration payload of an agent: from the content-hash cache when the
    source and the analyzer are unchanged, else by running analyzer.jac in-process
    (a few seconds once jaclang's own modules are compiled, ~20 s the first time)."""
    agent = os.path.abspath(agent)
    cache = os.path.join(CACHE_DIR, f"{os.path.splitext(os.path.basename(agent))[0]}-{_cache_key(agent)}.json")
    if use_cache and os.path.isfile(cache):
        with open(cache) as f:
            payload = json.load(f)
        print(f"[launcher] analysis from cache {cache}", file=sys.stderr)
        return payload
    from jaclang import JacRuntime as Jac  # type: ignore[import-not-found]
    (mod,) = Jac.jac_import("analyzer", base_path=HERE)
    sys.path.insert(0, HERE)
    from primitives import program_to_dict  # type: ignore[import-not-found]
    program = mod.analyze(agent)
    print(mod.describe(program), file=sys.stderr)
    payload = {"file": agent, "program": program_to_dict(program)}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(payload, f)
    return payload


def register(server: str, payload: Dict[str, Any], no_header_last: bool = False) -> Dict[str, Any]:
    """POST the program to the server. Raises on any failure: an unregistered
    program would make the server serve the agent as opaque text. `no_header_last`
    asks the server to keep this program's headers in front of the values
    (fact_check: header-last changes its output with the current model)."""
    url = f"http://{server}{REGISTER_PATH}"
    data = json.dumps(dict(payload, no_header_last=bool(no_header_last))).encode("utf-8")
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
    ap.add_argument("--no-cache", action="store_true", help="re-run the analysis even when a cached result matches")
    ap.add_argument("--no-header-last", action="store_true", help="register the program with its headers kept in front of the values")
    ap.add_argument("agent")
    ap.add_argument("agent_args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    agent = os.path.abspath(a.agent)
    if not os.path.isfile(agent):
        print(f"no such file: {agent}", file=sys.stderr)
        return 2

    payload = analyze_program(agent, use_cache=not a.no_cache)
    n = len(payload["program"]["sites"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[launcher] wrote {a.out}", file=sys.stderr)

    if a.no_register:
        print("[launcher] registration skipped (--no-register)", file=sys.stderr)
    else:
        reply = register(a.server, payload, no_header_last=a.no_header_last)
        print(f"[launcher] registered {n} call sites with {a.server}: {reply}", file=sys.stderr)

    if a.dry_run:
        print("[launcher] dry run: agent not started", file=sys.stderr)
        return 0
    return launch(agent, a.agent_args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
