"""The agent's only execution surface.

Everything the agent can run passes through run_command, so there is exactly
one audit point, one allowlist, and one timeout.

This does NOT prevent arbitrary code execution, and must not be described as if
it did: `pytest` imports and runs every conftest.py it collects, and `jac run`
executes Jac the agent just wrote. Both are the point of the tool. What the
allowlist does guarantee is that execution happens in a subprocess that is
cwd-pinned to the repository, with a scrubbed deterministic environment, no
stdin, a hard wall-clock deadline enforced by a process-group kill, and bounded
captured output. It stops accidental destruction (rm -rf, pip install,
pytest --basetemp=$HOME) and hangs, and it keeps both frameworks in the
comparison to an identical action space. Real isolation needs a container.

Which is available two ways: CODEAGENT_EXEC=docker sends every command through
`docker exec` into a container the caller has already started, and
CODEAGENT_EXEC=udocker does the same through udocker, which needs no daemon and
no root. Both leave the screening above untouched. See the "Execution backend"
block below.
"""

import os;
import shlex;
import shutil;
import signal;
import tempfile;
import time;
import from subprocess { Popen, DEVNULL, TimeoutExpired }
import from nodes.common { MAX_TOOL_CHARS, clip_ends, scrub, log_tool_call }

# program -> allowed subcommands. An empty list means the program takes any
# arguments. list[str] rather than set[str] so the rendered allowlist in error
# messages and in the `sem` string is deterministic.
# `git` is read-only by construction: no add, commit, checkout, reset, clean.
# `jac x` is deliberately absent -- it runs any installed console script.
glob ALLOWED_COMMANDS: dict[str, list[str]] = {
    "jac": ["check", "test", "run", "fmt", "code"],
    "pytest": [],
    "git": ["status", "diff", "log", "show", "ls-files"]
};

# Flags that turn an allowed program into arbitrary execution, an interactive
# hang, or a destructive operation.
glob DENIED_FLAGS: dict[str, list[str]] = {
    "jac": ["--debug", "--fix"],
    "pytest": [
        "--pdb", "--trace", "-p", "--plugins", "-c", "--config-file",
        "--rootdir", "--confcutdir", "--basetemp", "--import-mode",
        "-n", "--numprocesses", "--dist", "--forked",
        "--lf", "--last-failed", "--ff", "--failed-first"
    ],
    "git": [
        "-c", "-C", "--exec-path", "--git-dir", "--work-tree", "--namespace",
        "--ext-diff", "--textconv"
    ]
};

# Flags we inject ourselves; never model-controlled. Keyed by program, or by
# "program subcommand" where the flag is subcommand-specific.
glob FORCED_FLAGS: dict[str, list[str]] = {
    "pytest": ["-p", "no:cacheprovider", "-p", "no:randomly", "--color=no"],
    "git diff": ["--no-ext-diff", "--no-textconv", "--no-color"],
    "git log": ["--no-color", "-n", "20"],
    "git show": ["--no-ext-diff", "--no-color"]
};

# Flags supplied only when the command did not already decide the matter.
# Unlike FORCED_FLAGS these are a default, not an override: `pytest --tb=long`
# keeps the frames it asked for.
#
# `--tb=line` is the one that pays. A parametrised test failing six ways prints
# six near-identical tracebacks -- in one measured astropy run, 4,500 of the
# 5,600 tokens in a single pytest result, at 53-78% textual similarity to each
# other -- and the phase conversation then re-sends the whole block on every
# later call. The one-line form keeps file, line, exception type and message
# for every failure, which is what the agent acts on, at a twelfth of the size.
glob DEFAULT_FLAGS: dict[str, list[str]] = {
    "pytest": ["-q", "--tb=line"]
};

# What counts as the command having already decided. Keyed by the default it
# suppresses; a token matches on the part before "=", so `--tb=long` clashes
# with `--tb=line` and `-v` clashes with `-q`.
glob DEFAULT_FLAG_ALIASES: dict[str, list[str]] = {
    "-q": ["-q", "-qq", "--quiet", "-v", "-vv", "-vvv", "--verbose"],
    "--tb=line": ["--tb", "--full-trace"]
};

glob SHELL_METACHARS: list[str] = ["|", "&", ";", ">", "<", "`", "$(", "\n", "\r"];

# --- Execution backend -------------------------------------------------------
#
# "local" spawns the command on this machine, cwd-pinned to repo_root. That is
# the default and the only backend the agent uses on its own.
#
# "docker" runs it inside an already-running container whose /testbed is a bind
# mount of repo_root, and is what the SWE-bench bridge selects. The benchmark's
# repositories carry pinned, compiled dependencies that only exist inside their
# instance image, so a local `pytest` there fails on imports rather than on the
# bug -- the agent's Verifying phase would be measuring the harness. Screening
# is backend-independent: the same allowlist, the same denied flags, the same
# path confinement run first, and only the spawn differs.
glob EXEC_LOCAL: str = "local";
glob EXEC_DOCKER: str = "docker";
# Same contract as "docker", spawned through udocker instead: a pure-userspace
# runtime that needs no daemon, no root and no setuid helpers, which is the only
# way to reach the instance images on a shared machine where you are not in the
# docker group. It runs the image under PRoot, so /testbed is an ordinary host
# directory and the agent's file tools reach it with no bind mount at all.
glob EXEC_UDOCKER: str = "udocker";

# Where the SWE-bench images keep the environment their tests need. The eval
# script in every instance does `conda activate testbed`; putting that env's bin
# first is the no-shell equivalent.
glob DEFAULT_CONTAINER_PATH: str = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/condabin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
);

# `docker exec` kills nothing when the CLI it fronts is killed, so the deadline
# has to travel into the container. coreutils `timeout` is present in every
# SWE-bench image and reports 124 on TERM, 137 when it had to escalate.
glob TIMEOUT_EXIT_CODES: list[int] = [124, 137];
glob KILL_GRACE_SEC: int = 5;
# Slack on the host-side deadline so the in-container timer is what fires.
glob DOCKER_OVERHEAD_SEC: int = 15;

# Interpreters are not in the base allowlist: `jac x` is excluded for the same
# reason, and the agent's own repositories are Jac. They are added back only
# under the docker backend, where execution is contained and where the
# benchmark's repositories need them -- Django's suite runs through
# ./tests/runtests.py, not pytest, and is a third of SWE-bench Lite.
glob WORKSPACE_COMMANDS: dict[str, list[str]] = {"python": [], "python3": []};

# Only the flag that turns the interpreter into a prompt. Nothing else an
# interpreter can be told to do is reachable from outside the container anyway,
# and a REPL waiting on a stdin that is /dev/null just burns the timeout.
glob WORKSPACE_DENIED_FLAGS: dict[str, list[str]] = {
    "python": ["-i"],
    "python3": ["-i"]
};

# SIGKILL is POSIX-only and typeshed guards it behind a platform check, so
# resolve both signals once here rather than at each call site.
glob SIG_TERM: int = int(getattr(signal, "SIGTERM", 15));
glob SIG_KILL: int = int(getattr(signal, "SIGKILL", 9));

glob DEFAULT_TIMEOUT_SEC: int = 120;
glob MAX_TIMEOUT_SEC: int = 600;
glob MAX_CAPTURE_BYTES: int = 4 * 1024 * 1024;
glob POLL_SEC: float = 0.1;


"""The defaults for `key`/`prog` that `rest` has not already settled."""
def default_flags(prog: str, key: str, rest: list[str]) -> list[str] {
    heads = [tok.split("=", 1)[0] for tok in rest];
    out: list[str] = [];
    for flag in DEFAULT_FLAGS.get(key, DEFAULT_FLAGS.get(prog, [])) {
        aliases = DEFAULT_FLAG_ALIASES.get(flag, [flag]);
        if not [h for h in heads if h in aliases] {
            out.append(flag);
        }
    }
    return out;
}


def allowlist_text(allowed: dict[str, list[str]] | None = None) -> str {
    table = ALLOWED_COMMANDS if allowed is None else allowed;
    parts: list[str] = [];
    for (prog, subs) in sorted(table.items()) {
        if subs {
            for sub in subs {
                parts.append(f"{prog} {sub}");
            }
        } else {
            parts.append(prog);
        }
    }
    return ", ".join(parts);
}


# The first token after the program that is not an option. Handles
# `git --no-pager diff` and `pytest -q tests/` correctly.
def first_subcommand(rest: list[str]) -> str {
    for tok in rest {
        if not tok.startswith("-") {
            return tok;
        }
    }
    return "";
}


# Child environment built from a keep-list rather than by subtracting from
# os.environ. That is what drops PYTEST_ADDOPTS (which injects arbitrary pytest
# flags and would bypass DENIED_FLAGS entirely), PYTHONPATH, PYTHONBREAKPOINT,
# LD_PRELOAD, and every *_API_KEY -- the last also stops a runaway test from
# burning the benchmark's model budget.
def child_env(repo_root: str, backend: str = EXEC_LOCAL) -> dict[str, str] {
    keep: list[str] = [
        "HOME", "USER", "LOGNAME", "TMPDIR",
        "VIRTUAL_ENV", "CONDA_PREFIX", "LD_LIBRARY_PATH"
    ];
    if backend == EXEC_DOCKER {
        # Under the docker backend this environment belongs to the `docker`
        # CLI, not to the command being run -- and the CLI cannot find the
        # daemon without these. The command's own environment is built by
        # container_env and passed over with -e.
        keep.extend([
            "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
            "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY", "XDG_RUNTIME_DIR"
        ]);
    }
    if backend == EXEC_UDOCKER {
        # Likewise for udocker: UDOCKER_DIR is where its image and container
        # repository lives, and without it the CLI looks under a $HOME that the
        # keep-list may not even carry.
        keep.extend(["UDOCKER_DIR", "UDOCKER_TARBALL", "PROOT_NO_SECCOMP"]);
    }
    env: dict[str, str] = {};
    for key in keep {
        val = os.environ.get(key);
        if val is not None {
            env[key] = val;
        }
    }
    # Drop any PATH entry inside the repo, so a binary the agent wrote cannot
    # be picked up by name.
    safe_dirs: list[str] = [];
    for d in os.environ.get("PATH", "").split(os.pathsep) {
        if d and not os.path.realpath(d).startswith(repo_root + os.sep) {
            safe_dirs.append(d);
        }
    }
    env["PATH"] = os.pathsep.join(safe_dirs);
    env["LC_ALL"] = "C.UTF-8";
    env["LANG"] = "C.UTF-8";
    env["PYTHONHASHSEED"] = "0";
    env["PYTHONDONTWRITEBYTECODE"] = "1";
    env["PYTHONUNBUFFERED"] = "1";
    env["PYTHONIOENCODING"] = "utf-8";
    env["COLUMNS"] = "80";
    env["LINES"] = "24";
    env["TERM"] = "dumb";
    env["NO_COLOR"] = "1";
    env["GIT_CONFIG_GLOBAL"] = "/dev/null";
    env["GIT_CONFIG_NOSYSTEM"] = "1";
    env["GIT_TERMINAL_PROMPT"] = "0";
    env["GIT_PAGER"] = "cat";
    env["GIT_OPTIONAL_LOCKS"] = "0";
    return env;
}


# The environment the command itself runs under inside the container. Same
# determinism settings as child_env, plus the two things a bind-mounted
# workspace breaks: HOME (the image's is root's, which a non-root exec cannot
# write) and git's ownership check (the mount is owned by the host user, so git
# calls /testbed "dubious" and refuses every command).
def container_env(workdir: str, path: str) -> dict[str, str] {
    return {
        "PATH": path,
        "HOME": "/tmp",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "COLUMNS": "80",
        "LINES": "24",
        "TERM": "dumb",
        "NO_COLOR": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": workdir,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_PAGER": "cat",
        "GIT_OPTIONAL_LOCKS": "0"
    };
}


node VerifyCode {
    has repo_root: str = "";
    # Read from the environment rather than passed in, because the only caller
    # is AgentRuntime.retarget, which knows about a repository and nothing about
    # how the benchmark chose to host it.
    has backend: str = "";
    has container: str = "";
    has workdir: str = "";
    has container_path: str = "";
    has container_user: str = "";
    has udocker_bin: str = "";

    def postinit {
        self.repo_root = os.path.realpath(self.repo_root or os.getcwd());
        self.backend = self.backend or os.environ.get("CODEAGENT_EXEC", EXEC_LOCAL);
        self.container = self.container or os.environ.get("CODEAGENT_EXEC_CONTAINER", "");
        self.workdir = self.workdir or os.environ.get("CODEAGENT_EXEC_WORKDIR", "/testbed");
        self.container_path = (
            self.container_path
            or os.environ.get("CODEAGENT_EXEC_PATH", DEFAULT_CONTAINER_PATH)
        );
        self.container_user = (
            self.container_user or os.environ.get("CODEAGENT_EXEC_USER", "")
        );
        self.udocker_bin = (
            self.udocker_bin or os.environ.get("CODEAGENT_UDOCKER", "udocker")
        );
    }

    # True for every backend that runs the command inside the instance image.
    # Both container backends widen the allowlist by an interpreter, require a
    # container to have been configured, and need the mount point scrubbed out
    # of captured output -- none of which depends on which runtime spawns it.
    def in_container -> bool {
        return self.backend in (EXEC_DOCKER, EXEC_UDOCKER);
    }

    # What run_command will accept, which widens by one entry under the docker
    # backend. Every message that names the allowlist renders it from here, so
    # the model is never told about a command this run cannot actually run.
    def allowed -> dict[str, list[str]] {
        if not self.in_container() {
            return ALLOWED_COMMANDS;
        }
        table: dict[str, list[str]] = dict(ALLOWED_COMMANDS);
        table.update(WORKSPACE_COMMANDS);
        return table;
    }

    def denied_flags(prog: str) -> list[str] {
        if self.in_container() and prog in WORKSPACE_DENIED_FLAGS {
            return WORKSPACE_DENIED_FLAGS[prog];
        }
        return DENIED_FLAGS.get(prog, []);
    }

    def run_command(command: str, timeout_sec: int = DEFAULT_TIMEOUT_SEC) -> str {
        log_tool_call("run_command", {"command": command});
        (argv, refusal) = self._screen(command);
        if refusal {
            return refusal;
        }
        # What actually ran, not what was asked for. `_screen` appends flags of
        # its own, and `--tb=line` visibly changes the shape of the output: a
        # model that cannot see the flag reads a one-line traceback as pytest
        # having lost the frames rather than as a default it is free to
        # override. Rendered here, before the backend wrapper goes on -- that
        # part is plumbing and belongs to nobody's reasoning.
        head = f"$ {shlex.join([os.path.basename(argv[0])] + argv[1:])}\n";
        budget = max(1, min(timeout_sec, MAX_TIMEOUT_SEC));
        deadline = budget;
        if self.backend == EXEC_DOCKER {
            argv = self._wrap_docker(argv, budget);
            deadline = budget + DOCKER_OVERHEAD_SEC;
        } elif self.backend == EXEC_UDOCKER {
            argv = self._wrap_udocker(argv, budget);
            deadline = budget + DOCKER_OVERHEAD_SEC;
        }
        (code, out, err, status) = self._spawn(argv, deadline);
        # `docker exec` outlives nothing it started, so a container-side timeout
        # arrives as an exit code rather than as a killed host process.
        if status == "ok" and self.in_container() and code in TIMEOUT_EXIT_CODES {
            status = "timeout";
        }
        body = (
            "--- stdout ---\n" + (self._present(out).strip() or "(empty)")
            + "\n--- stderr ---\n" + (self._present(err).strip() or "(empty)")
        );
        if status == "timeout" {
            # Two different mechanisms do the killing -- a process-group signal
            # here, coreutils `timeout` in the container -- and the model reads
            # this text as evidence, so it names the one that actually fired.
            killed = "its whole process group was terminated";
            if self.in_container() {
                killed = "it was terminated inside the workspace container";
            }
            return clip_ends(
                head
                + f"Error: the command was killed after {budget}s and "
                + killed + ". This usually means it waited on "
                + "input, on the network, or on a server that never exits.\n"
                + "Output captured before the kill:\n" + body,
                MAX_TOOL_CHARS
            );
        }
        if status == "output_limit" {
            return clip_ends(
                head
                + f"Error: the command produced more than {MAX_CAPTURE_BYTES} "
                + "bytes of output and was stopped. Narrow what it prints.\n"
                + body,
                MAX_TOOL_CHARS
            );
        }
        if status == "spawn_error" {
            return head + f"Error: could not start the command: {out}";
        }
        # A nonzero exit is DATA, not a tool failure -- no "Error:" prefix.
        # Prefixing it sends the model off trying to fix the tool instead of
        # fixing the code that made the tests fail.
        return clip_ends(head + f"exit_code: {code}\n" + body, MAX_TOOL_CHARS);
    }

    # Returns (argv, "") to allow, or ([], "BLOCKED: ...") to refuse.
    def _screen(command: str) -> tuple[list[str], str] {
        stripped = command.strip();
        if not stripped {
            return ([], "BLOCKED: empty command.");
        }
        for meta in SHELL_METACHARS {
            if meta in stripped {
                return ([], (
                    f"BLOCKED: '{meta}' is not supported. Run exactly one program "
                    "per call -- there is no shell, so pipes, redirection, "
                    "chaining and command substitution do not work."
                ));
            }
        }
        try {
            argv: list[str] = shlex.split(stripped);
        } except ValueError as e {
            return ([], f"BLOCKED: could not parse the command ({str(e)}). Check your quoting.");
        }
        if not argv {
            return ([], "BLOCKED: empty command.");
        }
        allowed = self.allowed();
        listing = allowlist_text(allowed);
        prog = argv[0];
        # A bare program name only. This is stricter than basename(), which
        # would happily accept /tmp/attacker/git, and it also rejects the
        # interpreter shims: /usr/bin/env pytest, ./jac, sh -c, nohup.
        if os.sep in prog or prog.startswith(".") or prog.startswith("~") {
            return ([], (
                f"BLOCKED: '{prog}' must be a bare program name with no path. "
                f"Allowed commands: {listing}."
            ));
        }
        if prog not in allowed {
            return ([], (
                f"BLOCKED: command not allowed: '{prog}'. "
                f"run_command accepts exactly: {listing}. "
                "To change code, use write_file or replace_in_file instead."
            ));
        }
        subs = allowed[prog];
        rest = argv[1:];
        key = prog;
        if subs {
            sub = first_subcommand(rest);
            if not sub {
                return ([], f"BLOCKED: '{prog}' requires a subcommand. Allowed: {listing}.");
            }
            if sub not in subs {
                return ([], (
                    f"BLOCKED: '{prog} {sub}' is not allowed. "
                    f"Allowed: {listing}."
                ));
            }
            key = f"{prog} {sub}";
        }
        denied = self.denied_flags(prog);
        for tok in rest {
            bare = tok.split("=")[0];
            if bare in denied {
                return ([], (
                    f"BLOCKED: the flag '{bare}' is not allowed for '{prog}'."
                ));
            }
        }
        # Non-flag arguments must stay inside the repository. cwd-pinning alone
        # does not stop `pytest ../other_repo`, which would import and execute
        # code from outside the tree.
        for tok in rest {
            if tok.startswith("-") {
                continue;
            }
            if os.path.isabs(tok) or tok.split(os.sep)[0] == ".." {
                return ([], (
                    f"BLOCKED: the path '{tok}' is outside the repository. "
                    "Use paths relative to the repository root."
                ));
            }
        }
        # `jac run` with no file runs whatever jac.toml declares, which for a
        # servable project never returns.
        if key == "jac run" {
            files = [a for a in rest if not a.startswith("-") and a != "run"];
            if not files {
                return ([], (
                    "BLOCKED: 'jac run' needs an explicit file, e.g. "
                    "'jac run main.jac'."
                ));
            }
        }
        forced = default_flags(prog, key, rest);
        forced.extend(FORCED_FLAGS.get(key, FORCED_FLAGS.get(prog, [])));
        if self.in_container() {
            # Resolution belongs to the container's PATH: `pytest` here is the
            # one in the instance's conda env, which is exactly the point, and
            # host resolution would refuse a program the host never installed.
            if not self.container {
                return ([], (
                    "BLOCKED: no workspace container is configured, so nothing "
                    "can be run. Set CODEAGENT_EXEC_CONTAINER, or unset "
                    "CODEAGENT_EXEC to run commands locally."
                ));
            }
            bare: list[str] = [prog];
            bare.extend(rest);
            bare.extend(forced);
            return (bare, "");
        }
        resolved = shutil.which(prog);
        if resolved is None {
            return ([], f"BLOCKED: '{prog}' is not installed in this environment.");
        }
        real = os.path.realpath(resolved);
        if real.startswith(self.repo_root + os.sep) {
            return ([], f"BLOCKED: refusing to execute '{prog}' from inside the repository.");
        }
        final: list[str] = [real];
        final.extend(rest);
        final.extend(forced);
        return (final, "");
    }

    # Wrap a screened command so it runs in the workspace container. The host
    # process this produces is the `docker` CLI; the deadline is carried inside
    # by coreutils `timeout`, because killing the CLI leaves the command running.
    def _wrap_docker(argv: list[str], budget: int) -> list[str] {
        docker_bin = shutil.which("docker") or "docker";
        wrapped: list[str] = [docker_bin, "exec", "--workdir", self.workdir];
        if self.container_user {
            wrapped.extend(["--user", self.container_user]);
        }
        for (key, val) in sorted(container_env(self.workdir, self.container_path).items()) {
            wrapped.extend(["--env", f"{key}={val}"]);
        }
        wrapped.append(self.container);
        wrapped.extend(["timeout", f"--kill-after={KILL_GRACE_SEC}", str(budget)]);
        wrapped.extend(argv);
        return wrapped;
    }
    
    def _wrap_udocker(argv: list[str], budget: int) -> list[str] {
        udocker_bin = shutil.which(self.udocker_bin) or self.udocker_bin;
        wrapped: list[str] = [
            udocker_bin, "run", "--nobanner", f"--workdir={self.workdir}"
        ];
        if self.container_user {
            wrapped.append(f"--user={self.container_user}");
        }
        for (key, val) in sorted(container_env(self.workdir, self.container_path).items()) {
            wrapped.append(f"--env={key}={val}");
        }
        wrapped.append(self.container);
        wrapped.extend(["timeout", f"--kill-after={KILL_GRACE_SEC}", str(budget)]);
        wrapped.extend(argv);
        return wrapped;
    }

    # Captured output is quoted back to the model and stored in the run log, so
    # it must not carry anything that differs between machines or between runs.
    # Under the docker backend the repository answers to two names -- its host
    # path and its mount point -- and only the first is what scrub knows about.
    def _present(text: str) -> str {
        out = scrub(text, self.repo_root);
        if self.in_container() and self.workdir {
            out = out.replace(self.workdir + "/", "").replace(self.workdir, ".");
        }
        return out;
    }

    # Returns (exit_code, stdout, stderr, status) with status one of
    # "ok" | "timeout" | "output_limit" | "spawn_error".
    def _spawn(argv: list[str], budget: int) -> tuple[int, str, str, str] {
        # Temp files rather than pipes. capture_output=True buffers unbounded
        # output into RAM, and a grandchild that inherits the stdout pipe makes
        # communicate() block to the deadline even when the command itself
        # finished -- reporting a timeout with no output at all.
        out_f = tempfile.TemporaryFile(mode="w+b");
        err_f = tempfile.TemporaryFile(mode="w+b");
        status = "ok";
        code = -1;
        try {
            try {
                # start_new_session makes the child a process-group leader, so
                # killpg later takes its descendants with it.
                proc = Popen(
                    argv,
                    cwd=self.repo_root,
                    env=child_env(self.repo_root, self.backend),
                    stdin=DEVNULL,
                    stdout=out_f,
                    stderr=err_f,
                    start_new_session=True,
                    close_fds=True
                );
            } except OSError as e {
                return (-1, str(e), "", "spawn_error");
            }
            deadline = time.monotonic() + budget;
            while True {
                rc = proc.poll();
                if rc is not None {
                    code = rc;
                    break;
                }
                if time.monotonic() > deadline {
                    status = "timeout";
                    break;
                }
                if out_f.tell() + err_f.tell() > MAX_CAPTURE_BYTES {
                    status = "output_limit";
                    break;
                }
                time.sleep(POLL_SEC);
            }
            # Unconditional, including on a clean exit: a test that returns 0
            # while a fixture leaked a background process is exactly the case
            # that strands a port into the next task.
            self._kill_group(proc);
            out_f.seek(0);
            err_f.seek(0);
            out = out_f.read().decode("utf-8", errors="replace");
            err = err_f.read().decode("utf-8", errors="replace");
        } finally {
            out_f.close();
            err_f.close();
        }
        return (code, out, err, status);
    }

    def _kill_group(proc: Popen[bytes]) {
        try {
            pgid = os.getpgid(proc.pid);
        } except OSError {
            return;
        }
        try {
            os.killpg(pgid, SIG_TERM);
        } except OSError {
            return;
        }
        try {
            proc.wait(timeout=2.0);
            return;
        } except TimeoutExpired { }
        try {
            os.killpg(pgid, SIG_KILL);
            proc.wait(timeout=2.0);
        } except (OSError, TimeoutExpired) { }
    }
}


sem VerifyCode = "Runs toolchain commands to verify that code actually works.";

sem VerifyCode.run_command = "Run one allowed command in the repository and return its exit code and output. Use 'jac check <path>' to type-check Jac code after editing it, 'jac test <path>' or 'pytest' to run tests, and 'git status' or 'git diff' to review changes. A non-zero exit code is a real result about the code, not a failure of this tool. The commands always accepted are: jac check, jac test, jac run, jac fmt, jac code, pytest, git status, git diff, git log, git show, git ls-files; some workspaces additionally allow an interpreter such as 'python', and a refusal always names the exact list this workspace accepts. If a runner is not installed (exit 127 or 'No such file or directory'), do not conclude the tests cannot run: look for the runner script the repository itself ships and invoke it through the interpreter, e.g. 'python tests/runtests.py <module>' or 'python bin/test <path>'. There is no shell, so pipes, redirection, chaining with ';' or '&&', and command substitution do not work, and wildcards are NOT expanded -- write 'jac check nodes/' to check a directory rather than 'jac check nodes/*.jac'.";
sem VerifyCode.run_command.command = "The single command line to run, for example 'jac check nodes/explore.jac'. Paths must be relative to the repository root.";
sem VerifyCode.run_command.timeout_sec = "How many seconds to allow before the command is killed. Use the default of 120 unless you expect a long test run.";
