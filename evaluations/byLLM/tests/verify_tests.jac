import os;
import tempfile;
import time;
import from nodes.verify {
    VerifyCode, allowlist_text, first_subcommand, child_env, default_flags
}

glob PROJ: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."));

test "run_command reports a successful check" {
    v = VerifyCode(repo_root=PROJ);
    out = v.run_command("jac check nodes/explore.jac");
    assert out.startswith("$ jac check nodes/explore.jac");
    assert "exit_code: 0" in out;
    assert "--- stdout ---" in out;
    assert "--- stderr ---" in out;
}

# ---------------------------------------------------------------------------
# Default flags. A full traceback for a parametrised test that failed six ways
# is the largest single thing the agent puts in its own context window, and the
# phase conversation re-sends it on every later call -- so the terse form is
# the default and the frames are one word away.
# ---------------------------------------------------------------------------

test "pytest gets the terse defaults when the command did not decide" {
    assert default_flags("pytest", "pytest", ["tests/"]) == ["-q", "--tb=line"];
}

test "a command that decided keeps what it asked for" {
    # Long form beats the default, and does so by option name -- the value
    # differs, so a plain membership test would miss it.
    assert default_flags("pytest", "pytest", ["--tb=long"]) == ["-q"];
    assert default_flags("pytest", "pytest", ["--full-trace"]) == ["-q"];
    # Verbosity is the same argument in the other direction.
    assert default_flags("pytest", "pytest", ["-v"]) == ["--tb=line"];
    assert default_flags("pytest", "pytest", ["-q"]) == ["--tb=line"];
    assert default_flags("pytest", "pytest", ["-v", "--tb=short"]) == [];
}

test "nothing else gets pytest's defaults" {
    assert default_flags("jac", "jac test", ["tests/"]) == [];
    assert default_flags("git", "git diff", []) == [];
}

test "the echoed command is the one that actually ran" {
    # The header is what the model reasons from. Hiding `--tb=line` from it
    # would have the model read a one-line traceback as pytest having lost the
    # frames, rather than as a default it can override.
    v = VerifyCode(repo_root=PROJ);
    out = v.run_command("jac check nodes/explore.jac");
    # No path on the program even though the host resolved one.
    assert out.startswith("$ jac check nodes/explore.jac");
    boxed = VerifyCode(repo_root=PROJ, backend="docker", container="swe-probe");
    (screened, refusal) = boxed._screen("pytest tests/");
    assert refusal == "";
    assert screened == ["pytest", "tests/", "-q", "--tb=line", "-p",
                        "no:cacheprovider", "-p", "no:randomly", "--color=no"];
}

test "a failing command is data not a tool error" {
    # A failing command is DATA, not a tool error: no "Error:" prefix, so the
    # model fixes the code rather than trying to fix the tool.
    v = VerifyCode(repo_root=PROJ);
    out = v.run_command("jac check nodes/no_such_file.jac");
    assert "exit_code: 0" not in out;
    assert "Error: the command" not in out;
}

test "commands outside the allowlist are refused with the allowlist" {
    # Programs outside the allowlist, with the whole allowlist in the refusal.
    v = VerifyCode(repo_root=PROJ);
    for cmd in ["rm -rf /tmp/doomed", "pip install requests", "curl http://x", "bash -c ls"] {
        out = v.run_command(cmd);
        assert out.startswith("BLOCKED: ");
        assert "jac check" in out;
    }
}

test "shell metacharacters are refused" {
    # Shell metacharacters: chaining, pipes, redirection, substitution.
    v = VerifyCode(repo_root=PROJ);
    for cmd in [
        "git status; rm -rf /tmp/x",
        "jac check . && whoami",
        "jac check . | tee out.txt",
        "jac check > /tmp/out",
        "pytest $(whoami)"
    ] {
        assert v.run_command(cmd).startswith("BLOCKED: ");
    }
}

test "git stays read only" {
    # Disallowed subcommands of an allowed program: git stays read-only.
    v = VerifyCode(repo_root=PROJ);
    for cmd in ["git push", "git commit -m x", "git reset --hard", "git clean -fd"] {
        assert v.run_command(cmd).startswith("BLOCKED: ");
    }
}

test "argv zero must be a bare program name" {
    # argv[0] must be a bare name: no paths, no interpreter shims.
    v = VerifyCode(repo_root=PROJ);
    for cmd in ["/tmp/evil/jac check .", "./jac check .", "/usr/bin/env pytest", "~/jac check ."] {
        assert v.run_command(cmd).startswith("BLOCKED: ");
    }
}

test "git config injection flags are refused" {
    # `git -c core.pager=...` runs an arbitrary program. The subcommand must be
    # the first non-option token, which is what closes this.
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("git -c core.pager=whoami status").startswith("BLOCKED: ");
    assert v.run_command("git -C /home status").startswith("BLOCKED: ");
    assert v.run_command("git diff --ext-diff").startswith("BLOCKED: ");
}

test "destructive and interactive flags are refused" {
    # Flags that turn an allowed program destructive or interactive.
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("pytest --basetemp=/tmp/doomed").startswith("BLOCKED: ");
    assert v.run_command("pytest --pdb").startswith("BLOCKED: ");
    assert v.run_command("pytest -p evilplugin").startswith("BLOCKED: ");
    # --fix mutates source while posing as a read-only check.
    assert v.run_command("jac check --fix nodes/").startswith("BLOCKED: ");
    assert v.run_command("jac run --debug main.jac").startswith("BLOCKED: ");
}

test "jac x is never allowlisted" {
    # `jac x` runs any installed console script, so it is not allowlisted.
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("jac x pip install requests").startswith("BLOCKED: ");
}

test "arguments outside the repository are refused" {
    # cwd-pinning alone does not stop pytest importing code from outside.
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("pytest ../other_repo").startswith("BLOCKED: ");
    assert v.run_command("jac check /etc").startswith("BLOCKED: ");
}

test "jac run demands an explicit file" {
    # `jac run` with no file runs whatever jac.toml declares, possibly forever.
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("jac run").startswith("BLOCKED: ");
}

test "empty and unparseable commands are refused" {
    v = VerifyCode(repo_root=PROJ);
    assert v.run_command("").startswith("BLOCKED: ");
    assert v.run_command("   ").startswith("BLOCKED: ");
    assert v.run_command("jac check 'unbalanced").startswith("BLOCKED: ");
}

test "the allowlist renders deterministically" {
    # The allowlist renders deterministically, so the refusal text is stable
    # across runs -- it is part of the model's observation space.
    text = allowlist_text();
    assert text == allowlist_text();
    assert "git status" in text;
    assert "pytest" in text;
    assert "git push" not in text;
}

test "first_subcommand skips leading options" {
    assert first_subcommand(["--no-pager", "diff"]) == "diff";
    assert first_subcommand(["-q", "tests/"]) == "tests/";
    assert first_subcommand(["-c", "x"]) == "x";
    assert first_subcommand([]) == "";
}

test "the child environment drops injection vectors" {
    # The child env is built from a keep-list, so injection vectors are absent.
    env = child_env(PROJ);
    for leaked in [
        "PYTEST_ADDOPTS", "PYTHONPATH", "PYTHONBREAKPOINT", "LD_PRELOAD",
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GIT_EXTERNAL_DIFF"
    ] {
        assert leaked not in env;
    }
    assert env["PYTHONHASHSEED"] == "0";
    assert env["LC_ALL"] == "C.UTF-8";
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null";
    assert env["COLUMNS"] == "80";
    assert "PATH" in env;
}

test "a hang is killed at the deadline with partial output" {
    # A hang is killed at the deadline and reported WITH its partial output,
    # so the model does not simply re-run the identical command.
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "hang.jac"), "w") as f {
        f.write("import time;\nwith entry {\n    print(\"started\");\n    time.sleep(120);\n}\n");
    }
    v = VerifyCode(repo_root=tmp);
    began = time.time();
    out = v.run_command("jac run hang.jac", 5);
    elapsed = time.time() - began;
    assert out.startswith("$ jac run hang.jac");
    assert "Error: the command was killed after 5s" in out;
    assert "process group" in out;
    # Partial output is carried back, so the model does not simply re-run it.
    assert "Output captured before the kill:" in out;
    # The deadline is honoured rather than run to completion.
    assert elapsed < 60;
}


# --- The docker execution backend -------------------------------------------
# Constructed explicitly rather than through the environment, so these run the
# same way whether or not a docker daemon is reachable. Nothing below spawns a
# container: the contract worth pinning is that screening is unchanged and that
# the wrapper carries the deadline and the environment across the boundary.

test "screening is identical on both backends" {
    # The allowlist, the denied flags and the path confinement are the agent's
    # action space. Where a command runs must not widen any of them.
    boxed = VerifyCode(repo_root=PROJ, backend="docker", container="swe-probe");
    for cmd in [
        "rm -rf /tmp/doomed", "bash -c ls", "git push origin main",
        "git status; whoami", "pytest --pdb", "pytest ../other_repo",
        "/usr/bin/pytest", "jac run"
    ] {
        assert boxed.run_command(cmd).startswith("BLOCKED: ");
    }
}

test "an interpreter is available only inside a container" {
    # Django's suite runs through ./tests/runtests.py, so a Python workspace
    # needs an interpreter -- but only where execution is contained.
    local = VerifyCode(repo_root=PROJ);
    assert "python" not in local.allowed();
    assert local.run_command("python -c print(1)").startswith("BLOCKED: ");

    boxed = VerifyCode(repo_root=PROJ, backend="docker", container="swe-probe");
    assert "python" in boxed.allowed();
    assert "python" in allowlist_text(boxed.allowed());
    # Still not a way to get a prompt that would sit there until the deadline.
    assert boxed.run_command("python -i").startswith("BLOCKED: ");
}

test "the docker wrapper carries the deadline into the container" {
    boxed = VerifyCode(repo_root=PROJ, backend="docker", container="swe-probe");
    (screened, refusal) = boxed._screen("pytest -q");
    assert refusal == "";
    # Resolved by the container's PATH, not the host's: the whole point is to
    # get the pytest from the instance's conda env.
    assert screened[0] == "pytest";
    wrapped = boxed._wrap_docker(screened, 90);
    assert wrapped[1] == "exec";
    assert "--workdir" in wrapped;
    assert wrapped[wrapped.index("--workdir") + 1] == "/testbed";
    # The container name separates docker's own flags from the command, and the
    # deadline must land on the command side of it.
    cut = wrapped.index("swe-probe");
    assert wrapped[cut + 1 : cut + 4] == ["timeout", "--kill-after=5", "90"];
    assert wrapped[cut + 4 :] == ["pytest", "-q", "--tb=line", "-p",
                                  "no:cacheprovider", "-p", "no:randomly",
                                  "--color=no"];
    envs = [wrapped[i + 1] for i in range(len(wrapped)) if wrapped[i] == "--env"];
    # git refuses to touch a bind mount owned by another uid without this.
    assert "GIT_CONFIG_VALUE_0=/testbed" in envs;
    assert "PATH=" + boxed.container_path in envs;
    assert "HOME=/tmp" in envs;
}

test "the docker backend needs a container to run in" {
    orphan = VerifyCode(repo_root=PROJ, backend="docker");
    out = orphan.run_command("pytest -q");
    assert out.startswith("BLOCKED: ");
    assert "CODEAGENT_EXEC_CONTAINER" in out;
}

test "the docker CLI keeps the variables that find the daemon" {
    # A keep-list that drops DOCKER_HOST leaves the CLI unable to reach a
    # daemon that is not on the default socket, so every command would come
    # back as a spawn error rather than as a result about the code.
    had = os.environ.get("DOCKER_HOST");
    os.environ["DOCKER_HOST"] = "unix:///run/probe.sock";
    try {
        assert child_env(PROJ, "docker")["DOCKER_HOST"] == "unix:///run/probe.sock";
        assert "DOCKER_HOST" not in child_env(PROJ);
        for leaked in ["PYTEST_ADDOPTS", "PYTHONPATH", "OPENAI_API_KEY"] {
            assert leaked not in child_env(PROJ, "docker");
        }
    } finally {
        os.environ.pop("DOCKER_HOST", None);
        if had is not None {
            os.environ["DOCKER_HOST"] = had;
        }
    }
}


# --- The udocker execution backend ------------------------------------------
# Same contract as docker, spawned without a daemon or root. Screening must be
# untouched by it, or "where a command runs" would widen "what may run".

test "udocker is a container backend like docker" {
    boxed = VerifyCode(repo_root=PROJ, backend="udocker", container="swe-probe");
    assert boxed.in_container();
    assert "python" in boxed.allowed();
    assert boxed.denied_flags("python") == ["-i"];
}

test "screening is identical on the udocker backend" {
    boxed = VerifyCode(repo_root=PROJ, backend="udocker", container="swe-probe");
    for cmd in [
        "rm -rf /tmp/doomed", "bash -c ls", "git push origin main",
        "git status; whoami", "pytest --pdb", "pytest ../other_repo",
        "/usr/bin/pytest", "jac run", "python -i"
    ] {
        assert boxed.run_command(cmd).startswith("BLOCKED: ");
    }
}

test "the udocker wrapper carries the deadline into the container" {
    boxed = VerifyCode(repo_root=PROJ, backend="udocker", container="swe-probe");
    (screened, refusal) = boxed._screen("pytest -q");
    assert refusal == "";
    assert screened[0] == "pytest";
    wrapped = boxed._wrap_udocker(screened, 90);
    assert wrapped[1] == "run";
    # udocker spells these as single tokens, unlike docker's argv pairs.
    assert "--nobanner" in wrapped;
    assert "--workdir=/testbed" in wrapped;
    cut = wrapped.index("swe-probe");
    assert wrapped[cut + 1 : cut + 4] == ["timeout", "--kill-after=5", "90"];
    assert wrapped[cut + 4 :] == ["pytest", "-q", "--tb=line", "-p",
                                  "no:cacheprovider", "-p", "no:randomly",
                                  "--color=no"];
    assert "--env=GIT_CONFIG_VALUE_0=/testbed" in wrapped;
    assert "--env=HOME=/tmp" in wrapped;
}

test "the udocker backend needs a container to run in" {
    orphan = VerifyCode(repo_root=PROJ, backend="udocker");
    out = orphan.run_command("pytest -q");
    assert out.startswith("BLOCKED: ");
    assert "CODEAGENT_EXEC_CONTAINER" in out;
}

test "the udocker CLI keeps the variables that find its repository" {
    # UDOCKER_DIR is to udocker what DOCKER_HOST is to the docker CLI: drop it
    # and every command comes back as a spawn error rather than a result.
    had = os.environ.get("UDOCKER_DIR");
    os.environ["UDOCKER_DIR"] = "/probe/udocker";
    try {
        assert child_env(PROJ, "udocker")["UDOCKER_DIR"] == "/probe/udocker";
        assert "UDOCKER_DIR" not in child_env(PROJ);
        for leaked in ["PYTEST_ADDOPTS", "PYTHONPATH", "OPENAI_API_KEY"] {
            assert leaked not in child_env(PROJ, "udocker");
        }
    } finally {
        os.environ.pop("UDOCKER_DIR", None);
        if had is not None {
            os.environ["UDOCKER_DIR"] = had;
        }
    }
}
