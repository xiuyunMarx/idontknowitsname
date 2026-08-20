# Source this to point the agent at a SWE-bench instance instead of at the host.
#
#   source swebench-env.sh astropy__astropy-12907
#   jac run orchestrator.jac --task "$(cat task.txt)" --repo-path "$CODEAGENT_REPO"
#
# Why it exists: the benchmark repositories carry pinned, compiled dependencies
# that only exist inside their instance image. Run the Verifying phase against
# the host interpreter and it fails on imports rather than on the bug -- the
# agent then spends its remaining steps trying to `pip install hypothesis`,
# which the allowlist refuses by design, and the run ends having verified
# nothing. See the block comment at the top of nodes/verify.jac.
#
# udocker rather than docker: this account is not in the `docker` group, so the
# daemon socket is closed to it. udocker needs neither daemon nor root, and it
# unpacks the image under PRoot -- so /testbed is an ordinary host directory and
# the agent's file tools reach it with no bind mount.

_swebench_env() {
    local instance="$1"
    if [ -z "$instance" ]; then
        echo "usage: source swebench-env.sh <instance-id>   e.g. astropy__astropy-12907" >&2
        return 2
    fi

    # The agent imports jaclang.byllm, which only the bundled jac carries; the
    # conda env ships byllm as a separate top-level package and that import
    # fails there. Pin the interpreter rather than trusting PATH order.
    local jac_bin="${CODEAGENT_JAC:-$HOME/.local/bin/jac}"
    if [ ! -x "$jac_bin" ]; then
        echo "swebench-env: no jac at $jac_bin (override with CODEAGENT_JAC)" >&2
        return 1
    fi

    # udocker lives in the jaseci conda env. It has to be on PATH, not just
    # resolvable here: _wrap_udocker resolves it with shutil.which in the parent
    # process, and child_env passes PATH through to the spawned command.
    local udocker_dir="${CODEAGENT_UDOCKER_BIN_DIR:-$HOME/miniconda3/envs/jaseci/bin}"
    case ":$PATH:" in
        *":$udocker_dir:"*) ;;
        *) PATH="$udocker_dir:$PATH" ;;
    esac
    export PATH
    if ! command -v udocker >/dev/null 2>&1; then
        echo "swebench-env: udocker not found under $udocker_dir" >&2
        return 1
    fi

    # astropy__astropy-12907 -> swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
    local slug image container
    slug=$(printf '%s' "$instance" | sed 's/__/_1776_/' | tr '[:upper:]' '[:lower:]')
    image="swebench/sweb.eval.x86_64.${slug}:latest"
    container="codeagent-${instance}"

    if ! udocker ps 2>/dev/null | grep -q "'${container}'"; then
        if ! udocker images 2>/dev/null | grep -q "^${image}"; then
            echo "swebench-env: image not pulled: $image" >&2
            echo "  udocker pull $image" >&2
            return 1
        fi
        echo "swebench-env: creating container $container (first time, ~1 min)..."
        udocker create --name="$container" "$image" >/dev/null || return 1
    fi

    local croot
    croot=$(udocker inspect -p "$container" 2>/dev/null | tail -1)
    if [ ! -d "$croot/testbed" ]; then
        echo "swebench-env: $container has no /testbed (croot=$croot)" >&2
        return 1
    fi

    export CODEAGENT_EXEC=udocker
    export CODEAGENT_EXEC_CONTAINER="$container"
    export CODEAGENT_EXEC_WORKDIR=/testbed
    # The host path of that same /testbed. Pass it as --repo-path: the file
    # tools read and write it directly, only run_command goes through udocker.
    export CODEAGENT_REPO="$croot/testbed"
    export CODEAGENT_JAC="$jac_bin"

    echo "instance    $instance"
    echo "container   $CODEAGENT_EXEC_CONTAINER"
    echo "repo        $CODEAGENT_REPO"
    echo "jac         $CODEAGENT_JAC"
    echo "model       ${CODEAGENT_MODEL:-gpt-5 (default)}"
    echo
    echo "run:  \"\$CODEAGENT_JAC\" run orchestrator.jac --task '<issue text>' --repo-path \"\$CODEAGENT_REPO\""
}

_swebench_env "$@"
