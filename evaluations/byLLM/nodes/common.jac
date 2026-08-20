"""Shared foundation for the coding agent's LLM-facing tools.

Every tool in nodes/ returns `str`. byLLM's `normalize_tool_result` stringifies
anything that is not already a str, so a `list[str]` would reach the model as a
Python repr and an empty one as the literal "[]" -- indistinguishable from a
broken tool.

The string sub-contract, used uniformly by every tool:

  read success      the content, led by one context line
                    "# nodes/edit.jac - lines 1-54 of 54"
  mutation success  starts with "OK: ", naming what changed and by how much
  recoverable error starts with "Error: " and always names the next action
  policy refusal    starts with "BLOCKED: " -- a distinct prefix, because
                    "Error" means "try again differently" while "BLOCKED"
                    means "this class of action is unavailable, stop trying"

Tools never raise, never print, and never return "". byLLM catches exceptions
but logs a full traceback with absolute paths to stderr, which pollutes the
benchmark log; and an empty return is the classic ReAct-loop trigger.
"""

import os;
import re;

# Directories never worth walking in a source repo.
glob SKIP_DIRS: set[str] = {
    ".git", ".jac", ".venv", "venv", "env", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build"
};

# Subtrees no tool may ever write into. `.git/config` accepts `[diff] external`,
# which would turn a later `git diff` into arbitrary code execution.
glob NO_WRITE_DIRS: set[str] = {".git", ".jac"};

glob MAX_FILE_BYTES: int = 2 * 1024 * 1024;
glob MAX_LINE_CHARS: int = 400;
glob MAX_TOOL_CHARS: int = 20000;
glob MAX_GREP_MATCHES: int = 100;
glob MAX_LIST_ENTRIES: int = 300;

# Ordered record of every tool call, read by the eval harness. Mirrors the
# `tool_call_log` carried by every sibling byLLM implementation's tools.jac.
obj ToolCall {
    has name: str;
    has args: dict[str, str] = {};
}

glob tool_call_log: list[ToolCall] = [];


def log_tool_call(name: str, args: dict[str, str]) {
    tool_call_log.append(ToolCall(name=name, args=args));
}


def get_tool_calls() -> list[ToolCall] {
    return list(tool_call_log);
}


def reset_tool_log() {
    tool_call_log.clear();
}


"""Resolve `rel_path` under `repo_root`, or None if it escapes."""
def safe_path(repo_root: str, rel_path: str) -> str | None {
    base = os.path.realpath(repo_root or os.getcwd());
    joined = rel_path if os.path.isabs(rel_path) else os.path.join(base, rel_path);
    # realpath, not abspath: abspath normalizes `..` textually and would let a
    # symlink inside the repo pointing at /etc through. realpath also resolves
    # the existing prefix of a not-yet-created path, so new-file writes confine.
    cand = os.path.realpath(joined);
    if cand == base {
        return cand;
    }
    try {
        # commonpath compares path segments; startswith would accept a sibling
        # directory such as `<root>-evil`.
        if os.path.commonpath([base, cand]) != base {
            return None;
        }
    } except ValueError {
        # Raised on mixed absolute/relative input, or mixed drives on Windows.
        return None;
    }
    return cand;
}


"""True if `abs_path` lies in a subtree no tool may write to."""
def is_write_blocked(repo_root: str, abs_path: str) -> bool {
    base = os.path.realpath(repo_root or os.getcwd());
    rel = os.path.relpath(abs_path, base);
    first = rel.split(os.sep)[0];
    return first in NO_WRITE_DIRS;
}


"""Repo-relative form of `abs_path`, for output the model reads."""
def rel_to(repo_root: str, abs_path: str) -> str {
    base = os.path.realpath(repo_root or os.getcwd());
    try {
        return os.path.relpath(abs_path, base);
    } except ValueError {
        return abs_path;
    }
}


def refuse_path(path: str) -> str {
    return (
        f"BLOCKED: path '{path}' resolves outside the repository root and was "
        "refused. Use a path relative to the repository root."
    );
}


"""Keep the head of `text`; mark what was dropped."""
def clip(text: str, limit: int) -> str {
    if len(text) <= limit {
        return text;
    }
    dropped = len(text) - limit;
    return text[:limit] + f"\n... [truncated, {dropped} more characters]";
}


"""Keep the head AND tail of `text`.

Command output puts the verdict at the end and the traceback at the start,
so head-only truncation systematically discards the line that matters.
"""
def clip_ends(text: str, limit: int) -> str {
    if len(text) <= limit {
        return text;
    }
    half = limit // 2;
    omitted = len(text) - (2 * half);
    return text[:half] + f"\n... [{omitted} characters omitted] ...\n" + text[-half:];
}


"""Strip machine-specific detail from captured output.

Absolute paths, the home directory and wall-clock durations all differ
between runs and between machines; leaving them in would make the A/B
comparison sensitive to where the benchmark happens to be checked out.
"""
def scrub(text: str, repo_root: str) -> str {
    base = os.path.realpath(repo_root or os.getcwd());
    out = text.replace(base + os.sep, "").replace(base, ".");
    home = os.path.expanduser("~");
    if home and home != "/" {
        out = out.replace(home, "~");
    }
    out = re.sub(r"\b\d+\.\d+s\b", "X.XXs", out);
    out = re.sub(r"(?m)^(rootdir|cachedir|plugins|platform):.*$", r"\1: <scrubbed>", out);
    return out;
}
