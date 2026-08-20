import os;
import tempfile;
import from nodes.common {
    safe_path, is_write_blocked, rel_to, clip, clip_ends, scrub,
    log_tool_call, get_tool_calls, reset_tool_log
}

test "safe_path accepts a path inside the root" {
    root_dir = os.path.realpath(tempfile.mkdtemp());
    inside = safe_path(root_dir, "a/b.txt");
    assert inside is not None;
    assert str(inside).startswith(root_dir);
}

test "safe_path rejects traversal and absolute escapes" {
    root_dir = os.path.realpath(tempfile.mkdtemp());
    assert safe_path(root_dir, "../../etc/passwd") is None;
    assert safe_path(root_dir, "/etc/passwd") is None;
    assert safe_path(root_dir, "a/../../escape") is None;
}

test "safe_path rejects a symlink pointing outside the root" {
    # A symlink created inside the root pointing outside it must be refused.
    # This is what realpath buys over abspath.
    root_dir = os.path.realpath(tempfile.mkdtemp());
    outside = os.path.realpath(tempfile.mkdtemp());
    link = os.path.join(root_dir, "link");
    os.symlink(outside, link);
    assert safe_path(root_dir, "link/secret.txt") is None;
}

test "safe_path rejects a sibling directory sharing the prefix" {
    # commonpath compares segments; startswith would accept the sibling.
    base = os.path.realpath(tempfile.mkdtemp());
    root_dir = os.path.join(base, "repo");
    evil = os.path.join(base, "repo-evil");
    os.makedirs(root_dir);
    os.makedirs(evil);
    assert safe_path(root_dir, os.path.join(evil, "x.txt")) is None;
}

test "is_write_blocked guards dot git and dot jac" {
    root_dir = os.path.realpath(tempfile.mkdtemp());
    git_cfg = safe_path(root_dir, ".git/config");
    assert git_cfg is not None;
    assert is_write_blocked(root_dir, str(git_cfg));
    ok_file = safe_path(root_dir, "nodes/x.jac");
    assert not is_write_blocked(root_dir, str(ok_file));
}

test "clip and clip_ends mark what they dropped" {
    short = "hello";
    assert clip(short, 100) == short;
    assert clip_ends(short, 100) == short;
    long_text = "x" * 500;
    clipped = clip(long_text, 100);
    assert len(clipped) < 500;
    assert "truncated" in clipped;
    both = clip_ends(long_text, 100);
    assert "omitted" in both;
    assert both.startswith("x");
    assert both.endswith("x");
}

test "scrub removes absolute paths and durations" {
    root_dir = os.path.realpath(tempfile.mkdtemp());
    text = f"ran in {root_dir}/nodes took 1.23s";
    out = scrub(text, root_dir);
    assert root_dir not in out;
    assert "X.XXs" in out;
}

test "tool call log records and resets" {
    reset_tool_log();
    log_tool_call("read_file", {"file_path": "a.jac"});
    log_tool_call("grep", {"pattern": "x"});
    calls = get_tool_calls();
    assert len(calls) == 2;
    assert calls[0].name == "read_file";
    assert calls[1].args["pattern"] == "x";
    reset_tool_log();
    assert len(get_tool_calls()) == 0;
}
