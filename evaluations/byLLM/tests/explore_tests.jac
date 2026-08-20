import os;
import tempfile;
import from nodes.explore { ExploreCodeBase }
import from nodes.common { get_tool_calls, reset_tool_log }

glob PROJ: str = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."));


"""A repo holding one file too big to serve whole, and its line count.

Generated rather than pointed at a project file: which files sit either side of
the whole-file budget drifts every time one of them is edited, and a windowing
test that silently stops testing windowing is worse than no test.
"""
def big_repo(lines: int = 2000) -> tuple[str, int] {
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "big.txt"), "w") as f {
        f.write("\n".join([f"line {i:05d} " + "x" * 40 for i in range(1, lines + 1)]) + "\n");
    }
    return (tmp, lines);
}

test "read_file returns content with a range header" {
    ex = ExploreCodeBase(repo_root=PROJ);
    out = ex.read_file("nodes/explore.jac");
    assert isinstance(out, str);
    assert out.startswith("# nodes/explore.jac - lines 1-");
    assert "node ExploreCodeBase" in out;
}

test "read_file reports a missing file as actionable text" {
    # The regression test for the old print-and-return-"" bug: a failure must
    # reach the model as non-empty, actionable text.
    ex = ExploreCodeBase(repo_root=PROJ);
    out = ex.read_file("nodes/does_not_exist.jac");
    assert out.startswith("Error: ");
    assert len(out) > 20;
    assert "find_files" in out;
}

test "read tools refuse paths outside the root" {
    ex = ExploreCodeBase(repo_root=PROJ);
    assert ex.read_file("../../etc/passwd").startswith("BLOCKED: ");
    assert ex.read_file("/etc/passwd").startswith("BLOCKED: ");
    assert ex.ls_repo("..").startswith("BLOCKED: ");
}

test "read_file rejects a directory target" {
    ex = ExploreCodeBase(repo_root=PROJ);
    assert ex.read_file("nodes").startswith("Error: ");
    assert "ls_repo" in ex.read_file("nodes");
}

test "read_file describes an empty file instead of returning nothing" {
    # An empty file must not return "" -- the model cannot act on that.
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "empty.txt"), "w") as f {
        f.write("");
    }
    ex = ExploreCodeBase(repo_root=tmp);
    out = ex.read_file("empty.txt");
    assert out != "";
    assert "empty" in out;
}

test "read_file honours a line window on a file too big to serve whole" {
    (tmp, total) = big_repo();
    ex = ExploreCodeBase(repo_root=tmp);
    windowed = ex.read_file("big.txt", 2, 4);
    assert windowed.startswith(f"# big.txt - lines 2-4 of {total}");
    assert len(windowed.splitlines()) == 4;
    past = ex.read_file("big.txt", 100000);
    assert past.startswith("Error: ");
}

test "a file that fits comes back whole whatever window was asked for" {
    # Windowing a small file buys nothing and costs the loop guard its teeth:
    # the served-window memory keys on the range, so a dozen overlapping
    # windows of one small file are a dozen fresh reads and none of them a
    # repeat it can refuse. That is the shape the read_file scan loops took.
    ex = ExploreCodeBase(repo_root=PROJ);
    whole = ex.read_file("nodes/plan.jac");
    assert whole.startswith("# nodes/plan.jac - lines 1-116 of 116");
    ex.set_scope("second");
    assert ex.read_file("nodes/plan.jac", 40, 60).startswith(
        "# nodes/plan.jac - lines 1-116 of 116"
    );
    # A window past the end is still an error, not quietly forgiven.
    ex.set_scope("third");
    assert ex.read_file("nodes/plan.jac", 100000).startswith("Error: ");
}

test "ls_repo lists sorted entries" {
    ex = ExploreCodeBase(repo_root=PROJ);
    listing = ex.ls_repo("nodes");
    assert "explore.jac" in listing;
    assert "common.jac" in listing;
    # sorted order is load-bearing for benchmark determinism
    names = [n for n in listing.splitlines() if n.endswith(".jac")];
    assert names == sorted(names);
}

test "ls_repo hides skipped directories" {
    ex = ExploreCodeBase(repo_root=PROJ);
    listing = ex.ls_repo(".");
    assert "nodes/" in listing;
    assert ".jac/" not in listing;
}

test "ls_repo names an empty directory" {
    tmp = os.path.realpath(tempfile.mkdtemp());
    os.makedirs(os.path.join(tmp, "sub"));
    ex = ExploreCodeBase(repo_root=tmp);
    assert "empty directory" in ex.ls_repo("sub");
}

test "find_files locates by name and by path glob" {
    ex = ExploreCodeBase(repo_root=PROJ);
    found = ex.find_files("*.jac", ".");
    assert "nodes/verify.jac" in found;
    assert "nodes/plan.jac" in found;
    scoped = ex.find_files("nodes/*.jac", ".");
    assert "nodes/edit.jac" in scoped;
    assert ex.find_files("*.nosuchext", ".").startswith("No files matching");
}

test "grep finds a symbol across the tree" {
    ex = ExploreCodeBase(repo_root=PROJ);
    hits = ex.grep("node ExploreCodeBase", "nodes", "*.jac");
    assert "explore.jac:" in hits;
    assert isinstance(hits, str);
}

test "grep with no matches returns a sentence" {
    # No matches must be a sentence, never "[]" or "". Searched in a temp dir
    # so the pattern literal in this test file cannot match itself.
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "a.txt"), "w") as f {
        f.write("nothing interesting here\n");
    }
    ex = ExploreCodeBase(repo_root=tmp);
    out = ex.grep("absent_symbol", ".");
    assert out.startswith("No matches for");
    assert "searched 1 files" in out;
}

test "grep reports an invalid regex with a recovery hint" {
    ex = ExploreCodeBase(repo_root=PROJ);
    bad = ex.grep("def (", "nodes");
    assert bad.startswith("Error: invalid regular expression");
    assert "(?i)" in bad;
}

test "grep supports inline case insensitive flag" {
    # ignore_case was dropped in favour of the inline flag; prove it works.
    ex = ExploreCodeBase(repo_root=PROJ);
    hits = ex.grep("(?i)NODE EXPLORECODEBASE", "nodes", "*.jac");
    assert "explore.jac:" in hits;
}

test "grep emits repository relative paths" {
    # Every returned path is repo-relative, so output does not depend on where
    # the benchmark is checked out.
    ex = ExploreCodeBase(repo_root=PROJ);
    hits = ex.grep("import os", "nodes", "*.jac");
    for line in hits.splitlines() {
        assert not line.startswith("/");
    }
}


# --- served-window memory ----------------------------------------------------
#
# These cover the loop that cost a whole benchmark run: the model read one file,
# could not tell the result was the same one it already had, and read it again,
# 13 times over, until the phase budget ran out.

test "read_file answers a repeat with the next move, not the same bytes" {
    (tmp, total) = big_repo();
    ex = ExploreCodeBase(repo_root=tmp, scope="Exploring");
    first = ex.read_file("big.txt", 1, 20);
    assert first.startswith(f"# big.txt - lines 1-20 of {total}");
    again = ex.read_file("big.txt", 1, 20);
    assert again.startswith("Error: ");
    assert "already read" in again;
    # A refusal that names no next move is just a different dead end.
    assert "read_file(file_path='big.txt', start_line=21)" in again;
    assert "grep" in again;
}

test "a repeat of a fully read file is sent somewhere other than the file" {
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring");
    whole = ex.read_file("nodes/plan.jac");
    assert whole.startswith("# nodes/plan.jac - lines 1-116 of 116");
    again = ex.read_file("nodes/plan.jac");
    assert again.startswith("Error: ");
    assert "all 116 lines" in again;
    assert "grep" in again;
}

test "each phase gets its own served-window memory" {
    # Phases keep separate conversations, so a window Exploring read is genuinely
    # absent from Editing's messages and Editing must still be served it.
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring");
    assert ex.read_file("nodes/plan.jac").startswith("# nodes/plan.jac");
    ex.set_scope("Editing");
    out = ex.read_file("nodes/plan.jac");
    assert out.startswith("# nodes/plan.jac - lines 1-116 of 116");
}

test "writing the file re-opens it for reading" {
    # Editing reads, edits, then reads back to check the edit landed. The
    # size+mtime stamp in the key expires the entry on its own, so no tool has
    # to remember to invalidate anything.
    tmp = os.path.realpath(tempfile.mkdtemp());
    p = os.path.join(tmp, "a.txt");
    with open(p, "w") as f {
        f.write("one\ntwo\n");
    }
    ex = ExploreCodeBase(repo_root=tmp, scope="Editing");
    assert "one" in ex.read_file("a.txt");
    assert ex.read_file("a.txt").startswith("Error: ");
    with open(p, "w") as f {
        f.write("one\ntwo\nthree\n");
    }
    after = ex.read_file("a.txt");
    assert after.startswith("# a.txt - lines 1-3 of 3");
    assert "three" in after;
}

test "an oversized file pages forward on whole lines" {
    # The old behaviour cut at a character offset and reported "N more
    # characters", which names no line -- so no start_line resumed it and
    # repeating the identical call was the only move left.
    tmp = os.path.realpath(tempfile.mkdtemp());
    line = "x" * 200;
    with open(os.path.join(tmp, "big.txt"), "w") as f {
        f.write("\n".join([line for _ in range(400)]) + "\n");
    }
    ex = ExploreCodeBase(repo_root=tmp, scope="Exploring");
    page1 = ex.read_file("big.txt");
    assert "Continue with read_file(" in page1;
    # Every content line arrived whole: replace_in_file anchors on exact text.
    for ln in page1.splitlines()[1:] {
        assert ln == line or ln.startswith("... [stopped");
    }
    resume = int(page1.split("start_line=")[-1].strip().rstrip(")."));
    assert resume > 1;
    page2 = ex.read_file("big.txt", resume);
    assert page2.startswith(f"# big.txt - lines {resume}-");
}

test "a single over-budget line still advances" {
    # One line longer than the whole budget must not return an empty window
    # whose continuation hint points back at the call that produced it.
    tmp = os.path.realpath(tempfile.mkdtemp());
    with open(os.path.join(tmp, "wide.txt"), "w") as f {
        f.write(("y" * 60000) + "\nshort\n");
    }
    ex = ExploreCodeBase(repo_root=tmp, scope="Exploring");
    page1 = ex.read_file("wide.txt");
    assert page1.startswith("# wide.txt - lines 1-1 of 2");
    assert "start_line=2" in page1;
    assert ex.read_file("wide.txt", 2).startswith("# wide.txt - lines 2-2 of 2");
}

test "read_file logs the window it served, not just the path" {
    # Without start_line/end_line in the log, a run of paging calls and a run of
    # identical calls are the same sequence of entries -- which is how the loop
    # stayed invisible in the run log.
    reset_tool_log();
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring");
    ex.read_file("nodes/plan.jac", 5, 9);
    calls = get_tool_calls();
    assert len(calls) == 1;
    assert calls[0].args["file_path"] == "nodes/plan.jac";
    assert calls[0].args["start_line"] == "5";
    assert calls[0].args["end_line"] == "9";
    reset_tool_log();
}

test "read_file names the fix for a backwards window" {
    ex = ExploreCodeBase(repo_root=PROJ, scope="Exploring");
    out = ex.read_file("nodes/plan.jac", 20, 5);
    assert out.startswith("Error: ");
    assert "end_line" in out;
    assert "at least 20" in out;
}
