import os;
import tempfile;
import from nodes.edit { EditCode, syntax_error }

# Every test writes into a throwaway root, never the real repository.
def fresh() -> str {
    return os.path.realpath(tempfile.mkdtemp());
}

def put(root_dir: str, name: str, body: str) -> str {
    path = os.path.join(root_dir, name);
    with open(path, "w", encoding="utf-8", newline="") as f {
        f.write(body);
    }
    return path;
}

def read_raw(path: str) -> str {
    with open(path, "r", encoding="utf-8", newline="") as f {
        return f.read();
    }
}

test "write_file creates a new file" {
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    out = ed.write_file("a.txt", "x\ny\n");
    assert out.startswith("OK: Created a.txt");
    assert "2 lines" in out;
    assert read_raw(os.path.join(tmp, "a.txt")) == "x\ny\n";
}

test "write_file names the previous line count when overwriting" {
    # Overwriting must name the previous line count, so a clobber is visible.
    tmp = fresh();
    put(tmp, "b.txt", "1\n2\n3\n4\n5\n");
    ed = EditCode(repo_root=tmp);
    out = ed.write_file("b.txt", "just one line");
    assert out.startswith("OK: Overwrote b.txt");
    assert "was 5 lines" in out;
    assert "now 1 lines" in out;
}

test "write_file adds a trailing newline" {
    # A trailing newline is added, so diffs do not churn.
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    ed.write_file("c.txt", "no trailing newline");
    assert read_raw(os.path.join(tmp, "c.txt")).endswith("\n");
}

test "write_file creates missing parent directories" {
    # Missing parent directories are created, since there is no mkdir tool.
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    out = ed.write_file("deep/nested/d.txt", "hi\n");
    assert out.startswith("OK: Created");
    assert "created directory" in out;
    assert os.path.isfile(os.path.join(tmp, "deep/nested/d.txt"));
}

test "write_file refuses to escape the root" {
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    out = ed.write_file("../escape.txt", "should not exist");
    assert out.startswith("BLOCKED: ");
    assert not os.path.exists(os.path.join(os.path.dirname(tmp), "escape.txt"));
}

test "write_file refuses protected directories" {
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    assert ed.write_file(".git/config", "[diff]\n").startswith("BLOCKED: ");
    assert ed.write_file(".jac/x", "y").startswith("BLOCKED: ");
    assert not os.path.exists(os.path.join(tmp, ".git/config"));
}

test "write_file refuses to overwrite a binary file" {
    # Text must not be written over a binary file.
    tmp = fresh();
    binpath = os.path.join(tmp, "img.png");
    with open(binpath, "wb") as f {
        f.write(b"\x89PNG\x00\x00binary");
    }
    ed = EditCode(repo_root=tmp);
    assert ed.write_file("img.png", "text").startswith("Error: ");
}

test "replace_in_file replaces a unique anchor" {
    tmp = fresh();
    put(tmp, "e.txt", "alpha\nbeta\ngamma\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("e.txt", "beta", "BETA");
    assert out.startswith("OK: replaced 1 occurrence(s) in e.txt at line 2.");
    assert read_raw(os.path.join(tmp, "e.txt")) == "alpha\nBETA\ngamma\n";
}

test "replace_in_file reports a missing anchor" {
    tmp = fresh();
    put(tmp, "f.txt", "alpha\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("f.txt", "nowhere", "x");
    assert out.startswith("Error: no match for 'old'");
    assert "read_file" in out;
    assert read_raw(os.path.join(tmp, "f.txt")) == "alpha\n";
}

test "replace_in_file hints at surrounding whitespace" {
    # Near-match by surrounding whitespace: old.strip() does occur.
    tmp = fresh();
    put(tmp, "g.txt", "alpha\nbeta\ngamma\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("g.txt", "   beta   ", "BETA");
    assert out.startswith("Error: no match for 'old'");
    assert "near-match" in out;
}

test "replace_in_file hints at internal spacing drift" {
    # Near-match by internal spacing: the file uses different spacing than the
    # model remembers. This is the dominant anchored-edit failure mode.
    tmp = fresh();
    put(tmp, "g2.txt", "def foo():\n    x  =  1\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("g2.txt", "x = 1", "x = 2");
    assert out.startswith("Error: no match for 'old'");
    assert "whitespace normalization" in out;
    assert "line 2" in out;
}

test "replace_in_file refuses a duplicated anchor and leaves the file intact" {
    # A duplicated anchor must refuse AND leave the file byte-identical.
    tmp = fresh();
    body = "x = 1\ny = 2\nx = 1\nz = 3\nx = 1\n";
    path = put(tmp, "h.txt", body);
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("h.txt", "x = 1", "x = 9");
    assert out.startswith("Error: found 3 occurrences");
    assert "expected_count=3" in out;
    assert read_raw(path) == body;
}

test "replace_in_file replaces all when expected count matches" {
    # ... and expected_count is the sanctioned escape hatch.
    tmp = fresh();
    path = put(tmp, "i.txt", "x = 1\ny = 2\nx = 1\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("i.txt", "x = 1", "x = 9", 2);
    assert out.startswith("OK: replaced 2 occurrence(s)");
    assert read_raw(path) == "x = 9\ny = 2\nx = 9\n";
}

test "replace_in_file refuses an empty anchor" {
    tmp = fresh();
    path = put(tmp, "j.txt", "hello\n");
    ed = EditCode(repo_root=tmp);
    # Empty `old` would otherwise splice at byte 0.
    out = ed.replace_in_file("j.txt", "", "PREFIX");
    assert out.startswith("Error: 'old' must not be empty");
    assert read_raw(path) == "hello\n";
}

test "replace_in_file refuses an empty anchor on an empty file" {
    tmp = fresh();
    path = put(tmp, "k.txt", "");
    ed = EditCode(repo_root=tmp);
    # "".count("") == 1, so an empty file is the case that would sneak past a
    # bare uniqueness check.
    out = ed.replace_in_file("k.txt", "", "PREFIX");
    assert out.startswith("Error: 'old' must not be empty");
    assert read_raw(path) == "";
}

test "replace_in_file refuses an identical replacement" {
    tmp = fresh();
    path = put(tmp, "l.txt", "same\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("l.txt", "same", "same");
    assert out.startswith("Error: 'old' and 'new' are identical");
    assert read_raw(path) == "same\n";
}

test "replace_in_file preserves CRLF line endings" {
    # CRLF normalizes in and is restored out, so the model's LF anchor matches
    # and the file's line endings survive.
    tmp = fresh();
    path = put(tmp, "m.txt", "alpha\r\nbeta\r\ngamma\r\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("m.txt", "beta", "BETA");
    assert out.startswith("OK: replaced 1");
    assert read_raw(path) == "alpha\r\nBETA\r\ngamma\r\n";
}

test "replace_in_file points at write_file for a missing file" {
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("missing.txt", "a", "b");
    assert out.startswith("Error: ");
    assert "write_file" in out;
}

test "replace_in_file refuses to escape the root" {
    tmp = fresh();
    put(tmp, "n.txt", "x\n");
    ed = EditCode(repo_root=tmp);
    assert ed.replace_in_file("../n.txt", "x", "y").startswith("BLOCKED: ");
}


# ---------------------------------------------------------------------------
# Post-edit syntax check. A broken write that says "OK" sends the model looking
# for the damage through an import error three calls later; in one measured
# astropy run that search was twenty-four consecutive read_file calls walking
# backwards through a 4,473-line file one line at a time.
# ---------------------------------------------------------------------------

test "syntax_error finds the line in Python and says nothing about clean code" {
    assert syntax_error("def f():\n    return 1\n", "a.py") == "";
    broken = syntax_error("def f():\nreturn 1\n", "a.py");
    assert broken.startswith("line 2: IndentationError");
    assert syntax_error("def f(:\n", "a.py").startswith("line 1: SyntaxError");
}

test "syntax_error finds the line in Jac" {
    assert syntax_error("with entry {\n    print(1);\n}\n", "a.jac") == "";
    broken = syntax_error("with entry {\n    print(1)\n}\n", "a.jac");
    assert broken.startswith("line 3");
    assert "';'" in broken;
}

test "syntax_error keeps quiet about files it cannot parse" {
    # Not parsing a format is not the same as calling it broken.
    assert syntax_error("this is not code at all {{{", "notes.md") == "";
    assert syntax_error("}{", "data.json") == "";
}

test "write_file warns when the write breaks the parse" {
    tmp = fresh();
    ed = EditCode(repo_root=tmp);
    out = ed.write_file("m.py", "def f():\nreturn 1\n");
    # The write is still a write -- reverting behind the model's back would
    # leave it reasoning about a file that is not the one on disk.
    assert out.startswith("OK: Created m.py");
    assert "WARNING" in out;
    assert "line 2" in out;
    assert read_raw(os.path.join(tmp, "m.py")) == "def f():\nreturn 1\n";
    # Clean code says nothing.
    assert "WARNING" not in ed.write_file("n.py", "def f():\n    return 1\n");
}

test "replace_in_file warns when the edit breaks the parse" {
    # The exact shape of the astropy run's failure: an anchored edit that
    # dedents a method body, reported as OK and found three calls later.
    tmp = fresh();
    put(tmp, "m.py", "class A:\n    def f(self):\n        return 1\n");
    ed = EditCode(repo_root=tmp);
    out = ed.replace_in_file("m.py", "        return 1", "    return 1");
    assert out.startswith("OK: replaced 1 occurrence(s)");
    assert "WARNING" in out;
    assert "m.py no longer parses" in out;
    # A sound edit to the same file says nothing.
    tmp2 = fresh();
    put(tmp2, "m.py", "class A:\n    def f(self):\n        return 1\n");
    ed2 = EditCode(repo_root=tmp2);
    assert "WARNING" not in ed2.replace_in_file("m.py", "return 1", "return 2");
}
