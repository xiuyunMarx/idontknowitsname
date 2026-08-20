"""Mutating tools: create or overwrite a whole file, or replace exact text in one.

Two tools rather than one because the jobs differ: write_file is for new files
and total rewrites, replace_in_file is for surgical changes that must not
re-emit the surrounding file. Replacement is literal-substring only -- no regex,
no fuzzy matching, which is how editors silently corrupt files.
"""

import os;
import re;
import from nodes.common {
    MAX_FILE_BYTES,
    safe_path, rel_to, refuse_path, is_write_blocked, log_tool_call
}


"""The first syntax error in `source`, as "line N: message", or "" if it parses.

Both parsers run in-process -- Python's own `compile`, and the Jac parser the
runtime already carries -- so this costs microseconds and cannot be mistaken
for running the code. Anything else is left alone: a file we cannot parse is
not a file we can call broken.

Worth doing on every write because the alternative is the model finding out
three tool calls later, through an import error, and then hunting for the line
by hand. In one measured astropy run that hunt was twenty-four consecutive
read_file calls walking backwards through a 4,473-line file one line at a time,
and it cost a quarter of the run's tokens.
"""
def syntax_error(source: str, rel: str) -> str {
    ext = os.path.splitext(rel)[1].lower();
    if ext == ".py" {
        try {
            compile(source, rel, "exec");
        } except SyntaxError as e {
            where = f"line {e.lineno}" if e.lineno else "unknown line";
            return f"{where}: {type(e).__name__}: {str(e.msg)}";
        } except ValueError as e {
            # A NUL byte, and anything else compile() rejects before parsing.
            return f"unparseable: {str(e)}";
        }
        return "";
    }
    if ext == ".jac" {
        import from jaclang.jac0core.parser { parse }
        import from jaclang.jac0core.program { JacProgram }
        prog = JacProgram();
        (_, had_error) = parse(source, rel, prog);
        if not had_error {
            return "";
        }
        alerts = prog.errors_had;
        if not alerts {
            return "it no longer parses as Jac";
        }
        line = getattr(getattr(alerts[0], "loc", None), "first_line", 0);
        where = f"line {line}" if line else "unknown line";
        return f"{where}: {str(alerts[0].msg)}";
    }
    return "";
}


node EditCode {
    has repo_root: str = "";

    def postinit {
        self.repo_root = os.path.realpath(self.repo_root or os.getcwd());
    }

    def write_file(file_path: str, content: str) -> str {
        log_tool_call("write_file", {"file_path": file_path});
        target = safe_path(self.repo_root, file_path);
        if target is None {
            return refuse_path(file_path);
        }
        rel = rel_to(self.repo_root, target);
        if is_write_blocked(self.repo_root, target) {
            return (
                f"BLOCKED: '{rel}' is inside a protected directory (.git or "
                ".jac) and cannot be written."
            );
        }
        if os.path.isdir(target) {
            return f"Error: '{rel}' is a directory, not a file.";
        }
        existed = os.path.isfile(target);
        old_lines = 0;
        if existed {
            probe = self._probe_existing(target, rel);
            if probe {
                return probe;
            }
            old_lines = self._line_count(target);
        }
        # LLM-authored content almost never ends in a newline. Without this,
        # every file the agent touches loses its trailing newline and every
        # later diff carries a spurious "\ No newline at end of file".
        body = content;
        if body and not body.endswith("\n") {
            body = body + "\n";
        }
        made_dir = "";
        parent = os.path.dirname(target);
        if parent and not os.path.isdir(parent) {
            try {
                os.makedirs(parent, exist_ok=True);
            } except OSError as e {
                return f"Error: could not create directory for '{rel}': {str(e)}";
            }
            made_dir = f" (created directory {rel_to(self.repo_root, parent)}/)";
        }
        try {
            with open(target, "w", encoding="utf-8", errors="strict", newline="") as f {
                f.write(body);
            }
        } except OSError as e {
            return f"Error: could not write '{rel}': {str(e)}";
        }
        new_lines = len(body.splitlines());
        n_bytes = len(body.encode("utf-8"));
        warn = self._syntax_warning(body, rel);
        if existed {
            # Naming the previous line count is the cheap substitute for a
            # read-before-write gate: clobbering 400 lines with 12 is then
            # visible in the transcript instead of silent.
            return (
                f"OK: Overwrote {rel} (was {old_lines} lines, now {new_lines} "
                f"lines, {n_bytes} bytes).{made_dir}{warn}"
            );
        }
        return f"OK: Created {rel} ({new_lines} lines, {n_bytes} bytes).{made_dir}{warn}";
    }

    def replace_in_file(file_path: str, old: str, new: str, expected_count: int = 1) -> str {
        log_tool_call("replace_in_file", {"file_path": file_path});
        # Checked before the occurrence count: "".count("") is 1 on an empty
        # file, so an empty `old` would pass a uniqueness test and then splice
        # `new` in at byte 0.
        if not old {
            return (
                "Error: 'old' must not be empty. Use write_file to create a "
                "file or replace it entirely."
            );
        }
        if old == new {
            return "Error: 'old' and 'new' are identical; nothing to do.";
        }
        target = safe_path(self.repo_root, file_path);
        if target is None {
            return refuse_path(file_path);
        }
        rel = rel_to(self.repo_root, target);
        if is_write_blocked(self.repo_root, target) {
            return (
                f"BLOCKED: '{rel}' is inside a protected directory (.git or "
                ".jac) and cannot be written."
            );
        }
        if not os.path.isfile(target) {
            return (
                f"Error: '{rel}' does not exist, so there is nothing to edit. "
                "Create it with write_file instead."
            );
        }
        try {
            with open(target, "r", encoding="utf-8", errors="strict", newline="") as f {
                original = f.read();
            }
        } except UnicodeDecodeError {
            return f"Error: '{rel}' is not a UTF-8 text file (it may be binary).";
        } except OSError as e {
            return f"Error: could not read '{rel}': {str(e)}";
        }
        had_crlf = "\r\n" in original;
        work = original.replace("\r\n", "\n");
        found = work.count(old);
        if found == 0 {
            return f"Error: no match for 'old' in {rel}. Re-read the file with read_file and copy the exact text, including indentation." + self._diagnose_miss(work, old, rel);
        }
        if found != expected_count {
            return (
                f"Error: found {found} occurrences of 'old' in {rel} but "
                f"expected_count={expected_count}; the file was not changed. "
                "Add surrounding lines to 'old' to make it unique, or set "
                f"expected_count={found} to replace all of them."
            );
        }
        updated = work.replace(old, new);
        line_no = work[:work.index(old)].count("\n") + 1;
        warn = self._syntax_warning(updated, rel);
        if had_crlf {
            updated = updated.replace("\n", "\r\n");
        }
        try {
            with open(target, "w", encoding="utf-8", errors="strict", newline="") as f {
                f.write(updated);
            }
        } except OSError as e {
            return f"Error: could not write '{rel}': {str(e)}";
        }
        total = len(updated.splitlines());
        delta = total - len(original.splitlines());
        sign = "+" if delta >= 0 else "";
        return (
            f"OK: replaced {found} occurrence(s) in {rel} at line {line_no}. "
            f"File is now {total} lines ({sign}{delta}).{warn}"
        );
    }

    # A write that breaks the parse is still a write: the bytes are on disk,
    # and reverting them behind the model's back would leave it reasoning about
    # a file that is not the one it has. What the write must not be is silent.
    def _syntax_warning(source: str, rel: str) -> str {
        broken = syntax_error(source, rel);
        if not broken {
            return "";
        }
        return (
            f"\nWARNING: {rel} no longer parses -- {broken}. The write went "
            "through, so nothing else will work until this is fixed: read that "
            "line, correct it, and do not run the tests before it parses."
        );
    }

    # Explain a near-miss. Whitespace drift between what the model remembers
    # and what is on disk is the dominant anchored-edit failure mode, and a
    # bare "not found" just makes the model retry the same string.
    def _diagnose_miss(content: str, old: str, rel: str) -> str {
        stripped = old.strip();
        if stripped and stripped in content {
            return (
                " Note: a near-match exists that differs only in leading or "
                "trailing whitespace -- trim 'old' and retry."
            );
        }
        old_first = "";
        for line in old.splitlines() {
            if line.strip() {
                old_first = re.sub(r"[ \t]+", " ", line).strip();
                break;
            }
        }
        if old_first {
            hits: list[str] = [];
            for (i, line) in enumerate(content.splitlines()) {
                if re.sub(r"[ \t]+", " ", line).strip() == old_first {
                    hits.append(str(i + 1));
                    if len(hits) >= 5 {
                        break;
                    }
                }
            }
            if hits {
                return (
                    f" Note: line {', '.join(hits)} of {rel} matches the first "
                    "line of 'old' after whitespace normalization -- your 'old' "
                    "differs in indentation or internal spacing."
                );
            }
        }
        return "";
    }

    # Refuse to write text over a binary file. Returns "" to allow the write.
    def _probe_existing(target: str, rel: str) -> str {
        try {
            with open(target, "rb") as f {
                head = f.read(8192);
            }
        } except OSError {
            return "";
        }
        if b"\x00" in head {
            return (
                f"Error: '{rel}' looks like a binary file; refusing to "
                "overwrite it with text."
            );
        }
        return "";
    }

    def _line_count(target: str) -> int {
        try {
            if os.path.getsize(target) > MAX_FILE_BYTES {
                return 0;
            }
            with open(target, "r", encoding="utf-8", errors="ignore", newline="") as f {
                return len(f.read().splitlines());
            }
        } except OSError {
            return 0;
        }
    }
}


sem EditCode = "Create, overwrite, and edit files in the repository.";

sem EditCode.write_file = "Create a new file, or replace an existing file's contents entirely. Missing parent directories are created automatically, and a trailing newline is added if the content lacks one. To change part of a file without re-sending the rest, use replace_in_file instead.";
sem EditCode.write_file.file_path = "Path of the file to write, relative to the repository root.";
sem EditCode.write_file.content = "The complete new contents of the file.";

sem EditCode.replace_in_file = "Replace an exact piece of text in a file. The text is matched literally, not as a regular expression. By default it must occur exactly once, so include enough surrounding context to make it unique. If the text is not found, or occurs a different number of times than expected, the file is left unchanged and the error explains what to do next.";
sem EditCode.replace_in_file.file_path = "Path of the file to edit, relative to the repository root.";
sem EditCode.replace_in_file.old = "The exact existing text to find, copied character for character from the file including its indentation.";
sem EditCode.replace_in_file.new = "The text to put in its place.";
sem EditCode.replace_in_file.expected_count = "How many occurrences of 'old' you expect to replace. Leave it at 1 to require a unique match; set it higher only to intentionally replace every occurrence.";
