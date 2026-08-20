"""Read-only tools: orient in the repo, locate files, search their contents."""

import os;
import re;
import from fnmatch { fnmatch }
import from nodes.common {
    SKIP_DIRS, MAX_FILE_BYTES, MAX_LINE_CHARS, MAX_TOOL_CHARS,
    MAX_GREP_MATCHES, MAX_LIST_ENTRIES,
    safe_path, rel_to, refuse_path, clip, log_tool_call
}


# How many served windows to remember. Comfortably more than one run can issue,
# so the cap is a leak guard rather than a policy.
glob MAX_SERVED_WINDOWS: int = 128;


node ExploreCodeBase {
    # repo_root is node state, never a tool parameter: a parameter would be
    # model-controlled, which is no confinement at all. Node `has` fields do
    # not enter the tool JSON schema, so the model cannot address it.
    has repo_root: str = "";
    # Which phase is asking, and the windows already handed to it. Node state
    # for the same reason as repo_root: a model that can clear its own loop
    # memory has no loop memory.
    #
    # Scoped per phase because each phase keeps its own conversation. A window
    # Exploring read is absent from Editing's messages, so Editing must be
    # served it; a second identical read inside one phase is not memory, it is
    # a loop, and that is what gets rejected.
    has scope: str = "";
    has served: list[str] = [];

    def postinit {
        self.repo_root = os.path.realpath(self.repo_root or os.getcwd());
    }

    # Called by Phase.work at the top of every phase visit. Never a tool: only
    # the functions handed to `expose` become callable by the model.
    def set_scope(scope: str) {
        self.scope = scope;
    }

    def read_file(file_path: str, start_line: int = 1, end_line: int = 0) -> str {
        # The window is logged with the path. Without it a run of paging calls
        # and a run of identical calls are indistinguishable in the log, which
        # is the one distinction anyone reading it for a loop needs.
        log_tool_call("read_file", {
            "file_path": file_path,
            "start_line": str(start_line),
            "end_line": str(end_line)
        });
        target = safe_path(self.repo_root, file_path);
        if target is None {
            return refuse_path(file_path);
        }
        rel = rel_to(self.repo_root, target);
        if os.path.isdir(target) {
            return f"Error: '{rel}' is a directory, not a file. Use ls_repo to list it.";
        }
        if not os.path.isfile(target) {
            return (
                f"Error: no such file: '{rel}'. Use ls_repo to list a directory "
                "or find_files to locate a file by name."
            );
        }
        try {
            # stat, not getsize: st_mtime_ns and st_size together stamp the
            # version of the file a window was served from, so an edit
            # invalidates the served-window memory below without any explicit
            # cache busting from the editing tools.
            st = os.stat(target);
        } except OSError as e {
            return f"Error: could not stat '{rel}': {str(e)}";
        }
        if st.st_size > MAX_FILE_BYTES {
            return (
                f"Error: '{rel}' is {st.st_size} bytes, over the "
                f"{MAX_FILE_BYTES}-byte limit. Use grep to search it, or read a "
                "window with start_line/end_line."
            );
        }
        try {
            # errors="strict" so a non-UTF-8 file fails loudly and recoverably
            # rather than silently corrupting on a later write-back.
            # newline="" disables universal-newline translation, so bytes
            # round-trip and CRLF files stay CRLF.
            with open(target, "r", encoding="utf-8", errors="strict", newline="") as f {
                content = f.read();
            }
        } except UnicodeDecodeError {
            return f"Error: '{rel}' is not a UTF-8 text file (it may be binary).";
        } except OSError as e {
            return f"Error: could not read '{rel}': {str(e)}";
        }
        if not content {
            return f"({rel} is empty -- 0 bytes)";
        }
        lines = content.splitlines();
        total = len(lines);
        first = max(1, start_line);
        last = total if end_line <= 0 else min(total, end_line);
        if first > total {
            return (
                f"Error: start_line {first} is past the end of '{rel}' "
                f"({total} lines). Use start_line between 1 and {total}."
            );
        }
        if last < first {
            return (
                f"Error: end_line {last} is before start_line {first} in '{rel}'. "
                f"end_line must be 0 to read to the end, or at least {first}."
            );
        }
        # Header and footer are quoted back with the content, so they come out
        # of the budget before any line goes in.
        budget = max(MAX_TOOL_CHARS - (200 + 2 * len(rel)), 1000);
        # A file that fits whole is served whole, whatever window was asked
        # for. Windowing a small file buys nothing and costs a great deal: the
        # served-window memory below keys on the range, so a dozen overlapping
        # windows of one 318-line file are a dozen fresh reads and not one of
        # them a repeat it can refuse. That is the shape the scan loops took --
        # in one measured run, twenty-four reads walking backwards through a
        # file one line at a time. Served whole, the second read of it is a
        # repeat, and gets the next move instead of the same bytes. The window
        # arguments are still validated above, so a start_line past the end is
        # reported rather than quietly forgiven.
        if len(content) <= budget {
            first = 1;
            last = total;
        }
        (window, shown_last) = self._fill(lines, first, last, budget);

        # A window is identified by what it shows AND by the version of the
        # file it showed, so the entry expires by itself the moment the file is
        # written to and a re-read after an edit is served normally.
        key = f"{self.scope}|{rel}|{first}-{shown_last}|{st.st_size}:{st.st_mtime_ns}";
        if key in self.served {
            return self._already_read(rel, first, shown_last, total);
        }
        self.served.append(key);
        if len(self.served) > MAX_SERVED_WINDOWS {
            self.served = self.served[len(self.served) - MAX_SERVED_WINDOWS:];
        }

        # No line-number gutter: the model copies gutter text into an edit
        # anchor, which is the top cause of failed anchored edits. Line numbers
        # for navigation come from grep, which emits `path:line:`.
        header = f"# {rel} - lines {first}-{shown_last} of {total}";
        out: str = header + "\n" + "\n".join(window);
        if shown_last < last {
            out += (
                f"\n... [stopped at the {MAX_TOOL_CHARS}-character limit; "
                f"{last - shown_last} more lines in the range you asked for]. "
                f"Continue with read_file(file_path='{rel}', "
                f"start_line={shown_last + 1})."
            );
        }
        return out;
    }

    # Fill `budget` characters with whole lines, and report the last line that
    # fitted. Whole lines, because a mid-line cut is what made truncation
    # unrecoverable: "[truncated, N more characters]" names no line, so no
    # start_line resumes it and repeating the identical call is the only move
    # the truncation leaves. Lines are never shortened either -- replace_in_file
    # anchors on exact text, and a silently clipped line yields an anchor that
    # cannot match.
    def _fill(
        lines: list[str], first: int, last: int, budget: int
    ) -> tuple[list[str], int] {
        out: list[str] = [];
        used = 0;
        n = first;
        while n <= last {
            cost = len(lines[n - 1]) + 1;
            # `and out` emits the first line whatever its length: a single
            # over-budget line would otherwise produce an empty window whose
            # continuation hint points back at the call that just failed.
            if used + cost > budget and out {
                break;
            }
            out.append(lines[n - 1]);
            used += cost;
            n += 1;
        }
        return (out, n - 1);
    }

    # A repeat is answered with the next move, not with the same bytes. Handing
    # the identical window back is what let the model spin: nothing in the
    # result distinguishes it from a fresh read, so asking again is the obvious
    # retry. Naming the exact next call gives it somewhere else to go.
    def _already_read(rel: str, first: int, shown_last: int, total: int) -> str {
        head = (
            f"Error: you already read '{rel}' lines {first}-{shown_last} in this "
            "phase and the file has not changed since, so re-reading returns "
            "the same bytes. "
        );
        if shown_last < total {
            return head + (
                f"{total - shown_last} of its {total} lines are still unread: "
                f"read_file(file_path='{rel}', start_line={shown_last + 1}). "
                "To jump straight to a symbol instead, use grep."
            );
        }
        return head + (
            f"You have all {total} lines of it. Use grep to locate a symbol "
            "across the repository, or move on to the next step of the plan."
        );
    }

    def ls_repo(dir_path: str = ".") -> str {
        log_tool_call("ls_repo", {"dir_path": dir_path});
        target = safe_path(self.repo_root, dir_path);
        if target is None {
            return refuse_path(dir_path);
        }
        rel = rel_to(self.repo_root, target);
        if not os.path.isdir(target) {
            return f"Error: '{rel}' is not a directory. Use read_file to read a file.";
        }
        try {
            # sorted() is not cosmetic: raw os.listdir order is filesystem hash
            # order, so it differs per machine and after any file creation --
            # which would make the two frameworks explore in different orders.
            names = sorted(os.listdir(target));
        } except OSError as e {
            return f"Error: could not list '{rel}': {str(e)}";
        }
        entries: list[str] = [];
        for name in names {
            if name in SKIP_DIRS {
                continue;
            }
            full = os.path.join(target, name);
            entries.append(name + ("/" if os.path.isdir(full) else ""));
        }
        if not entries {
            return f"({rel} is an empty directory)";
        }
        shown = entries[:MAX_LIST_ENTRIES];
        out: str = "\n".join(shown);
        if len(entries) > MAX_LIST_ENTRIES {
            out += (
                f"\n... [{len(entries) - MAX_LIST_ENTRIES} more entries; "
                "list a subdirectory to narrow]"
            );
        }
        return out;
    }

    def find_files(name_glob: str, dir_path: str = ".") -> str {
        log_tool_call("find_files", {"name_glob": name_glob, "dir_path": dir_path});
        base_dir = safe_path(self.repo_root, dir_path);
        if base_dir is None {
            return refuse_path(dir_path);
        }
        if not os.path.isdir(base_dir) {
            return f"Error: '{rel_to(self.repo_root, base_dir)}' is not a directory.";
        }
        hits: list[str] = [];
        for full in self._walk_files(base_dir) {
            rel = rel_to(self.repo_root, full);
            # Match the bare name and the repo-relative path, so both "*.jac"
            # and "nodes/*.jac" behave the way the model expects.
            if fnmatch(os.path.basename(full), name_glob) or fnmatch(rel, name_glob) {
                hits.append(rel);
                if len(hits) >= MAX_LIST_ENTRIES {
                    break;
                }
            }
        }
        if not hits {
            return (
                f"No files matching '{name_glob}' under "
                f"'{rel_to(self.repo_root, base_dir)}'."
            );
        }
        return clip("\n".join(hits), MAX_TOOL_CHARS);
    }

    def grep(pattern: str, path: str = ".", file_glob: str = "*") -> str {
        log_tool_call("grep", {"pattern": pattern, "path": path, "file_glob": file_glob});
        try {
            regex = re.compile(pattern);
        } except re.error as e {
            return (
                f"Error: invalid regular expression '{pattern}': {str(e)}. "
                "For a case-insensitive search prefix the pattern with (?i)."
            );
        }
        target = safe_path(self.repo_root, path);
        if target is None {
            return refuse_path(path);
        }
        targets: list[str] = [];
        if os.path.isfile(target) {
            targets = [target];
        } elif os.path.isdir(target) {
            targets = self._walk_files(target);
        } else {
            return f"Error: no such file or directory: '{rel_to(self.repo_root, target)}'.";
        }
        hits: list[str] = [];
        searched = 0;
        truncated = False;
        for full in targets {
            if not fnmatch(os.path.basename(full), file_glob) {
                continue;
            }
            try {
                if os.path.getsize(full) > MAX_FILE_BYTES {
                    continue;
                }
                # errors="ignore" is safe here and nowhere else: grep never
                # writes back, so a mangled character costs at most a 0-match.
                with open(full, "r", encoding="utf-8", errors="ignore") as f {
                    content = f.read();
                }
            } except OSError {
                continue;
            }
            if chr(0) in content {
                continue;
            }
            searched += 1;
            rel = rel_to(self.repo_root, full);
            for (i, line) in enumerate(content.splitlines()) {
                if regex.search(line) {
                    if len(hits) >= MAX_GREP_MATCHES {
                        truncated = True;
                        break;
                    }
                    text = line.strip();
                    if len(text) > MAX_LINE_CHARS {
                        text = text[:MAX_LINE_CHARS] + " ...";
                    }
                    hits.append(f"{rel}:{i + 1}: {text}");
                }
            }
            if truncated {
                break;
            }
        }
        if not hits {
            return (
                f"No matches for '{pattern}' in "
                f"'{rel_to(self.repo_root, target)}' (file_glob='{file_glob}', "
                f"searched {searched} files)."
            );
        }
        out: str = "\n".join(hits);
        if truncated {
            out += (
                f"\n... [stopped at {MAX_GREP_MATCHES} matches; narrow the "
                "pattern, the path, or file_glob]"
            );
        }
        return clip(out, MAX_TOOL_CHARS);
    }

    def _walk_files(root_dir: str) -> list[str] {
        found: list[str] = [];
        for (dir_path, dir_names, file_names) in os.walk(root_dir) {
            # Prune in place -- filtering the yielded list afterwards still
            # descends into every skipped directory. sorted() for determinism.
            dir_names[:] = sorted(
                [d for d in dir_names if d not in SKIP_DIRS and not d.startswith(".")]
            );
            for name in sorted(file_names) {
                found.append(os.path.join(dir_path, name));
            }
        }
        return found;
    }
}


sem ExploreCodeBase = "Read-only access to the repository: read files, list directories, locate files by name, and search file contents.";

sem ExploreCodeBase.read_file = "Read a text file from the repository. Returns the file's contents preceded by a header naming the path and the line range shown, as 'lines 12-340 of 900'. There are no line-number prefixes on the content, so text copied from here can be used directly as an edit anchor. A file small enough to fit comes back whole no matter what window you ask for, so read it once and work from that; a large one comes back one window at a time, and when the result ends by naming the next start_line, call read_file again with that start_line to continue. Repeating a call you have already made returns an error rather than the same text, so page forward with start_line or narrow the search with grep instead.";
sem ExploreCodeBase.read_file.file_path = "Path to the file, relative to the repository root.";
sem ExploreCodeBase.read_file.start_line = "First line to show, 1-based. Use 1 to start at the top of the file.";
sem ExploreCodeBase.read_file.end_line = "Last line to show, 1-based and inclusive. Use 0 to read to the end of the file.";

sem ExploreCodeBase.ls_repo = "List the immediate contents of one directory. Directory names are suffixed with '/'. This is not recursive -- use find_files to locate a file anywhere in the tree.";
sem ExploreCodeBase.ls_repo.dir_path = "Directory to list, relative to the repository root. Use '.' for the root itself.";

sem ExploreCodeBase.find_files = "Find files anywhere under a directory by matching their name against a glob pattern. Use this to locate a file when you know its name but not where it lives.";
sem ExploreCodeBase.find_files.name_glob = "Glob pattern matched against each file's name and its repository-relative path, for example '*.jac' or 'nodes/*.jac'.";
sem ExploreCodeBase.find_files.dir_path = "Directory to search under, relative to the repository root. Use '.' to search the whole repository.";

sem ExploreCodeBase.grep = "Search file contents for a regular expression. Returns one line per match, formatted as 'path:line: text'. Use this to find where a symbol is defined or used before reading whole files.";
sem ExploreCodeBase.grep.pattern = "Python regular expression to search for. Prefix it with (?i) for a case-insensitive search, for example '(?i)todo'.";
sem ExploreCodeBase.grep.path = "File or directory to search, relative to the repository root. A directory is searched recursively. Use '.' for the whole repository.";
sem ExploreCodeBase.grep.file_glob = "Glob restricting which file names are searched, for example '*.jac'. Use '*' to search every file.";
