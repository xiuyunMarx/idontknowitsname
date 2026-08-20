"""Writes the recorded LLM calls out as one readable file per call.

Holds no state: everything it writes comes from the records `logger.log_usage`
accumulated during the run. A call is the unit because a call is what gets
read -- a single transcript of a twenty-call run is a file nobody scrolls
through, while `call-014.txt` is one prompt, one answer, and the tools that
went with it.

Nothing here goes to stdout, which the eval harness reads for the agent's
answer; the only thing that reaches the terminal is a one-line note saying
where the files landed.
"""

import json;
import os;
import sys;
import from typing { TextIO }

import from logger.log_usage { LLMCall, LLMToolCall, token_usage }


# Written under the working directory, which is where the run was launched
# from and not the repository being edited -- nothing here can end up in a
# patch taken from the workspace.
glob DEFAULT_LOG_DIR: str = "llm_calls";

glob RULE: str = "=" * 78;
glob SUBRULE: str = "-" * 78;

# What a backslash can legally stand for inside a JSON string. A backslash
# followed by anything else is left exactly as it was found.
glob ESCAPES: dict[str, str] = {
    "n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f",
    "\"": "\"", "'": "'", "\\": "\\", "/": "/"
};


"""Send one already-formatted line to stderr.

The checker sees `sys.stderr` as `TextIO | Any`; the cast pins it down here so
no call site has to carry an unresolved-expression warning.
"""
def write_err(line: str) {
    stream: TextIO = sys.stderr as TextIO;
    stream.write(line);
}


"""Turn escape sequences back into the characters they stand for.

Walks the string a character at a time rather than calling
`str.replace("\\n", "\n")`, which cannot tell a line break from the two
characters that follow the backslash in an escaped backslash.
"""
def unescape(text: str) -> str {
    out: list[str] = [];
    i = 0;
    while i < len(text) {
        ch = text[i];
        if ch == "\\" and i + 1 < len(text) and text[i + 1] in ESCAPES {
            out.append(ESCAPES[text[i + 1]]);
            i += 2;
            continue;
        }
        out.append(ch);
        i += 1;
    }
    return "".join(out);
}


"""`text` with its escapes resolved -- but only if escapes are all it has.

Text that already carries a real line break is handed back untouched. It is
not an escaped blob, and rewriting the `\\n` inside a code sample the model
quoted would corrupt the very thing the log exists to show.
"""
def readable(text: str) -> str {
    if "\n" in text or "\\n" not in text {
        return text;
    }
    return unescape(text);
}


"""`text` as log lines, each one indented by `pad`. Empty text yields nothing."""
def indent(text: str, pad: str) -> list[str] {
    return [pad + ln for ln in readable(text).splitlines()];
}


"""One tool call: its name, then every argument as its own labelled block.

Arguments arrive as the raw JSON string the provider emitted, which puts the
entire body of a written file on one line with its newlines spelled out.
Decoding the JSON is what unspells them, and it is exact where a text
substitution would only be a guess. A blob that will not parse is printed as
it came, with its escapes resolved as far as they can be.
"""
def render_tool_call(index: int, tc: LLMToolCall) -> list[str] {
    lines: list[str] = ["", f"[{index}] {tc.name}"];
    args: dict[str, object] = {};
    parsed = True;
    if tc.arguments {
        try {
            args = json.loads(tc.arguments) as dict[str, object];
        } except Exception {
            parsed = False;
        }
    }
    # A provider is free to emit anything; only a JSON object has arguments to
    # take apart, and anything else is shown as it arrived.
    if not parsed or not isinstance(args, dict) {
        lines.extend(indent(tc.arguments, "      "));
        return lines;
    }
    if not args {
        lines.append("      (no arguments)");
    }
    for (key, value) in args.items() {
        text: str = value if isinstance(value, str) else json.dumps(value, indent=2);
        if len(text.splitlines()) > 1 {
            lines.append(f"      {key}:");
            lines.extend(indent(text, "        "));
        } else {
            lines.append(f"      {key}: {text}");
        }
    }
    return lines;
}


"""One call as the whole text of its file: header, prompt, output, tool calls."""
def render_call(index: int, total: int, call: LLMCall) -> str {
    offered = ", ".join(call.tools_offered) if call.tools_offered else "(none)";
    lines: list[str] = [
        RULE,
        f"call {index} of {total}",
        f"model          {call.model}",
        f"tokens         {call.prompt_tokens} prompt + {call.completion_tokens} completion",
        f"tools offered  {offered}",
        RULE,
        "",
        SUBRULE,
        f"PROMPT -- {len(call.messages)} message(s)",
        SUBRULE
    ];
    if not call.messages {
        lines.append("");
        lines.append("    (not captured)");
    }
    for (i, m) in enumerate(call.messages) {
        lines.append("");
        lines.append(f"[{i + 1}] {m.role}");
        body = indent(m.content, "    ");
        lines.extend(body if body else ["    (empty)"]);
    }

    lines.extend(["", SUBRULE, "OUTPUT", SUBRULE, ""]);
    out = indent(call.output, "    ");
    lines.extend(out if out else ["    (no text -- tool-call turn)"]);

    lines.extend(["", SUBRULE, f"TOOL CALLS -- {len(call.tools_called)}", SUBRULE]);
    if not call.tools_called {
        lines.append("");
        lines.append("    (none)");
    }
    for (i, tc) in enumerate(call.tools_called) {
        lines.extend(render_tool_call(i + 1, tc));
    }
    return "\n".join(lines) + "\n";
}


"""The one-page view: every call on a line, and what the run cost in total."""
def render_index(calls: list[LLMCall], totals: dict[str, int], names: list[str]) -> str {
    lines: list[str] = [RULE, f"LLM CALL LOG -- {len(calls)} call(s)", RULE, ""];
    for (i, call) in enumerate(calls) {
        called = ", ".join([tc.name for tc in call.tools_called]) if call.tools_called else "-";
        lines.append(
            f"{names[i]}  {call.model}  "
            f"{call.prompt_tokens}+{call.completion_tokens} tok  "
            f"tools: {called}"
        );
    }
    lines.extend([
        "",
        RULE,
        f"TOTAL {totals['llm_calls']} call(s), {totals['prompt_tokens']} prompt "
        f"+ {totals['completion_tokens']} completion tokens",
        RULE
    ]);
    return "\n".join(lines) + "\n";
}


"""Write every call the run made to `folder`, one file each, plus an index.

Destination is `folder`, else $CODEAGENT_LLM_LOG, else `llm_calls/`. Returns
the directory written, or "" if it could not be written -- in which case the
whole transcript goes to stderr instead, since losing the log outright is the
worse outcome.

`call-*.txt` files already in the directory are cleared first, and only those:
a short run would otherwise inherit the tail of a longer one, where it reads
as part of the run just finished.
"""
def log_LLM_calls(folder: str = "") -> str {
    # Settling again is cheap next to losing the final call: `solve` settles
    # before it returns, but this function is also callable on its own.
    token_usage.settle();
    calls = token_usage.calls;
    totals = token_usage.totals();
    names = [f"call-{(i + 1):03d}.txt" for i in range(len(calls))];
    dest = folder or os.environ.get("CODEAGENT_LLM_LOG", "") or DEFAULT_LOG_DIR;

    try {
        os.makedirs(dest, exist_ok=True);
        for stale in os.listdir(dest) {
            if stale.startswith("call-") and stale.endswith(".txt") {
                os.remove(os.path.join(dest, stale));
            }
        }
        for (i, call) in enumerate(calls) {
            with open(os.path.join(dest, names[i]), "w") as fh {
                fh.write(render_call(i + 1, len(calls), call));
            }
        }
        with open(os.path.join(dest, "index.txt"), "w") as fh {
            fh.write(render_index(calls, totals, names));
        }
    } except OSError as e {
        write_err(f"[llm-log] cannot write {dest}: {str(e)}\n");
        write_err(render_index(calls, totals, names));
        for (i, call) in enumerate(calls) {
            write_err(render_call(i + 1, len(calls), call));
        }
        return "";
    }
    write_err(f"[llm-log] {len(calls)} call(s) written to {dest}{os.sep}\n");
    return dest;
}
