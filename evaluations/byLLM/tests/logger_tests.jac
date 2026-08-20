import os;
import tempfile;
import from logger.log_usage { LLMMessage, LLMToolCall, LLMCall, token_usage }
import from logger.log_LLM_history { unescape, readable, render_tool_call, log_LLM_calls }


def sample_calls() -> list[LLMCall] {
    return [
        LLMCall(
            model="gpt-4o",
            messages=[LLMMessage(role="user", content="with entry {\\n    print(\"hello\");\\n}")],
            output="",
            tools_offered=["write_file"],
            tools_called=[LLMToolCall(
                name="write_file",
                arguments="{\"file_path\": \"greet.jac\", \"content\": \"line one\\nline two\"}"
            )],
            prompt_tokens=100,
            completion_tokens=10
        ),
        LLMCall(
            model="gpt-4o",
            messages=[LLMMessage(role="user", content="done?")],
            output="Yes.",
            tools_offered=[],
            tools_called=[],
            prompt_tokens=20,
            completion_tokens=5
        )
    ];
}

# ---------------------------------------------------------------------------
# Escape handling. The point of the log is to be read, and a provider hands
# back a written file's whole body on one line with its newlines spelled out.
# ---------------------------------------------------------------------------

test "unescape resolves a sequence without eating an escaped backslash" {
    assert unescape("a\\nb") == "a\nb";
    assert unescape("a\\tb") == "a\tb";
    # `\\n` is a backslash followed by the letter n, not a line break.
    assert unescape("C:\\\\next") == "C:\\next";
    # An escape that means nothing is left exactly as it was found.
    assert unescape("a\\qb") == "a\\qb";
}

test "readable leaves text that already has real line breaks alone" {
    # Code the model quoted: rewriting the sequence inside it would corrupt
    # the very thing the log exists to show.
    quoted = "print(\"a\\nb\")\nprint(\"c\")";
    assert readable(quoted) == quoted;
    # A one-line blob is the escaped case, and does get resolved.
    assert readable("a\\nb") == "a\nb";
}

test "tool arguments are decoded from JSON, not text-substituted" {
    tc = LLMToolCall(
        name="write_file",
        arguments="{\"path\": \"g.jac\", \"body\": \"one\\ntwo\", \"force\": true}"
    );
    lines = render_tool_call(1, tc);
    assert "[1] write_file" in lines;
    # A single-line value stays on the label's line; a multi-line one becomes
    # its own indented block.
    assert "      path: g.jac" in lines;
    assert "      body:" in lines;
    assert "        one" in lines;
    assert "        two" in lines;
    assert "      force: true" in lines;
}

test "a blob that is not JSON is still printed" {
    lines = render_tool_call(1, LLMToolCall(name="broken", arguments="{oops\\nhere}"));
    assert "      {oops" in lines;
    assert "      here}" in lines;
}

# ---------------------------------------------------------------------------
# One file per call.
# ---------------------------------------------------------------------------

test "each call is written to its own file, with an index beside them" {
    token_usage.calls = sample_calls();
    dest = os.path.join(tempfile.mkdtemp(), "llm_calls");
    assert log_LLM_calls(dest) == dest;

    assert sorted(os.listdir(dest)) == ["call-001.txt", "call-002.txt", "index.txt"];
    with open(os.path.join(dest, "call-001.txt")) as f {
        first = f.read();
    }
    assert "call 1 of 2" in first;
    assert "tokens         100 prompt + 10 completion" in first;
    # The escaped prompt and the escaped tool argument both come out as lines.
    assert "    with entry {\n        print(\"hello\");" in first;
    assert "        line one\n        line two" in first;

    with open(os.path.join(dest, "index.txt")) as f {
        index = f.read();
    }
    assert "call-001.txt" in index;
    assert "TOTAL 2 call(s), 120 prompt + 15 completion tokens" in index;
    token_usage.calls = [];
}

test "a shorter run does not inherit the tail of a longer one" {
    dest = os.path.join(tempfile.mkdtemp(), "llm_calls");
    os.makedirs(dest, exist_ok=True);
    with open(os.path.join(dest, "call-009.txt"), "w") as f {
        f.write("left over from a nine call run\n");
    }
    # Not ours, so not ours to delete.
    with open(os.path.join(dest, "notes.txt"), "w") as f {
        f.write("keep me\n");
    }
    token_usage.calls = sample_calls();
    _ = log_LLM_calls(dest);
    assert "call-009.txt" not in os.listdir(dest);
    assert "notes.txt" in os.listdir(dest);
    token_usage.calls = [];
}
