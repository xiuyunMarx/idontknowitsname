"""Live rendering of a phase's LLM stream.

byLLM hands `stream_handler` one StreamEvent per thing that happens inside a
ReAct lap -- reasoning tokens, answer tokens, tool calls and their results.

The handler form rather than bare `stream=True` is the whole point. With tools
in play, `stream=True` alone returns a *generator* from `invoke`, and the phase
graph's `str(run_phase(...))` would stringify the generator object rather than
the answer -- every phase summary would become "<generator object ...>". Given a
handler, byLLM drives the same streaming loop but reassembles the chunks and
returns the final answer string, so the graph is untouched and the tokens are
merely observed on the way past.

Turning streaming on has a second effect that is not cosmetic:
`max_tool_result_length` is only consulted on the streaming path, so until now
the orchestrator's setting bound nothing and tool results entered the
conversation at whatever size the tools themselves emitted. Streaming makes that
cap real, which is what brings this side in line with the sibling
implementations (see MAX_TOOL_RESULT_CHARS in orchestrator.jac).

Output goes to stderr, not stdout: the SWE-bench shim treats stdout as a
transcript rather than a channel -- byLLM, litellm and httpx all log there -- and
the driver captures both streams into agent.log regardless.
"""

import sys;
import from typing { TextIO }


# Tool results are echoed as a length rather than a body. They are already
# recorded in full by log_LLM_history, and a phase that reads three files would
# otherwise repeat every byte of them here -- in a run that made 188 LLM calls
# that is the difference between a readable transcript and an unreadable one.
glob TOOL_RESULT_PREVIEW_CHARS: int = 160;


"""One line of the stream, on stderr, flushed so it interleaves in real time."""
def emit(text: str) {
    stream: TextIO = sys.stderr as TextIO;
    stream.write(text);
    stream.flush();
}


"""Collapse whitespace so a multi-line tool result stays on one line."""
def one_line(text: str, limit: int) -> str {
    flat = " ".join(str(text).split());
    if len(flat) <= limit {
        return flat;
    }
    return flat[:limit] + f"... [+{len(flat) - limit} chars]";
}


"""Render one StreamEvent.

Passed to byLLM as `stream_handler`. byLLM swallows whatever this raises (it
logs at debug and carries on), so a bug here costs the transcript rather than
the run -- but the broad guard is kept anyway, since the alternative is a silent
gap in the log that looks like the model going quiet.
"""
def render_stream_event(event: object) -> None {
    try {
        kind = str(getattr(event, "event_type", "") or "");
        data: any = getattr(event, "data", None) or {};
        if kind == "thought" or kind == "chunk" {
            emit(str(data.get("content", "") or ""));
        } elif kind == "tool_call" {
            # Keys are byLLM's: `tool`/`args` here, `tool`/`result` below --
            # not the `name`/`arguments`/`content` an OpenAI tool-call payload
            # uses. Guessing the latter renders every call as `?()`.
            name = str(data.get("tool", "") or "?");
            args = one_line(str(data.get("args", "") or ""), TOOL_RESULT_PREVIEW_CHARS);
            emit(f"\n  -> {name}({args})\n");
        } elif kind == "tool_result" {
            # Already cut to max_tool_result_length by the streaming loop, so
            # this length is the size the model actually received.
            body = str(data.get("result", "") or "");
            emit(f"  <- {len(body)} chars: {one_line(body, TOOL_RESULT_PREVIEW_CHARS)}\n");
        } elif kind == "stream_reset" {
            # A transport retry discarded the partial answer; say so, or the
            # duplicated text that follows reads as the model repeating itself.
            emit("\n  [stream reset -- retrying]\n");
        }
    } except Exception as e {
        emit(f"\n  [stream handler error: {type(e).__name__}]\n");
    }
}
