"""Usage recording: what each round trip to the LLM backend cost, and what it
carried.

Recording is passive. litellm invokes `TokenUsage.track` from its
success-callback pool, so nothing on the agent's own path -- no phase, no tool,
no walker ability -- has to remember to log. The records accumulated here are
the only input to `logger.log_LLM_history`, and the source of the token counts
`solve` reports.

Mirrors the TokenUsage tracker in the sibling byLLM implementations so the eval
harness reads the same shape on both sides of the comparison.
"""

import time;
import litellm;


"""One message in the prompt handed to the provider."""
obj LLMMessage {
    has role: str = "";
    has content: str = "";
}

"""One tool invocation the model asked for. `arguments` is the raw JSON string
the provider emitted, kept verbatim so a malformed argument blob shows up in
the log instead of being swallowed by a parse."""
obj LLMToolCall {
    has name: str = "";
    has arguments: str = "";
}

"""One completed round trip to the LLM backend."""
obj LLMCall {
    has model: str = "";
    has messages: list[LLMMessage] = [];
    has output: str = "";
    has tools_offered: list[str] = [];
    has tools_called: list[LLMToolCall] = [];
    has prompt_tokens: int = 0;
    has completion_tokens: int = 0;
    # A SUBSET of prompt_tokens, never an addition to it: the provider counts a
    # cache hit in both figures and bills it at a discount. Summing the two
    # double-counts the cached half of every prompt.
    has cached_tokens: int = 0;
}


# The extractors below read litellm's untyped callback payload, so their
# locals are `any` by necessity -- the shape varies by provider and by whether
# the call streamed. Every field is optional-with-fallback: a provider that
# omits one costs that line of the log, never the run.

"""The prompt handed to the provider, flattened to role/content pairs."""
def extract_messages(kwargs: dict[str, object]) -> list[LLMMessage] {
    raw: any = kwargs.get("messages") or [];
    out: list[LLMMessage] = [];
    for m in raw {
        if isinstance(m, dict) {
            out.append(LLMMessage(
                role=str(m.get("role", "")),
                content=str(m.get("content", "") or "")
            ));
        } else {
            out.append(LLMMessage(
                role=str(getattr(m, "role", "") or ""),
                content=str(getattr(m, "content", "") or "")
            ));
        }
    }
    return out;
}


"""The assistant text of one completion; "" when the turn was tools-only."""
def extract_output(resp: object) -> str {
    choices: any = getattr(resp, "choices", None);
    if not choices {
        return "";
    }
    msg: any = getattr(choices[0], "message", None);
    if msg is None {
        return "";
    }
    return str(getattr(msg, "content", "") or "");
}


"""Names of the tools the provider was offered on this call.

byLLM passes the phase's toolset straight through, so this is what identifies
which phase a call belongs to -- Editing offers write_file, Verifying does not.
"""
def extract_tools_offered(kwargs: dict[str, object]) -> list[str] {
    opts: any = kwargs.get("optional_params");
    nested: any = opts.get("tools") if isinstance(opts, dict) else None;
    specs: any = kwargs.get("tools") or nested or [];
    names: list[str] = [];
    for t in specs {
        fn: any = t.get("function") if isinstance(t, dict) else getattr(t, "function", None);
        if isinstance(fn, dict) {
            names.append(str(fn.get("name", "")));
        } elif fn is not None {
            names.append(str(getattr(fn, "name", "") or ""));
        }
    }
    return names;
}


"""True when this payload is one chunk of a stream rather than a finished call.

litellm fires the success callback for every streamed chunk *and* once more for
the completed call, so without this the tracker counts chunks. A measured
one-sentence completion produced 28 chunks and 29 records, of which 27 carried
no usage at all; `llm_calls` became a chunk count and the per-call token
averages became meaningless. Only the aggregate is a call.

Discriminated on the choice rather than the class name -- a chunk's choice
carries `delta`, a finished one carries `message` -- so a provider shipping its
own response type is still classified correctly.
"""
def is_stream_chunk(response: object) -> bool {
    choices: any = getattr(response, "choices", None);
    if not choices {
        return False;
    }
    first: any = choices[0];
    return hasattr(first, "delta") and not hasattr(first, "message");
}


"""Prompt tokens the provider served from its own cache.

Nested a level below `usage` and optional at every step -- a provider that does
no caching omits `prompt_tokens_details` entirely and litellm passes the absence
through, so this reads defensively for the same reason the extractors above do:
a provider that omits it should cost this one figure, never the run.

It matters here because the phase graph is close to the best case for prompt
caching. Every visit to a phase reissues the same system preamble, and a ReAct
lap resends the whole conversation each iteration, so the shared prefix is large
and repeated -- 2176 of 2414 prompt tokens on a measured second call. Reporting
prompt_tokens alone therefore overstates spend by whatever the cache absorbed.
"""
def extract_cached_tokens(usage: object) -> int {
    details: any = getattr(usage, "prompt_tokens_details", None);
    if details is None {
        return 0;
    }
    cached: any = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None);
    return int(cached or 0);
}


"""The tools the model actually invoked on this call."""
def extract_tools_called(resp: object) -> list[LLMToolCall] {
    out: list[LLMToolCall] = [];
    choices: any = getattr(resp, "choices", None);
    if not choices {
        return out;
    }
    msg: any = getattr(choices[0], "message", None);
    if msg is None {
        return out;
    }
    calls: any = getattr(msg, "tool_calls", None);
    if not calls {
        return out;
    }
    for c in calls {
        fn: any = getattr(c, "function", None);
        out.append(LLMToolCall(
            name=str(getattr(fn, "name", "") or ""),
            arguments=str(getattr(fn, "arguments", "") or "")
        ));
    }
    return out;
}


obj TokenUsage {
    has calls: list[LLMCall] = [];
    # Completion ids already recorded. See the dedupe in `track`.
    has seen_ids: set[str] = set();

    def track(kwargs: dict[str, object], response: object, _start: object, _end: object) {
        resp = kwargs.get("complete_streaming_response");
        if resp is None {
            # A streamed call arrives here once per chunk before it arrives
            # once as a whole; only the whole is a call. See is_stream_chunk.
            if is_stream_chunk(response) {
                return;
            }
            resp = response;
        }
        # A sync streaming request reaches this callback twice for the one
        # finished call: litellm's run_success_logging_and_cache_storage
        # (streaming_handler.py) runs the async success handler and then the
        # sync one, and both dispatch to litellm.success_callback. The two
        # carry the same aggregated response, so the provider's own completion
        # id separates a redelivery from a second call. Ids are unique per
        # completion, so this cannot merge two real calls; an empty id (a local
        # or mock backend that issues none) falls through and is recorded,
        # which risks a double rather than a silent loss.
        call_id = str(getattr(resp, "id", "") or "");
        if call_id {
            if call_id in self.seen_ids {
                return;
            }
            self.seen_ids.add(call_id);
        }
        usage = getattr(resp, "usage", None);
        prompt_tokens = 0;
        completion_tokens = 0;
        cached_tokens = 0;
        if usage is not None {
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0);
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0);
            cached_tokens = extract_cached_tokens(usage);
        }
        self.calls.append(LLMCall(
            model=str(getattr(resp, "model", None) or kwargs.get("model") or "?"),
            messages=extract_messages(kwargs),
            output=extract_output(resp),
            tools_offered=extract_tools_offered(kwargs),
            tools_called=extract_tools_called(resp),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens
        ));
    }

    def settle(timeout_s: float = 3.0, quiet_s: float = 0.3) {
        # litellm runs success callbacks on a background pool, so the records
        # can lag the last call; wait until they stop moving.
        deadline = time.monotonic() + timeout_s;
        last = len(self.calls);
        quiet_since = time.monotonic();
        while time.monotonic() < deadline {
            time.sleep(0.05);
            if len(self.calls) != last {
                last = len(self.calls);
                quiet_since = time.monotonic();
            } elif time.monotonic() - quiet_since >= quiet_s {
                return;
            }
        }
    }

    def totals() -> dict[str, int] {
        prompt = 0;
        completion = 0;
        cached = 0;
        for call in self.calls {
            prompt += call.prompt_tokens;
            completion += call.completion_tokens;
            cached += call.cached_tokens;
        }
        return {
            "llm_calls": len(self.calls),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "cached_tokens": cached
        };
    }
}

glob token_usage: TokenUsage = TokenUsage();


"""Start recording. Guarded rather than unconditional: `solve` calls this once
per run, and a second registration would have litellm invoke the callback twice
per call, doubling every count."""
def register_token_tracking() {
    if token_usage.track not in litellm.success_callback {
        litellm.success_callback.append(token_usage.track);
    }
}


"""What the run cost, once the in-flight callbacks have landed.

Settling is the reason this wrapper exists. Totals read straight off the
tracker can miss the final call, and the miss is silent -- the run simply looks
one call cheaper than it was.
"""
def usage_totals() -> dict[str, int] {
    token_usage.settle();
    return token_usage.totals();
}
