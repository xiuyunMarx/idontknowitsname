"""Pretty-print the prompts captured in ./out by the three framework runners.

Usage:
    python pretty_print.py                # all three, messages only
    python pretty_print.py byllm dspy     # a subset, in the order given
    python pretty_print.py --schema       # also print the response_format schema
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

FRAMEWORKS = {
    "byllm": ("byLLM", "byllm_prompt.json"),
    "dspy": ("DSPy", "dspy_prompt.json"),
    "nooa": ("NVIDIA nooa", "nooa_prompt.json"),
}

WIDTH = 100


def rule(char: str, label: str = "") -> str:
    if not label:
        return char * WIDTH
    head = f"{char * 3} {label} "
    return head + char * max(0, WIDTH - len(head))


def as_text(content) -> str:
    """Flatten a message body: plain string, or a list of content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", json.dumps(part, indent=2)))
            else:
                parts.append(str(part))
        return "\n\n".join(parts)
    return json.dumps(content, indent=2)


def show(key: str, with_schema: bool) -> None:
    label, filename = FRAMEWORKS[key]
    path = os.path.join(OUT, filename)
    if not os.path.exists(path):
        print(f"{label}: {path} is missing — run the {key} case first.\n")
        return

    request = json.load(open(path))
    params = {
        k: v
        for k, v in request.items()
        if k in ("model", "temperature", "max_tokens") and v is not None
    }

    print(rule("="))
    print(f"  {label}    {'  '.join(f'{k}={v}' for k, v in params.items())}")
    print(rule("="))

    for message in request["messages"]:
        print()
        print(rule("-", message.get("role", "?").upper()))
        print(as_text(message.get("content")))

    schema = request.get("response_format")
    if schema is not None:
        name = schema.get("json_schema", {}).get("name") or schema.get("title") or "response_format"
        if with_schema:
            print()
            print(rule("-", f"RESPONSE_FORMAT ({name})"))
            print(json.dumps(schema, indent=2))
        else:
            print()
            print(rule("-", f"RESPONSE_FORMAT ({name}, use --schema to expand)"))
    print()


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    with_schema = "--schema" in sys.argv[1:]
    keys = args or list(FRAMEWORKS)
    unknown = [k for k in keys if k not in FRAMEWORKS]
    if unknown:
        sys.exit(f"unknown framework(s): {', '.join(unknown)}; pick from {', '.join(FRAMEWORKS)}")
    for key in keys:
        show(key, with_schema)


main()
