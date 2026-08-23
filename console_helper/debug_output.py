import sys

_RESET = "\033[0m"
_COLORS = {
    "DEBUG": "\033[36m",  # cyan
    "LOG": "\033[32m",    # green
    "WARN": "\033[33m",   # yellow
    "ERROR": "\033[31m",  # red
}


def _emit(level: str, msg: str, stream=sys.stdout):
    color = _COLORS[level]
    print(f"{color}[{level}]{_RESET} {msg}", file=stream)


def console_debug(msg: str):
    _emit("DEBUG", msg)


def console_log(msg: str):
    _emit("LOG", msg)


def console_warn(msg: str):
    _emit("WARN", msg)


def console_error(msg: str):
    _emit("ERROR", msg, stream=sys.stderr)
