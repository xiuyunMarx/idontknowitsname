"""Sweep deep_research concurrency on an already running server."""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("deep_research", "deepr", inputs=100, levels="4,8,12,16"))
