"""Sweep doc_analysis concurrency on an already running server."""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("doc_analysis", "finance", inputs=150, levels="4,8,12,16,20,24,32"))
