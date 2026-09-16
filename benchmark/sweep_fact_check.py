"""Sweep fact_check concurrency on an already running server."""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("fact_check", "fact", inputs=120, levels="8,12,16,20,24,28,32"))
