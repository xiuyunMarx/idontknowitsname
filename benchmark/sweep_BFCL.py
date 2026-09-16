"""Sweep BFCL_agent concurrency: python -m benchmark.sweep_BFCL [arm] [--c ...] [--sessions N] [--fresh]"""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("BFCL_agent", "bfcl", inputs=100, levels="10,20,30,40,50,60"))
