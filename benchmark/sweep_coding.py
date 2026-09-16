"""Sweep coding_agent concurrency: python -m benchmark.sweep_coding [arm] [--c ...] [--sessions N] [--fresh]"""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("coding_agent", "coding", inputs=157, levels="16,20,24,28,32,36,40,44,48,52,56,60,64"))
