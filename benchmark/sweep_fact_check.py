"""Sweep fact_check concurrency: python -m benchmark.sweep_fact_check [arm] [--c ...] [--sessions N] [--fresh]"""
import sys

from benchmark.sweep_common import sweep

if __name__ == "__main__":
    sys.exit(sweep("fact_check", "fact", inputs=120, levels="6,8,10,12,14,16,18,20"))
