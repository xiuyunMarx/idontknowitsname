"""Fetch Flowise marketplace templates and Langflow starter projects -> data/raw/{flowise,langflow}/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, github_raw, github_tree, log, write_if_missing  # noqa: E402

FLOWISE_DIRS = ["chatflows", "agentflows", "agentflowsv2"]
FLOWISE_BASE = "packages/server/marketplaces/"
LANGFLOW_DIR = "src/backend/base/langflow/initial_setup/starter_projects/"


def flowise():
    paths = github_tree("FlowiseAI/Flowise", "main")
    for d in FLOWISE_DIRS:
        pre = FLOWISE_BASE + d + "/"
        files = [p for p in paths if p.startswith(pre) and p.endswith(".json")]
        out = RAW / "flowise" / d
        n = 0
        for p in files:
            if write_if_missing(out / Path(p).name, lambda p=p: github_raw("FlowiseAI/Flowise", p)):
                n += 1
        log(f"flowise/{d}: {len(files)} files, {n} new")


def langflow():
    paths = github_tree("langflow-ai/langflow", "main")
    files = [p for p in paths if p.startswith(LANGFLOW_DIR) and p.endswith(".json")]
    out = RAW / "langflow"
    n = 0
    for p in files:
        if write_if_missing(out / Path(p).name, lambda p=p: github_raw("langflow-ai/langflow", p)):
            n += 1
    log(f"langflow: {len(files)} files, {n} new")


if __name__ == "__main__":
    flowise()
    langflow()
