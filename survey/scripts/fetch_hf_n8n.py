"""Download the three HF n8n workflow dumps (cross-check corpus) into data/raw/hf/<name>/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, log  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

FILES = [
    ("mbakgun/n8nbuilder-n8n-workflows-dataset", "train.jsonl", "mbakgun"),
    ("ruh-ai/n8n-workflow-dataset", "filtered dataset.jsonl", "ruh-ai"),
    ("npv2k1/n8n-workflow", "dataset.json", "npv2k1"),
]

if __name__ == "__main__":
    for repo, fname, tag in FILES:
        out = RAW / "hf" / tag
        out.mkdir(parents=True, exist_ok=True)
        if (out / fname).exists():
            log(f"{tag}: have {fname}")
            continue
        p = hf_hub_download(repo_id=repo, filename=fname, repo_type="dataset", local_dir=str(out))
        log(f"{tag}: {p} ({Path(p).stat().st_size / 1e6:.1f} MB)")
