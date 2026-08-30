"""Fetch Dify DSL corpora: official template feed + community GitHub collections -> data/raw/dify/<src>/*.yml"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, get, get_json, github_raw, github_tree, log, write_if_missing  # noqa: E402

OUT = RAW / "dify"
REPOS = [  # (repo, branch, tag)
    ("svcvit/Awesome-Dify-Workflow", "main", "svcvit"),
    ("wwwzhouhui/dify-for-dsl", "main", "wwwzhouhui"),
    ("difyhub/workflows", "main", "difyhub"),
    ("Winson-030/dify-DSL", "main", "winson"),
]


def safe(name):
    return re.sub(r"[^A-Za-z0-9_.\-一-鿿]+", "_", name)[:120]


def official():
    out = OUT / "tmpl_dify"
    out.mkdir(parents=True, exist_ok=True)
    try:
        d = get_json("https://tmpl.dify.ai/apps?language=en-US")
    except Exception as e:  # noqa: BLE001
        log(f"tmpl.dify.ai list failed: {e}")
        return
    apps = d.get("recommended_apps") or d.get("apps") or d.get("data") or []
    log(f"tmpl.dify.ai: {len(apps)} apps")
    n = 0
    for a in apps:
        app = a.get("app") or a
        aid = a.get("app_id") or app.get("id")
        name = app.get("name") or str(aid)
        if not aid:
            continue
        path = out / f"{safe(name)}__{aid}.yml"

        def fetch(aid=aid):
            dd = get_json(f"https://tmpl.dify.ai/apps/{aid}", sleep=0.3)
            return (dd.get("export_data") or "").encode("utf-8")
        try:
            if write_if_missing(path, fetch):
                n += 1
        except Exception as e:  # noqa: BLE001
            log(f"tmpl.dify.ai app {aid} failed: {e}")
    log(f"tmpl_dify: fetched {n} new")


def repos():
    for repo, branch, tag in REPOS:
        out = OUT / tag
        out.mkdir(parents=True, exist_ok=True)
        try:
            paths = github_tree(repo, branch)
        except Exception as e:  # noqa: BLE001
            log(f"{repo}: tree failed: {e}")
            continue
        ymls = [p for p in paths if p.lower().endswith((".yml", ".yaml")) and not p.startswith(".github/")]
        n = 0
        for p in ymls:
            dst = out / safe(p.replace("/", "__"))
            try:
                if write_if_missing(dst, lambda p=p: github_raw(repo, p, branch)):
                    n += 1
            except Exception as e:  # noqa: BLE001
                log(f"{repo}:{p} failed: {e}")
        log(f"{tag}: {len(ymls)} yml files, {n} new")


if __name__ == "__main__":
    official()
    repos()
