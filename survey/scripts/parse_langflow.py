"""Parse Langflow starter projects into call-site rows (model ids are mostly empty in exports)."""
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, RAW, graph_hash, log, longest_site_path, write_jsonl  # noqa: E402

SITE_TYPES = {"Agent": "agent", "LanguageModelComponent": "chat", "StructuredOutput": "extractor", "BatchRunComponent": "chat"}


def tval(tpl, k):
    v = tpl.get(k)
    if isinstance(v, dict):
        v = v.get("value")
    return v if isinstance(v, str) else ""


if __name__ == "__main__":
    wrows, srows, fails = [], [], []
    for path in sorted(glob.glob(str(RAW / "langflow" / "*.json"))):
        wid = f"langflow:{Path(path).stem}"
        name = Path(path).stem
        try:
            j = json.load(open(path, encoding="utf-8"))
            nodes = (j.get("data") or {}).get("nodes") or []
            rows, types, site_ids = [], [], set()
            for nd in nodes:
                data = nd.get("data") or {}
                t = data.get("type") or ""
                types.append((t, ""))
                is_site = t in SITE_TYPES or (t.endswith("Model") and t != "LanguageModelComponent")
                if not is_site:
                    continue
                tpl = ((data.get("node") or {}).get("template") or {})
                prov = tval(tpl, "provider") or (t.replace("Model", "").lower() if t.endswith("Model") and t not in SITE_TYPES else "")
                m = tval(tpl, "model_name") or tval(tpl, "model")
                if isinstance(tpl.get("model"), dict) and isinstance(tpl["model"].get("value"), dict):
                    mv = tpl["model"]["value"]
                    m = mv.get("model_name") or mv.get("name") or m
                    prov = mv.get("provider") or prov
                rows.append(dict(node_id=nd.get("id"), node_name=data.get("display_name") or t, node_type=t, node_version="",
                                 role=SITE_TYPES.get(t, "chat"), provider_raw=(prov or "unknown").lower(), model_raw=m,
                                 resolution="explicit" if m else ("default" if prov else "unknown"), via_router=False))
                site_ids.add(nd.get("id"))
            by_id = {nd.get("id"): (nd.get("data") or {}).get("type", "") for nd in nodes}
            edges, main_edges = [], []
            for e in (j.get("data") or {}).get("edges") or []:
                s, t_ = e.get("source"), e.get("target")
                edges.append((by_id.get(s, ""), by_id.get(t_, ""), "main"))
                main_edges.append((s, t_))
            wrows.append(dict(corpus="langflow", source="starter_projects", workflow_id=wid, name=name, n_nodes=len(types),
                              graph_hash=graph_hash(types, edges), chain_len=longest_site_path(list(by_id), main_edges, site_ids),
                              has_router=False, views=None, template_id=None))
            for r in rows:
                r.update(corpus="langflow", source="starter_projects", workflow_id=wid, name=name)
            srows += rows
        except Exception as e:  # noqa: BLE001
            fails.append(dict(corpus="langflow", workflow_id=wid, error=repr(e)))
    write_jsonl(PARSED / "langflow_workflows.jsonl", wrows)
    write_jsonl(PARSED / "langflow_callsites.jsonl", srows)
    write_jsonl(PARSED / "langflow_failures.jsonl", fails)
    log(f"langflow: {len(wrows)} workflows, {len(srows)} call-site rows, {len(fails)} failures")
