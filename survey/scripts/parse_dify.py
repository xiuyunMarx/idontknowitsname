"""Parse Dify DSL YAML into call-site rows + workflow rows."""
import glob
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, RAW, graph_hash, log, longest_site_path, write_jsonl  # noqa: E402

MODEL_NODES = {"llm": "chat", "question-classifier": "classifier", "parameter-extractor": "extractor", "agent": "agent"}


def norm_provider(p):
    if not p:
        return "unknown"
    p = str(p)
    if "/" in p:  # langgenius/<plugin>/<provider>
        p = p.split("/")[-1]
    return p.lower()


def model_of(data):
    t = data.get("type")
    if t == "agent":
        m = ((data.get("agent_parameters") or {}).get("model") or {}).get("value") or {}
        return norm_provider(m.get("provider")), m.get("model") or ""
    m = data.get("model") or {}
    if isinstance(m, dict):
        return norm_provider(m.get("provider")), m.get("name") or ""
    return "unknown", ""


def parse_file(path, source):
    d = yaml.safe_load(open(path, encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("not a mapping")
    app = d.get("app") or {}
    mode = app.get("mode")
    name = app.get("name") or Path(path).stem
    wid = f"{source}:{Path(path).stem}"
    rows, types, edges, site_ids, main_edges = [], [], [], set(), []
    g = ((d.get("workflow") or {}).get("graph") or {})
    nodes = g.get("nodes") or []
    if nodes:
        by_id = {}
        for nd in nodes:
            data = nd.get("data") or {}
            t = data.get("type") or ""
            by_id[nd.get("id")] = t
            types.append((t, ""))
            if t in MODEL_NODES:
                prov, m = model_of(data)
                res = "explicit" if m else ("unknown" if prov == "unknown" else "default")
                rows.append(dict(node_id=nd.get("id"), node_name=data.get("title"), node_type=t, node_version="",
                                 role=MODEL_NODES[t], provider_raw=prov, model_raw=m, resolution=res, via_router=False))
                site_ids.add(nd.get("id"))
            elif t == "knowledge-retrieval":
                rm = ((data.get("multiple_retrieval_config") or {}).get("reranking_model") or {})
                if rm.get("model"):
                    rows.append(dict(node_id=nd.get("id"), node_name=data.get("title"), node_type=t, node_version="",
                                     role="reranker", provider_raw=norm_provider(rm.get("provider")), model_raw=rm.get("model"),
                                     resolution="explicit", via_router=False))
        for e in g.get("edges") or []:
            s, t_ = e.get("source"), e.get("target")
            edges.append((by_id.get(s, ""), by_id.get(t_, ""), "main"))
            main_edges.append((s, t_))
    else:
        mc = (d.get("model_config") or {}).get("model")
        if isinstance(mc, dict):
            rows.append(dict(node_id="model_config", node_name="model_config", node_type=mode or "chat", node_version="",
                             role="agent" if mode == "agent-chat" else "chat", provider_raw=norm_provider(mc.get("provider")),
                             model_raw=mc.get("name") or "", resolution="explicit" if mc.get("name") else "default", via_router=False))
            site_ids.add("model_config")
            types.append((mode or "chat", ""))
    has_router = any(r["node_type"] == "question-classifier" for r in rows)
    wrow = dict(corpus="dify", source=source, workflow_id=wid, name=name, n_nodes=len(types), graph_hash=graph_hash(types, edges),
                chain_len=longest_site_path(list(site_ids) + [n.get("id") for n in nodes], main_edges, site_ids),
                has_router=has_router, views=None, template_id=None, app_mode=mode)
    for r in rows:
        r.update(corpus="dify", source=source, workflow_id=wid, name=name)
    return wrow, rows


if __name__ == "__main__":
    wrows, srows, fails = [], [], []
    for path in sorted(glob.glob(str(RAW / "dify" / "*" / "*.y*ml"))):
        source = Path(path).parent.name
        try:
            w, rs = parse_file(path, source)
            wrows.append(w)
            srows += rs
        except Exception as e:  # noqa: BLE001
            fails.append(dict(corpus="dify", workflow_id=path, error=repr(e)))
    write_jsonl(PARSED / "dify_workflows.jsonl", wrows)
    write_jsonl(PARSED / "dify_callsites.jsonl", srows)
    write_jsonl(PARSED / "dify_failures.jsonl", fails)
    log(f"dify: {len(wrows)} workflows, {len(srows)} call-site rows, {len(fails)} failures")
