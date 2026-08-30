"""Print the raw model-bearing nodes of a workflow for manual verification: inspect_wf.py <corpus> <workflow_id>"""
import glob
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RAW, read_jsonl  # noqa: E402

LC = "@n8n/n8n-nodes-langchain."


def n8n(wid):
    for r in read_jsonl(RAW / "n8n_api" / "workflows.jsonl"):
        if str(r.get("id")) == wid:
            break
    else:
        print("not found"); return
    print(f"# n8n {wid}: {r['name']}")
    by = {n["name"]: n for n in r["nodes"]}
    for n in r["nodes"]:
        t = n.get("type", "")
        s = t[len(LC):] if t.startswith(LC) else t
        p = n.get("parameters") or {}
        keep = {k: p[k] for k in ("model", "modelName", "modelId", "resource", "operation") if k in p}
        if t.startswith(LC) and not any(s.startswith(x) for x in ("memory", "tool", "outputParser", "vectorStore", "documentDefault", "textSplitter", "chatTrigger", "mcp", "retriever")):
            print(f"  {n['name']!r:45s} {s:28s} v{n.get('typeVersion')} {json.dumps(keep, ensure_ascii=False)[:110]}")
        elif t in ("n8n-nodes-base.openAi",):
            print(f"  {n['name']!r:45s} {t:28s} v{n.get('typeVersion')} {json.dumps(keep, ensure_ascii=False)[:110]}")
    for src, kinds in (r.get("connections") or {}).items():
        for kind, outs in (kinds or {}).items():
            if kind == "ai_languageModel":
                tg = [e["node"] for out in outs for e in out]
                print(f"    LM {src!r} -> {tg}")


def dify(wid):
    src, stem = wid.split(":", 1)
    for f in glob.glob(str(RAW / "dify" / src / "*")):
        if Path(f).stem == stem:
            d = yaml.safe_load(open(f, encoding="utf-8"))
            break
    else:
        print("not found"); return
    print(f"# dify {wid}: {(d.get('app') or {}).get('name')} mode={(d.get('app') or {}).get('mode')}")
    g = ((d.get("workflow") or {}).get("graph") or {})
    for nd in g.get("nodes") or []:
        data = nd.get("data") or {}
        t = data.get("type")
        m = data.get("model") if t != "agent" else ((data.get("agent_parameters") or {}).get("model") or {}).get("value")
        if t in ("llm", "agent", "question-classifier", "parameter-extractor") or m:
            print(f"  {str(data.get('title'))[:40]!r:45s} {t:22s} {json.dumps(m, ensure_ascii=False)[:120] if m else ''}")
    mc = (d.get("model_config") or {}).get("model")
    if mc:
        print("  model_config:", json.dumps(mc)[:120])


def flowise(wid):
    src, stem = wid.split(":", 1)
    j = json.load(open(RAW / "flowise" / src / f"{stem}.json", encoding="utf-8"))
    print(f"# flowise {wid}")
    for nd in j.get("nodes") or []:
        data = nd.get("data") or {}
        inputs = data.get("inputs") or {}
        cat = data.get("category")
        if cat in ("Chat Models", "LLMs", "Embeddings") or data.get("name") in ("agentAgentflow", "llmAgentflow", "conditionAgentAgentflow"):
            cfg = inputs.get("agentModelConfig") or inputs.get("llmModelConfig") or inputs.get("conditionAgentModelConfig") or {}
            print(f"  {nd.get('id'):32s} {cat!s:18s} {data.get('name'):26s} model={inputs.get('modelName') or inputs.get('agentModel') or inputs.get('llmModel') or inputs.get('conditionAgentModel')} {cfg.get('modelName', '') if isinstance(cfg, dict) else ''}")
        refs = [f"{k}={v}" for k, v in inputs.items() if isinstance(v, str) and ".data.instance" in v and ("model" in k.lower() or k in ("llm",))]
        if refs:
            print(f"  {nd.get('id'):32s} {cat!s:18s} {data.get('name'):26s} uses {refs}")


def langflow(wid):
    stem = wid.split(":", 1)[1]
    j = json.load(open(RAW / "langflow" / f"{stem}.json", encoding="utf-8"))
    print(f"# langflow {wid}")
    for nd in (j.get("data") or {}).get("nodes") or []:
        data = nd.get("data") or {}
        t = data.get("type")
        if t in ("Agent", "LanguageModelComponent", "StructuredOutput", "BatchRunComponent") or (t or "").endswith("Model"):
            tpl = ((data.get("node") or {}).get("template") or {})
            vals = {k: (tpl[k].get("value") if isinstance(tpl.get(k), dict) else None) for k in ("provider", "model_name", "model", "agent_llm") if k in tpl}
            print(f"  {nd.get('id')[:40]:42s} {t:26s} {json.dumps(vals, default=str)[:100]}")


if __name__ == "__main__":
    corpus, wid = sys.argv[1], sys.argv[2]
    {"n8n_api": n8n, "dify": dify, "flowise": flowise, "langflow": langflow}[corpus](wid)
