"""Parse Flowise marketplace templates (chatflows, agentflows v1, agentflows v2) into call-site rows."""
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, RAW, graph_hash, log, longest_site_path, write_jsonl  # noqa: E402

MODEL_CATS = {"Chat Models", "LLMs"}
AGENT_CATS = {"Agents", "Multi Agents", "Sequential Agents"}
CONSUMER_CATS = AGENT_CATS | {"Chains", "Engine"}
V2 = {  # node name -> (model key, config key, role)
    "agentAgentflow": ("agentModel", "agentModelConfig", "agent"),
    "llmAgentflow": ("llmModel", "llmModelConfig", "chat"),
    "conditionAgentAgentflow": ("conditionAgentModel", "conditionAgentModelConfig", "classifier"),
}


def provider_of(model_node_name):
    n = model_node_name or ""
    n = re.sub(r"_LlamaIndex$", "", n)
    n = re.sub(r"^(chat|azureChat|awsChat)", "", n)
    low = n.lower()
    for k, v in (("openai", "openai"), ("anthropic", "anthropic"), ("googlegenerativeai", "google"), ("googlevertex", "vertex"),
                 ("ollama", "ollama"), ("openrouter", "openrouter"), ("groq", "groq"), ("mistral", "mistral"),
                 ("huggingface", "huggingface"), ("replicate", "replicate"), ("localai", "localai"), ("bedrock", "bedrock"),
                 ("deepseek", "deepseek"), ("xai", "xai"), ("cohere", "cohere"), ("together", "together"),
                 ("fireworks", "fireworks"), ("cerebras", "cerebras"), ("nvidianim", "nvidia"), ("perplexity", "perplexity")):
        if k in low:
            prov = v
            break
    else:
        prov = low or "unknown"
    if model_node_name and model_node_name.lower().startswith("azure"):
        prov = "azure_openai"
    return prov


def parse_v1(j, source, wid, name):
    nodes = j.get("nodes") or []
    rows, types, site_ids = [], [], set()
    model_nodes = {}
    for nd in nodes:
        data = nd.get("data") or {}
        types.append((data.get("name") or "", data.get("version") or ""))
        cat = data.get("category")
        inputs = data.get("inputs") or {}
        if cat in MODEL_CATS:
            m = inputs.get("modelName") or inputs.get("model") or ""
            m = m if isinstance(m, str) else ""
            model_nodes[nd.get("id")] = dict(provider_raw=provider_of(data.get("name")), model_raw=m,
                                             resolution="explicit" if m else "default")
        elif cat == "Embeddings":
            m = inputs.get("modelName") or ""
            rows.append(dict(node_id=nd.get("id"), node_name=data.get("label"), node_type=data.get("name"), node_version=data.get("version"),
                             role="embedding", provider_raw=provider_of(data.get("name")), model_raw=m if isinstance(m, str) else "",
                             resolution="explicit" if m else "default", via_router=False))
    for nd in nodes:
        data = nd.get("data") or {}
        cat = data.get("category")
        if cat not in CONSUMER_CATS and data.get("name") not in ("seqLLMNode", "seqConditionAgent"):
            continue
        inputs = data.get("inputs") or {}
        refs = []
        for k, v in inputs.items():
            if isinstance(v, str):
                for mid in re.findall(r"\{\{([^.}]+)\.data\.instance\}\}", v):
                    if mid in model_nodes:
                        refs.append(mid)
        role = "agent" if cat in AGENT_CATS else "chat"
        if data.get("name") in ("seqCondition", "seqConditionAgent"):
            role = "classifier"
        if not refs:
            continue  # consumer without a model input (e.g. retriever-only) -> not a generation site
        for mid in refs:
            mn = model_nodes[mid]
            rows.append(dict(node_id=nd.get("id"), node_name=data.get("label"), node_type=data.get("name"), node_version=data.get("version"),
                             role=role, via_router=False, **mn))
        site_ids.add(nd.get("id"))
    edges, main_edges = [], []
    by_id = {nd.get("id"): (nd.get("data") or {}).get("name", "") for nd in nodes}
    for e in j.get("edges") or []:
        s, t = e.get("source"), e.get("target")
        if s in model_nodes:
            continue
        edges.append((by_id.get(s, ""), by_id.get(t, ""), "main"))
        main_edges.append((s, t))
    return rows, types, edges, main_edges, site_ids


def parse_v2(j, source, wid, name):
    nodes = j.get("nodes") or []
    rows, types, site_ids = [], [], set()
    for nd in nodes:
        data = nd.get("data") or {}
        nm = data.get("name") or ""
        types.append((nm, data.get("version") or ""))
        if nm in V2:
            mk, ck, role = V2[nm]
            inputs = data.get("inputs") or {}
            prov = provider_of(inputs.get(mk) or "")
            cfg = inputs.get(ck) or {}
            m = cfg.get("modelName") if isinstance(cfg, dict) else ""
            m = m if isinstance(m, str) else ""
            rows.append(dict(node_id=nd.get("id"), node_name=data.get("label"), node_type=nm, node_version=data.get("version"),
                             role=role, provider_raw=prov if inputs.get(mk) else "unknown", model_raw=m,
                             resolution="explicit" if m else ("default" if inputs.get(mk) else "unknown"), via_router=False))
            site_ids.add(nd.get("id"))
    by_id = {nd.get("id"): (nd.get("data") or {}).get("name", "") for nd in nodes}
    edges, main_edges = [], []
    for e in j.get("edges") or []:
        s, t = e.get("source"), e.get("target")
        edges.append((by_id.get(s, ""), by_id.get(t, ""), "main"))
        main_edges.append((s, t))
    return rows, types, edges, main_edges, site_ids


if __name__ == "__main__":
    wrows, srows, fails = [], [], []
    for path in sorted(glob.glob(str(RAW / "flowise" / "*" / "*.json"))):
        source = Path(path).parent.name
        wid = f"{source}:{Path(path).stem}"
        name = Path(path).stem
        try:
            j = json.load(open(path, encoding="utf-8"))
            fn = parse_v2 if source == "agentflowsv2" else parse_v1
            rows, types, edges, main_edges, site_ids = fn(j, source, wid, name)
            ids = [nd.get("id") for nd in j.get("nodes") or []]
            wrows.append(dict(corpus="flowise", source=source, workflow_id=wid, name=name, n_nodes=len(types),
                              graph_hash=graph_hash(types, edges), chain_len=longest_site_path(ids, main_edges, site_ids),
                              has_router=any(r["role"] == "classifier" for r in rows), views=None, template_id=None))
            for r in rows:
                r.update(corpus="flowise", source=source, workflow_id=wid, name=name)
            srows += rows
        except Exception as e:  # noqa: BLE001
            fails.append(dict(corpus="flowise", workflow_id=wid, error=repr(e)))
    write_jsonl(PARSED / "flowise_workflows.jsonl", wrows)
    write_jsonl(PARSED / "flowise_callsites.jsonl", srows)
    write_jsonl(PARSED / "flowise_failures.jsonl", fails)
    log(f"flowise: {len(wrows)} workflows, {len(srows)} call-site rows, {len(fails)} failures")
