"""Parse n8n workflow JSON (api.n8n.io details or HF dumps) into call-site rows + workflow rows.

Call site = a node that invokes an LLM at runtime (agent, chain, extractor, classifier, or a
direct vendor node). Its model comes from the LM sub-node wired in through an
`ai_languageModel` connection (possibly via a `modelSelector` router), or from the node's own
`modelId`/`model` parameter for direct vendor nodes.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, RAW, graph_hash, log, longest_site_path, read_jsonl, write_jsonl  # noqa: E402

LC = "@n8n/n8n-nodes-langchain."

INVOKERS = {  # invoking node short name -> role (None = takes a model but is not a generation site)
    "agent": "agent", "agentTool": "agent", "chainLlm": "chat", "chainSummarization": "chat",
    "chainRetrievalQa": "chat", "informationExtractor": "extractor", "textClassifier": "classifier",
    "sentimentAnalysis": "classifier", "outputParserAutofixing": None,
}
DIRECT = {  # direct vendor nodes (carry their own model id) -> provider
    "openAi": "openai", "googleGemini": "google", "anthropic": "anthropic", "ollama": "ollama",
    "mistralAi": "mistral", "perplexity": "perplexity", "xAiGrok": "xai", "deepSeek": "deepseek",
    "cohere": "cohere", "groq": "groq", "openRouter": "openrouter", "vercelAiGateway": "vercel",
    "azureOpenAi": "azure_openai",
}
LEGACY_DIRECT = {"n8n-nodes-base.openAi": "openai"}
LM_PROVIDER = {
    "lmChatOpenAi": "openai", "lmOpenAi": "openai", "lmChatGoogleGemini": "google",
    "lmChatGoogleVertex": "vertex", "lmChatAnthropic": "anthropic", "lmChatOllama": "ollama",
    "lmOllama": "ollama", "lmChatOpenRouter": "openrouter", "lmChatGroq": "groq",
    "lmChatDeepSeek": "deepseek", "lmChatMistralCloud": "mistral", "lmChatAzureOpenAi": "azure_openai",
    "lmCohere": "cohere", "lmChatCohere": "cohere", "lmOpenHuggingFaceInference": "huggingface",
    "lmChatXAiGrok": "xai", "lmChatAwsBedrock": "bedrock", "lmChatVercelAiGateway": "vercel",
    "lmChatPerplexity": "perplexity", "lmChatOpenAiCompatible": "openai_compatible",
    "lmChatAzureAiFoundry": "azure_openai",
}
IMAGE_AUDIO = {"image", "audio", "video", "file", "vision"}


def model_value(p):
    """(model_raw, resolution) from a parameters dict."""
    val = None
    for k in ("model", "modelName", "modelId"):
        if k in p and p[k] not in (None, ""):
            val = p[k]
            break
    if val is None:
        return "", "default"
    if isinstance(val, dict):
        val = val.get("value") or val.get("cachedResultName") or ""
    if not isinstance(val, str):
        return str(val), "explicit"
    s = val.strip()
    if "{{" in s:
        return s, "expression"
    if s.startswith("="):
        s = s[1:].strip()
    if s.startswith("models/"):
        s = s[len("models/"):]
    if not s:
        return "", "default"
    return s, "explicit"


def lm_provider(short, p):
    prov = LM_PROVIDER.get(short)
    if prov is None:
        prov = re.sub(r"^lm(Chat)?", "", short).lower() or "unknown"
    opts = p.get("options") or {}
    if prov == "openai" and isinstance(opts, dict) and opts.get("baseURL"):
        prov = "openai_compatible"
    return prov


def parse_workflow(wf, corpus, source, wid, name, views=None):
    nodes = [n for n in (wf.get("nodes") or []) if isinstance(n, dict)]
    conns = wf.get("connections") or {}
    by_name = {n.get("name"): n for n in nodes}
    rows = []

    def short(n):
        t = n.get("type", "")
        return t[len(LC):] if t.startswith(LC) else None

    def iter_edges():
        """(src_name, kind, dst_name) over a connections block, tolerating malformed shapes."""
        if not isinstance(conns, dict):
            return
        for src, kinds in conns.items():
            if not isinstance(kinds, dict):
                continue
            for kind, outs in kinds.items():
                for out in outs or []:
                    for e in out or []:
                        if isinstance(e, dict) and e.get("node"):
                            yield src, kind, e["node"]

    lm_targets = {}
    for src, kind, dst in iter_edges():
        if kind == "ai_languageModel":
            lm_targets.setdefault(src, []).append(dst)

    def consumers_of(lm_name, depth=0):
        res = []
        for tgt in lm_targets.get(lm_name, []):
            tn = by_name.get(tgt)
            if tn is None:
                continue
            if short(tn) == "modelSelector" and depth < 3:
                res += [(c, True) for c, _ in consumers_of(tgt, depth + 1)]
            else:
                res.append((tgt, False))
        return res

    site_names = set()
    attached = {}
    for n in nodes:
        s = short(n)
        p = n.get("parameters") or {}
        if not s or not (s.startswith("lm") or s.startswith("embeddings") or s.startswith("reranker")):
            continue
        m, res = model_value(p)
        if s.startswith("embeddings") or s.startswith("reranker"):
            rows.append(dict(node_id=n.get("id"), node_name=n.get("name"), node_type=s, node_version=n.get("typeVersion"),
                             role="embedding" if s.startswith("embeddings") else "reranker",
                             provider_raw=re.sub(r"^(embeddings|reranker)", "", s).lower(),
                             model_raw=m, resolution=res, via_router=False, lm_node=s))
            continue
        prov = lm_provider(s, p)
        cons = consumers_of(n.get("name"))
        if not cons:  # LM sub-node not wired to any consumer: declared but never invoked -> excluded from headline
            rows.append(dict(node_id=n.get("id"), node_name=n.get("name"), node_type=s, node_version=n.get("typeVersion"),
                             role="dangling", provider_raw=prov, model_raw=m, resolution=res, via_router=False, lm_node=s))
            continue
        for cname, via_router in cons:
            cn = by_name.get(cname, n)
            cs = short(cn) or cn.get("type", "")
            if cs.startswith("outputParser"):
                continue
            role = INVOKERS.get(cs, "chat")
            if role is None:
                continue
            r = dict(node_id=cn.get("id"), node_name=cname, node_type=cs, node_version=cn.get("typeVersion"),
                     role=role, provider_raw=prov, model_raw=m, resolution=res, via_router=via_router, lm_node=s)
            attached.setdefault(cname, []).append(r)
            site_names.add(cname)
    for rs in attached.values():
        rows += rs
    for n in nodes:
        s = short(n)
        p = n.get("parameters") or {}
        t = n.get("type", "")
        if s in INVOKERS and INVOKERS[s] and n.get("name") not in attached:
            rows.append(dict(node_id=n.get("id"), node_name=n.get("name"), node_type=s, node_version=n.get("typeVersion"),
                             role=INVOKERS[s], provider_raw="unknown", model_raw="", resolution="unknown", via_router=False, lm_node=""))
            site_names.add(n.get("name"))
        elif s in DIRECT or t in LEGACY_DIRECT:
            prov = DIRECT.get(s) or LEGACY_DIRECT.get(t)
            resource = p.get("resource") or "text"
            operation = p.get("operation") or ""
            m, res = model_value(p)
            role = "image_audio" if (resource in IMAGE_AUDIO and not (resource == "image" and operation == "analyze")) else "chat"
            if resource == "assistant" and not m:
                res = "unknown"
            rows.append(dict(node_id=n.get("id"), node_name=n.get("name"), node_type=s or t, node_version=n.get("typeVersion"),
                             role=role, provider_raw=prov, model_raw=m, resolution=res, via_router=False, lm_node="", resource=resource))
            if role != "image_audio":
                site_names.add(n.get("name"))

    types = [(n.get("type", ""), n.get("typeVersion", "")) for n in nodes if n.get("type") != "n8n-nodes-base.stickyNote"]
    edges, main_edges = [], []
    for src, kind, dst in iter_edges():
        st = (by_name.get(src) or {}).get("type", "")
        dt = (by_name.get(dst) or {}).get("type", "")
        edges.append((st, dt, kind))
        if kind == "main":
            main_edges.append((src, dst))
    wrow = dict(corpus=corpus, source=source, workflow_id=wid, name=name, n_nodes=len(types),
                graph_hash=graph_hash(types, edges),
                chain_len=longest_site_path([n.get("name") for n in nodes], main_edges, site_names),
                has_router=any(short(n) == "modelSelector" for n in nodes), views=views,
                template_id=(wf.get("meta") or {}).get("templateId"))
    for r in rows:
        r.update(corpus=corpus, source=source, workflow_id=wid, name=name)
    return wrow, rows


def iter_api():
    seen = set()
    for f in sorted((RAW / "n8n_api").glob("workflows*.jsonl")):
        for r in read_jsonl(f):
            if "error" in r or r["id"] in seen:
                continue
            seen.add(r["id"])
            yield "n8n_api", "api", str(r["id"]), r.get("name"), r, r.get("totalViews")


def iter_hf():
    src = RAW / "hf"
    specs = [("mbakgun", src / "mbakgun" / "train.jsonl", "output"),
             ("ruh-ai", src / "ruh-ai" / "filtered dataset.jsonl", "json"),
             ("npv2k1", src / "npv2k1" / "dataset.json", None)]
    for tag, path, key in specs:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            it = read_jsonl(path)
        else:
            d = json.load(open(path, encoding="utf-8"))
            it = d if isinstance(d, list) else d.get("data", [])
        for i, row in enumerate(it):
            wf = row.get(key) if key else None
            if wf is None:
                for k in ("output", "json", "workflow", "workflow_json", "content"):
                    if k in row:
                        wf = row[k]
                        break
            if isinstance(wf, str):
                try:
                    wf = json.loads(wf)
                except Exception:  # noqa: BLE001
                    continue
            if not isinstance(wf, dict) or "nodes" not in wf:
                continue
            yield "n8n_hf", tag, f"{tag}:{i}", wf.get("name"), wf, None


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    wrows, srows, fails = [], [], []
    its = ([iter_api()] if which in ("all", "api") else []) + ([iter_hf()] if which in ("all", "hf") else [])
    for it in its:
        for corpus, source, wid, name, wf, views in it:
            try:
                w, rs = parse_workflow(wf, corpus, source, wid, name, views)
                wrows.append(w)
                srows += rs
            except Exception as e:  # noqa: BLE001
                fails.append(dict(corpus=corpus, workflow_id=wid, error=repr(e)))
    tag = "" if which == "all" else "_" + which
    write_jsonl(PARSED / f"n8n{tag}_workflows.jsonl", wrows)
    write_jsonl(PARSED / f"n8n{tag}_callsites.jsonl", srows)
    write_jsonl(PARSED / f"n8n{tag}_failures.jsonl", fails)
    log(f"n8n[{which}]: {len(wrows)} workflows, {len(srows)} call-site rows, {len(fails)} failures")
