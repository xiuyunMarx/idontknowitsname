"""Normalise (provider_raw, model_raw) -> vendor, family, size_b, tier, open_weight, provider_class.

Rules are regex-driven and deliberately conservative: anything unmatched keeps its raw string
as `model_norm` (still counts as a distinct model) and is listed for review.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import PARSED, log, read_jsonl, write_jsonl  # noqa: E402

PROVIDER_CLASS = {
    "self_host": {"ollama", "lmstudio", "localai", "xinference", "openai_compatible", "openai_api_compatible", "vllm",
                  "gpustack", "openllm", "lemonade", "nvidia"},
    "open_hosted": {"huggingface", "groq", "together", "fireworks", "siliconflow", "deepinfra", "replicate", "openrouter",
                    "modelscope", "cerebras", "sambanova", "novita", "hyperbolic"},
    "closed_api": {"openai", "azure_openai", "azureopenai", "anthropic", "google", "googlegemini", "gemini", "vertex", "vertex_ai",
                   "xai", "mistral", "mistralcloud", "cohere", "bedrock", "perplexity", "vercel", "volcengine_maas", "tongyi",
                   "moonshot", "minimax", "zhipuai", "baichuan", "wenxin", "spark", "hunyuan", "yi", "stepfun", "deepseek",
                   "azure_ai_studio", "alibabacloud", "chatglm"},
}
PROVIDER_ALIAS = {"googlegemini": "google", "gemini": "google", "azureopenai": "azure_openai", "mistralcloud": "mistral",
                  "vertex_ai": "vertex", "openai_api_compatible": "openai_compatible", "alibabacloud": "tongyi"}

# (regex on lowercase model id, vendor, family, size_b or None, open_weight)
RULES = [
    # OpenAI (closed)
    (r"^(chatgpt-4o|gpt-4o)", "openai", "gpt-4o", None, False),
    (r"^gpt-4\.1", "openai", "gpt-4.1", None, False),
    (r"^gpt-5", "openai", "gpt-5", None, False),
    (r"^gpt-4", "openai", "gpt-4", None, False),
    (r"^gpt-3\.5|^text-davinci|^davinci", "openai", "gpt-3.5", None, False),
    (r"^o[134](-|$)", "openai", "o-series", None, False),
    (r"^gpt-oss(?:-safeguard)?-?(\d+)b", "openai", "gpt-oss", "g1", True),
    (r"^typhoon[\w.-]*?(\d+)b", "scb10x", "typhoon", "g1", True),
    (r"^dots-", "rednote", "dots", None, True),
    (r"^text-embedding|^embedding", "openai", "embedding", None, False),
    # Anthropic (incl. Bedrock ARN-style ids: anthropic.claude-3-5-sonnet-20240620-v1:0, us.anthropic.claude-...)
    (r"^(us\.|eu\.|apac\.)?anthropic\.claude|^claude", "anthropic", "claude", None, False),
    (r"^gpt-?35-turbo", "openai", "gpt-3.5", None, False),
    (r"^gpt4o|gpt-?4o(?!-mini)|^4o$", "openai", "gpt-4o", None, False),   # azure deployment names: intelgpt4o, 4o
    (r"gpt-?4o-?mini", "openai", "gpt-4o", None, False),
    (r"gpt-?35|gpt-?3\.5", "openai", "gpt-3.5", None, False),
    (r"^gpt-?4(?!o|\.)", "openai", "gpt-4", None, False),
    (r"textembedding|text-embedding|^embed", None, "embedding", None, False),
    (r"^gpt-image|^dall-e|^sdxl|^flux|^stable-diffusion|^imagen|^whisper|^tts-", None, "image_audio", None, None),
    (r"^falcon[\w.-]*?(\d+)b", "tii", "falcon", "g1", True),
    (r"^allam[\w.-]*?(\d+)b", "sdaia", "allam", "g1", True),
    (r"^compound", "groq", "compound", None, False),
    # Google
    (r"^gemini", "google", "gemini", None, False),
    (r"^gemma-?\d?-?(\d+)b", "google", "gemma", "g1", True),
    (r"^gemma", "google", "gemma", None, True),
    # xAI / Mistral / Cohere / Perplexity
    (r"^grok", "xai", "grok", None, False),
    (r"^mistral-(large|medium|small|tiny|nemo)", "mistral", "mistral-api", None, False),
    (r"mixtral-?8x(\d+)b", "mistral", "mixtral", "g1x8", True),
    (r"^gpt-?5-?mini|^gpt5mini", "openai", "gpt-5", None, False),
    (r"^exaone[\w.-]*?(\d+(?:\.\d+)?)b", "lg", "exaone", "g1", True),
    (r"^exaone", "lg", "exaone", None, True),
    (r"^amazon\.titan|^amazon\.nova|^nova-", "amazon", "titan/nova", None, False),
    (r"^(us\.|eu\.)?meta\.llama[\w.-]*?(\d+)b", "meta", "llama", "g2", True),
    (r"^mistral(-|:|$)", "mistral", "mistral", 7.0, True),
    (r"^codestral|^ministral|^pixtral|^magistral|^devstral", "mistral", "mistral-api", None, False),
    (r"^command", "cohere", "command", None, False),
    (r"^rerank", "cohere", "rerank", None, False),
    (r"^sonar|^llama-3\.1-sonar", "perplexity", "sonar", None, False),
    # DeepSeek
    (r"^deepseek-r1-distill-(qwen|llama)-(\d+)b", "deepseek", "r1-distill", "g2", True),
    (r"^deepseek-r1|^deepseek-reasoner", "deepseek", "deepseek-r1", 671.0, True),
    (r"^deepseek-?(chat|v3|coder|v2)", "deepseek", "deepseek-v3", 671.0, True),
    (r"^deepseek", "deepseek", "deepseek", None, True),
    # Qwen
    (r"^qwen[\w.-]*?(\d+(?:\.\d+)?)b", "alibaba", "qwen", "g1", True),
    (r"^qwen", "alibaba", "qwen", None, True),
    (r"^qwq", "alibaba", "qwq", 32.0, True),
    (r"^qvq", "alibaba", "qvq", 72.0, True),
    # Meta Llama
    (r"^(meta-)?llama[\w.-]*?(\d+(?:\.\d+)?)b", "meta", "llama", "g2", True),
    (r"^(meta-)?llama3\.2$", "meta", "llama", 3.0, True),
    (r"^(meta-)?llama3\.1$", "meta", "llama", 8.0, True),
    (r"^(meta-)?llama3$", "meta", "llama", 8.0, True),
    (r"^(meta-)?llama", "meta", "llama", None, True),
    # Microsoft Phi
    (r"^phi-?\d[\w.-]*?(\d+(?:\.\d+)?)b", "microsoft", "phi", "g1", True),
    (r"^phi", "microsoft", "phi", 3.8, True),
    # Others open
    (r"^nemotron[\w.-]*?(\d+)b", "nvidia", "nemotron", "g1", True),
    (r"^nemotron", "nvidia", "nemotron", None, True),
    (r"^glm|^chatglm", "zhipu", "glm", None, True),
    (r"^yi-", "01ai", "yi", None, True),
    (r"^kimi", "moonshot", "kimi", None, True),
    (r"^moonshot", "moonshot", "moonshot-api", None, False),
    (r"^internlm", "shanghai-ai-lab", "internlm", None, True),
    (r"^baichuan", "baichuan", "baichuan", None, True),
    (r"^doubao", "bytedance", "doubao", None, False),
    (r"^hunyuan", "tencent", "hunyuan", None, False),
    (r"^ernie", "baidu", "ernie", None, False),
    (r"^abab|^minimax", "minimax", "minimax", None, False),
    (r"^step-", "stepfun", "step", None, False),
    (r"^spark", "iflytek", "spark", None, False),
    (r"^deepcoder[\w.-]*?(\d+)b", "agentica", "deepcoder", "g1", True),
    (r"^tinyllama", "tinyllama", "tinyllama", 1.1, True),
    (r"^dolphin[\w.-]*?(\d+)b", "cognitive-computations", "dolphin", "g1", True),
    (r"^lfm[\w.-]*?(\d+(?:\.\d+)?)b", "liquid", "lfm", "g1", True),
    (r"^ling-", "inclusionai", "ling", None, True),
    (r"^c4ai-aya[\w.-]*?(\d+)b", "cohere", "aya", "g1", True),
    (r"^wayfarer[\w.-]*?(\d+)b", "latitude", "wayfarer", "g1", True),
    (r"^nanonets-ocr", "nanonets", "nanonets-ocr", 3.0, True),
    (r"^bespoke-minicheck", "bespoke", "minicheck", 7.0, True),
    (r"^sherlock", "openrouter", "stealth", None, None),
    (r"^gpt-realtime", "openai", "gpt-realtime", None, False),
    (r"^(nomic|bge|mxbai|all-minilm|e5|gte|jina)", None, "embedding", None, True),
    (r"^(text-embedding|embed-)", None, "embedding", None, False),
]


def _size_from(m, spec):
    if spec is None:
        return None
    if isinstance(spec, float):
        return spec
    if spec == "g1":
        return float(m.group(1))
    if spec == "g2":
        return float(m.group(2))
    if spec == "g1x8":
        return float(m.group(1)) * 8 * 0.83
    return None


def clean_model(raw):
    s = (raw or "").strip().lower()
    s = re.sub(r"^models/", "", s)
    s = s.split("?")[0]
    # strip hosted-route prefixes like "openai/gpt-4o", "google/gemini-2.0-flash", "meta-llama/llama-3.3-70b-instruct"
    if "/" in s and not s.startswith("deepseek-ai/"):
        s = s.split("/")[-1]
    elif s.startswith("deepseek-ai/"):
        s = s.split("/")[-1]
    s = s.replace("_", "-")
    s = re.sub(r":(free|latest|nitro|online|floor|exacto|thinking)$", "", s)   # OpenRouter routing / Ollama tag suffixes without size info
    return s


PLACEHOLDER = re.compile(r"^(free|auto$|your-|replace-with|ai-model$|\{your|model-name|<|your_|xxx|todo|changeme|example)")


def tier_of(vendor, family, size_b, name, open_weight):
    if size_b is not None:
        return "xs" if size_b <= 2 else "s" if size_b <= 9 else "m" if size_b <= 35 else "l"
    n = name or ""
    if re.search(r"nano|mini|flash-lite|flash|lite|haiku|small|tiny|8b|7b|3b|1b", n):
        return "s"
    if re.search(r"pro|opus|o1|o3|gpt-5(?!-mini|-nano)|gpt-4\.1$|gpt-4o$|gpt-4$|gpt-4-turbo|large|sonnet|reasoner|r1|v3|max|ultra|70b|72b|405b|deepseek-chat", n):
        return "l"
    return "m"


def normalise(provider_raw, model_raw, resolution):
    prov = PROVIDER_ALIAS.get((provider_raw or "unknown").lower(), (provider_raw or "unknown").lower())
    pclass = next((k for k, v in PROVIDER_CLASS.items() if prov in v), "unknown")
    name = clean_model(model_raw)
    out = dict(provider=prov, provider_class=pclass, model_clean=name, vendor=None, family=None, size_b=None,
               tier=None, open_weight=None, model_norm=None, matched=False)
    if name and PLACEHOLDER.search(name) and resolution == "explicit":
        resolution = "default"          # template placeholder ("your-model-name"): the author left the choice to the deployer
        out["placeholder"] = True
    out["resolution"] = resolution
    if resolution in ("default", "unknown", "expression") or not name:
        out["model_norm"] = f"{prov}:{resolution}"
        out["vendor"] = prov if pclass == "closed_api" else None
        return out
    for rx, vendor, family, size_spec, open_w in RULES:
        m = re.search(rx, name)
        if m:
            size = _size_from(m, size_spec)
            out.update(vendor=vendor or prov, family=family, size_b=size, open_weight=open_w, matched=True)
            break
    if out["matched"]:
        out["tier"] = tier_of(out["vendor"], out["family"], out["size_b"], name, out["open_weight"])
    else:
        out["vendor"] = prov
        if pclass == "self_host":
            out["open_weight"] = True
        out["tier"] = tier_of(prov, None, None, name, None)
    out["model_norm"] = name
    return out


if __name__ == "__main__":
    files = sorted(PARSED.glob("*_callsites.jsonl"))
    rows, unmatched = [], {}
    for f in files:
        for r in read_jsonl(f):
            n = normalise(r.get("provider_raw"), r.get("model_raw"), r.get("resolution"))
            r.update(n)
            rows.append(r)
            if not n["matched"] and r.get("resolution") == "explicit" and r.get("role") in ("chat", "agent", "classifier", "extractor"):
                unmatched[n["model_clean"]] = unmatched.get(n["model_clean"], 0) + 1
    write_jsonl(PARSED / "callsites_norm.jsonl", rows)
    top = sorted(unmatched.items(), key=lambda x: -x[1])
    json.dump(dict(top), open(PARSED / "unmatched_models.json", "w"), indent=1, ensure_ascii=False)
    log(f"normalised {len(rows)} rows from {len(files)} files; {len(unmatched)} unmatched explicit ids "
        f"({sum(unmatched.values())} rows); top: {top[:15]}")
