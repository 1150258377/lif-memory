from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


@dataclass
class EmbeddingConfig:
    enabled: bool
    provider: str = "api"
    base_url: str = "https://api.openai.com/v1"
    model: str = DEFAULT_EMBEDDING_MODEL
    api_key: str = ""
    source_path: str = ""
    device: str = "cpu"
    model_class: str = "auto"
    timeout: int = 45


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_embedding_config(project_dir: Path | None = None) -> EmbeddingConfig:
    """Load an OpenAI-compatible embedding config without requiring it.

    Private values should live in environment variables or config/*.local.json.
    The default is disabled, so the sparse field keeps working offline.
    """
    project_dir = project_dir or Path.cwd()
    local = _read_json(project_dir / "config" / "embedding.local.json")
    provider = (os.environ.get("LIF_EMBEDDING_PROVIDER") or str(local.get("provider") or "api")).strip().lower()
    base_url = os.environ.get("LIF_EMBEDDING_BASE_URL") or str(local.get("base_url") or "https://api.openai.com/v1")
    model = (
        os.environ.get("LIF_EMBEDDING_MODEL_PATH")
        or str(local.get("model_path") or "")
        or os.environ.get("LIF_EMBEDDING_MODEL")
        or str(local.get("model") or (DEFAULT_EMBEDDING_MODEL if provider == "api" else ""))
    )
    api_key = os.environ.get("LIF_EMBEDDING_API_KEY") or str(local.get("api_key") or "")
    default_flag_source = Path(r"C:\Users\Administrator\Downloads\FlagEmbedding-master\FlagEmbedding-master")
    source_path = (
        os.environ.get("LIF_FLAGEMBEDDING_SOURCE")
        or str(local.get("source_path") or "")
        or (str(default_flag_source) if default_flag_source.exists() else "")
    )
    device = os.environ.get("LIF_EMBEDDING_DEVICE") or str(local.get("device") or "cpu")
    model_class = os.environ.get("LIF_EMBEDDING_MODEL_CLASS") or str(local.get("model_class") or "auto")
    enabled_flag = str(local.get("enabled", os.environ.get("LIF_EMBEDDING_ENABLED", "1"))).lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if provider in {"flag", "flagembedding", "local", "sentence_transformers", "transformers"}:
        enabled = enabled_flag and bool(model)
    else:
        provider = "api"
        enabled = enabled_flag and bool(api_key)
    return EmbeddingConfig(
        enabled=enabled,
        provider=provider,
        base_url=base_url.rstrip("/"),
        model=model,
        api_key=api_key,
        source_path=source_path,
        device=device,
        model_class=model_class,
    )


def _norm_dense(values: Iterable[float]) -> list[float]:
    vec = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def dense_cosine(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    if n <= 0:
        return 0.0
    return max(0.0, sum(a[i] * b[i] for i in range(n)))


_LOCAL_EMBEDDERS: dict[tuple[str, str, str, str, str], tuple[str, Any]] = {}


def _vector_to_list(raw: Any) -> list[float]:
    if isinstance(raw, dict):
        for key in ("dense_vecs", "sentence_embedding", "embeddings"):
            if key in raw:
                return _vector_to_list(raw[key])
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        raw = raw[0]
    if not isinstance(raw, list):
        return []
    return _norm_dense(float(v) for v in raw)


def _load_local_embedder(config: EmbeddingConfig) -> tuple[str, Any]:
    key = (config.provider, config.model, config.source_path, config.device, config.model_class)
    if key in _LOCAL_EMBEDDERS:
        return _LOCAL_EMBEDDERS[key]

    if config.source_path:
        source = Path(config.source_path)
        if source.exists():
            sys.path.insert(0, str(source))

    flag_error: Exception | None = None
    if config.provider in {"flag", "flagembedding", "local"}:
        try:
            from FlagEmbedding import BGEM3FlagModel, FlagAutoModel, FlagModel

            model_lower = config.model.lower()
            class_hint = config.model_class.lower()
            if class_hint == "m3" or "bge-m3" in model_lower:
                model = BGEM3FlagModel(config.model, use_fp16=False, device=config.device)
            elif class_hint == "flagmodel":
                model = FlagModel(config.model, use_fp16=False, device=config.device)
            else:
                model = FlagAutoModel.from_finetuned(config.model, use_fp16=False, devices=[config.device])
            _LOCAL_EMBEDDERS[key] = ("flag", model)
            return _LOCAL_EMBEDDERS[key]
        except Exception as exc:
            flag_error = exc

    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.model, device=config.device)
        _LOCAL_EMBEDDERS[key] = ("sentence_transformers", model)
        return _LOCAL_EMBEDDERS[key]
    except Exception as st_exc:
        if flag_error is not None:
            raise RuntimeError(f"FlagEmbedding load failed: {flag_error}; SentenceTransformer fallback failed: {st_exc}") from st_exc
        raise


def _embed_local(config: EmbeddingConfig, text: str) -> list[float]:
    kind, model = _load_local_embedder(config)
    if kind == "flag":
        if hasattr(model, "encode_queries"):
            raw = model.encode_queries([text], batch_size=1, max_length=512, convert_to_numpy=True)
        else:
            raw = model.encode([text], batch_size=1, max_length=512, convert_to_numpy=True)
        return _vector_to_list(raw)
    raw = model.encode([text], batch_size=1, normalize_embeddings=True)
    return _vector_to_list(raw)


class EmbeddingCache:
    def __init__(self, path: Path, config: EmbeddingConfig):
        self.path = path
        self.config = config
        self.items: dict[str, Any] = _read_json(path)
        if not isinstance(self.items, dict):
            self.items = {}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def key_for(self, namespace: str, text: str) -> str:
        raw = f"{self.config.provider}\n{self.config.model}\n{self.config.source_path}\n{namespace}\n{text[:12000]}".encode("utf-8", errors="ignore")
        return hashlib.sha256(raw).hexdigest()

    def embed(self, namespace: str, text: str) -> list[float]:
        if not self.config.enabled:
            return []
        key = self.key_for(namespace, text)
        cached = self.items.get(key)
        if isinstance(cached, list) and cached:
            return [float(v) for v in cached]

        if self.config.provider == "api":
            url = self.config.base_url + "/embeddings"
            payload = json.dumps({"model": self.config.model, "input": text[:12000]}, ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vec = _norm_dense(data["data"][0]["embedding"])
        else:
            vec = _embed_local(self.config, text[:12000])
        self.items[key] = vec
        return vec


def hydrate_note_embeddings(notes: list[Any], vault: Path, project_dir: Path | None = None) -> tuple[bool, str]:
    config = load_embedding_config(project_dir)
    if not config.enabled:
        return False, "embedding disabled or missing key"
    cache = EmbeddingCache(vault / "lif_embedding_cache.json", config)
    changed = False
    for note in notes:
        if getattr(note, "dense_vector", None):
            continue
        text = f"{getattr(note, 'title', '')}\n{getattr(note, 'text', '')}"
        try:
            note.dense_vector = cache.embed(getattr(note, "rel_path", "note"), text)
            changed = True
            time.sleep(0.02)
        except Exception as exc:
            return False, f"embedding failed: {exc}"
    if changed:
        cache.save()
    return True, f"embedded {len(notes)} notes with {config.model}"


def embed_query(query: str, vault: Path, project_dir: Path | None = None) -> list[float]:
    config = load_embedding_config(project_dir)
    if not config.enabled:
        return []
    cache = EmbeddingCache(vault / "lif_embedding_cache.json", config)
    vec = cache.embed("query", query)
    cache.save()
    return vec


NEURON_CONFIG: dict[str, dict[str, float]] = {
    "semantic_density": {"threshold": 1.8, "leak": 0.86, "reset": 0.35},
    "novelty": {"threshold": 1.2, "leak": 0.80, "reset": 0.20},
    "conflict": {"threshold": 1.1, "leak": 0.84, "reset": 0.25},
    "action_pressure": {"threshold": 1.4, "leak": 0.82, "reset": 0.25},
    "integration": {"threshold": 2.2, "leak": 0.90, "reset": 0.45},
}


def _default_domain_state() -> dict[str, Any]:
    return {
        "version": 1,
        "concepts": {},
        "relations": {},
        "neurons": {
            name: {"voltage": 0.0, "spikes": 0, "threshold": cfg["threshold"], "leak": cfg["leak"]}
            for name, cfg in NEURON_CONFIG.items()
        },
        "spike_events": [],
        "reward_history": [],
        "last_observation": {},
    }


def load_domain_state(vault: Path) -> dict[str, Any]:
    path = vault / "lif_domain_state.json"
    state = _read_json(path)
    if not state:
        state = _default_domain_state()
    defaults = _default_domain_state()
    for key, value in defaults.items():
        state.setdefault(key, value)
    for name, cfg in NEURON_CONFIG.items():
        state["neurons"].setdefault(
            name, {"voltage": 0.0, "spikes": 0, "threshold": cfg["threshold"], "leak": cfg["leak"]}
        )
    return state


def save_domain_state(vault: Path, state: dict[str, Any]) -> None:
    path = vault / "lif_domain_state.json"
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _bump(mapping: dict[str, Any], key: str, amount: float) -> None:
    item = mapping.setdefault(key, {"count": 0, "energy": 0.0, "last_seen": ""})
    item["count"] = int(item.get("count", 0)) + 1
    item["energy"] = round(float(item.get("energy", 0.0)) + amount, 4)
    item["last_seen"] = time.strftime("%Y-%m-%d %H:%M:%S")


def _neuron_step(state: dict[str, Any], signals: dict[str, float]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, cfg in NEURON_CONFIG.items():
        neuron = state["neurons"].setdefault(name, {})
        threshold = float(neuron.get("threshold", cfg["threshold"]))
        leak = float(neuron.get("leak", cfg["leak"]))
        voltage = float(neuron.get("voltage", 0.0)) * leak + float(signals.get(name, 0.0))
        fired = voltage >= threshold
        if fired:
            neuron["spikes"] = int(neuron.get("spikes", 0)) + 1
            voltage *= float(cfg.get("reset", 0.25))
        neuron["voltage"] = round(max(0.0, voltage), 4)
        neuron["threshold"] = threshold
        neuron["leak"] = leak
        out[name] = {"voltage": neuron["voltage"], "threshold": threshold, "spiked": fired}
    return out


def update_domain_after_query(
    vault: Path,
    query: str,
    topic: str,
    hits: list[Any],
    field_energy: float,
    voltage: float,
    spiked: bool,
) -> dict[str, Any]:
    state = load_domain_state(vault)
    _bump(state["concepts"], topic, field_energy)

    modifier_counts: dict[str, int] = {}
    top_titles = []
    for hit in hits[:8]:
        title = getattr(getattr(hit, "note", None), "title", "")
        if title:
            top_titles.append(title)
            relation_key = f"{topic}::{title}"
            _bump(state["relations"], relation_key, float(getattr(hit, "score", 0.0)))
        for mod in getattr(hit, "modifiers", []):
            modifier_counts[mod] = modifier_counts.get(mod, 0) + 1

    signals = {
        "semantic_density": min(2.0, field_energy),
        "novelty": 0.25 * modifier_counts.get("novelty", 0),
        "conflict": 0.35 * modifier_counts.get("blocker", 0),
        "action_pressure": 0.30 * modifier_counts.get("action", 0),
        "integration": min(2.5, field_energy * 0.55 + voltage * 0.25 + (0.8 if spiked else 0.0)),
    }
    neurons = _neuron_step(state, signals)
    observation = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "query": query,
        "topic": topic,
        "field_energy": round(field_energy, 4),
        "voltage": round(voltage, 4),
        "spiked": bool(spiked),
        "top_titles": top_titles,
        "modifier_counts": modifier_counts,
        "signals": signals,
        "neurons": neurons,
    }
    state["last_observation"] = observation
    if spiked or any(n["spiked"] for n in neurons.values()):
        state["spike_events"].append(observation)
        state["spike_events"] = state["spike_events"][-200:]
    save_domain_state(vault, state)
    return observation


POSITIVE_WORDS = ("好", "有用", "准确", "继续", "深入", "保存", "对", "清楚", "有价值", "很好")
NEGATIVE_WORDS = ("不对", "没用", "错误", "重复", "空泛", "不准确", "偏了", "幻觉", "没有价值")


def reward_from_feedback(text: str) -> float:
    lower = text.lower()
    score = 0.0
    for word in POSITIVE_WORDS:
        if word in lower:
            score += 0.25
    for word in NEGATIVE_WORDS:
        if word in lower:
            score -= 0.35
    return max(-1.0, min(1.0, score))


def reward_update(vault: Path, feedback: str, params: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    state = load_domain_state(vault)
    reward = reward_from_feedback(feedback)
    obs = state.get("last_observation") or {}
    delta: dict[str, float] = {}
    if reward != 0.0:
        voltage = float(obs.get("voltage", 0.0) or 0.0)
        spiked = bool(obs.get("spiked", False))
        if reward > 0:
            delta["dense_weight"] = float(params.get("dense_weight", 0.45)) + 0.03 * reward
            delta["sparse_weight"] = float(params.get("sparse_weight", 0.55)) - 0.015 * reward
            if not spiked and voltage > 0:
                delta["threshold"] = float(params.get("threshold", 5.0)) - 0.08 * reward
        else:
            delta["semantic_sigma"] = float(params.get("semantic_sigma", 0.55)) + 0.05 * abs(reward)
            delta["threshold"] = float(params.get("threshold", 5.0)) + 0.08 * abs(reward)
    record = {
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        "feedback": feedback,
        "reward": round(reward, 4),
        "delta": delta,
        "observation": obs,
    }
    state["reward_history"].append(record)
    state["reward_history"] = state["reward_history"][-500:]
    save_domain_state(vault, state)
    return delta, record


def domain_summary(vault: Path) -> dict[str, Any]:
    state = load_domain_state(vault)
    concepts = sorted(state.get("concepts", {}).items(), key=lambda item: item[1].get("energy", 0), reverse=True)[:12]
    relations = sorted(state.get("relations", {}).items(), key=lambda item: item[1].get("energy", 0), reverse=True)[:12]
    return {
        "concepts": [{"name": k, **v} for k, v in concepts],
        "relations": [{"name": k, **v} for k, v in relations],
        "neurons": state.get("neurons", {}),
        "last_observation": state.get("last_observation", {}),
        "reward_count": len(state.get("reward_history", [])),
        "spike_count": len(state.get("spike_events", [])),
    }
