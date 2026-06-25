"""
LIF-Memory 网页交互调参器 v2 — 持续进化版
- 首次扫描后场状态持久化，后续直接加载
- LIF 电位跨会话继承，不重置
- 对话历史全程传给 LLM，真正多轮进化
- 洞察追加写入进化日志 MD
零外部依赖，Python 标准库实现
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime, date

from lif_field_learning import (
    domain_summary,
    embed_query,
    hydrate_note_embeddings,
    reward_update,
    update_domain_after_query,
)

# ── 默认参数 ─────────────────────────────────────────────────────────────────

DEFAULT_PARAMS = {
    "threshold": 5.0,
    "semantic_sigma": 0.55,
    "time_sigma": 30.0,
    "graph_alpha": 0.35,
    "graph_steps": 2,
    "fast_decay": 0.76,
    "slow_decay": 0.94,
    "fast_weight": 0.65,
    "slow_weight": 0.35,
    "dense_weight": 0.45,
    "sparse_weight": 0.55,
    "topic_weights": {
        "LIF链路": 1.0, "连续问题场": 1.0, "AI记忆": 1.0,
        "论文闭环": 1.0, "负阻": 1.0, "求职": 1.0, "健康恢复": 1.0,
    },
}

PARAM_BOUNDS = {
    "threshold":      (1.0, 15.0),
    "semantic_sigma": (0.1, 2.0),
    "time_sigma":     (1.0, 365.0),
    "graph_alpha":    (0.0, 0.9),
    "graph_steps":    (0, 5),
    "fast_decay":     (0.3, 0.99),
    "slow_decay":     (0.5, 0.99),
    "fast_weight":    (0.1, 0.9),
    "slow_weight":    (0.1, 0.9),
    "dense_weight":   (0.0, 1.0),
    "sparse_weight":  (0.0, 1.0),
}

PROVIDER_PRESETS = {
    "qwen":     {"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus",     "env": "DASHSCOPE_API_KEY"},
    "deepseek": {"base_url": "https://api.deepseek.com",                          "model": "deepseek-chat", "env": "DEEPSEEK_API_KEY"},
    "kimi":     {"base_url": "https://api.moonshot.cn/v1",                        "model": "moonshot-v1-8k","env": "MOONSHOT_API_KEY"},
    "zhipu":    {"base_url": "https://open.bigmodel.cn/api/paas/v4",              "model": "glm-4-flash",   "env": "ZHIPUAI_API_KEY"},
}


def llm_local_config_paths() -> list[Path]:
    paths = [
        Path(__file__).resolve().parent / "config" / "llm.local.json",
        Path.home() / ".lif-memory" / "llm.local.json",
    ]
    env_path = os.environ.get("LIF_LLM_CONFIG")
    if env_path:
        paths.append(Path(env_path))
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve() if path.is_absolute() else path)
        if key not in seen:
            unique.append(path)
            seen.add(key)
    return unique


def llm_local_config_path() -> Path:
    for path in llm_local_config_paths():
        if path.exists():
            return path
    return llm_local_config_paths()[0]


def _read_llm_local_config() -> dict:
    merged: dict = {}
    merged_keys: dict = {}
    for path in llm_local_config_paths():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            keys = data.get("api_keys", {})
            if isinstance(keys, dict):
                merged_keys.update(keys)
            merged.update({k: v for k, v in data.items() if k != "api_keys"})
        except Exception:
            continue
    if merged_keys:
        merged["api_keys"] = merged_keys
    return merged


def build_llm_config(
    provider: str = "deepseek",
    *,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    prefer_local_provider: bool = True,
) -> dict:
    local_cfg = _read_llm_local_config()
    local_provider = str(local_cfg.get("provider") or "").strip()
    if prefer_local_provider and local_provider:
        provider = local_provider
    if provider not in PROVIDER_PRESETS:
        provider = "deepseek"

    preset = PROVIDER_PRESETS[provider]
    api_keys = local_cfg.get("api_keys", {})
    if not isinstance(api_keys, dict):
        api_keys = {}

    resolved_key = (
        api_key
        if api_key is not None
        else os.environ.get(preset["env"], "")
        or str(api_keys.get(provider) or local_cfg.get("api_key") or "")
    )

    return {
        "provider": provider,
        "base_url": (base_url or str(local_cfg.get("base_url") or preset["base_url"])).strip(),
        "model": (model or str(local_cfg.get("model") or preset["model"])).strip(),
        "api_key": str(resolved_key or "").strip(),
        "config_path": str(llm_local_config_path()),
    }


def save_llm_local_config(provider: str, base_url: str, model: str, api_key: str = "") -> dict:
    if provider not in PROVIDER_PRESETS:
        raise ValueError("unsupported provider")
    base_url = base_url.strip() or PROVIDER_PRESETS[provider]["base_url"]
    model = model.strip() or PROVIDER_PRESETS[provider]["model"]

    local_cfg = _read_llm_local_config()
    api_keys = local_cfg.get("api_keys", {})
    if not isinstance(api_keys, dict):
        api_keys = {}
    if api_key.strip():
        api_keys[provider] = api_key.strip()

    local_cfg.update({
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_keys": api_keys,
    })
    last_error: Exception | None = None
    for path in llm_local_config_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(local_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            cfg = build_llm_config(provider, model=model, base_url=base_url, prefer_local_provider=False)
            cfg["config_path"] = str(path)
            return cfg
        except Exception as e:
            last_error = e
    raise last_error or OSError("cannot write llm config")


def public_llm_config() -> dict:
    cfg = STATE.llm_cfg if STATE else build_llm_config()
    return {
        "provider": cfg.get("provider", "deepseek"),
        "base_url": cfg.get("base_url", ""),
        "model": cfg.get("model", ""),
        "has_key": bool(cfg.get("api_key")),
        "config_path": str(cfg.get("config_path") or llm_local_config_path()),
        "providers": {
            name: {"base_url": p["base_url"], "model": p["model"], "env": p["env"]}
            for name, p in PROVIDER_PRESETS.items()
        },
    }

# ── 场状态 ────────────────────────────────────────────────────────────────────

class FieldState:
    """持久化的连续问题场状态：向量缓存 + LIF 电位 + 会话记忆"""

    def __init__(self, state_path: Path):
        self.state_path = state_path
        self.vault_hash: str = ""
        self.note_count: int = 0
        self.note_vectors: list[dict] = []     # 序列化的 NoteObservation
        self.adjacency: dict[str, list[int]] = {}
        # LIF 电位（按 query topic 分别维护）
        self.voltage: dict[str, dict] = {}     # topic -> {v_fast, v_slow, last_date}
        # 长期对话历史（LLM 上下文）
        self.llm_history: list[dict] = []      # [{role, content}]
        # 进化日志
        self.evolution_log: list[dict] = []    # [{ts, query, insight, changes}]
        self.scan_ts: str = ""
        self._load()

    def _load(self):
        if not self.state_path.exists():
            return
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.vault_hash   = d.get("vault_hash", "")
            self.note_count   = d.get("note_count", 0)
            self.note_vectors = d.get("note_vectors", [])
            self.adjacency    = d.get("adjacency", {})
            self.voltage      = d.get("voltage", {})
            self.llm_history  = d.get("llm_history", [])
            self.evolution_log= d.get("evolution_log", [])
            self.scan_ts      = d.get("scan_ts", "")
        except Exception:
            pass

    def save(self):
        d = {
            "vault_hash":   self.vault_hash,
            "note_count":   self.note_count,
            "note_vectors": self.note_vectors,
            "adjacency":    self.adjacency,
            "voltage":      self.voltage,
            "llm_history":  self.llm_history,
            "evolution_log":self.evolution_log,
            "scan_ts":      self.scan_ts,
        }
        self.state_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def needs_rescan(self, vault: Path) -> bool:
        """检查知识库是否有新文件"""
        try:
            md_files = list(vault.rglob("*.md"))
            count = len(md_files)
            h = hashlib.md5(str(sorted(str(f) for f in md_files)).encode()).hexdigest()
            return h != self.vault_hash or count != self.note_count
        except Exception:
            return True

    def get_voltage(self, topic: str) -> tuple[float, float]:
        v = self.voltage.get(topic, {})
        return float(v.get("v_fast", 0.0)), float(v.get("v_slow", 0.0))

    def set_voltage(self, topic: str, v_fast: float, v_slow: float):
        self.voltage[topic] = {
            "v_fast": round(v_fast, 4),
            "v_slow": round(v_slow, 4),
            "last_date": date.today().isoformat(),
        }

    def push_llm(self, role: str, content: str):
        self.llm_history.append({"role": role, "content": content})
        # 保留最近 40 轮（防止 context 过长）
        if len(self.llm_history) > 40:
            # 保留 system prompt（第0条）+ 最近 39 条
            self.llm_history = self.llm_history[-40:]

    def append_log(self, query: str, insight: str, changes: list[str]):
        self.evolution_log.append({
            "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "insight": insight,
            "changes": changes,
        })


# ── 全局状态 ──────────────────────────────────────────────────────────────────

class SessionStore:
    """管理多个对话会话，持久化到 lif_sessions.json"""

    def __init__(self, sessions_path: Path):
        self.sessions_path = sessions_path
        self.sessions: dict[str, dict] = {}   # sid -> {id, title, ts, messages:[{role,content,ts,type}]}
        self.active_sid: str = ""
        self._load()
        if not self.active_sid or self.active_sid not in self.sessions:
            self._new_session()

    def _load(self):
        if self.sessions_path.exists():
            try:
                d = json.loads(self.sessions_path.read_text(encoding="utf-8"))
                self.sessions   = d.get("sessions", {})
                self.active_sid = d.get("active_sid", "")
            except Exception:
                pass

    def _save(self):
        try:
            self.sessions_path.write_text(
                json.dumps({"sessions": self.sessions, "active_sid": self.active_sid},
                           ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def _new_session(self, title: str = "") -> str:
        import uuid
        sid = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:4]
        self.sessions[sid] = {
            "id":       sid,
            "title":    title or f"对话 {datetime.now().strftime('%m/%d %H:%M')}",
            "ts":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [],
        }
        self.active_sid = sid
        self._save()
        return sid

    def new_session(self, title: str = "") -> dict:
        sid = self._new_session(title)
        return self.sessions[sid]

    def switch_session(self, sid: str) -> bool:
        if sid in self.sessions:
            self.active_sid = sid
            self._save()
            return True
        return False

    def delete_session(self, sid: str):
        if sid in self.sessions:
            del self.sessions[sid]
            if self.active_sid == sid:
                if self.sessions:
                    self.active_sid = sorted(self.sessions.keys())[-1]
                else:
                    self._new_session()
            self._save()

    def add_message(self, role: str, content: str, msg_type: str = "text"):
        """msg_type: text | debate | system"""
        sess = self.sessions.get(self.active_sid)
        if not sess:
            return
        msg = {
            "role":    role,
            "content": content,
            "ts":      datetime.now().strftime("%H:%M:%S"),
            "type":    msg_type,
        }
        sess["messages"].append(msg)
        sess["updated_ts"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 用第一条用户消息自动命名会话
        if role == "user" and sess["title"].startswith("对话 ") and len(sess["messages"]) == 1:
            sess["title"] = content[:20] + ("…" if len(content) > 20 else "")
        self._save()

    def get_messages(self, sid: str = "") -> list[dict]:
        sid = sid or self.active_sid
        return list(self.sessions.get(sid, {}).get("messages", []))

    def list_sessions(self) -> list[dict]:
        items = []
        for s in self.sessions.values():
            messages = s.get("messages", [])
            last = messages[-1] if messages else {}
            preview = str(last.get("content", "")).replace("\n", " ").strip()
            items.append({
                "id": s["id"],
                "title": s["title"],
                "ts": s["ts"],
                "updated_ts": s.get("updated_ts", s["ts"]),
                "count": len(messages),
                "preview": preview[:90],
                "active": s["id"] == self.active_sid,
            })
        return sorted(items, key=lambda x: x["updated_ts"], reverse=True)[:30]


# ── 精华结论库 ────────────────────────────────────────────────────────────────

class ConclusionsStore:
    """
    双层记忆的第二层：存储每次辩论/洞察提炼出的精华结论。
    每条结论是一个向量化的知识点，查询时以更高权重参与场重建。
    持久化到 lif_conclusions.json。
    """

    MAX_CONCLUSIONS = 200  # 最多保留200条，淘汰最低分的

    def __init__(self, path: Path):
        self.path = path
        self.conclusions: list[dict] = []   # [{id, ts, query, synthesis, key_point, importance, vector}]
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text(encoding="utf-8"))
                self.conclusions = d.get("conclusions", [])
            except Exception:
                pass

    def _save(self):
        try:
            self.path.write_text(
                json.dumps({"conclusions": self.conclusions}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    def add(self, query: str, synthesis: str, key_point: str,
            importance: float, vector: dict | None = None) -> dict:
        import uuid
        entry = {
            "id":         uuid.uuid4().hex[:8],
            "ts":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query":      query,
            "synthesis":  synthesis,
            "key_point":  key_point,          # 一句话核心结论
            "importance": round(min(1.0, max(0.0, importance)), 3),
            "vector":     vector or {},        # 稀疏词频向量，用于场查询
            "recall_count": 0,                 # 被检索到的次数
        }
        self.conclusions.append(entry)
        # 超出上限时淘汰最低分
        if len(self.conclusions) > self.MAX_CONCLUSIONS:
            self.conclusions.sort(key=lambda x: x["importance"] + x["recall_count"] * 0.05, reverse=True)
            self.conclusions = self.conclusions[:self.MAX_CONCLUSIONS]
        self._save()
        return entry

    def search(self, query_vec: dict, top_k: int = 3, min_importance: float = 0.4) -> list[dict]:
        """余弦相似度检索结论，返回 top_k 个高相关+高重要性的结论"""
        if not self.conclusions or not query_vec:
            return []

        def cosine(a: dict, b: dict) -> float:
            keys = set(a) & set(b)
            if not keys:
                return 0.0
            dot = sum(a[k] * b[k] for k in keys)
            na  = sum(v*v for v in a.values()) ** 0.5
            nb  = sum(v*v for v in b.values()) ** 0.5
            return dot / (na * nb + 1e-9)

        scored = []
        for c in self.conclusions:
            if c["importance"] < min_importance:
                continue
            sim = cosine(query_vec, c.get("vector", {}))
            score = sim * 0.6 + c["importance"] * 0.4
            if score > 0.05:
                scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        result = [c for _, c in scored[:top_k]]
        # 更新 recall_count
        for c in result:
            c["recall_count"] = c.get("recall_count", 0) + 1
        if result:
            self._save()
        return result

    def get_all(self, sort_by: str = "ts") -> list[dict]:
        if sort_by == "importance":
            return sorted(self.conclusions, key=lambda x: x["importance"], reverse=True)
        return sorted(self.conclusions, key=lambda x: x["ts"], reverse=True)

    def delete(self, cid: str) -> bool:
        before = len(self.conclusions)
        self.conclusions = [c for c in self.conclusions if c["id"] != cid]
        if len(self.conclusions) < before:
            self._save()
            return True
        return False

    def clear(self):
        self.conclusions = []
        self._save()


class AppState:
    def __init__(self, vault: Path, params_path: Path, state_path: Path,
                 log_path: Path, sessions_path: Path, conclusions_path: Path,
                 llm_cfg: dict):
        self.vault = vault
        self.params_path = params_path
        self.log_path = log_path
        self.llm_cfg = llm_cfg
        self.params: dict = self._load_params()
        self.field = FieldState(state_path)
        self.sessions = SessionStore(sessions_path)
        self.conclusions = ConclusionsStore(conclusions_path)
        self.lock = threading.Lock()
        self.status: str = "idle"
        self.last_query: str = ""

    def _load_params(self) -> dict:
        if self.params_path.exists():
            try:
                saved = json.loads(self.params_path.read_text(encoding="utf-8"))
                merged = dict(DEFAULT_PARAMS)
                merged.update(saved)
                return merged
            except Exception:
                pass
        return dict(DEFAULT_PARAMS)

    def save_params(self):
        self.params_path.write_text(json.dumps(self.params, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_msg(self, role: str, content: str, msg_type: str = "text"):
        self.sessions.add_message(role, content, msg_type)

    # 兼容旧调用
    def add_ui_msg(self, role: str, content: str):
        self.add_msg(role, content)

    def apply_delta(self, delta: dict) -> list[str]:
        changes = []
        for key, value in delta.items():
            if key == "topic_weights" and isinstance(value, dict):
                for topic, w in value.items():
                    if topic in self.params["topic_weights"]:
                        old = self.params["topic_weights"][topic]
                        new_w = round(max(0.1, min(3.0, float(w))), 3)
                        self.params["topic_weights"][topic] = new_w
                        if abs(new_w - old) > 0.001:
                            changes.append(f"topic_weights[{topic}]: {old:.3f} → {new_w:.3f}")
            elif key in PARAM_BOUNDS:
                lo, hi = PARAM_BOUNDS[key]
                old = self.params.get(key)
                new_v = round(max(lo, min(hi, float(value))), 4)
                self.params[key] = new_v
                if old != new_v:
                    changes.append(f"{key}: {old} → {new_v}")
        if changes:
            self.save_params()
        return changes


STATE: AppState | None = None


# ── LLM 调用 ──────────────────────────────────────────────────────────────────

def call_llm(messages: list[dict]) -> str:
    cfg = STATE.llm_cfg
    if not cfg.get("api_key"):
        return "[未配置 API Key，LLM 功能不可用]"
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": cfg["model"], "messages": messages, "temperature": 0.7, "max_tokens": 1500}
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {cfg['api_key']}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[LLM 调用失败: {e}]"


def extract_json(text: str) -> dict:
    import re
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        s, e = cleaned.find("{"), cleaned.rfind("}")
        if s >= 0 and e > s:
            try:
                return json.loads(cleaned[s:e+1])
            except Exception:
                pass
    return {}


# ── 知识库扫描（首次 or 增量）────────────────────────────────────────────────

def scan_vault(vault: Path):
    """扫描知识库，序列化向量存入 FieldState"""
    STATE.status = "scanning"
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import continuous_problem_field as cpf
        today = date.today()
        params = STATE.params

        notes = cpf.read_notes(vault, today, 365, 3000, all_notes=True)
        cpf.diffuse_graph(notes, int(params.get("graph_steps", 2)), float(params.get("graph_alpha", 0.35)))
        if float(params.get("dense_weight", 0.0)) > 0:
            hydrate_note_embeddings(notes, vault, Path(__file__).resolve().parent)

        # 序列化 notes 向量
        serialized = []
        for n in notes:
            serialized.append({
                "rel_path": n.rel_path,
                "title":    n.title,
                "day":      n.day.isoformat(),
                "snippet":  n.snippet,
                "tags":     n.tags,
                "links":    n.links,
                "vector":   n.vector,
                "graph_vector": n.graph_vector,
                "dense_vector": getattr(n, "dense_vector", []),
                "centrality": n.centrality,
                "text_len": len(n.text),
            })

        md_files = list(vault.rglob("*.md"))
        h = hashlib.md5(str(sorted(str(f) for f in md_files)).encode()).hexdigest()

        with STATE.lock:
            STATE.field.vault_hash   = h
            STATE.field.note_count   = len(notes)
            STATE.field.note_vectors = serialized
            STATE.field.scan_ts      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        STATE.field.save()
        return len(notes)
    except Exception as e:
        return f"扫描失败: {e}"
    finally:
        STATE.status = "idle"


def restore_notes_from_field():
    """从持久化数据恢复 NoteObservation 对象列表"""
    sys.path.insert(0, str(Path(__file__).parent))
    import continuous_problem_field as cpf
    from datetime import date as date_cls
    notes = []
    for d in STATE.field.note_vectors:
        day_str = d.get("day", "2000-01-01")
        try:
            day = date_cls.fromisoformat(day_str)
        except Exception:
            day = date_cls.today()
        n = cpf.NoteObservation(
            path=Path(d.get("rel_path", "")),
            rel_path=d.get("rel_path", ""),
            title=d.get("title", ""),
            day=day,
            text="",  # text 不序列化，snippet 够用
            snippet=d.get("snippet", ""),
            tags=d.get("tags", []),
            links=d.get("links", []),
            vector=d.get("vector", {}),
        )
        n.graph_vector = d.get("graph_vector", {})
        n.dense_vector = d.get("dense_vector", [])
        n.centrality   = d.get("centrality", 0.0)
        notes.append(n)
    return notes


# ── 核心：场查询 + LIF 积分（继承电位）─────────────────────────────────────

def query_field(query: str) -> str:
    """
    在持久场上做查询，继承上次的 LIF 电位，
    把 insight + 主动追问发给用户
    """
    import continuous_problem_field as cpf
    from collections import defaultdict

    params = STATE.params
    today  = date.today()

    notes = restore_notes_from_field()
    if not notes:
        return "知识库尚未扫描，请先点击「扫描知识库」。"

    query_dense = []
    if float(params.get("dense_weight", 0.0)) > 0:
        try:
            query_dense = embed_query(query, STATE.vault, Path(__file__).resolve().parent)
        except Exception:
            query_dense = []
    hits, field_energy, daily_current, daily_completion = cpf.reconstruct_field(
        notes=notes,
        query=query,
        today=today,
        top_k=8,
        time_sigma=float(params.get("time_sigma", 30.0)),
        semantic_sigma=float(params.get("semantic_sigma", 0.55)),
        all_notes=True,
        query_dense=query_dense,
        dense_weight=float(params.get("dense_weight", 0.45)),
        sparse_weight=float(params.get("sparse_weight", 0.55)),
    )

    if not hits:
        return f"在场中未找到与「{query}」相关的笔记。尝试换个关键词，或调大 semantic_sigma。"

    topic = cpf.infer_topic(query)

    # 继承上次电位，继续积分
    v_fast, v_slow = STATE.field.get_voltage(topic)
    threshold  = float(params.get("threshold", 5.0))
    fast_decay = float(params.get("fast_decay", 0.76))
    slow_decay = float(params.get("slow_decay", 0.94))
    fast_w     = float(params.get("fast_weight", 0.65))
    slow_w     = float(params.get("slow_weight", 0.35))

    total_input      = sum(d for d in daily_current.values())
    total_completion = sum(d for d in daily_completion.values())
    v_fast = max(0.0, v_fast * fast_decay + total_input * 4.0 - total_completion)
    v_slow = max(0.0, v_slow * slow_decay + total_input * 4.0 * 0.42 - total_completion * 0.30)
    v = fast_w * v_fast + slow_w * v_slow
    spiked = v >= threshold

    # 存回电位
    STATE.field.set_voltage(topic, v_fast, v_slow)
    field_observation = update_domain_after_query(STATE.vault, query, topic, hits, field_energy, v, spiked)

    # 构建证据摘要
    snippets = []
    for i, hit in enumerate(hits[:5], 1):
        snippets.append(f"{i}. [{hit.note.title}] 相似度={hit.semantic:.3f}\n   {hit.note.snippet}")
    evidence_text = "\n".join(snippets)

    voltage_info = f"V={v:.3f} (fast={v_fast:.3f}, slow={v_slow:.3f}, threshold={threshold})"
    spike_flag   = "⚡ 已触发 insight spike" if spiked else f"📈 电位积累中 ({v:.2f}/{threshold})"

    # 构建系统提示（只在首次注入，之后靠历史）
    system_prompt = {
        "role": "system",
        "content": (
            "你是 LIF-Memory 的持续进化洞察引擎。你有完整的对话历史，每次都在前一轮的基础上深化。\n"
            "你的目标：\n"
            "1. 基于知识库证据找出真正的问题张力（不是复述笔记）\n"
            "2. 每次洞察要比上一次更深，利用对话历史避免重复\n"
            "3. 洞察后主动向用户提问，验证或深化这个方向\n"
            "4. 如果发现新的关联，明确指出它和之前话题的连接\n"
            "不要每次都重新介绍自己，直接继续对话。"
        )
    }

    # 构建本轮用户消息
    user_content = (
        f"【新查询】{query}\n\n"
        f"【场状态】{spike_flag}\n{voltage_info}\n场能量={field_energy:.3f}\n\n"
        f"【相关证据（Top5）】\n{evidence_text}\n\n"
        "请在上述历史对话基础上，给出更深一层的洞察，并主动提问。"
    )

    # 组装消息：system + 历史 + 本轮
    messages = [system_prompt] + STATE.field.llm_history + [{"role": "user", "content": user_content}]
    reply    = call_llm(messages)

    # 存入持久历史
    STATE.field.push_llm("user", user_content)
    STATE.field.push_llm("assistant", reply)
    STATE.field.append_log(query, reply, [voltage_info])
    STATE.field.save()

    # 追加写入进化日志 MD
    append_evolution_log(query, reply, voltage_info, spiked)

    return reply


def llm_analyze_and_tune(feedback: str, last_query: str) -> tuple[str, dict]:
    """LLM 理解反馈 → 给出深化洞察 + 调参建议"""
    params = STATE.params
    system_prompt = {
        "role": "system",
        "content": (
            "你是 LIF-Memory 的调参助手，也是持续对话的洞察引擎。\n"
            "根据用户反馈：\n"
            "1. 深化分析，不重复之前说过的内容\n"
            "2. 判断是否需要调参，输出 JSON\n\n"
            "可调参数及范围：\n"
            "- threshold (1-15): 触发阈值，越低越灵敏\n"
            "- semantic_sigma (0.1-2.0): 语义宽度，越大越宽泛\n"
            "- time_sigma (1-365): 时间窗口天数\n"
            "- fast_decay (0.3-0.99): 近期衰减\n"
            "- slow_decay (0.5-0.99): 长期衰减\n"
            "- topic_weights: 各主题权重 (0.1-3.0)\n\n"
            "输出 JSON：\n"
            "{\n"
            '  "reply": "对用户说的话，继续深化洞察或追问",\n'
            '  "param_delta": {"threshold": null或新值, "semantic_sigma": null或新值, "topic_weights": {}},\n'
            '  "reason": "调参原因"\n'
            "}"
        )
    }

    user_content = (
        f"【用户反馈】{feedback}\n"
        f"【当前参数】{json.dumps(params, ensure_ascii=False)}"
    )

    messages = [system_prompt] + STATE.field.llm_history + [{"role": "user", "content": user_content}]
    raw   = call_llm(messages)
    data  = extract_json(raw)

    reply = data.get("reply", raw)
    delta = {k: v for k, v in (data.get("param_delta") or {}).items() if v is not None}

    STATE.field.push_llm("user", user_content)
    STATE.field.push_llm("assistant", reply)
    STATE.field.save()

    return reply, delta


# ── 进化日志写入 ──────────────────────────────────────────────────────────────

def append_evolution_log(query: str, insight: str, voltage_info: str, spiked: bool):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mark = "⚡ SPIKE" if spiked else "📈"
    block = (
        f"\n## {ts} {mark}\n\n"
        f"**查询**：{query}\n\n"
        f"**场状态**：{voltage_info}\n\n"
        f"**洞察**：\n\n{insight}\n\n"
        f"---\n"
    )
    try:
        with open(STATE.log_path, "a", encoding="utf-8") as f:
            if STATE.log_path.stat().st_size == 0:
                f.write("# LIF-Memory 进化日志\n\n> 此文件由系统自动追加，记录每一次洞察的完整过程。\n")
            f.write(block)
    except Exception:
        pass


# ── 多智能体辩论引擎 ──────────────────────────────────────────────────────────

def agent_search_field(agent: dict, query: str, notes: list) -> tuple[str, float]:
    """
    每个角色基于自己的立场，生成搜索关键词，独立进入连续场查询，
    返回 (专属证据文本, 场能量)
    """
    import continuous_problem_field as cpf
    from datetime import date

    # 让 LLM 生成角色专属的搜索视角
    kw_prompt = (
        f"你是「{agent['name']}」，立场：{agent['stance']}，关注：{agent['focus']}。\n"
        f"原始问题：{query}\n\n"
        f"从你的视角，生成 3-5 个最能揭示你关注维度的搜索关键词（中文，空格分隔）。\n"
        f"只输出关键词，不要其他内容。"
    )
    kw_raw = call_llm([{"role": "user", "content": kw_prompt}])
    agent_query = kw_raw.strip().replace("\n", " ")[:120]

    params = STATE.params
    today  = date.today()

    try:
        query_dense = []
        if float(params.get("dense_weight", 0.0)) > 0:
            try:
                query_dense = embed_query(agent_query, STATE.vault, Path(__file__).resolve().parent)
            except Exception:
                query_dense = []
        hits, field_energy, _, _ = cpf.reconstruct_field(
            notes=notes,
            query=agent_query,
            today=today,
            top_k=5,
            time_sigma=float(params.get("time_sigma", 30.0)),
            semantic_sigma=float(params.get("semantic_sigma", 0.65)),  # 稍宽，让角色探索更广
            all_notes=True,
            query_dense=query_dense,
            dense_weight=float(params.get("dense_weight", 0.45)),
            sparse_weight=float(params.get("sparse_weight", 0.55)),
        )
        if not hits:
            return f"（{agent['name']} 在场中未找到相关证据）", 0.0

        snippets = []
        for i, hit in enumerate(hits[:4], 1):
            snippets.append(f"{i}. [{hit.note.title}] {hit.note.snippet}")
        return "\n".join(snippets), field_energy
    except Exception as e:
        return f"（搜索失败: {e}）", 0.0


def spawn_agents(query: str, evidence_text: str) -> list[dict]:
    """
    让 LLM 根据问题和证据，自动孵化 3 个具有强烈对比视角的角色。
    返回 [{name, stance, personality, focus, search_keywords}, ...]
    """
    prompt = f"""你是角色设计师。根据以下问题，设计 3 个具有强烈对比视角的思考角色。
每个角色必须有真实冲突，代表不同的认识论立场。

问题：{query}

初始证据摘要：
{evidence_text[:400]}

输出 JSON 数组，每个角色包含：
{{
  "name": "角色名（2-4字，有个性）",
  "stance": "该角色的核心立场（一句话，要有锋芒）",
  "personality": "激进质疑者|系统建构者|现实锚定者|直觉跳跃者|历史归因者",
  "focus": "最关注的维度（行动可能性|深层原因|情绪真相|结构矛盾|时间模式）",
  "search_angle": "这个角色会用什么不同的角度去搜索知识库（一句话描述搜索策略）"
}}

只输出 JSON 数组。"""

    raw = call_llm([{"role": "user", "content": prompt}])
    try:
        import re as _re
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = _re.sub(r"\s*```$", "", cleaned).strip()
        agents = json.loads(cleaned)
        if isinstance(agents, list) and len(agents) >= 2:
            return agents[:3]
    except Exception:
        pass
    return [
        {"name": "质疑者", "stance": "现有证据根本不足以支撑结论", "personality": "激进质疑者",
         "focus": "逻辑漏洞", "search_angle": "寻找证据中的矛盾和反例"},
        {"name": "建构者", "stance": "应该从系统结构找根本原因", "personality": "系统建构者",
         "focus": "深层原因", "search_angle": "寻找重复出现的结构性模式"},
        {"name": "锚定者", "stance": "理论没用，最紧迫的是一个可做的行动", "personality": "现实锚定者",
         "focus": "行动可能性", "search_angle": "寻找已完成事项和可复制的小行动"},
    ]


def agent_speak(agent: dict, query: str, agent_evidence: str,
                round_num: int, debate_history: list[dict]) -> str:
    """
    每个角色发言，拥有自己独立搜索到的证据，以及完整辩论上下文。
    debate_history 是 [{role:'角色名', content:'发言'}] 的列表。
    """
    # 把辩论历史格式化
    history_text = ""
    for h in debate_history[-6:]:  # 最近6条，保持上下文连贯
        history_text += f"\n{h['role']}：{h['content']}"

    if round_num == 1:
        sys_p = (
            f"你现在扮演「{agent['name']}」。\n"
            f"立场：{agent['stance']}\n"
            f"思维风格：{agent['personality']}\n"
            f"关注维度：{agent['focus']}\n\n"
            f"你已独立搜索了知识库，找到了以下专属证据（其他角色看不到这些）：\n"
            f"{agent_evidence}\n\n"
            f"规则：\n"
            f"- 完全代入角色，不说'作为AI'\n"
            f"- 基于你的专属证据形成观点，可以说'我在知识库里发现了'\n"
            f"- 观点尖锐，有真实立场，不超过 150 字"
        )
        user_p = f"问题：{query}\n\n请基于你独立找到的证据，给出第一轮观点。"
    else:
        sys_p = (
            f"你现在扮演「{agent['name']}」。\n"
            f"立场：{agent['stance']}\n"
            f"思维风格：{agent['personality']}\n\n"
            f"你的专属证据（其他角色没有看到这些）：\n"
            f"{agent_evidence}\n\n"
            f"规则：\n"
            f"- 结合你的专属证据，反驳对方最薄弱的一个点\n"
            f"- 或者用你的证据补充一个对方完全忽视的维度\n"
            f"- 不要和稀泥，要有锋芒，不超过 130 字"
        )
        user_p = (
            f"问题：{query}\n\n"
            f"目前的辩论：{history_text}\n\n"
            f"你的反驳或补充（记住你有对方没有的证据）："
        )

    # 角色自己的对话历史（单角色视角）
    agent_msgs = [{"role": "system", "content": sys_p},
                  {"role": "user",   "content": user_p}]
    return call_llm(agent_msgs)


def synthesize_debate(query: str, agents: list[dict], debate_rounds: list[dict],
                      all_evidence: dict[str, str], history: list[dict]) -> tuple[str, dict, dict]:
    """
    综合辩论，all_evidence = {agent_name: evidence_text}
    提炼突破性洞察 + 调参建议
    """
    debate_text = ""
    for r in debate_rounds:
        debate_text += f"\n【{r['agent']}·第{r['round']}轮】{r['content']}"

    evidence_summary = ""
    for name, ev in all_evidence.items():
        evidence_summary += f"\n{name} 的专属证据：\n{ev[:200]}\n"

    sys_p = (
        "你是辩论综合者和 LIF-Memory 调参助手。\n"
        "注意：每个角色持有不同的专属证据，辩论的张力部分来自信息差。\n\n"
        "任务：\n"
        "1. 找出辩论中真正的突破点——信息差碰撞后产生的新洞察\n"
        "2. 指出哪个角色的专属证据最关键，为什么\n"
        "3. 提出一个融合所有角色视角才能看到的综合命题\n"
        "4. 向用户提出下一轮最值得探索的问题\n"
        "5. 判断是否需要调参\n\n"
        "输出 JSON：\n"
        "{\n"
        '  "breakthrough": "突破性洞察（200字内，要有张力）",\n'
        '  "key_evidence": "哪个角色的证据最关键及原因",\n'
        '  "hidden_tension": "辩论揭示的隐藏张力",\n'
        '  "synthesis": "综合命题（一句话，有冲击力）",\n'
        '  "follow_up": "下一轮最值得探索的问题",\n'
        '  "param_delta": {"threshold": null或新值, "semantic_sigma": null或新值, "topic_weights": {}}\n'
        "}"
    )
    user_p = (
        f"问题：{query}\n\n"
        f"各角色专属证据摘要：{evidence_summary}\n\n"
        f"完整辩论记录：{debate_text}"
    )
    messages = [{"role": "system", "content": sys_p}] + history[-8:] + [{"role": "user", "content": user_p}]
    raw  = call_llm(messages)
    data = extract_json(raw)

    if not data:
        return raw, {}, {}

    lines = []
    if data.get("breakthrough"):
        lines.append(f"**突破洞察**：{data['breakthrough']}\n")
    if data.get("key_evidence"):
        lines.append(f"**关键证据来源**：{data['key_evidence']}\n")
    if data.get("hidden_tension"):
        lines.append(f"**隐藏张力**：{data['hidden_tension']}\n")
    if data.get("synthesis"):
        lines.append(f"**综合命题**：{data['synthesis']}\n")
    if data.get("follow_up"):
        lines.append(f"\n🔍 {data['follow_up']}")

    text  = "\n".join(lines) if lines else raw
    delta = {k: v for k, v in (data.get("param_delta") or {}).items() if v is not None}
    return text, delta, data


def _auto_save_conclusion(query: str, synthesis: str, raw_data: dict, evidence_text: str):
    """
    辩论结束后，LLM 判断本次最重要的结论，自动存入 ConclusionsStore。
    raw_data 来自 synthesize_debate 的 JSON 解析结果。
    """
    if not STATE or not STATE.llm_cfg.get("api_key"):
        return
    try:
        import continuous_problem_field as cpf

        # 如果 raw_data 里已有 synthesis 字段，直接用；否则调 LLM 提炼
        key_point = raw_data.get("synthesis", "").strip()
        importance = 0.7  # 默认重要性

        if not key_point:
            # LLM 提炼一句话结论
            prompt = (
                "从以下辩论综合中提炼出最重要的一句话结论，要有洞察力和冲击力。\n"
                "只输出一句话，不超过60字，不加引号。\n\n"
                f"问题：{query}\n\n综合：{synthesis[:600]}"
            )
            key_point = call_llm([{"role": "user", "content": prompt}]).strip()[:120]

        # 计算重要性分数（基于是否触发了 breakthrough）
        if raw_data.get("breakthrough"):
            importance = 0.85
        if raw_data.get("hidden_tension"):
            importance = min(1.0, importance + 0.05)

        # 为结论构建稀疏词向量，用于未来的结论场检索
        combined_text = f"{query} {key_point} {synthesis[:200]}"
        con_vec = cpf.vectorize(combined_text)

        STATE.conclusions.add(
            query=query,
            synthesis=synthesis[:500],
            key_point=key_point,
            importance=importance,
            vector=con_vec,
        )
    except Exception:
        pass


def _build_conclusions_context(query_vec: dict) -> str:
    """
    从结论库中检索与当前查询相关的历史结论，构建注入文本。
    """
    if not STATE:
        return ""
    hits = STATE.conclusions.search(query_vec, top_k=3, min_importance=0.4)
    if not hits:
        return ""
    lines = ["【历史精华结论（高权重参考）】"]
    for i, c in enumerate(hits, 1):
        lines.append(f"{i}. [{c['ts'][:10]}] {c['key_point']}"
                     f"（重要性={c['importance']:.2f}，原始问题：{c['query'][:30]}）")
    return "\n".join(lines)


def run_debate(query: str, evidence_text: str, history: list[dict]) -> tuple[str, list[dict], dict]:
    """
    完整辩论流程：
    孵化角色 → 各角色独立搜索场 → 2轮辩论（上下文连贯）→ 综合 → 提炼结论
    """
    import continuous_problem_field as cpf
    from datetime import date

    # 1. 恢复场数据（一次，所有角色共用同一批 notes）
    notes = restore_notes_from_field()

    # 2. 孵化角色（基于初始证据）
    agents = spawn_agents(query, evidence_text)

    # 3. 每个角色独立搜索，建立专属证据库
    agent_evidence: dict[str, str] = {}
    agent_field_energy: dict[str, float] = {}
    for agent in agents:
        ev, fe = agent_search_field(agent, query, notes)
        agent_evidence[agent["name"]] = ev
        agent_field_energy[agent["name"]] = fe

    # 4. 两轮辩论，共享上下文
    debate_rounds  = []
    debate_history = []  # [{role, content}] 全局辩论历史

    for round_num in (1, 2):
        for agent in agents:
            ev = agent_evidence[agent["name"]]
            content = agent_speak(agent, query, ev, round_num, debate_history)
            debate_rounds.append({
                "agent":    agent["name"],
                "round":    round_num,
                "content":  content,
                "evidence": ev,
            })
            # 把发言加入全局历史，下一个角色可以看到
            debate_history.append({"role": agent["name"], "content": content})

    # 5. 综合
    synthesis, delta, raw_data = synthesize_debate(
        query, agents, debate_rounds, agent_evidence, history
    )

    # 6. 自动提炼并存入结论库
    _auto_save_conclusion(query, synthesis, raw_data, evidence_text)

    # 7. 格式化输出
    agent_colors = ["🔴", "🔵", "🟡", "🟢"]
    lines = ["### 🧠 多角色辩论场\n"]
    lines.append(f"**问题**：{query}\n")

    role_tags = []
    for i, a in enumerate(agents):
        fe = agent_field_energy.get(a["name"], 0)
        role_tags.append(f"{agent_colors[i%4]} {a['name']}（{a['stance'][:18]}… | 场能量={fe:.2f}）")
    lines.append("**参与角色**：\n" + "\n".join(role_tags) + "\n")
    lines.append("---\n")

    for rn, round_label in ((1, "第一轮 · 各持专属证据发言"), (2, "第二轮 · 交锋反驳")):
        lines.append(f"**{round_label}**\n")
        for r in debate_rounds:
            if r["round"] == rn:
                i = next((j for j, a in enumerate(agents) if a["name"] == r["agent"]), 0)
                lines.append(f"{agent_colors[i%4]} **{r['agent']}**：{r['content']}\n")
        lines.append("")

    lines.append("---\n**综合**\n")
    lines.append(synthesis)

    return "\n".join(lines), debate_rounds, delta



# ── HTTP 处理器 ───────────────────────────────────────────────────────────────

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>LIF-Memory 持续进化</title>
<style>
:root{--bg:#0f172a;--panel:rgba(15,23,42,0.88);--panel2:rgba(30,41,59,0.92);--text:#e5e7eb;--muted:#94a3b8;--accent:#38bdf8;--accent2:#a78bfa;--accent3:#34d399;--warn:#fbbf24;--border:rgba(148,163,184,0.18);--radius:14px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:radial-gradient(circle at 15% 20%,rgba(56,189,248,0.12),transparent 38%),radial-gradient(circle at 85% 80%,rgba(167,139,250,0.12),transparent 38%),#0f172a;color:var(--text);font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;min-height:100vh;display:flex;flex-direction:column}
header{padding:16px 28px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:var(--panel);backdrop-filter:blur(16px);position:sticky;top:0;z-index:10}
header h1{font-size:17px;font-weight:600;background:linear-gradient(90deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.chip{font-size:11px;padding:3px 10px;border-radius:999px;border:1px solid rgba(56,189,248,0.28);color:#bae6fd;background:rgba(14,165,233,0.08);white-space:nowrap}
.chip.green{border-color:rgba(52,211,153,0.35);color:var(--accent3);background:rgba(52,211,153,0.08)}
.chip.purple{border-color:rgba(167,139,250,0.35);color:var(--accent2);background:rgba(167,139,250,0.08)}
#statusChip{margin-left:auto}
.layout{display:grid;grid-template-columns:286px 1fr 310px;flex:1;height:calc(100vh - 57px)}
/* 左：会话列表 */
.sessions-panel{display:flex;flex-direction:column;border-right:1px solid var(--border);background:#0b1220;overflow:hidden}
.sessions-header{padding:14px 14px 10px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px}
.sessions-title{font-size:14px;font-weight:650;color:var(--text);flex:1}
.new-chat-btn{height:32px;border-radius:8px;border:1px solid rgba(56,189,248,.32);background:rgba(56,189,248,.10);color:#dff7ff;font-size:13px;cursor:pointer;display:flex;align-items:center;gap:6px;justify-content:center;transition:all .2s;line-height:1;padding:0 10px;white-space:nowrap}
.new-chat-btn:hover{background:rgba(56,189,248,.18);border-color:var(--accent)}
.sessions-search{padding:10px 12px 8px;border-bottom:1px solid rgba(148,163,184,0.10)}
#sessionSearch{width:100%;height:34px;border-radius:8px;border:1px solid var(--border);background:rgba(15,23,42,.88);color:var(--text);outline:none;padding:0 10px;font-size:12.5px}
#sessionSearch:focus{border-color:var(--accent)}
.sessions-list{flex:1;overflow-y:auto;padding:8px}
.session-empty{padding:26px 16px;color:var(--muted);font-size:12px;line-height:1.7;text-align:center;border:1px dashed rgba(148,163,184,.20);border-radius:8px;margin:8px;background:rgba(15,23,42,.45)}
.sess-item{padding:10px 34px 10px 11px;border-radius:8px;cursor:pointer;margin-bottom:6px;transition:all .18s;border:1px solid transparent;position:relative;background:transparent}
.sess-item:hover{background:rgba(30,41,59,.70);border-color:var(--border)}
.sess-item.active{background:rgba(56,189,248,.12);border-color:rgba(56,189,248,.38)}
.sess-title{font-size:13px;font-weight:560;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:4px;line-height:1.35}
.sess-preview{font-size:11.5px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;margin-bottom:5px}
.sess-meta{font-size:10.5px;color:#64748b;display:flex;gap:8px}
.sess-del{position:absolute;right:7px;top:9px;width:22px;height:22px;border-radius:5px;border:none;background:transparent;color:var(--muted);cursor:pointer;font-size:13px;display:none;align-items:center;justify-content:center}
.sess-item:hover .sess-del{display:flex}
.sess-del:hover{color:#f87171;background:rgba(248,113,113,.1)}
/* 左：对话 */
.chat{display:flex;flex-direction:column;border-right:1px solid var(--border)}
.msgs{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;animation:fi .25s ease}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:translateY(0)}}
.msg.user{flex-direction:row-reverse}
.av{width:30px;height:30px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:13px;flex-shrink:0}
.msg.assistant .av{background:linear-gradient(135deg,#0ea5e9,#7c3aed)}
.msg.user .av{background:linear-gradient(135deg,var(--accent3),#059669)}
.bub{max-width:78%;padding:11px 15px;border-radius:12px;font-size:13.5px;line-height:1.75;white-space:pre-wrap;word-break:break-word}
.msg.assistant .bub{background:var(--panel2);border:1px solid var(--border);border-top-left-radius:3px}
.msg.user .bub{background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.22);border-top-right-radius:3px}
.ts{font-size:10px;color:var(--muted);margin-top:3px}
.msg.user .ts{text-align:right}
.voltage-bar{margin:4px 0 0 40px;height:4px;background:rgba(148,163,184,0.1);border-radius:2px;overflow:hidden;width:200px}
.voltage-fill{height:100%;background:linear-gradient(90deg,var(--accent3),var(--warn));border-radius:2px;transition:width .6s ease}
.spike-badge{margin:2px 0 0 40px;font-size:11px;color:var(--warn);font-weight:600}
/* 输入区 */
.input-area{padding:14px 20px;border-top:1px solid var(--border);background:var(--panel)}
.scan-row{display:flex;gap:8px;margin-bottom:10px}
#qInput{flex:1;background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:9px 13px;color:var(--text);font-size:13.5px;outline:none;transition:border .2s}
#qInput:focus{border-color:var(--accent)}
.btn{padding:9px 18px;border:none;border-radius:9px;cursor:pointer;font-size:13px;font-weight:500;transition:all .18s}
.btn-p{background:linear-gradient(135deg,var(--accent),var(--accent2));color:#0f172a}
.btn-p:hover{opacity:.82;transform:translateY(-1px)}
.btn-p:disabled{opacity:.4;cursor:not-allowed;transform:none}
.btn-s{background:var(--panel2);color:var(--text);border:1px solid var(--border);padding:7px 13px;font-size:12px}
.btn-s:hover{border-color:var(--accent);color:var(--accent)}
.chips-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px}
.fb{padding:5px 11px;border-radius:999px;font-size:12px;cursor:pointer;border:1px solid var(--border);background:var(--panel2);color:var(--muted);transition:all .18s}
.fb:hover{border-color:var(--accent2);color:var(--accent2)}
#fbInput{width:100%;background:var(--panel2);border:1px solid var(--border);border-radius:9px;padding:9px 13px;color:var(--text);font-size:13.5px;outline:none;resize:none;min-height:56px;margin:6px 0;transition:border .2s}
#fbInput:focus{border-color:var(--accent)}
.send-row{display:flex;gap:7px;justify-content:flex-end}
/* 右：参数+状态 */
.side{overflow-y:auto;padding:18px 16px}
.stitle{font-size:11px;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px}
.pcard{background:var(--panel2);border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin-bottom:8px;transition:border .4s,box-shadow .4s}
.pcard.hl{border-color:var(--accent3);box-shadow:0 0 12px rgba(52,211,153,.2)}
.pname{font-size:12px;font-weight:500;color:var(--accent);margin-bottom:1px}
.pdesc{font-size:10px;color:var(--muted);margin-bottom:6px}
.pval{font-size:20px;font-weight:700;font-variant-numeric:tabular-nums}
.pbar{height:3px;background:rgba(148,163,184,.12);border-radius:2px;margin-top:6px;overflow:hidden}
.pbarf{height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));transition:width .5s}
.tw-row{display:flex;align-items:center;gap:6px;padding:5px 0;border-bottom:1px solid var(--border);font-size:12px}
.tw-row:last-child{border-bottom:none}
.tw-bar{flex:1;height:3px;background:rgba(148,163,184,.1);border-radius:2px;overflow:hidden}
.tw-bf{height:100%;background:var(--accent3);border-radius:2px;transition:width .5s}
.tw-v{color:var(--accent3);font-weight:600;font-variant-numeric:tabular-nums;min-width:32px;text-align:right}
.llm-card{display:flex;flex-direction:column;gap:8px}
.form-field{display:flex;flex-direction:column;gap:4px}
.form-field label{font-size:10.5px;color:var(--muted)}
.form-field input,.form-field select{width:100%;height:34px;border-radius:8px;border:1px solid var(--border);background:rgba(15,23,42,.88);color:var(--text);outline:none;padding:0 10px;font-size:12.5px}
.form-field input:focus,.form-field select:focus{border-color:var(--accent)}
.form-field input[type=password]{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
.llm-actions{display:flex;gap:7px}
.llm-actions .btn{flex:1}
.llm-status{font-size:11px;color:var(--muted);line-height:1.55;border-top:1px solid rgba(148,163,184,.12);padding-top:8px;word-break:break-word}
.llm-status.ok{color:var(--accent3)}
.llm-status.bad{color:#f87171}
/* 电位状态 */
.vstate{background:var(--panel2);border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin-bottom:8px}
.vrow{display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px}
.vlabel{color:var(--muted)}
.vnum{font-variant-numeric:tabular-nums;color:var(--accent)}
.vnum.spike{color:var(--warn)}
.neuron-list{display:flex;flex-direction:column;gap:7px}
.neuron-row{font-size:11px}
.neuron-head{display:flex;justify-content:space-between;margin-bottom:3px}
.neuron-name{color:var(--text)}
.neuron-val{color:var(--muted);font-variant-numeric:tabular-nums}
.neuron-row.spike .neuron-name,.neuron-row.spike .neuron-val{color:var(--warn)}
/* change badge */
.cbadge{display:inline-block;padding:3px 9px;border-radius:999px;background:rgba(52,211,153,.1);border:1px solid rgba(52,211,153,.3);color:var(--accent3);font-size:11px;margin:2px}
/* scan state */
.scan-info{font-size:11px;color:var(--muted);margin-bottom:8px}
.pulse{animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.log-btn{width:100%;margin-top:10px;background:transparent;border:1px solid rgba(167,139,250,.3);color:var(--accent2);border-radius:9px;padding:8px;cursor:pointer;font-size:12px;transition:all .2s}
.log-btn:hover{background:rgba(167,139,250,.08)}
/* 辩论模式开关 */
.mode-row{display:flex;align-items:center;gap:10px;margin:8px 0 4px}
.toggle-label{display:flex;align-items:center;gap:8px;cursor:pointer;user-select:none}
.toggle-label input{display:none}
.toggle-track{width:36px;height:20px;background:rgba(148,163,184,.2);border-radius:10px;position:relative;transition:background .2s;flex-shrink:0}
.toggle-label input:checked + .toggle-track{background:linear-gradient(90deg,var(--accent),var(--accent2))}
.toggle-thumb{position:absolute;width:14px;height:14px;background:#fff;border-radius:50%;top:3px;left:3px;transition:left .2s;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.toggle-label input:checked + .toggle-track .toggle-thumb{left:19px}
.toggle-text{font-size:13px;color:var(--text);font-weight:500}
.mode-hint{font-size:11px;color:var(--muted)}
/* 辩论气泡 */
.debate-block{margin:8px 0 8px 40px;background:var(--panel2);border:1px solid var(--border);border-radius:12px;overflow:hidden}
.debate-header{padding:10px 14px 6px;font-size:12px;font-weight:600;color:var(--muted);border-bottom:1px solid var(--border);background:rgba(15,23,42,.5)}
.debate-round{padding:6px 14px}
.debate-round-title{font-size:11px;color:var(--accent2);font-weight:600;margin:8px 0 4px;text-transform:uppercase;letter-spacing:.05em}
.agent-line{display:flex;gap:8px;margin-bottom:8px;align-items:flex-start}
.agent-tag{flex-shrink:0;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;border:1px solid}
.agent-tag.r0{color:#f87171;border-color:rgba(248,113,113,.4);background:rgba(248,113,113,.08)}
.agent-tag.r1{color:#60a5fa;border-color:rgba(96,165,250,.4);background:rgba(96,165,250,.08)}
.agent-tag.r2{color:#fbbf24;border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.08)}
.agent-content{font-size:13px;line-height:1.65;color:var(--text);flex:1}
.synthesis-block{padding:10px 14px;border-top:1px solid var(--border);background:rgba(52,211,153,.05)}
.synthesis-block strong{color:var(--accent3)}
/* 加载动画 */
.ldots::after{content:'';animation:ld 1.2s infinite}
@keyframes ld{0%{content:''}33%{content:'.'}66%{content:'..'}100%{content:'...'}}
/* 结论库 */
.conc-card{background:var(--panel2);border:1px solid var(--border);border-radius:10px;padding:9px 11px;position:relative;transition:border .2s}
.conc-card:hover{border-color:rgba(52,211,153,.35)}
.conc-kp{font-size:12.5px;color:var(--text);line-height:1.6;margin-bottom:4px}
.conc-meta{font-size:10px;color:var(--muted);display:flex;gap:8px;flex-wrap:wrap}
.conc-imp{font-weight:700;color:var(--accent3)}
.conc-del{position:absolute;top:6px;right:6px;background:transparent;border:none;color:var(--muted);cursor:pointer;font-size:12px;opacity:0;transition:opacity .2s}
.conc-card:hover .conc-del{opacity:1}
.conc-del:hover{color:#f87171}
.btn-sort{background:transparent;border:1px solid var(--border);border-radius:6px;padding:3px 10px;font-size:11px;color:var(--muted);cursor:pointer;transition:all .18s}
.btn-sort.active,.btn-sort:hover{border-color:var(--accent3);color:var(--accent3);background:rgba(52,211,153,.08)}
.conc-new-badge{font-size:10px;background:rgba(52,211,153,.15);border:1px solid rgba(52,211,153,.3);color:var(--accent3);border-radius:999px;padding:1px 6px;margin-left:4px}
/* 时间线 modal */
.tl-modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:100;align-items:center;justify-content:center;backdrop-filter:blur(6px)}
.tl-modal.open{display:flex}
.tl-box{background:#0f172a;border:1px solid var(--border);border-radius:18px;width:min(800px,94vw);max-height:88vh;display:flex;flex-direction:column;box-shadow:0 24px 80px rgba(0,0,0,.7)}
.tl-head{padding:16px 20px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.tl-head h2{flex:1;font-size:15px;font-weight:600;background:linear-gradient(90deg,var(--accent3),var(--accent));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.tl-close{width:30px;height:30px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;font-size:16px;display:flex;align-items:center;justify-content:center}
.tl-close:hover{color:var(--text);border-color:var(--accent)}
.tl-body{flex:1;overflow-y:auto;padding:20px 24px;font-size:13.5px;line-height:1.85;color:var(--text)}
.tl-body h1,.tl-body h2{color:var(--accent);margin:18px 0 8px}
.tl-body h3{color:var(--accent3);margin:14px 0 6px}
.tl-body strong{color:var(--text)}
.tl-body blockquote{border-left:3px solid var(--accent2);padding-left:12px;color:var(--muted);margin:8px 0}
.tl-body hr{border:none;border-top:1px solid var(--border);margin:16px 0}
.tl-body p{margin-bottom:8px}
.tl-body ul,.tl-body ol{padding-left:20px;margin-bottom:8px}
.tl-foot{padding:12px 20px;border-top:1px solid var(--border);display:flex;gap:8px;align-items:center}
.tl-meta{font-size:11px;color:var(--muted);flex:1}
</style>
</head>
<body>
<header>
  <h1>⚡ LIF-Memory</h1>
  <span class="chip">连续进化场</span>
  <span class="chip purple" id="noteChip">笔记：加载中</span>
  <span class="chip green" id="statusChip">就绪</span>
</header>
<div class="layout">
  <!-- 左：会话列表 -->
  <div class="sessions-panel">
    <div class="sessions-header">
      <span class="sessions-title">历史对话</span>
      <button class="new-chat-btn" onclick="newChat()" title="新建对话">＋ 新对话</button>
    </div>
    <div class="sessions-search">
      <input id="sessionSearch" placeholder="搜索历史对话" oninput="filterSessions()"/>
    </div>
    <div class="sessions-list" id="sessionsList"></div>
  </div>
  <!-- 中：对话 -->
  <div class="chat">
    <div class="msgs" id="msgs">
      <div class="msg assistant">
        <div class="av">⚡</div>
        <div>
          <div class="bub">你好！我是 LIF-Memory 持续进化助手。<br><br>我会记住我们所有的对话，每次洞察都在前一次的基础上深化。<br><br>先在下方输入问题，我会在你的知识库场中为你查找张力。</div>
          <div class="ts">系统</div>
        </div>
      </div>
    </div>
    <div class="input-area">
      <div class="scan-row">
        <input id="qInput" placeholder="输入问题（例如：我最近的健康状态如何）"/>
        <button class="btn btn-p" id="queryBtn" onclick="doQuery()">查询场</button>
        <button class="btn btn-s" onclick="doScan()">🔄 重扫库</button>
      </div>
      <div class="mode-row">
        <label class="toggle-label">
          <input type="checkbox" id="debateToggle" checked/>
          <span class="toggle-track"><span class="toggle-thumb"></span></span>
          <span class="toggle-text">🧠 多角色辩论模式</span>
        </label>
        <span class="mode-hint" id="modeHint">3个角色自动孵化 · 2轮交锋 · 综合突破洞察</span>
      </div>
      <div class="chips-row">
        <span class="fb" onclick="qf('这个洞察很准，继续深化')">✅ 准确，深化</span>
        <span class="fb" onclick="qf('太宽泛，没抓住核心')">😐 太宽泛</span>
        <span class="fb" onclick="qf('触发太早，证据不足')">⏰ 太早</span>
        <span class="fb" onclick="qf('触发太晚了')">🐌 太晚</span>
        <span class="fb" onclick="qf('洞察深度不够，只是在复述')">🔍 深度不够</span>
        <span class="fb" onclick="qf('和之前的话题有什么关联？')">🔗 找关联</span>
      </div>
      <textarea id="fbInput" placeholder="输入反馈或追问……" rows="2"></textarea>
      <div class="send-row">
        <button class="btn btn-s" onclick="doExport()">📄 导出日志</button>
        <button class="btn btn-s" id="distillBtn" onclick="doDistill()">🧬 整理时间线</button>
        <button class="btn btn-p" id="fbBtn" onclick="doFeedback()">发送 →</button>
      </div>
    </div>
  </div>
  <!-- 右：参数+电位 -->
  <div class="side">
    <div class="stitle">大模型 API</div>
    <div class="vstate llm-card">
      <div class="form-field">
        <label for="llmProvider">Provider</label>
        <select id="llmProvider" onchange="applyProviderPreset()"></select>
      </div>
      <div class="form-field">
        <label for="llmModel">模型</label>
        <input id="llmModel" placeholder="例如 deepseek-chat"/>
      </div>
      <div class="form-field">
        <label for="llmBaseUrl">Base URL</label>
        <input id="llmBaseUrl" placeholder="https://api.deepseek.com"/>
      </div>
      <div class="form-field">
        <label for="llmApiKey">API Key</label>
        <input id="llmApiKey" type="password" placeholder="粘贴 API Key；留空则不修改已保存 key"/>
      </div>
      <div class="llm-actions">
        <button class="btn btn-p" onclick="saveLlmConfig()">保存配置</button>
        <button class="btn btn-s" onclick="testLlmConfig()">测试连接</button>
      </div>
      <div class="llm-status" id="llmStatus">加载配置中…</div>
    </div>
    <div class="stitle">场电位</div>
    <div class="vstate" id="voltageState">
      <div class="vrow"><span class="vlabel">快轨 V_fast</span><span class="vnum" id="vf">—</span></div>
      <div class="vrow"><span class="vlabel">慢轨 V_slow</span><span class="vnum" id="vs">—</span></div>
      <div class="vrow"><span class="vlabel">综合 V</span><span class="vnum" id="vv">—</span></div>
      <div class="vrow"><span class="vlabel">阈值 θ</span><span class="vnum" id="vt">—</span></div>
      <div class="pbar" style="margin-top:8px"><div class="pbarf" id="vbar" style="width:0%"></div></div>
    </div>
    <div class="stitle">LIF 神经元群</div>
    <div class="vstate neuron-list" id="neuronList">
      <div class="vrow"><span class="vlabel">等待查询</span><span class="vnum">—</span></div>
    </div>
    <div class="stitle">参数</div>
    <div id="paramCards"></div>
    <div class="stitle" style="margin-top:12px">主题权重</div>
    <div id="topicW"></div>
    <div class="scan-info" id="scanInfo">扫描时间：—</div>
    <button class="log-btn" onclick="openLog()">📋 打开进化日志</button>

    <!-- 精华结论库 -->
    <div class="stitle" style="margin-top:16px">
      精华结论库
      <span id="conclusionCount" style="color:var(--accent3);font-weight:700;margin-left:4px">0</span>
      <button onclick="clearConclusions()" style="float:right;background:transparent;border:none;color:var(--muted);cursor:pointer;font-size:11px" title="清空结论库">✕ 清空</button>
    </div>
    <div id="conclusionSort" style="display:flex;gap:6px;margin-bottom:8px">
      <button class="btn-sort active" onclick="loadConclusions('ts')" data-sort="ts">最新</button>
      <button class="btn-sort" onclick="loadConclusions('importance')" data-sort="importance">最重要</button>
    </div>
    <div id="conclusionsList" style="display:flex;flex-direction:column;gap:6px"></div>
  </div>
</div>

<!-- 认知演化时间线 Modal -->
<div class="tl-modal" id="tlModal" onclick="if(event.target===this)closeTL()">
  <div class="tl-box">
    <div class="tl-head">
      <h2>🧬 认知演化时间线</h2>
      <button class="tl-close" onclick="closeTL()">✕</button>
    </div>
    <div class="tl-body" id="tlBody">
      <div style="text-align:center;color:var(--muted);padding:40px 0">
        点击「整理时间线」，LLM 会把你所有的对话日志整理成可读的认知演化叙事
      </div>
    </div>
    <div class="tl-foot">
      <span class="tl-meta" id="tlMeta"></span>
      <button class="btn btn-s" onclick="doDistill()" id="redistillBtn">🔄 重新整理</button>
      <button class="btn btn-s" onclick="exportTL()">💾 下载 MD</button>
    </div>
  </div>
</div>

<script>
const PM = {
  threshold:      {desc:"触发阈值",   min:1,   max:15,  unit:""},
  semantic_sigma: {desc:"语义宽度",   min:0.1, max:2.0, unit:""},
  time_sigma:     {desc:"时间窗(天)", min:1,   max:365, unit:"d"},
  fast_decay:     {desc:"近期衰减",   min:0.3, max:0.99,unit:""},
  slow_decay:     {desc:"长期衰减",   min:0.5, max:0.99,unit:""},
  dense_weight:   {desc:"Dense语义权重", min:0, max:1, unit:""},
  sparse_weight:  {desc:"稀疏解释权重", min:0, max:1, unit:""},
};
let params={}, lastQuery="", allSessions=[], activeSessionId="", sessionsBootstrapped=false;
let llmProviders={}, currentConclusionSort='ts';

async function loadLlmConfig(){
  const status=document.getElementById('llmStatus');
  try{
    const r=await fetch('/api/llm-config');
    const d=await r.json();
    llmProviders=d.providers||{};
    const providerEl=document.getElementById('llmProvider');
    providerEl.innerHTML='';
    for(const name of Object.keys(llmProviders)){
      const opt=document.createElement('option');
      opt.value=name; opt.textContent=name;
      providerEl.appendChild(opt);
    }
    providerEl.value=d.provider||'deepseek';
    document.getElementById('llmModel').value=d.model||'';
    document.getElementById('llmBaseUrl').value=d.base_url||'';
    const keyEl=document.getElementById('llmApiKey');
    keyEl.value='';
    keyEl.placeholder=d.has_key?'已配置，留空表示不修改':'粘贴 API Key';
    status.className='llm-status '+(d.has_key?'ok':'bad');
    status.textContent=d.has_key
      ? `已配置 ${d.provider} / ${d.model}。保存位置：${d.config_path||'本机配置'}`
      : `未配置 API Key。请选择 provider，填入 API Key 后保存。保存位置：${d.config_path||'本机配置'}`;
  }catch(e){
    status.className='llm-status bad';
    status.textContent='读取配置失败：'+e;
  }
}

function applyProviderPreset(){
  const p=document.getElementById('llmProvider').value;
  const preset=llmProviders[p]||{};
  document.getElementById('llmModel').value=preset.model||'';
  document.getElementById('llmBaseUrl').value=preset.base_url||'';
  const status=document.getElementById('llmStatus');
  status.className='llm-status';
  status.textContent='已切换 provider。填入 API Key 后保存；留空则沿用该 provider 已保存的 key。';
}

async function saveLlmConfig(){
  const status=document.getElementById('llmStatus');
  status.className='llm-status';
  status.textContent='保存中…';
  const payload={
    provider:document.getElementById('llmProvider').value,
    model:document.getElementById('llmModel').value.trim(),
    base_url:document.getElementById('llmBaseUrl').value.trim(),
    api_key:document.getElementById('llmApiKey').value.trim(),
  };
  try{
    const r=await fetch('/api/llm-config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
    const d=await r.json();
    if(!r.ok||!d.ok) throw new Error(d.error||'保存失败');
    document.getElementById('llmApiKey').value='';
    await loadLlmConfig();
    const cfg=d.config||{};
    status.className='llm-status ok';
    status.textContent=`已保存：${cfg.provider||payload.provider} / ${cfg.model||payload.model}。保存位置：${cfg.config_path||'本机配置'}`;
  }catch(e){
    status.className='llm-status bad';
    status.textContent='保存失败：'+e.message;
  }
}

async function testLlmConfig(){
  const status=document.getElementById('llmStatus');
  status.className='llm-status';
  status.textContent='测试连接中…';
  try{
    const r=await fetch('/api/llm-test',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
    const d=await r.json();
    status.className='llm-status '+(d.ok?'ok':'bad');
    status.textContent=d.ok?'连接成功：'+d.reply:'连接失败：'+d.reply;
  }catch(e){
    status.className='llm-status bad';
    status.textContent='测试失败：'+e;
  }
}

async function fetchStatus(){
  const r=await fetch('/api/status');
  const d=await r.json();
  params=d.params||{};
  renderParams(params);
  document.getElementById('noteChip').textContent=`笔记：${d.note_count||0}`;
  document.getElementById('scanInfo').textContent=`上次扫描：${d.scan_ts||'—'}`;
  document.getElementById('statusChip').textContent=d.status==='scanning'?'扫描中…':d.status==='thinking'?'思考中…':'就绪';
  renderNeurons(d.field_state?.neurons||{});
  if(d.voltage&&lastQuery){
    const topic=d.voltage_topic||'';
    const vd=d.voltage[topic]||{};
    updateVoltage(vd.v_fast,vd.v_slow,params.threshold);
  }
}

function renderParams(p){
  const el=document.getElementById('paramCards');
  el.innerHTML='';
  for(const[k,m] of Object.entries(PM)){
    const v=p[k]??0;
    const pct=((v-m.min)/(m.max-m.min)*100).toFixed(1);
    el.innerHTML+=`<div class="pcard" id="pc_${k}">
      <div class="pname">${k}</div><div class="pdesc">${m.desc}</div>
      <div class="pval">${typeof v==='number'?v.toFixed(3):v}${m.unit}</div>
      <div class="pbar"><div class="pbarf" style="width:${pct}%"></div></div></div>`;
  }
  const tw=p.topic_weights||{};
  const twEl=document.getElementById('topicW');
  twEl.innerHTML='';
  for(const[t,w] of Object.entries(tw)){
    const pct=((w-0.1)/(3.0-0.1)*100).toFixed(1);
    twEl.innerHTML+=`<div class="tw-row">
      <span style="flex:0 0 80px;color:var(--text)">${t}</span>
      <div class="tw-bar"><div class="tw-bf" style="width:${pct}%"></div></div>
      <span class="tw-v">${Number(w).toFixed(2)}</span></div>`;
  }
}

function renderNeurons(neurons){
  const el=document.getElementById('neuronList');
  if(!el) return;
  const entries=Object.entries(neurons||{});
  if(!entries.length){
    el.innerHTML='<div class="vrow"><span class="vlabel">等待查询</span><span class="vnum">—</span></div>';
    return;
  }
  el.innerHTML='';
  for(const[name,n] of entries){
    const v=Number(n.voltage||0), th=Number(n.threshold||1);
    const pct=Math.min(100,(v/Math.max(th,0.001))*100).toFixed(1);
    const spiked=v>=th;
    el.innerHTML+=`<div class="neuron-row ${spiked?'spike':''}">
      <div class="neuron-head"><span class="neuron-name">${name}</span><span class="neuron-val">${v.toFixed(3)} / ${th.toFixed(2)}</span></div>
      <div class="pbar"><div class="pbarf" style="width:${pct}%"></div></div>
    </div>`;
  }
}

function updateVoltage(vf,vs,theta){
  if(vf==null) return;
  const fast_w=params.fast_weight||0.65, slow_w=params.slow_weight||0.35;
  const v=fast_w*(vf||0)+slow_w*(vs||0);
  const th=theta||5;
  const pct=Math.min(100,(v/th*100)).toFixed(1);
  document.getElementById('vf').textContent=(vf||0).toFixed(3);
  document.getElementById('vs').textContent=(vs||0).toFixed(3);
  document.getElementById('vv').textContent=v.toFixed(3);
  document.getElementById('vt').textContent=th;
  document.getElementById('vbar').style.width=pct+'%';
  if(v>=th){
    document.getElementById('vv').classList.add('spike');
  }else{
    document.getElementById('vv').classList.remove('spike');
  }
}

function addMsg(role,content,ts='',vf,vs,spiked){
  const wrap=document.getElementById('msgs');
  const div=document.createElement('div');
  div.className='msg '+role;
  const av=role==='assistant'?'⚡':'🧑';
  const html=content.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');
  div.innerHTML=`<div class="av">${av}</div>
    <div><div class="bub">${html}</div><div class="ts">${ts||new Date().toLocaleTimeString()}</div></div>`;
  wrap.appendChild(div);
  if(role==='assistant'&&vf!=null){
    const barDiv=document.createElement('div');
    barDiv.className='voltage-bar';
    const fast_w=params.fast_weight||0.65,slow_w=params.slow_weight||0.35;
    const v=fast_w*vf+slow_w*vs;
    const th=params.threshold||5;
    const pct=Math.min(100,v/th*100).toFixed(1);
    barDiv.innerHTML=`<div class="voltage-fill" style="width:${pct}%"></div>`;
    wrap.appendChild(barDiv);
    if(spiked){
      const badge=document.createElement('div');
      badge.className='spike-badge';
      badge.textContent='⚡ Insight Spike 触发';
      wrap.appendChild(badge);
    }
  }
  wrap.scrollTop=wrap.scrollHeight;
}

function rmLoading(id){const el=document.getElementById(id);if(el)el.remove();}

function addChanges(changes){
  if(!changes||!changes.length)return;
  const wrap=document.getElementById('msgs');
  const div=document.createElement('div');
  div.style.padding='4px 0 4px 40px';
  div.innerHTML=changes.map(c=>`<span class="cbadge">⚙ ${c}</span>`).join('');
  wrap.appendChild(div);wrap.scrollTop=wrap.scrollHeight;
}

function hlParam(changes){
  for(const ch of changes){
    const k=ch.split(':')[0].trim();
    const el=document.getElementById('pc_'+k);
    if(el){el.classList.add('hl');setTimeout(()=>el.classList.remove('hl'),3000);}
  }
}

function qf(t){document.getElementById('fbInput').value=t;}

async function doScan(){
  document.getElementById('statusChip').textContent='扫描中…';
  addMsg('assistant','正在扫描知识库，建立连续场……（首次较慢）');
  addLoading('scanLoad');
  const r=await fetch('/api/scan',{method:'POST'});
  const d=await r.json();
  rmLoading('scanLoad');
  addMsg('assistant',`✅ 扫描完成，共加载 ${d.note_count} 篇笔记。场已建立，现在可以开始查询了。`);
  await fetchStatus();
}

async function doQuery(){
  const q=document.getElementById('qInput').value.trim();
  if(!q)return;
  lastQuery=q;
  addMsg('user',q);
  document.getElementById('qInput').value='';
  document.getElementById('queryBtn').disabled=true;
  const debateOn=document.getElementById('debateToggle').checked;
  addLoading('qLoad', debateOn ? '孵化角色，开始辩论' : '查询场');
  const r=await fetch('/api/query',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({query:q, debate:debateOn})});
  const d=await r.json();
  rmLoading('qLoad');
  document.getElementById('queryBtn').disabled=false;
  if(debateOn){
    addDebateMsg(d.reply||'[无结果]', d.v_fast, d.v_slow, d.spiked);
  } else {
    addMsg('assistant', d.reply||'[无结果]', '', d.v_fast, d.v_slow, d.spiked);
  }
  if(d.changes&&d.changes.length){addChanges(d.changes);hlParam(d.changes);}
  if(d.v_fast!=null) updateVoltage(d.v_fast,d.v_slow,params.threshold);
  if(d.field_observation?.neurons) renderNeurons(Object.fromEntries(Object.entries(d.field_observation.neurons).map(([k,v])=>[k,{voltage:v.voltage,threshold:v.threshold}])));
  // 如果有新结论（辩论后自动提炼或 spike），刷新结论库面板
  if(d.conclusions_count!=null) loadConclusions(currentConclusionSort);
  await fetchStatus();
  await loadSessions(false);
}

function addDebateMsg(raw, vf, vs, spiked){
  // 把纯文本辩论结果渲染成结构化气泡
  const wrap=document.getElementById('msgs');
  const container=document.createElement('div');
  container.style.animation='fi .3s ease';

  // 解析各段
  const lines = raw.split('\n');
  let headerLines=[], round1=[], round2=[], synthLines=[], inR1=false, inR2=false, inSynth=false;
  for(const line of lines){
    if(line.includes('第一轮') || line.includes('初始观点')){inR1=true;inR2=false;inSynth=false;continue;}
    if(line.includes('第二轮') || line.includes('交锋反驳')){inR2=true;inR1=false;inSynth=false;continue;}
    if(line.includes('综合') && !inSynth && (inR2||round2.length)){inSynth=true;inR1=false;inR2=false;continue;}
    if(inR1) round1.push(line);
    else if(inR2) round2.push(line);
    else if(inSynth) synthLines.push(line);
    else headerLines.push(line);
  }

  const agentColors=['r0','r1','r2'];
  function renderRound(lines){
    let html='';
    let agentIdx=-1;
    const seen={};
    for(const line of lines){
      const m=line.match(/^[🔴🔵🟡🟢]\s*\*\*(.+?)\*\*[：:](.*)/);
      if(m){
        const name=m[1].trim();
        if(!(name in seen)){seen[name]=Object.keys(seen).length;}
        const ci=seen[name]%3;
        html+=`<div class="agent-line">
          <span class="agent-tag ${agentColors[ci]}">${name}</span>
          <span class="agent-content">${m[2].trim()}</span></div>`;
      } else if(line.trim()){
        // 普通行
        const cleaned=line.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
        if(cleaned.trim()) html+=`<div style="font-size:12px;color:var(--muted);margin-bottom:4px">${cleaned}</div>`;
      }
    }
    return html;
  }

  const r1html=renderRound(round1);
  const r2html=renderRound(round2);
  const synthRaw=synthLines.join('\n').trim()
    .replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>').replace(/\n/g,'<br>');

  // 如果解析失败，fallback 到普通气泡
  if(!r1html && !r2html && !synthRaw){
    addMsg('assistant', raw, '', vf, vs, spiked);
    return;
  }

  let html=`<div class="debate-block">
    <div class="debate-header">🧠 多角色辩论场</div>`;
  if(r1html){
    html+=`<div class="debate-round">
      <div class="debate-round-title">第一轮 · 初始观点</div>${r1html}</div>`;
  }
  if(r2html){
    html+=`<div class="debate-round" style="border-top:1px solid var(--border)">
      <div class="debate-round-title">第二轮 · 交锋反驳</div>${r2html}</div>`;
  }
  if(synthRaw){
    html+=`<div class="synthesis-block">${synthRaw}</div>`;
  }
  html+=`</div>`;
  container.innerHTML=html;
  wrap.appendChild(container);

  // 电位条
  if(vf!=null){
    const fw=params.fast_weight||0.65, sw=params.slow_weight||0.35;
    const v=fw*vf+sw*vs, th=params.threshold||5;
    const pct=Math.min(100,v/th*100).toFixed(1);
    const bar=document.createElement('div');
    bar.className='voltage-bar';bar.style.marginLeft='0';
    bar.innerHTML=`<div class="voltage-fill" style="width:${pct}%"></div>`;
    wrap.appendChild(bar);
    if(spiked){
      const badge=document.createElement('div');
      badge.className='spike-badge';badge.style.marginLeft='0';
      badge.textContent='⚡ Insight Spike 触发';
      wrap.appendChild(badge);
    }
  }
  wrap.scrollTop=wrap.scrollHeight;
}

function addLoading(id, label='思考中'){
  const wrap=document.getElementById('msgs');
  const div=document.createElement('div');
  div.className='msg assistant';div.id=id;
  div.innerHTML=`<div class="av">⚡</div><div><div class="bub ldots">${label}</div></div>`;
  wrap.appendChild(div);wrap.scrollTop=wrap.scrollHeight;
}

async function doFeedback(){
  const fb=document.getElementById('fbInput').value.trim();
  if(!fb)return;
  addMsg('user',fb);
  document.getElementById('fbInput').value='';
  document.getElementById('fbBtn').disabled=true;
  addLoading('fbLoad','分析反馈，调参中');
  const r=await fetch('/api/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({feedback:fb})});
  const d=await r.json();
  rmLoading('fbLoad');
  document.getElementById('fbBtn').disabled=false;
  addMsg('assistant',d.reply||'[无回复]');
  if(d.changes&&d.changes.length){addChanges(d.changes);hlParam(d.changes);}
  await fetchStatus();
  await loadSessions(false);
}

async function doExport(){
  const r=await fetch('/api/export',{method:'POST'});
  const d=await r.json();
  const blob=new Blob([d.markdown],{type:'text/markdown'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='LIF-Memory-进化日志-'+new Date().toISOString().slice(0,10)+'.md';
  a.click();
  addMsg('assistant','✅ 进化日志已导出为 Markdown 文件。');
}

function openLog(){
  fetch('/api/log_path').then(r=>r.json()).then(d=>{
    addMsg('assistant',`📋 进化日志路径：\n${d.path}\n\n可用 Obsidian 打开，每次洞察都会自动追加到这个文件。`);
  });
}

// ── 会话管理 ──────────────────────────────────────────────────────────────────
async function loadSessions(restoreActive=false){
  const r=await fetch('/api/sessions');
  const d=await r.json();
  allSessions=d.sessions||[];
  activeSessionId=d.active||'';
  renderSessionsList(allSessions);
  if(restoreActive && activeSessionId && !sessionsBootstrapped){
    sessionsBootstrapped=true;
    const r2=await fetch('/api/sessions/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid:activeSessionId})});
    const d2=await r2.json();
    restoreMessages(d2.messages);
  }
}

function renderSessionsList(sessions){
  const el=document.getElementById('sessionsList');
  el.innerHTML='';
  if(!sessions||!sessions.length){
    el.innerHTML='<div class="session-empty">暂无历史对话<br>在中间输入问题后，这里会自动保存并显示每一轮对话。</div>';
    return;
  }
  for(const s of sessions){
    const div=document.createElement('div');
    div.className='sess-item'+(s.active?' active':'');
    div.dataset.sid=s.id;
    div.dataset.search=((s.title||'')+' '+(s.preview||'')).toLowerCase();
    const preview=s.preview||'还没有消息';
    const time=(s.updated_ts||s.ts||'').slice(5,16);
    div.innerHTML=`
      <div class="sess-title">${escHtml(s.title||'未命名对话')}</div>
      <div class="sess-preview">${escHtml(preview)}</div>
      <div class="sess-meta"><span>${time}</span><span>${s.count||0} 条消息</span></div>
      <button class="sess-del" onclick="deleteSession(event,'${s.id}')" title="删除">×</button>`;
    div.addEventListener('click', e=>{
      if(e.target.classList.contains('sess-del')) return;
      switchSession(s.id);
    });
    el.appendChild(div);
  }
}

function filterSessions(){
  const kw=(document.getElementById('sessionSearch')?.value||'').trim().toLowerCase();
  if(!kw){renderSessionsList(allSessions);return;}
  renderSessionsList(allSessions.filter(s=>((s.title||'')+' '+(s.preview||'')).toLowerCase().includes(kw)));
}

function escHtml(s){ return String(s??'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

async function newChat(){
  const r=await fetch('/api/sessions/new',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({})});
  const d=await r.json();
  activeSessionId=d.session?.id||'';
  document.getElementById('msgs').innerHTML='';
  addMsg('assistant','新对话已开始，请输入你的问题。');
  await loadSessions(false);
}

async function switchSession(sid){
  const r=await fetch('/api/sessions/switch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});
  const d=await r.json();
  if(d.ok){
    activeSessionId=sid;
    restoreMessages(d.messages);
    await loadSessions(false);
  }
}

async function deleteSession(e, sid){
  e.stopPropagation();
  if(!confirm('删除这个对话？')) return;
  const r=await fetch('/api/sessions/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid})});
  const d=await r.json();
  allSessions=d.sessions||[];
  activeSessionId=d.active||'';
  renderSessionsList(allSessions);
  // 如果删的是当前，刷新消息区
  const newActive=activeSessionId;
  const r2=await fetch('/api/sessions/messages',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({sid:newActive})});
  const d2=await r2.json();
  restoreMessages(d2.messages);
}

function restoreMessages(messages){
  const wrap=document.getElementById('msgs');
  wrap.innerHTML='';
  if(!messages||!messages.length){
    addMsg('assistant','新对话已开始，请输入你的问题。');
    return;
  }
  for(const m of messages){
    if(m.type==='debate'){
      addMsg(m.role, m.content, m.ts);  // 辩论内容暂时用普通气泡恢复，保留可读性
    } else {
      addMsg(m.role, m.content, m.ts);
    }
  }
}

loadLlmConfig();
fetchStatus();
loadSessions(true);
loadConclusions('ts');
setInterval(fetchStatus,8000);
setInterval(loadSessions,15000);
setInterval(()=>loadConclusions(currentConclusionSort),30000);

// ── 结论库 ────────────────────────────────────────────────────────────────
async function loadConclusions(sort='ts'){
  currentConclusionSort=sort;
  // 更新排序按钮样式
  document.querySelectorAll('.btn-sort').forEach(b=>{
    b.classList.toggle('active', b.dataset.sort===sort);
  });
  const r=await fetch('/api/conclusions?sort='+sort);
  const d=await r.json();
  renderConclusions(d.conclusions||[], d.count||0);
}

function renderConclusions(list, count){
  document.getElementById('conclusionCount').textContent=count;
  const el=document.getElementById('conclusionsList');
  el.innerHTML='';
  if(!list.length){
    el.innerHTML='<div style="font-size:11px;color:var(--muted);text-align:center;padding:12px 0">暂无精华结论<br>完成一次辩论后自动提炼</div>';
    return;
  }
  const now=Date.now();
  for(const c of list){
    const ageMs=now - new Date(c.ts).getTime();
    const isNew=ageMs<3600000; // 1小时内
    const impPct=Math.round(c.importance*100);
    const impColor=c.importance>=0.8?'var(--warn)':c.importance>=0.6?'var(--accent3)':'var(--accent)';
    const div=document.createElement('div');
    div.className='conc-card';
    div.innerHTML=`
      <div class="conc-kp">${escHtml(c.key_point)}${isNew?'<span class="conc-new-badge">NEW</span>':''}</div>
      <div class="conc-meta">
        <span class="conc-imp" style="color:${impColor}">重要性 ${impPct}%</span>
        <span>${c.ts.slice(0,10)}</span>
        <span style="color:var(--muted);max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${escHtml(c.query)}">Q: ${escHtml(c.query.slice(0,20))}</span>
        ${c.recall_count>0?`<span>被引用${c.recall_count}次</span>`:''}
      </div>
      <button class="conc-del" onclick="deleteConclusion(event,'${c.id}')" title="删除">✕</button>`;
    el.appendChild(div);
  }
}

async function deleteConclusion(e, cid){
  e.stopPropagation();
  const r=await fetch('/api/conclusions/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:cid})});
  const d=await r.json();
  document.getElementById('conclusionCount').textContent=d.count;
  await loadConclusions(currentConclusionSort);
}

async function clearConclusions(){
  if(!confirm('清空所有精华结论？此操作不可撤销。')) return;
  await fetch('/api/conclusions/clear',{method:'POST'});
  await loadConclusions(currentConclusionSort);
}

// ── 认知演化时间线 ────────────────────────────────────────────────────────────
let tlMarkdown = '';

async function doDistill(){
  const btn=document.getElementById('distillBtn');
  const rbtn=document.getElementById('redistillBtn');
  if(btn) btn.disabled=true;
  if(rbtn) rbtn.disabled=true;
  // 打开 modal 并显示进度
  document.getElementById('tlModal').classList.add('open');
  document.getElementById('tlBody').innerHTML=
    '<div style="text-align:center;padding:60px 0"><div class="bub ldots" style="display:inline-block">LLM 正在整理你的认知演化历程</div></div>';
  document.getElementById('tlMeta').textContent='';

  const r=await fetch('/api/distill',{method:'POST'});
  const d=await r.json();

  if(btn) btn.disabled=false;
  if(rbtn) rbtn.disabled=false;

  if(d.error){
    document.getElementById('tlBody').innerHTML=
      `<div style="color:#f87171;padding:20px">${escHtml(d.error)}</div>`;
    return;
  }
  tlMarkdown=d.markdown||'';
  document.getElementById('tlBody').innerHTML=renderMarkdown(tlMarkdown);
  document.getElementById('tlMeta').textContent=
    `共整理 ${d.log_count||0} 条对话、${d.conclusions_count||0} 条精华结论 · 已保存到知识库`;
}

function closeTL(){
  document.getElementById('tlModal').classList.remove('open');
}

function exportTL(){
  if(!tlMarkdown) return;
  const blob=new Blob([tlMarkdown],{type:'text/markdown'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);
  a.download='LIF-Memory-精华时间线-'+new Date().toISOString().slice(0,10)+'.md';
  a.click();
}

function renderMarkdown(md){
  // 轻量 MD 渲染：标题/粗体/斜体/引用/分割线/段落
  let html = md
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    // 标题
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm,  '<h2>$1</h2>')
    .replace(/^# (.+)$/gm,   '<h1>$1</h1>')
    // 粗体斜体
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g,     '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g,         '<em>$1</em>')
    // 引用块
    .replace(/^&gt; (.+)$/gm, '<blockquote>$1</blockquote>')
    // 分割线
    .replace(/^---+$/gm, '<hr>')
    // 无序列表
    .replace(/^[*-] (.+)$/gm, '<li>$1</li>')
    // 换行 → <br> or <p>
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  return '<p>' + html + '</p>';
}
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", "/index.html"):
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif p == "/api/status":
            voltage = dict(STATE.field.voltage)
            self.send_json({
                "status":     STATE.status,
                "note_count": STATE.field.note_count,
                "scan_ts":    STATE.field.scan_ts,
                "params":     STATE.params,
                "voltage":    voltage,
                "field_state": domain_summary(STATE.vault),
            })
        elif p == "/api/llm-config":
            self.send_json(public_llm_config())
        elif p == "/api/field-state":
            self.send_json(domain_summary(STATE.vault))
        elif p == "/api/log_path":
            self.send_json({"path": str(STATE.log_path)})
        elif p == "/api/conclusions":
            sort = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("sort", ["ts"])[0]
            self.send_json({
                "conclusions": STATE.conclusions.get_all(sort_by=sort),
                "count":       len(STATE.conclusions.conclusions),
            })
        elif p == "/api/sessions":
            self.send_json({"sessions": STATE.sessions.list_sessions(),
                            "active": STATE.sessions.active_sid})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        p    = urllib.parse.urlparse(self.path).path
        body = self.read_body()

        if p == "/api/scan":
            def do():
                count = scan_vault(STATE.vault)
            threading.Thread(target=do, daemon=True).start()
            # 等待扫描完成（最多120s）
            for _ in range(120):
                if STATE.status == "idle" and STATE.field.note_count > 0:
                    break
                time.sleep(1)
            self.send_json({"note_count": STATE.field.note_count, "scan_ts": STATE.field.scan_ts})

        elif p == "/api/query":
            query = body.get("query", "").strip()
            debate_mode = body.get("debate", True)  # 默认开启辩论
            if not query:
                self.send_json({"error": "empty"}, 400); return
            STATE.status = "thinking"
            STATE.last_query = query

            if not STATE.field.note_vectors:
                scan_vault(STATE.vault)

            try:
                import continuous_problem_field as cpf
                params = STATE.params
                today  = date.today()
                notes  = restore_notes_from_field()
                query_dense = []
                if float(params.get("dense_weight", 0.0)) > 0:
                    try:
                        query_dense = embed_query(query, STATE.vault, Path(__file__).resolve().parent)
                    except Exception:
                        query_dense = []
                hits, field_energy, daily_current, daily_completion = cpf.reconstruct_field(
                    notes=notes, query=query, today=today,
                    top_k=8,
                    time_sigma=float(params.get("time_sigma", 30.0)),
                    semantic_sigma=float(params.get("semantic_sigma", 0.55)),
                    all_notes=True,
                    query_dense=query_dense,
                    dense_weight=float(params.get("dense_weight", 0.45)),
                    sparse_weight=float(params.get("sparse_weight", 0.55)),
                )
                topic = cpf.infer_topic(query)
                v_fast, v_slow = STATE.field.get_voltage(topic)
                threshold  = float(params.get("threshold", 5.0))
                fast_decay = float(params.get("fast_decay", 0.76))
                slow_decay = float(params.get("slow_decay", 0.94))
                fast_w     = float(params.get("fast_weight", 0.65))
                slow_w     = float(params.get("slow_weight", 0.35))
                total_input      = sum(daily_current.values())
                total_completion = sum(daily_completion.values())
                v_fast = max(0.0, v_fast * fast_decay + total_input * 4.0 - total_completion)
                v_slow = max(0.0, v_slow * slow_decay + total_input * 4.0 * 0.42 - total_completion * 0.30)
                v      = fast_w * v_fast + slow_w * v_slow
                spiked = v >= threshold
                STATE.field.set_voltage(topic, v_fast, v_slow)
                field_observation = update_domain_after_query(STATE.vault, query, topic, hits, field_energy, v, spiked)

                snippets = []
                for i, hit in enumerate(hits[:5], 1):
                    snippets.append(f"{i}. [{hit.note.title}] 相似={hit.semantic:.3f}\n   {hit.note.snippet}")
                evidence_text = "\n".join(snippets) if snippets else "（未找到相关证据，将基于问题本身讨论）"

                # 检索历史精华结论，注入证据（比原始 vault 证据权重高）
                import continuous_problem_field as cpf
                query_vec      = cpf.vectorize(query)
                conclusions_ctx = _build_conclusions_context(query_vec)
                if conclusions_ctx:
                    evidence_text = conclusions_ctx + "\n\n【知识库证据（原始）】\n" + evidence_text

                voltage_info  = f"V={v:.3f} fast={v_fast:.3f} slow={v_slow:.3f} θ={threshold} {'⚡SPIKE' if spiked else ''}"

                param_changes = []
                if debate_mode:
                    # 多智能体辩论模式
                    reply, debate_rounds, delta = run_debate(
                        query, evidence_text, STATE.field.llm_history[-16:]
                    )
                    param_changes = STATE.apply_delta(delta) if delta else []
                    compressed = f"[辩论] 问题：{query}\n综合命题：{reply.split('综合命题')[1][:100] if '综合命题' in reply else reply[:120]}"
                    STATE.field.push_llm("user", f"问题：{query}\n证据：{evidence_text[:300]}")
                    STATE.field.push_llm("assistant", compressed)
                    # 存入会话历史
                    STATE.add_msg("user", query)
                    STATE.add_msg("assistant", reply, "debate")
                else:
                    system_p = {"role": "system", "content": (
                        "你是 LIF-Memory 持续进化洞察引擎，拥有完整对话历史。"
                        "每次洞察比上一次更深，不重复，主动追问。"
                    )}
                    user_c = (f"【查询】{query}\n【场状态】{voltage_info}\n"
                              f"【证据】\n{evidence_text}\n\n请洞察并追问。")
                    messages = [system_p] + STATE.field.llm_history[-20:] + [{"role":"user","content":user_c}]
                    reply = call_llm(messages)
                    STATE.field.push_llm("user", user_c)
                    STATE.field.push_llm("assistant", reply)
                    # 存入会话历史
                    STATE.add_msg("user", query)
                    STATE.add_msg("assistant", reply)
                    # 非辩论模式：spike 时自动保存结论
                    if spiked:
                        _auto_save_conclusion(query, reply, {}, evidence_text)

                STATE.field.append_log(query, reply, [voltage_info] + param_changes)
                STATE.field.save()
                append_evolution_log(query, reply, voltage_info, spiked)
                self.send_json({
                    "reply": reply, "v_fast": v_fast, "v_slow": v_slow,
                    "v": v, "spiked": spiked, "field_energy": field_energy,
                    "changes": param_changes,
                    "field_observation": field_observation,
                    "conclusions_count": len(STATE.conclusions.conclusions),
                })
            except Exception as e:
                self.send_json({"reply": f"[错误: {e}]", "v_fast": 0, "v_slow": 0, "v": 0, "spiked": False})
            finally:
                STATE.status = "idle"

        elif p == "/api/feedback":
            fb = body.get("feedback", "").strip()
            if not fb:
                self.send_json({"error": "empty"}, 400); return
            STATE.status = "thinking"
            try:
                reply, delta = llm_analyze_and_tune(fb, STATE.last_query)
                changes = STATE.apply_delta(delta) if delta else []
                reward_delta, reward_record = reward_update(STATE.vault, fb, STATE.params)
                reward_changes = STATE.apply_delta(reward_delta) if reward_delta else []
                changes.extend(reward_changes)
                if changes:
                    STATE.field.append_log(f"[调参反馈] {fb}", reply, changes)
                    STATE.field.save()
                    append_evolution_log(f"调参反馈：{fb}", reply, ", ".join(changes), False)
                STATE.add_msg("user", fb)
                STATE.add_msg("assistant", reply)
                self.send_json({"reply": reply, "changes": changes, "params": STATE.params, "reward": reward_record})
            except Exception as e:
                self.send_json({"reply": f"[错误: {e}]", "changes": []})
            finally:
                STATE.status = "idle"

        elif p == "/api/llm-config":
            try:
                provider = str(body.get("provider") or STATE.llm_cfg.get("provider") or "deepseek").strip()
                base_url = str(body.get("base_url") or "").strip()
                model    = str(body.get("model") or "").strip()
                api_key  = str(body.get("api_key") or "").strip()
                cfg = save_llm_local_config(provider, base_url, model, api_key)
                with STATE.lock:
                    STATE.llm_cfg = cfg
                self.send_json({
                    "ok": True,
                    "message": "已保存到本地配置，并已更新当前服务。",
                    "config": public_llm_config(),
                })
            except Exception as e:
                self.send_json({"ok": False, "error": str(e)}, 400)

        elif p == "/api/llm-test":
            try:
                reply = call_llm([
                    {"role": "system", "content": "你是连接测试助手。"},
                    {"role": "user", "content": "请只回复 OK。"},
                ])
                ok = not reply.startswith("[LLM") and "失败" not in reply
                self.send_json({"ok": ok, "reply": reply})
            except Exception as e:
                self.send_json({"ok": False, "reply": f"[测试失败: {e}]"})

        elif p == "/api/export":
            md = generate_export_md()
            self.send_json({"markdown": md})

        elif p == "/api/distill":
            STATE.status = "thinking"
            try:
                md = distill_evolution_narrative()
                timeline_path = STATE.vault / "LIF-Memory-精华时间线.md"
                self.send_json({
                    "markdown": md,
                    "path":     str(timeline_path),
                    "log_count": len(STATE.field.evolution_log),
                    "conclusions_count": len(STATE.conclusions.conclusions),
                })
            except Exception as e:
                self.send_json({"error": str(e), "markdown": ""})
            finally:
                STATE.status = "idle"

        elif p == "/api/conclusions/delete":
            cid = body.get("id", "")
            ok  = STATE.conclusions.delete(cid)
            self.send_json({"ok": ok, "count": len(STATE.conclusions.conclusions)})

        elif p == "/api/conclusions/clear":
            STATE.conclusions.clear()
            self.send_json({"ok": True})

        elif p == "/api/sessions":
            self.send_json({"sessions": STATE.sessions.list_sessions(),
                            "active": STATE.sessions.active_sid})

        elif p == "/api/sessions/new":
            title = body.get("title", "")
            sess  = STATE.sessions.new_session(title)
            self.send_json({"session": sess, "messages": []})

        elif p == "/api/sessions/switch":
            sid = body.get("sid", "")
            ok  = STATE.sessions.switch_session(sid)
            msgs = STATE.sessions.get_messages(sid) if ok else []
            self.send_json({"ok": ok, "messages": msgs})

        elif p == "/api/sessions/delete":
            sid = body.get("sid", "")
            STATE.sessions.delete_session(sid)
            self.send_json({"ok": True, "sessions": STATE.sessions.list_sessions(),
                            "active": STATE.sessions.active_sid})

        elif p == "/api/sessions/messages":
            sid  = body.get("sid", STATE.sessions.active_sid)
            msgs = STATE.sessions.get_messages(sid)
            self.send_json({"messages": msgs})

        else:
            self.send_response(404); self.end_headers()


def generate_export_md() -> str:
    lines = [
        "# LIF-Memory 进化日志（导出）",
        f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"知识库：{STATE.vault}",
        f"笔记数：{STATE.field.note_count}",
        "\n---\n",
        "## 完整进化记录\n",
    ]
    for entry in STATE.field.evolution_log:
        lines.append(f"### {entry['ts']}")
        lines.append(f"\n**查询**：{entry['query']}\n")
        lines.append(f"**状态**：{', '.join(entry['changes'])}\n")
        lines.append(f"{entry['insight']}\n")
        lines.append("---\n")
    lines.append("## 当前参数\n\n```json")
    lines.append(json.dumps(STATE.params, ensure_ascii=False, indent=2))
    lines.append("```\n")
    return "\n".join(lines)


def distill_evolution_narrative() -> str:
    """
    让 LLM 对全部进化日志做二次消化：
    把散乱的流水账整理成一条有时间节点的认知演化叙事，
    写入 LIF-Memory-精华时间线.md，同时返回 Markdown 文本。
    """
    log_entries = STATE.field.evolution_log
    conclusions = STATE.conclusions.get_all(sort_by="ts")

    if not log_entries and not conclusions:
        return "# 精华时间线\n\n暂无记录，请先进行几轮对话后再整理。"

    # ── 构建原始素材 ──────────────────────────────────────────────────────────
    raw_log_text = ""
    for e in log_entries[-60:]:   # 最多取60条，避免超 token
        raw_log_text += (
            f"\n[{e['ts']}] 查询：{e['query']}\n"
            f"  场状态：{', '.join(e['changes'][:2])}\n"
            f"  洞察摘要：{e['insight'][:300]}\n"
        )

    conclusions_text = ""
    for c in conclusions[-30:]:
        conclusions_text += (
            f"\n[{c['ts'][:10]}] 重要性={c['importance']:.2f} "
            f"| {c['key_point']}\n"
            f"  原问题：{c['query'][:60]}\n"
        )

    system_p = {
        "role": "system",
        "content": (
            "你是一位善于整理思想演化历程的编辑。\n"
            "你会收到一份原始的「对话日志」和「精华结论列表」。\n"
            "你的任务：把它们整理成一篇让人能读懂的「认知演化叙事」，格式要求：\n\n"
            "1. 以时间为轴，按日期分组（同一天的归在一起）\n"
            "2. 每个时间节点：\n"
            "   - 说清楚「这段时间在探索什么问题」\n"
            "   - 「这次对话产生了什么认知跃变/新洞察」\n"
            "   - 如果有精华结论与该时间点关联，引用它\n"
            "3. 在文末写一段「认知演化总结」：回顾整体轨迹，指出最重要的转折点\n"
            "4. 输出 Markdown 格式，标题层级清晰，可直接在 Obsidian 打开\n"
            "5. 语言自然流畅，像在讲一个思维成长的故事，不要像流水账\n\n"
            "不要复述原文，要提炼和叙述。"
        ),
    }
    user_p = {
        "role": "user",
        "content": (
            f"以下是原始对话日志（共 {len(log_entries)} 条）：\n"
            f"{raw_log_text}\n\n"
            f"以下是精华结论（共 {len(conclusions)} 条）：\n"
            f"{conclusions_text}\n\n"
            "请整理成认知演化叙事 Markdown 文档。"
        ),
    }

    narrative = call_llm([system_p, user_p])

    # ── 写入文件 ──────────────────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_md = (
        f"# LIF-Memory 认知演化时间线\n\n"
        f"> 由 LLM 于 {ts} 自动整理，原始对话 {len(log_entries)} 条，精华结论 {len(conclusions)} 条。\n\n"
        f"---\n\n"
        f"{narrative}\n\n"
        f"---\n\n"
        f"*此文件由「整理日志」功能自动生成，可随时重新生成以更新内容。*\n"
    )

    timeline_path = STATE.vault / "LIF-Memory-精华时间线.md"
    try:
        timeline_path.write_text(full_md, encoding="utf-8")
    except Exception:
        pass

    return full_md


# ── 启动 ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LIF-Memory 持续进化网页调参器")
    p.add_argument("--vault",        type=Path, default=Path("."))
    p.add_argument("--port",         type=int,  default=7860)
    p.add_argument("--llm-provider", choices=list(PROVIDER_PRESETS.keys()), default="deepseek")
    p.add_argument("--llm-api-key",  type=str,  default=None)
    p.add_argument("--llm-model",    type=str,  default=None)
    return p.parse_args()


def main():
    global STATE
    args   = parse_args()
    vault  = args.vault.resolve()
    llm_cfg = build_llm_config(
        args.llm_provider,
        api_key=args.llm_api_key,
        model=args.llm_model,
        prefer_local_provider=not (args.llm_api_key or args.llm_model),
    )

    params_path       = vault / "lif_field_params.json"
    state_path        = vault / "lif_field_state.json"
    log_path          = vault / "LIF-Memory-进化日志.md"
    sessions_path     = vault / "lif_sessions.json"
    conclusions_path  = vault / "lif_conclusions.json"

    STATE = AppState(vault=vault, params_path=params_path, state_path=state_path,
                     log_path=log_path, sessions_path=sessions_path,
                     conclusions_path=conclusions_path, llm_cfg=llm_cfg)

    # 启动时自动检查是否需要（重）扫描
    if STATE.field.needs_rescan(vault):
        print("📂 检测到知识库变化，后台扫描中…")
        threading.Thread(target=lambda: scan_vault(vault), daemon=True).start()
    else:
        print(f"✅ 加载已有场状态，共 {STATE.field.note_count} 篇笔记")

    url = f"http://127.0.0.1:{args.port}"
    print(f"\n⚡ LIF-Memory 持续进化服务")
    print(f"   知识库：{vault}")
    print(f"   LLM：{llm_cfg['provider']} / {llm_cfg['model']}")
    print(f"   API Key：{'已配置 ✓' if llm_cfg.get('api_key') else '❌ 未配置，可在网页右侧填写'}")
    print(f"   进化日志：{log_path}")
    print(f"\n   👉 {url}\n")

    try:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    except Exception:
        pass

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


if __name__ == "__main__":
    main()
