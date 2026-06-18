from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import llm_adapter


VERSION = "0.8.1"

DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-/\u4e00-\u9fff]+)")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}")

IGNORED_PARTS = {
    ".git",
    ".obsidian",
    ".trash",
    "__pycache__",
    ".venv",
    "node_modules",
    "examples",
    "tests",
    "AI_Compression",
}

DEFAULT_ROLE_SPECS: dict[str, dict[str, object]] = {
    "cartographer": {
        "name": "Cartographer",
        "mission": "还原知识库的等效迷宫地图：主题、通道、中心节点、孤岛节点。",
        "focus": ["topic", "edge", "hub", "bridge", "orphan"],
        "weight": 1.0,
    },
    "skeptic": {
        "name": "Skeptic",
        "mission": "审查哪些判断证据不足、哪些连接是幻觉、哪些主线应该降级。",
        "focus": ["claim", "contradiction", "evidence_gap", "risk", "downgrade"],
        "weight": 1.15,
    },
    "executor": {
        "name": "Executor",
        "mission": "把知识库压成可执行任务：今天能做什么、输入是什么、输出是什么、完成标准是什么。",
        "focus": ["task", "next_action", "completion_target", "priority"],
        "weight": 1.25,
    },
    "linker": {
        "name": "Linker",
        "mission": "寻找跨笔记、跨主题、跨时间的连接，专门修复局部输入链接不起来的问题。",
        "focus": ["cross_link", "concept_alignment", "missing_edge", "bridge"],
        "weight": 1.1,
    },
    "lif_scheduler": {
        "name": "LIF Scheduler",
        "mission": "从多视角图谱中计算今日 spike：行动压力、解释张力、阻塞、紧急性、可完成性。",
        "focus": ["lif_score", "spike", "urgency", "blocker", "completion"],
        "weight": 1.3,
    },
}

ACTION_WORDS = ["今天", "下一步", "接下来", "应该", "需要", "必须", "写", "做", "测", "跑", "整理", "投递", "生成", "执行", "闭环"]
BLOCKER_WORDS = ["卡", "不知道", "缺", "没有", "不够", "失败", "问题", "害怕", "焦虑", "难受", "迷茫", "链接不起来", "消失"]
COMPLETION_WORDS = ["完成", "已经", "成功", "跑通", "保存", "提交", "测出来", "生成", "产生事件"]
URGENT_WORDS = ["今天", "现在", "今晚", "明天", "最近", "截止", "提交", "答辩", "马上"]


@dataclass
class SearchDoc:
    doc_id: str
    path: str
    title: str
    day: str | None
    text: str
    links: list[str]
    tags: list[str]
    tokens: Counter[str]
    action_hits: int = 0
    blocker_hits: int = 0
    completion_hits: int = 0
    urgency_hits: int = 0

    def preview(self, max_chars: int = 900) -> dict[str, object]:
        return {
            "doc_id": self.doc_id,
            "path": self.path,
            "title": self.title,
            "day": self.day,
            "links": self.links[:12],
            "tags": self.tags[:12],
            "signals": {
                "action": self.action_hits,
                "blocker": self.blocker_hits,
                "completion": self.completion_hits,
                "urgency": self.urgency_hits,
            },
            "excerpt": short_text(self.text, max_chars),
        }


@dataclass
class RoleRun:
    role: str
    iteration: int
    docs: list[str]
    query_terms: list[str]
    result: dict[str, object]

    def to_packet(self) -> dict[str, object]:
        return {
            "role": self.role,
            "iteration": self.iteration,
            "docs": self.docs,
            "query_terms": self.query_terms,
            "result": self.result,
        }


@dataclass
class MultiPerspectiveGraph:
    nodes: dict[str, dict[str, object]] = field(default_factory=dict)
    edges: list[dict[str, object]] = field(default_factory=list)
    claims: list[dict[str, object]] = field(default_factory=list)
    tasks: list[dict[str, object]] = field(default_factory=list)
    tensions: list[dict[str, object]] = field(default_factory=list)
    role_runs: list[RoleRun] = field(default_factory=list)

    def to_packet(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "nodes": sorted(self.nodes.values(), key=lambda item: str(item.get("id", ""))),
            "edges": self.edges,
            "claims": self.claims,
            "tasks": self.tasks,
            "tensions": self.tensions,
            "role_runs": [run.to_packet() for run in self.role_runs],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use multiple LLM roles to explore an Obsidian vault and merge their JSON outputs into a graph-LIF daily spike."
    )
    parser.add_argument("--vault", type=Path, default=Path("."), help="Obsidian vault path.")
    parser.add_argument("--days", type=int, default=30, help="Only include dated notes from the latest N days. Use 0 to disable date filtering.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--max-notes", type=int, default=400, help="Maximum notes to index after filtering.")
    parser.add_argument("--roles", type=str, default="cartographer,skeptic,executor,linker,lif_scheduler", help="Comma-separated role keys.")
    parser.add_argument("--iterations", type=int, default=1, help="Exploration rounds per role.")
    parser.add_argument("--docs-per-role", type=int, default=8, help="Documents sampled for each role iteration.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible exploration.")
    parser.add_argument("--focus", type=str, default="", help="Optional focus query, for example: 论文闭环, Obsidian知识库, 负阻.")
    parser.add_argument("--theta", type=float, default=6.0, help="LIF spike threshold for merged tasks/topics.")
    parser.add_argument("--output", type=Path, default=Path("LLM-MazeGraph 今日Spike.md"), help="Markdown report output path.")
    parser.add_argument("--json-output", type=Path, default=Path("llm_maze_graph.json"), help="JSON graph output path.")
    parser.add_argument("--skip-llm", action="store_true", help="Build search environment and deterministic graph only; useful for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Print Markdown instead of writing files.")
    llm_adapter.add_cli_args(parser)
    parser.add_argument("--version", action="version", version=f"LLM-Maze-Explorer {VERSION}")
    return parser.parse_args()


def ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def parse_cutoff(value: str | None) -> date:
    if value is None:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def note_date(path: Path) -> date | None:
    match = DATE_RE.match(path.stem)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def relative_path(path: Path, vault: Path) -> str:
    try:
        return str(path.relative_to(vault)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def short_text(text: str, limit: int = 800) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 1] + "…"


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def count_hits(text: str, words: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for word in words if word.lower() in lower)


def tokenize(text: str) -> Counter[str]:
    tokens: Counter[str] = Counter()
    for raw in TOKEN_RE.findall(text):
        token = raw.strip().lower()
        if len(token) < 2:
            continue
        tokens[token] += 1
    return tokens


def extract_title(path: Path, text: str) -> str:
    for line in normalize_text(text).splitlines():
        line = line.strip()
        if line.startswith("#"):
            title = line.lstrip("#").strip()
            if title:
                return title
    return path.stem


def candidate_notes(vault: Path, cutoff: date, days: int, max_notes: int) -> list[Path]:
    paths = [path for path in vault.rglob("*.md") if path.is_file() and not ignored_path(path)]
    filtered: list[Path] = []
    for path in paths:
        day = note_date(path)
        if days > 0 and day is not None:
            if day > cutoff or (cutoff - day).days >= days:
                continue
        filtered.append(path)
    filtered.sort(key=lambda path: (note_date(path) or date.min, path.stat().st_mtime), reverse=True)
    return filtered[:max_notes]


def build_search_environment(vault: Path, cutoff: date, days: int, max_notes: int) -> list[SearchDoc]:
    docs: list[SearchDoc] = []
    for path in candidate_notes(vault, cutoff, days, max_notes):
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        links = sorted(set(link.strip() for link in WIKILINK_RE.findall(raw) if link.strip()))[:30]
        tags = sorted(set(TAG_RE.findall(raw)))[:30]
        day = note_date(path)
        docs.append(
            SearchDoc(
                doc_id="doc:" + relative_path(path, vault),
                path=relative_path(path, vault),
                title=extract_title(path, raw),
                day=day.isoformat() if day else None,
                text=text,
                links=links,
                tags=tags,
                tokens=tokenize(text + " " + path.stem),
                action_hits=count_hits(text, ACTION_WORDS),
                blocker_hits=count_hits(text, BLOCKER_WORDS),
                completion_hits=count_hits(text, COMPLETION_WORDS),
                urgency_hits=count_hits(text, URGENT_WORDS),
            )
        )
    return docs


def global_terms(docs: list[SearchDoc], limit: int = 80) -> list[str]:
    counts: Counter[str] = Counter()
    for doc in docs:
        counts.update(doc.tokens)
    noisy = {"这个", "就是", "然后", "现在", "感觉", "问题", "但是", "因为", "所以", "the", "and", "for", "with"}
    return [term for term, _ in counts.most_common(limit * 2) if term not in noisy][:limit]


def score_doc(doc: SearchDoc, terms: Iterable[str], role_key: str, focus: str) -> float:
    score = 0.0
    all_text = (doc.title + " " + doc.path + " " + doc.text).lower()
    for term in terms:
        score += doc.tokens.get(term.lower(), 0) * 0.4
        if term.lower() in all_text:
            score += 0.8
    if focus and focus.lower() in all_text:
        score += 3.0
    if role_key == "skeptic":
        score += doc.blocker_hits * 0.9
    elif role_key == "executor":
        score += doc.action_hits * 0.9 + doc.urgency_hits * 0.5
    elif role_key == "lif_scheduler":
        score += doc.blocker_hits * 0.7 + doc.action_hits * 0.7 + doc.urgency_hits * 0.5
    elif role_key == "linker":
        score += len(doc.links) * 0.3
    else:
        score += len(doc.links) * 0.2 + len(doc.tags) * 0.1
    return score


def choose_docs_for_role(
    docs: list[SearchDoc],
    role_key: str,
    focus: str,
    iteration: int,
    docs_per_role: int,
    rng: random.Random,
) -> tuple[list[SearchDoc], list[str]]:
    terms = global_terms(docs, 40)
    role_terms = [str(item).lower() for item in DEFAULT_ROLE_SPECS[role_key]["focus"]]
    focus_terms = tokenize(focus).most_common(8)
    query_terms = role_terms + [term for term, _ in focus_terms] + rng.sample(terms, min(8, len(terms)))

    ranked = sorted(docs, key=lambda doc: score_doc(doc, query_terms, role_key, focus), reverse=True)
    deterministic = ranked[: max(2, docs_per_role // 2)]
    remaining = [doc for doc in docs if doc not in deterministic]
    random_count = max(0, docs_per_role - len(deterministic))
    sampled = rng.sample(remaining, min(random_count, len(remaining))) if remaining else []

    if iteration % 2 == 1:
        # Odd rounds bias toward weak or less linked notes to force frontier exploration.
        sparse = sorted(remaining, key=lambda doc: (len(doc.links), doc.blocker_hits + doc.action_hits), reverse=False)
        sampled = sparse[:random_count]

    return (deterministic + sampled)[:docs_per_role], query_terms


def role_prompt(role_key: str, selected_docs: list[SearchDoc], query_terms: list[str], focus: str) -> list[dict[str, str]]:
    role = DEFAULT_ROLE_SPECS[role_key]
    docs_payload = [doc.preview() for doc in selected_docs]
    system = (
        "你是 LIF-Memory / MazeGraph 系统中的一个探索型智能体。"
        "你不是闲聊助手。你的任务是把 Obsidian 知识库片段还原成结构化、可合并的 JSON。"
        "不要输出 Markdown，不要输出解释性散文，只输出 JSON object。"
    )
    user = {
        "role": role["name"],
        "mission": role["mission"],
        "focus_query": focus,
        "query_terms": query_terms,
        "docs": docs_payload,
        "required_schema": {
            "role": "string",
            "summary": "string; no more than 120 Chinese characters",
            "nodes": [
                {
                    "id": "stable id, such as concept:论文闭环 or task:整理证据链",
                    "label": "human-readable label",
                    "kind": "concept|claim|task|risk|evidence|question",
                    "confidence": "0..1",
                    "source_docs": ["doc ids or paths"],
                }
            ],
            "edges": [
                {
                    "source": "node id",
                    "target": "node id",
                    "relation": "supports|blocks|requires|contradicts|refines|same_as|suggests|evidence_for",
                    "confidence": "0..1",
                    "source_docs": ["doc ids or paths"],
                }
            ],
            "claims": [
                {
                    "claim": "string",
                    "status": "supported|weak|contradicted|unknown",
                    "evidence": ["short evidence strings"],
                    "missing_evidence": ["what is still needed"],
                    "source_docs": ["doc ids or paths"],
                }
            ],
            "tasks": [
                {
                    "task": "one executable task",
                    "topic": "string",
                    "priority": "P0|P1|P2",
                    "input": ["needed inputs"],
                    "output": "completion output",
                    "completion_standard": "how to judge done",
                    "source_docs": ["doc ids or paths"],
                }
            ],
            "tensions": [
                {
                    "topic": "string",
                    "description": "unresolved tension or blocker",
                    "severity": "0..1",
                    "source_docs": ["doc ids or paths"],
                }
            ],
            "lif_votes": [
                {
                    "topic": "string",
                    "urgency": "0..1",
                    "blocker": "0..1",
                    "actionability": "0..1",
                    "evidence_strength": "0..1",
                    "recommended_spike": "boolean",
                    "reason": "short reason",
                }
            ],
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def extract_json_object(text: str) -> dict[str, object]:
    return llm_adapter.extract_json_object(text)


def call_role_llm(config: llm_adapter.LLMConfig, role_key: str, docs: list[SearchDoc], query_terms: list[str], focus: str) -> dict[str, object]:
    content = llm_adapter.call_chat_completions(config, role_prompt(role_key, docs, query_terms, focus))
    data = extract_json_object(content)
    data.setdefault("role", role_key)
    return data


def deterministic_role_result(role_key: str, selected_docs: list[SearchDoc], query_terms: list[str], focus: str) -> dict[str, object]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    tensions: list[dict[str, object]] = []
    claims: list[dict[str, object]] = []
    topic_counter: Counter[str] = Counter()

    for doc in selected_docs:
        topic = doc.title[:32]
        topic_id = "concept:" + topic.replace(" ", "_")
        nodes.append({"id": topic_id, "label": topic, "kind": "concept", "confidence": 0.45, "source_docs": [doc.path]})
        topic_counter[topic] += 1
        for link in doc.links[:5]:
            link_id = "concept:" + link.replace(" ", "_")
            nodes.append({"id": link_id, "label": link, "kind": "concept", "confidence": 0.35, "source_docs": [doc.path]})
            edges.append({"source": topic_id, "target": link_id, "relation": "suggests", "confidence": 0.35, "source_docs": [doc.path]})
        if doc.blocker_hits:
            tensions.append({"topic": topic, "description": "规则层检测到阻塞/不确定信号。", "severity": min(1.0, doc.blocker_hits * 0.25), "source_docs": [doc.path]})
        if doc.action_hits:
            tasks.append(
                {
                    "task": f"把《{topic}》压成一张观察卡并连接到已有图谱。",
                    "topic": topic,
                    "priority": "P1",
                    "input": [doc.path],
                    "output": "一张 observation JSON/card",
                    "completion_standard": "包含事实、判断、链接、下一步。",
                    "source_docs": [doc.path],
                }
            )

    top_topic = topic_counter.most_common(1)[0][0] if topic_counter else focus or role_key
    claims.append(
        {
            "claim": f"{role_key} 视角认为当前应围绕 {top_topic} 建立图谱连接。",
            "status": "weak",
            "evidence": [doc.path for doc in selected_docs[:3]],
            "missing_evidence": ["需要 LLM 深度审查以避免规则误判。"],
            "source_docs": [doc.path for doc in selected_docs[:5]],
        }
    )
    return {
        "role": role_key,
        "summary": "规则回退模式：仅基于关键词、链接和行动/阻塞信号生成粗图谱。",
        "nodes": nodes,
        "edges": edges,
        "claims": claims,
        "tasks": tasks,
        "tensions": tensions,
        "lif_votes": [],
    }


def stable_node_id(raw: object, fallback_prefix: str, fallback_label: str) -> str:
    if isinstance(raw, str) and raw.strip():
        value = raw.strip()
    else:
        value = f"{fallback_prefix}:{fallback_label}"
    value = re.sub(r"\s+", "_", value)
    if ":" not in value:
        value = f"{fallback_prefix}:{value}"
    return value[:120]


def list_of_dicts(value: object) -> list[dict[str, object]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def merge_role_result(graph: MultiPerspectiveGraph, role: str, iteration: int, docs: list[SearchDoc], query_terms: list[str], result: dict[str, object]) -> None:
    graph.role_runs.append(RoleRun(role=role, iteration=iteration, docs=[doc.path for doc in docs], query_terms=query_terms, result=result))

    for node in list_of_dicts(result.get("nodes")):
        label = str(node.get("label") or node.get("id") or "unknown").strip()[:80]
        node_id = stable_node_id(node.get("id"), "concept", label)
        existing = graph.nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": label,
                "kind": str(node.get("kind") or "concept"),
                "confidence_sum": 0.0,
                "role_votes": [],
                "source_docs": [],
            },
        )
        confidence = safe_float(node.get("confidence"), 0.5)
        existing["confidence_sum"] = safe_float(existing.get("confidence_sum"), 0.0) + confidence
        role_votes = existing.setdefault("role_votes", [])
        if isinstance(role_votes, list):
            role_votes.append({"role": role, "confidence": confidence})
        source_docs = existing.setdefault("source_docs", [])
        if isinstance(source_docs, list):
            for source in as_str_list(node.get("source_docs")):
                if source not in source_docs:
                    source_docs.append(source)

    for edge in list_of_dicts(result.get("edges")):
        source = stable_node_id(edge.get("source"), "concept", "source")
        target = stable_node_id(edge.get("target"), "concept", "target")
        graph.edges.append(
            {
                "source": source,
                "target": target,
                "relation": str(edge.get("relation") or "suggests"),
                "confidence": safe_float(edge.get("confidence"), 0.5),
                "role": role,
                "source_docs": as_str_list(edge.get("source_docs")),
            }
        )

    for claim in list_of_dicts(result.get("claims")):
        claim = dict(claim)
        claim["role"] = role
        graph.claims.append(claim)

    for task in list_of_dicts(result.get("tasks")):
        task = dict(task)
        task["role"] = role
        graph.tasks.append(task)

    for tension in list_of_dicts(result.get("tensions")):
        tension = dict(tension)
        tension["role"] = role
        graph.tensions.append(tension)

    for vote in list_of_dicts(result.get("lif_votes")):
        topic = str(vote.get("topic") or "unknown").strip()
        if not topic:
            continue
        node_id = stable_node_id("concept:" + topic, "concept", topic)
        existing = graph.nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": topic,
                "kind": "concept",
                "confidence_sum": 0.0,
                "role_votes": [],
                "source_docs": [],
            },
        )
        lif_votes = existing.setdefault("lif_votes", [])
        if isinstance(lif_votes, list):
            vote = dict(vote)
            vote["role"] = role
            lif_votes.append(vote)


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def compute_lif_spikes(graph: MultiPerspectiveGraph, theta: float) -> list[dict[str, object]]:
    task_by_topic: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    tension_by_topic: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    claim_by_topic: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for task in graph.tasks:
        topic = str(task.get("topic") or task.get("task") or "unknown")
        task_by_topic[topic].append(task)
    for tension in graph.tensions:
        topic = str(tension.get("topic") or "unknown")
        tension_by_topic[topic].append(tension)
    for claim in graph.claims:
        text = str(claim.get("claim") or "unknown")[:40]
        claim_by_topic[text].append(claim)

    candidates: dict[str, dict[str, object]] = {}

    for node in graph.nodes.values():
        label = str(node.get("label") or node.get("id"))
        votes = node.get("lif_votes") if isinstance(node.get("lif_votes"), list) else []
        role_votes = node.get("role_votes") if isinstance(node.get("role_votes"), list) else []
        confidence_sum = safe_float(node.get("confidence_sum"), 0.0)
        urgency = max([safe_float(vote.get("urgency"), 0.0) for vote in votes], default=0.0)
        blocker = max([safe_float(vote.get("blocker"), 0.0) for vote in votes], default=0.0)
        actionability = max([safe_float(vote.get("actionability"), 0.0) for vote in votes], default=0.0)
        evidence_strength = max([safe_float(vote.get("evidence_strength"), 0.0) for vote in votes], default=0.0)
        vote_bonus = sum(1 for vote in votes if vote.get("recommended_spike") is True) * 1.2

        voltage = (
            confidence_sum * 0.9
            + len(role_votes) * 0.6
            + urgency * 1.4
            + blocker * 1.7
            + actionability * 1.5
            + evidence_strength * 0.8
            + vote_bonus
        )
        candidates[label] = {
            "topic": label,
            "voltage": round(voltage, 3),
            "theta": theta,
            "spike": voltage >= theta,
            "urgency": round(urgency, 3),
            "blocker": round(blocker, 3),
            "actionability": round(actionability, 3),
            "evidence_strength": round(evidence_strength, 3),
            "role_count": len(role_votes),
            "source_docs": node.get("source_docs", [])[:8] if isinstance(node.get("source_docs"), list) else [],
            "tasks": task_by_topic.get(label, [])[:3],
            "tensions": tension_by_topic.get(label, [])[:3],
            "reason": "merged role votes + LIF factors",
        }

    for topic, tasks in task_by_topic.items():
        existing = candidates.setdefault(
            topic,
            {
                "topic": topic,
                "voltage": 0.0,
                "theta": theta,
                "spike": False,
                "urgency": 0.0,
                "blocker": 0.0,
                "actionability": 0.0,
                "evidence_strength": 0.0,
                "role_count": 0,
                "source_docs": [],
                "tasks": [],
                "tensions": [],
                "reason": "task-only topic",
            },
        )
        priority_bonus = max({"P0": 2.8, "P1": 1.6, "P2": 0.8}.get(str(task.get("priority")), 1.0) for task in tasks)
        existing["voltage"] = round(safe_float(existing.get("voltage"), 0.0) + len(tasks) * 1.0 + priority_bonus, 3)
        existing["actionability"] = max(safe_float(existing.get("actionability"), 0.0), 0.8)
        existing["tasks"] = tasks[:3]
        existing["spike"] = safe_float(existing["voltage"]) >= theta

    for topic, tensions in tension_by_topic.items():
        existing = candidates.setdefault(
            topic,
            {
                "topic": topic,
                "voltage": 0.0,
                "theta": theta,
                "spike": False,
                "urgency": 0.0,
                "blocker": 0.0,
                "actionability": 0.0,
                "evidence_strength": 0.0,
                "role_count": 0,
                "source_docs": [],
                "tasks": [],
                "tensions": [],
                "reason": "tension-only topic",
            },
        )
        severity = max(safe_float(item.get("severity"), 0.5) for item in tensions)
        existing["voltage"] = round(safe_float(existing.get("voltage"), 0.0) + len(tensions) * 0.9 + severity * 1.4, 3)
        existing["blocker"] = max(safe_float(existing.get("blocker"), 0.0), severity)
        existing["tensions"] = tensions[:3]
        existing["spike"] = safe_float(existing["voltage"]) >= theta

    spikes = sorted(candidates.values(), key=lambda item: safe_float(item.get("voltage"), 0.0), reverse=True)
    return spikes


def render_markdown(docs: list[SearchDoc], graph: MultiPerspectiveGraph, spikes: list[dict[str, object]], top_k: int = 8) -> str:
    lines: list[str] = [
        "# LLM-MazeGraph 今日 Spike",
        "",
        f"- Version: `{VERSION}`",
        f"- Indexed docs: {len(docs)}",
        f"- Role runs: {len(graph.role_runs)}",
        f"- Merged nodes: {len(graph.nodes)}",
        f"- Merged edges: {len(graph.edges)}",
        f"- Claims: {len(graph.claims)}",
        f"- Tasks: {len(graph.tasks)}",
        f"- Tensions: {len(graph.tensions)}",
        "",
        "## Pipeline",
        "",
        "```text",
        "Raw notes -> searchable environment -> multi-role LLM exploration -> JSON merge -> multi-perspective graph -> LIF score -> today spike",
        "```",
        "",
        "## Top LIF Spikes",
        "",
        "| Topic | V | θ | Spike | Urgency | Blocker | Actionability | Evidence | Roles |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
    ]

    for spike in spikes[:top_k]:
        lines.append(
            f"| {spike['topic']} | {spike['voltage']} | {spike['theta']} | {spike['spike']} | "
            f"{spike['urgency']} | {spike['blocker']} | {spike['actionability']} | {spike['evidence_strength']} | {spike['role_count']} |"
        )

    lines.extend(["", "## 今日主卡候选", ""])
    for spike in spikes[: min(3, len(spikes))]:
        lines.extend([f"### {spike['topic']}", "", f"- Voltage: `{spike['voltage']}` / theta `{spike['theta']}`"])
        tasks = spike.get("tasks") if isinstance(spike.get("tasks"), list) else []
        tensions = spike.get("tensions") if isinstance(spike.get("tensions"), list) else []
        if tasks:
            best = tasks[0]
            lines.extend(
                [
                    f"- Task: {best.get('task', '')}",
                    f"- Output: {best.get('output', '')}",
                    f"- Done when: {best.get('completion_standard', '')}",
                ]
            )
        if tensions:
            lines.append(f"- Main tension: {tensions[0].get('description', '')}")
        source_docs = spike.get("source_docs") if isinstance(spike.get("source_docs"), list) else []
        if source_docs:
            lines.append(f"- Sources: {', '.join(str(item) for item in source_docs[:5])}")
        lines.append("")

    lines.extend(["## Role Summaries", ""])
    for run in graph.role_runs:
        summary = str(run.result.get("summary", ""))
        lines.append(f"- **{run.role} #{run.iteration}**: {summary}")

    lines.extend(["", "## Recent Tasks From Roles", ""])
    for task in graph.tasks[:12]:
        lines.append(f"- [{task.get('priority', 'P2')}] {task.get('task', '')} → {task.get('output', '')}")

    lines.extend(["", "## Boundary", "", "LLM 负责探索、对齐、提出 JSON；LIF 分数负责选择今日 spike；原始笔记仍然保留完整信息。"])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    vault = args.vault.resolve()
    cutoff = parse_cutoff(args.today)
    rng = random.Random(args.seed)
    docs = build_search_environment(vault, cutoff, args.days, args.max_notes)
    roles = [role.strip() for role in args.roles.split(",") if role.strip()]
    unknown_roles = [role for role in roles if role not in DEFAULT_ROLE_SPECS]
    if unknown_roles:
        raise SystemExit(f"Unknown roles: {', '.join(unknown_roles)}")

    config = None if args.skip_llm else llm_adapter.config_from_args(args)
    graph = MultiPerspectiveGraph()

    for iteration in range(args.iterations):
        for role in roles:
            selected_docs, query_terms = choose_docs_for_role(
                docs=docs,
                role_key=role,
                focus=args.focus,
                iteration=iteration,
                docs_per_role=args.docs_per_role,
                rng=rng,
            )
            if config is None:
                result = deterministic_role_result(role, selected_docs, query_terms, args.focus)
            else:
                try:
                    result = call_role_llm(config, role, selected_docs, query_terms, args.focus)
                except Exception as exc:
                    result = deterministic_role_result(role, selected_docs, query_terms, args.focus)
                    result["llm_error"] = str(exc)
            merge_role_result(graph, role, iteration, selected_docs, query_terms, result)

    spikes = compute_lif_spikes(graph, args.theta)
    packet = graph.to_packet()
    packet["search_environment"] = {
        "doc_count": len(docs),
        "top_terms": global_terms(docs, 40),
    }
    packet["lif_spikes"] = spikes

    markdown = render_markdown(docs, graph, spikes)

    if args.dry_run:
        print(markdown)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
