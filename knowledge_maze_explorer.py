from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import llm_adapter
except Exception:  # pragma: no cover - the explorer still works without LLM review.
    llm_adapter = None


VERSION = "0.8.0"

CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")

IGNORED_PARTS = {
    ".git",
    ".obsidian",
    ".trash",
    ".venv",
    "__pycache__",
    "node_modules",
    "examples",
    "tests",
}

ACTION_WORDS = [
    "下一步",
    "接下来",
    "必须",
    "应该",
    "需要",
    "写",
    "整理",
    "验证",
    "测试",
    "补",
    "投递",
    "完成",
    "推进",
]

CONFLICT_WORDS = [
    "卡",
    "失败",
    "不稳",
    "不知道",
    "缺",
    "不够",
    "质疑",
    "害怕",
    "焦虑",
    "延毕",
    "混乱",
    "矛盾",
    "问题",
]

COMPLETION_WORDS = [
    "完成",
    "已经",
    "跑通",
    "成功",
    "出结果",
    "测出来",
    "写完",
    "提交",
    "保存",
    "确认",
]

SPECIFICITY_MARKERS = [
    "Hz",
    "mV",
    "V",
    "MHz",
    "USRP",
    "KS1092",
    "SSVEP",
    "EEG",
    "PSD",
    "ADC",
    "LIF",
    "JSON",
    "GitHub",
]

TOPIC_TERMS: dict[str, list[str]] = {
    "论文闭环": ["论文", "盲审", "答辩", "第三章", "第四章", "摘要", "主线", "证据链", "创新点"],
    "LIF链路": ["LIF", "事件", "spike", "神经元", "阈值", "事件率", "稀疏", "高斯恢复"],
    "后向散射": ["后向散射", "backscatter", "USRP", "915", "边带", "反射系数", "MOS", "无线"],
    "EEG实验": ["EEG", "脑电", "SSVEP", "EO", "EC", "KS1092", "电极", "事件率", "PSD"],
    "负阻": ["负阻", "NDR", "negative resistance", "偏置", "斜率", "抵消", "放大"],
    "AI记忆": ["LIF-Memory", "Obsidian", "知识库", "记忆", "agent", "智能体", "LLM", "多视角", "迷宫"],
    "行动阻塞": ["焦虑", "动不了", "害怕", "羞耻", "延毕", "失败", "拖延", "压力", "难受"],
    "求职转化": ["简历", "求职", "投递", "实习", "岗位", "AI+嵌入式", "项目表达", "面试"],
}

ROLE_TERMS: dict[str, list[str]] = {
    "claim": ["我认为", "本质", "定义", "核心", "应该表述", "主张", "创新点"],
    "evidence": ["数据", "测得", "结果", "波形", "PSD", "事件率", "截图", "实验", "跑通", "成功"],
    "blocker": CONFLICT_WORDS,
    "action": ACTION_WORDS,
    "completion": COMPLETION_WORDS,
    "question": ["为什么", "如何", "怎么办", "是不是", "能不能", "到底", "是否"],
}


@dataclass(frozen=True)
class ExplorationView:
    name: str
    title: str
    goal: str
    seed_terms: list[str]
    preferred_topics: list[str]
    threshold: float
    risk_terms: list[str] = field(default_factory=list)
    output_policy: str = "spike"


DEFAULT_VIEWS: dict[str, ExplorationView] = {
    "thesis_closure": ExplorationView(
        name="thesis_closure",
        title="论文闭环视角",
        goal="从知识库中还原论文主线、已有证据和缺失证据。",
        seed_terms=["论文", "主线", "创新点", "证据", "第四章", "盲审", "答辩", "LIF", "后向散射"],
        preferred_topics=["论文闭环", "LIF链路", "后向散射", "EEG实验"],
        threshold=7.5,
        risk_terms=["缺", "不够", "质疑", "不会写", "拼凑"],
    ),
    "experiment_auditor": ExplorationView(
        name="experiment_auditor",
        title="实验审计视角",
        goal="判断哪些实验结果可靠，哪些只是现象，哪些需要补测。",
        seed_terms=["实验", "数据", "测试", "波形", "阈值", "事件率", "KS1092", "USRP", "PSD", "不稳"],
        preferred_topics=["EEG实验", "LIF链路", "后向散射", "负阻"],
        threshold=7.0,
        risk_terms=["不稳", "失败", "缺", "没有数据", "没测"],
    ),
    "theory_builder": ExplorationView(
        name="theory_builder",
        title="理论建构视角",
        goal="抽取 LIF、事件驱动、后向散射、负阻之间的理论关系。",
        seed_terms=["本质", "定义", "机制", "自组织", "涌现", "事件", "LIF", "负阻", "后向散射"],
        preferred_topics=["LIF链路", "后向散射", "负阻", "AI记忆"],
        threshold=6.8,
        risk_terms=["不理解", "矛盾", "质疑", "已有"],
    ),
    "action_blocker": ExplorationView(
        name="action_blocker",
        title="行动阻塞视角",
        goal="识别导致行动停滞的真实阻塞因素，并压缩成一个最小行动。",
        seed_terms=["焦虑", "害怕", "延毕", "动不了", "羞耻", "失败", "接下来", "今天", "任务"],
        preferred_topics=["行动阻塞", "论文闭环", "EEG实验", "求职转化"],
        threshold=6.5,
        risk_terms=["崩", "难受", "动不了", "害怕", "羞耻"],
        output_policy="recover_or_isolate",
    ),
    "career_transfer": ExplorationView(
        name="career_transfer",
        title="求职转化视角",
        goal="把项目转化成 AI + 嵌入式、信号处理、硬件系统方向的简历表达。",
        seed_terms=["简历", "求职", "投递", "实习", "AI", "嵌入式", "项目", "GitHub", "作品"],
        preferred_topics=["求职转化", "AI记忆", "LIF链路", "EEG实验"],
        threshold=6.2,
        risk_terms=["不会写", "没项目", "没工作", "没有机会"],
    ),
    "reviewer": ExplorationView(
        name="reviewer",
        title="审稿人质疑视角",
        goal="以审稿人视角攻击系统漏洞、证据不足和概念不清处。",
        seed_terms=["质疑", "不够", "创新", "证明", "对比", "消融", "数据", "为什么", "是否"],
        preferred_topics=["论文闭环", "LIF链路", "后向散射", "负阻", "EEG实验"],
        threshold=7.2,
        risk_terms=["不够", "缺", "质疑", "没有", "失败", "不稳", "拼凑"],
        output_policy="criticize",
    ),
}


@dataclass
class NoteBlock:
    path: Path
    index: int
    text: str
    outgoing_links: list[str]
    topics: list[str]
    roles: list[str]
    specificity: float


@dataclass
class EvidenceHit:
    path: str
    block_index: int
    snippet: str
    score: float
    topics: list[str]
    roles: list[str]
    matched_terms: list[str]
    outgoing_links: list[str]

    def packet(self) -> dict[str, object]:
        return {
            "path": self.path,
            "block_index": self.block_index,
            "snippet": self.snippet,
            "score": round(self.score, 3),
            "topics": self.topics,
            "roles": self.roles,
            "matched_terms": self.matched_terms,
            "outgoing_links": self.outgoing_links[:8],
        }


@dataclass
class MazeNode:
    node_id: str
    label: str
    node_type: str
    voltage: float = 0.0
    threshold: float = 0.0
    evidence_count: int = 0
    dominant_role: str = "evidence"
    spike: bool = False

    def packet(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "label": self.label,
            "node_type": self.node_type,
            "voltage": round(self.voltage, 3),
            "threshold": round(self.threshold, 3),
            "evidence_count": self.evidence_count,
            "dominant_role": self.dominant_role,
            "spike": self.spike,
        }


@dataclass
class MazeEdge:
    source: str
    target: str
    relation: str
    weight: float
    evidence_paths: list[str] = field(default_factory=list)

    def packet(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 3),
            "evidence_paths": self.evidence_paths[:6],
        }


@dataclass
class ViewReport:
    view: str
    title: str
    goal: str
    nodes: list[MazeNode]
    edges: list[MazeEdge]
    evidence: list[EvidenceHit]
    claims: list[dict[str, object]]
    spikes: list[dict[str, object]]
    llm_review: dict[str, object] | None = None

    def packet(self) -> dict[str, object]:
        return {
            "view": self.view,
            "title": self.title,
            "goal": self.goal,
            "nodes": [node.packet() for node in self.nodes],
            "edges": [edge.packet() for edge in self.edges],
            "evidence": [hit.packet() for hit in self.evidence],
            "claims": self.claims,
            "spikes": self.spikes,
            "llm_review": self.llm_review,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore an Obsidian vault as a multi-view LIF knowledge maze."
    )
    parser.add_argument("--vault", type=Path, default=Path("."), help="Obsidian vault root.")
    parser.add_argument(
        "--views",
        type=str,
        default="all",
        help="Comma-separated view names, or all. Built-ins: " + ",".join(DEFAULT_VIEWS),
    )
    parser.add_argument("--steps", type=int, default=24, help="Top evidence blocks explored per view.")
    parser.add_argument("--max-files", type=int, default=1200, help="Maximum markdown files to scan.")
    parser.add_argument("--max-blocks-per-file", type=int, default=80, help="Maximum text blocks per file.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 知识迷宫探索报告.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print Markdown instead of writing files.")
    parser.add_argument("--min-score", type=float, default=0.35, help="Minimum block score kept as evidence.")
    parser.add_argument("--version", action="version", version=f"LIF-Memory Knowledge Maze Explorer {VERSION}")
    if llm_adapter is not None:
        llm_adapter.add_cli_args(parser)
    else:
        parser.add_argument("--llm-review", action="store_true", help="Ignored when llm_adapter.py is unavailable.")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = WIKI_LINK_RE.sub(r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


def relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def discover_markdown_files(vault: Path, max_files: int) -> list[Path]:
    files: list[Path] = []
    for path in vault.rglob("*.md"):
        if path.is_file() and not ignored_path(path):
            files.append(path)
            if len(files) >= max_files:
                break
    return sorted(files)


def split_blocks(text: str, max_blocks: int) -> list[str]:
    text = normalize_text(text)
    raw_blocks = re.split(r"\n{2,}|[。！？!?；;]\s*", text)
    blocks = [re.sub(r"\s+", " ", block).strip(" -\t") for block in raw_blocks]
    return [block for block in blocks if len(block) >= 8][:max_blocks]


def matched_words(text: str, words: Iterable[str]) -> list[str]:
    lower = text.lower()
    hits: list[str] = []
    for word in words:
        if word.lower() in lower and word not in hits:
            hits.append(word)
    return hits


def contains_any(text: str, words: Iterable[str]) -> bool:
    return bool(matched_words(text, words))


def extract_links(raw_text: str) -> list[str]:
    links = [match.group(1).strip() for match in WIKI_LINK_RE.finditer(raw_text)]
    deduped: list[str] = []
    for link in links:
        if link and link not in deduped:
            deduped.append(link)
    return deduped


def infer_topics(text: str) -> list[str]:
    topics = [topic for topic, terms in TOPIC_TERMS.items() if contains_any(text, terms)]
    return topics or ["未归类问题"]


def infer_roles(text: str) -> list[str]:
    roles = [role for role, terms in ROLE_TERMS.items() if contains_any(text, terms)]
    return roles or ["evidence"]


def specificity_score(text: str) -> float:
    number_hits = len(re.findall(r"\d+(?:\.\d+)?", text))
    marker_hits = len(matched_words(text, SPECIFICITY_MARKERS))
    symbol_hits = sum(symbol in text for symbol in ["->", "→", "/", "%", "×", "=", ":"])
    return min(1.0, 0.08 * number_hits + 0.12 * marker_hits + 0.08 * symbol_hits)


def build_blocks(vault: Path, max_files: int, max_blocks_per_file: int) -> list[NoteBlock]:
    blocks: list[NoteBlock] = []
    for path in discover_markdown_files(vault, max_files):
        raw_text = safe_read_text(path)
        if not raw_text:
            continue
        links = extract_links(raw_text)
        for index, text in enumerate(split_blocks(raw_text, max_blocks_per_file)):
            blocks.append(
                NoteBlock(
                    path=path,
                    index=index,
                    text=text,
                    outgoing_links=links,
                    topics=infer_topics(text),
                    roles=infer_roles(text),
                    specificity=specificity_score(text),
                )
            )
    return blocks


def score_for_view(block: NoteBlock, view: ExplorationView) -> tuple[float, list[str]]:
    matched_seed = matched_words(block.text, view.seed_terms)
    matched_risk = matched_words(block.text, view.risk_terms)
    preferred_topics = [topic for topic in block.topics if topic in view.preferred_topics]
    role_bonus = 0.0
    if "blocker" in block.roles:
        role_bonus += 0.45
    if "action" in block.roles:
        role_bonus += 0.30
    if "evidence" in block.roles:
        role_bonus += 0.25
    if "question" in block.roles:
        role_bonus += 0.18
    if "completion" in block.roles:
        role_bonus -= 0.12

    score = (
        0.22 * len(matched_seed)
        + 0.34 * len(preferred_topics)
        + 0.18 * len(matched_risk)
        + role_bonus
        + 0.50 * block.specificity
    )
    score *= 1.0 + min(0.45, math.log1p(len(block.text)) / 20.0)
    terms = [*matched_seed, *matched_risk, *preferred_topics]
    return score, list(dict.fromkeys(terms))


def short_snippet(text: str, limit: int = 150) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def edge_relation_for_roles(roles: list[str]) -> str:
    if "blocker" in roles:
        return "blocks"
    if "action" in roles:
        return "suggests_action_for"
    if "question" in roles:
        return "questions"
    if "completion" in roles:
        return "partly_closes"
    return "supports"


def build_edges(evidence: list[EvidenceHit]) -> list[MazeEdge]:
    edge_map: dict[tuple[str, str, str], MazeEdge] = {}
    for hit in evidence:
        topics = hit.topics
        if len(topics) < 2:
            continue
        relation = edge_relation_for_roles(hit.roles)
        for index, source in enumerate(topics):
            for target in topics[index + 1 :]:
                key = (source, target, relation)
                edge = edge_map.get(key)
                if edge is None:
                    edge = MazeEdge(source=source, target=target, relation=relation, weight=0.0)
                    edge_map[key] = edge
                edge.weight += max(0.2, hit.score)
                if hit.path not in edge.evidence_paths:
                    edge.evidence_paths.append(hit.path)
    return sorted(edge_map.values(), key=lambda edge: edge.weight, reverse=True)[:18]


def build_nodes(evidence: list[EvidenceHit], view: ExplorationView) -> list[MazeNode]:
    topic_voltage: defaultdict[str, float] = defaultdict(float)
    topic_count: Counter[str] = Counter()
    topic_roles: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for hit in evidence:
        for topic in hit.topics:
            topic_voltage[topic] += hit.score
            topic_count[topic] += 1
            for role in hit.roles:
                topic_roles[topic][role] += 1

    nodes: list[MazeNode] = []
    for topic, voltage in topic_voltage.items():
        dominant_role = topic_roles[topic].most_common(1)[0][0] if topic_roles[topic] else "evidence"
        adjusted_threshold = view.threshold
        if topic in view.preferred_topics:
            adjusted_threshold *= 0.92
        if dominant_role == "blocker":
            adjusted_threshold *= 0.88
        nodes.append(
            MazeNode(
                node_id=topic,
                label=topic,
                node_type="topic",
                voltage=voltage,
                threshold=adjusted_threshold,
                evidence_count=topic_count[topic],
                dominant_role=dominant_role,
                spike=voltage >= adjusted_threshold,
            )
        )
    return sorted(nodes, key=lambda node: node.voltage, reverse=True)[:12]


def build_claims(nodes: list[MazeNode], evidence: list[EvidenceHit], view: ExplorationView) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    evidence_by_topic: defaultdict[str, list[EvidenceHit]] = defaultdict(list)
    for hit in evidence:
        for topic in hit.topics:
            evidence_by_topic[topic].append(hit)

    for node in nodes[:5]:
        hits = sorted(evidence_by_topic[node.label], key=lambda hit: hit.score, reverse=True)[:3]
        if not hits:
            continue
        if node.dominant_role == "blocker":
            claim_text = f"{node.label} 是当前视角下的高压阻塞点，需要被隔离、降级或补证据。"
        elif node.dominant_role == "action":
            claim_text = f"{node.label} 已经积累到可执行层面，应转成一个最小下一步。"
        else:
            claim_text = f"{node.label} 是当前视角下反复出现的结构性主题。"
        confidence = min(0.95, 0.45 + node.evidence_count * 0.06 + node.voltage / max(view.threshold * 2.5, 1.0))
        claims.append(
            {
                "claim": claim_text,
                "topic": node.label,
                "confidence": round(confidence, 3),
                "supporting_evidence": [hit.packet() for hit in hits],
            }
        )
    return claims


def build_spikes(nodes: list[MazeNode], view: ExplorationView) -> list[dict[str, object]]:
    spikes: list[dict[str, object]] = []
    for node in nodes:
        if not node.spike:
            continue
        if view.output_policy == "criticize":
            action = f"对「{node.label}」写出一条审稿人质疑：缺什么对比、消融或定义。"
        elif view.output_policy == "recover_or_isolate":
            action = f"把「{node.label}」压缩成一个 30 分钟内可完成的小动作，不重新规划全局。"
        else:
            action = f"围绕「{node.label}」生成一张证据卡：一句结论、三条证据、一个下一步。"
        spikes.append(
            {
                "topic": node.label,
                "voltage": round(node.voltage, 3),
                "threshold": round(node.threshold, 3),
                "policy": view.output_policy,
                "suggested_action": action,
            }
        )
    return spikes[:5]


def explore_view(vault: Path, blocks: list[NoteBlock], view: ExplorationView, steps: int, min_score: float) -> ViewReport:
    scored: list[tuple[float, list[str], NoteBlock]] = []
    for block in blocks:
        score, terms = score_for_view(block, view)
        if score >= min_score:
            scored.append((score, terms, block))
    scored.sort(key=lambda item: item[0], reverse=True)

    evidence: list[EvidenceHit] = []
    for score, terms, block in scored[:steps]:
        evidence.append(
            EvidenceHit(
                path=relative_path(block.path, vault),
                block_index=block.index,
                snippet=short_snippet(block.text),
                score=score,
                topics=block.topics,
                roles=block.roles,
                matched_terms=terms,
                outgoing_links=block.outgoing_links,
            )
        )

    nodes = build_nodes(evidence, view)
    edges = build_edges(evidence)
    claims = build_claims(nodes, evidence, view)
    spikes = build_spikes(nodes, view)
    return ViewReport(
        view=view.name,
        title=view.title,
        goal=view.goal,
        nodes=nodes,
        edges=edges,
        evidence=evidence,
        claims=claims,
        spikes=spikes,
    )


def select_views(value: str) -> list[ExplorationView]:
    if value.strip().lower() == "all":
        return list(DEFAULT_VIEWS.values())
    result: list[ExplorationView] = []
    missing: list[str] = []
    for name in [part.strip() for part in value.split(",") if part.strip()]:
        view = DEFAULT_VIEWS.get(name)
        if view is None:
            missing.append(name)
        else:
            result.append(view)
    if missing:
        raise SystemExit(f"Unknown view(s): {', '.join(missing)}. Available: {', '.join(DEFAULT_VIEWS)}")
    return result


def llm_review_prompt(report: ViewReport) -> list[dict[str, str]]:
    compact = {
        "view": report.view,
        "goal": report.goal,
        "nodes": [node.packet() for node in report.nodes[:8]],
        "edges": [edge.packet() for edge in report.edges[:8]],
        "claims": report.claims[:4],
        "spikes": report.spikes,
    }
    system = (
        "你是 LIF-Memory v0.8 的知识迷宫审查器。"
        "你只能根据输入的证据审查，不要编造不存在的文件或实验。"
        "必须只输出 JSON object。"
    )
    user = {
        "task": "Review this multi-view knowledge maze report.",
        "required_schema": {
            "is_coherent": "boolean",
            "strongest_node": "string",
            "main_missing_evidence": "string",
            "merge_hint": "string",
            "risk": "string",
            "next_action": "string",
        },
        "report": compact,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def maybe_review_with_llm(args: argparse.Namespace, reports: list[ViewReport]) -> None:
    if not getattr(args, "llm_review", False) or llm_adapter is None:
        return
    config = llm_adapter.config_from_args(args)
    for report in reports:
        try:
            raw = llm_adapter.call_chat_completions(config, llm_review_prompt(report))
            report.llm_review = llm_adapter.extract_json_object(raw)
        except Exception as exc:  # Keep deterministic output available even when LLM fails.
            report.llm_review = {"error": str(exc)}


def merge_global_maze(reports: list[ViewReport]) -> dict[str, object]:
    node_map: dict[str, MazeNode] = {}
    edge_map: dict[tuple[str, str, str], MazeEdge] = {}
    for report in reports:
        for node in report.nodes:
            current = node_map.get(node.node_id)
            if current is None:
                node_map[node.node_id] = MazeNode(
                    node_id=node.node_id,
                    label=node.label,
                    node_type=node.node_type,
                    voltage=node.voltage,
                    threshold=node.threshold,
                    evidence_count=node.evidence_count,
                    dominant_role=node.dominant_role,
                    spike=node.spike,
                )
            else:
                current.voltage += node.voltage
                current.evidence_count += node.evidence_count
                current.threshold = min(current.threshold, node.threshold)
                current.spike = current.spike or node.spike
        for edge in report.edges:
            key = (edge.source, edge.target, edge.relation)
            current = edge_map.get(key)
            if current is None:
                edge_map[key] = MazeEdge(
                    source=edge.source,
                    target=edge.target,
                    relation=edge.relation,
                    weight=edge.weight,
                    evidence_paths=list(edge.evidence_paths),
                )
            else:
                current.weight += edge.weight
                for path in edge.evidence_paths:
                    if path not in current.evidence_paths:
                        current.evidence_paths.append(path)

    nodes = sorted(node_map.values(), key=lambda node: node.voltage, reverse=True)
    edges = sorted(edge_map.values(), key=lambda edge: edge.weight, reverse=True)
    return {
        "nodes": [node.packet() for node in nodes[:20]],
        "edges": [edge.packet() for edge in edges[:25]],
    }


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def render_markdown(vault: Path, reports: list[ViewReport], global_maze: dict[str, object], args: argparse.Namespace) -> str:
    lines: list[str] = []
    lines.append("# LIF-Memory 知识迷宫探索报告")
    lines.append("")
    lines.append(f"- Version：`{VERSION}`")
    lines.append(f"- Vault：`{vault.resolve()}`")
    lines.append(f"- Generated：`{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Views：`{', '.join(report.view for report in reports)}`")
    lines.append(f"- Steps per view：`{args.steps}`")
    lines.append("")
    lines.append("## 总览")
    summary_rows = []
    for report in reports:
        top_node = report.nodes[0] if report.nodes else None
        summary_rows.append(
            [
                report.title,
                top_node.label if top_node else "-",
                round(top_node.voltage, 2) if top_node else "-",
                len(report.spikes),
                len(report.evidence),
            ]
        )
    lines.append(markdown_table(["视角", "最高电压节点", "V", "spike数", "证据块"], summary_rows))
    lines.append("")
    lines.append("## 全局等效迷宫")
    lines.append("")
    global_nodes = global_maze.get("nodes", [])
    if global_nodes:
        rows = [
            [node["label"], node["voltage"], node["evidence_count"], "yes" if node["spike"] else "no"]
            for node in global_nodes[:10]
        ]
        lines.append(markdown_table(["节点", "综合V", "证据数", "是否spike"], rows))
    else:
        lines.append("未发现可合并节点。")
    lines.append("")

    global_edges = global_maze.get("edges", [])
    if global_edges:
        rows = [
            [edge["source"], edge["relation"], edge["target"], edge["weight"]]
            for edge in global_edges[:10]
        ]
        lines.append(markdown_table(["source", "relation", "target", "weight"], rows))
        lines.append("")

    for report in reports:
        lines.append(f"## {report.title} `{report.view}`")
        lines.append("")
        lines.append(report.goal)
        lines.append("")
        if report.nodes:
            rows = [
                [
                    node.label,
                    round(node.voltage, 2),
                    round(node.threshold, 2),
                    node.evidence_count,
                    node.dominant_role,
                    "yes" if node.spike else "no",
                ]
                for node in report.nodes
            ]
            lines.append(markdown_table(["节点", "V", "阈值", "证据数", "主角色", "spike"], rows))
        else:
            lines.append("没有找到足够强的节点。")
        lines.append("")

        if report.spikes:
            lines.append("### Spike 建议")
            lines.append("")
            for spike in report.spikes:
                lines.append(
                    f"- **{spike['topic']}**：V={spike['voltage']} / θ={spike['threshold']}；{spike['suggested_action']}"
                )
            lines.append("")

        if report.claims:
            lines.append("### 视角结论")
            lines.append("")
            for claim in report.claims:
                lines.append(f"- **{claim['topic']}**：{claim['claim']} 置信度 `{claim['confidence']}`")
            lines.append("")

        if report.edges:
            lines.append("### 关键路径")
            lines.append("")
            rows = [[edge.source, edge.relation, edge.target, round(edge.weight, 2)] for edge in report.edges[:8]]
            lines.append(markdown_table(["source", "relation", "target", "weight"], rows))
            lines.append("")

        if report.evidence:
            lines.append("### Top evidence")
            lines.append("")
            for hit in report.evidence[:8]:
                lines.append(
                    f"- `{hit.path}`#{hit.block_index} | score={round(hit.score, 2)} | "
                    f"topics={','.join(hit.topics)} | roles={','.join(hit.roles)}：{hit.snippet}"
                )
            lines.append("")

        if report.llm_review is not None:
            lines.append("### LLM review")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(report.llm_review, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")

    lines.append("## 推荐运行方式")
    lines.append("")
    lines.append("```powershell")
    lines.append('python "04 项目库\\P2_LIF-Memory\\knowledge_maze_explorer.py" --vault "." --views all --steps 24 --output "LIF-Memory 知识迷宫探索报告.md" --json-output "lif_knowledge_maze.json"')
    lines.append("```")
    lines.append("")
    lines.append("启用 LLM 语义审查：")
    lines.append("")
    lines.append("```powershell")
    lines.append('python "04 项目库\\P2_LIF-Memory\\knowledge_maze_explorer.py" --vault "." --views thesis_closure,experiment_auditor,reviewer --llm-review --llm-provider deepseek')
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    vault = args.vault.resolve()
    blocks = build_blocks(vault, args.max_files, args.max_blocks_per_file)
    views = select_views(args.views)
    reports = [explore_view(vault, blocks, view, args.steps, args.min_score) for view in views]
    maybe_review_with_llm(args, reports)
    global_maze = merge_global_maze(reports)

    packet = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "vault": str(vault),
        "settings": {
            "views": [view.name for view in views],
            "steps": args.steps,
            "max_files": args.max_files,
            "max_blocks_per_file": args.max_blocks_per_file,
            "min_score": args.min_score,
        },
        "global_maze": global_maze,
        "view_reports": [report.packet() for report in reports],
    }

    markdown = render_markdown(vault, reports, global_maze, args)
    if args.dry_run:
        print(markdown)
    else:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote Markdown report: {args.output}")

    if args.json_output:
        args.json_output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON packet: {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
