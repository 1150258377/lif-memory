from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable


VERSION = "0.8.0"

DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-/\u4e00-\u9fff]+)")

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

ACTION_WORDS = [
    "今天",
    "接下来",
    "下一步",
    "需要",
    "必须",
    "应该",
    "先",
    "测试",
    "修改",
    "写",
    "整理",
    "验证",
    "闭环",
    "投递",
    "跑",
    "生成",
    "压缩",
    "执行",
]

BLOCKER_WORDS = [
    "卡",
    "不知道",
    "缺",
    "没有",
    "不够",
    "失败",
    "问题",
    "害怕",
    "拖延",
    "焦虑",
    "混乱",
    "断开",
    "链接不起来",
    "消失",
    "太大",
    "迷茫",
]

COMPLETION_WORDS = [
    "完成",
    "已经",
    "跑通",
    "成功",
    "保存",
    "提交",
    "测出来",
    "产生事件",
    "生成",
]

CONCEPT_RULES: dict[str, list[str]] = {
    "MazeGraph等效迷宫": ["迷宫", "等效迷宫", "还原", "MazeGraph", "地图", "图谱", "知识图谱"],
    "Obsidian知识库": ["Obsidian", "obsidan", "知识库", "vault", "笔记"],
    "LIF注意力触发": ["LIF", "spike", "脉冲", "电压", "阈值", "触发"],
    "智能体执行": ["智能体", "agent", "AI执行", "执行器", "自动执行", "Codex"],
    "论文闭环": ["论文", "第三章", "第四章", "主线", "证据", "盲审", "答辩"],
    "LIF链路": ["后向散射", "USRP", "EEG", "脑电", "SSVEP", "事件率", "边带", "链路"],
    "负阻实验": ["负阻", "NDR", "斜率", "偏置", "增益", "眼电"],
    "KS1092脑电前端": ["KS1092", "RLD", "LL", "RL", "电极", "额头", "耳垂"],
    "AI求职": ["求职", "简历", "实习", "岗位", "大模型", "投递", "面试"],
    "健康恢复": ["焦虑", "身体", "睡眠", "难受", "害怕", "压力", "崩", "恢复"],
}

CONCEPT_ACTION_TARGETS = {
    "MazeGraph等效迷宫": "生成一份图谱报告：节点、边、孤岛、前沿问题各保留 10 条以内。",
    "Obsidian知识库": "先输出全库索引和主题地图，不要直接把整库交给一个智能体。",
    "LIF注意力触发": "把 LIF 限定为注意力/行动触发器，不再让它承担完整记忆压缩。",
    "智能体执行": "为 top spike 生成一个可执行任务卡，输入、输出、完成标准都要写清楚。",
    "论文闭环": "写出一个论文证据块：一句结论、一张图说明、一个限制条件。",
    "LIF链路": "整理 EEG→LIF→Tag→USRP 的最小链路证据。",
    "负阻实验": "把负阻降级为隔离验证：可用、不可用、暂不作为主线三选一。",
    "KS1092脑电前端": "补齐电极位置、共模/参考、噪声和事件截图的实验记录。",
    "AI求职": "把项目压成一个 AI+嵌入式 简历条目。",
    "健康恢复": "先做一个 10 分钟恢复动作，再只保留一个任务。",
}


@dataclass
class NoteObservation:
    note_id: str
    path: str
    day: str | None
    title: str
    summary: str
    concepts: list[str]
    wikilinks: list[str]
    tags: list[str]
    actions: list[str]
    blockers: list[str]
    completions: list[str]

    def to_packet(self) -> dict[str, object]:
        return {
            "note_id": self.note_id,
            "path": self.path,
            "day": self.day,
            "title": self.title,
            "summary": self.summary,
            "concepts": self.concepts,
            "wikilinks": self.wikilinks,
            "tags": self.tags,
            "actions": self.actions,
            "blockers": self.blockers,
            "completions": self.completions,
        }


@dataclass
class GraphNode:
    node_id: str
    label: str
    kind: str
    weight: float = 0.0
    evidence_count: int = 0
    blocker_count: int = 0
    action_count: int = 0
    completion_count: int = 0
    sources: list[str] = field(default_factory=list)

    def to_packet(self) -> dict[str, object]:
        return {
            "id": self.node_id,
            "label": self.label,
            "kind": self.kind,
            "weight": round(self.weight, 3),
            "evidence_count": self.evidence_count,
            "blocker_count": self.blocker_count,
            "action_count": self.action_count,
            "completion_count": self.completion_count,
            "sources": self.sources[:12],
        }


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0
    evidence: list[str] = field(default_factory=list)

    def to_packet(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "weight": round(self.weight, 3),
            "evidence": self.evidence[:8],
        }


@dataclass
class MazeGraph:
    observations: list[NoteObservation] = field(default_factory=list)
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[tuple[str, str, str], GraphEdge] = field(default_factory=dict)

    def node(self, node_id: str, label: str, kind: str) -> GraphNode:
        current = self.nodes.get(node_id)
        if current is None:
            current = GraphNode(node_id=node_id, label=label, kind=kind)
            self.nodes[node_id] = current
        return current

    def add_edge(self, source: str, target: str, relation: str, evidence: str, weight: float = 1.0) -> None:
        key = (source, target, relation)
        edge = self.edges.get(key)
        if edge is None:
            edge = GraphEdge(source=source, target=target, relation=relation)
            self.edges[key] = edge
        edge.weight += weight
        if evidence not in edge.evidence:
            edge.evidence.append(evidence)

    def to_packet(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "stats": {
                "observations": len(self.observations),
                "nodes": len(self.nodes),
                "edges": len(self.edges),
            },
            "observations": [item.to_packet() for item in self.observations],
            "nodes": [node.to_packet() for node in sorted(self.nodes.values(), key=lambda n: (-n.weight, n.label))],
            "edges": [edge.to_packet() for edge in sorted(self.edges.values(), key=lambda e: (-e.weight, e.source, e.target))],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MazeGraph layer from an Obsidian vault, then run LIF-style attention scheduling on graph nodes."
    )
    parser.add_argument("--vault", type=Path, default=Path("."), help="Obsidian vault path.")
    parser.add_argument("--days", type=int, default=30, help="Only include dated notes from the latest N days. Use 0 to disable date filtering.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--max-notes", type=int, default=300, help="Maximum notes to scan after filtering.")
    parser.add_argument("--theta", type=float, default=5.0, help="Graph-node spike threshold.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of graph spikes/frontier nodes to render.")
    parser.add_argument("--output", type=Path, default=Path("MazeGraph-LIF 报告.md"), help="Markdown report output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON graph output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print Markdown instead of writing output files.")
    parser.add_argument("--version", action="version", version=f"MazeGraph-LIF {VERSION}")
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


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    return text


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def matched_words(text: str, words: Iterable[str]) -> list[str]:
    lower = text.lower()
    hits: list[str] = []
    for word in words:
        if word.lower() in lower and word not in hits:
            hits.append(word)
    return hits


def split_blocks(text: str) -> list[str]:
    text = normalize_text(text)
    raw_blocks = re.split(r"[\n。！？!?；;]+", text)
    blocks = [re.sub(r"\s+", " ", block).strip(" -\t") for block in raw_blocks]
    return [block for block in blocks if len(block) >= 4]


def short_text(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def concept_id(label: str) -> str:
    return "concept:" + re.sub(r"\s+", "_", label.strip())


def note_id_for_path(path: Path, vault: Path) -> str:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = path
    return "note:" + str(rel).replace("\\", "/")


def relative_path(path: Path, vault: Path) -> str:
    try:
        return str(path.relative_to(vault)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def extract_title(path: Path, text: str) -> str:
    for line in normalize_text(text).splitlines():
        clean = line.strip()
        if clean.startswith("#"):
            return clean.lstrip("#").strip() or path.stem
    return path.stem


def extract_concepts(text: str) -> list[str]:
    concepts: list[str] = []
    for concept, words in CONCEPT_RULES.items():
        if contains_any(text, words):
            concepts.append(concept)
    return concepts


def extract_lines_with_words(blocks: list[str], words: Iterable[str], limit: int = 5) -> list[str]:
    lines = [short_text(block, 120) for block in blocks if contains_any(block, words)]
    return lines[:limit]


def observe_note(path: Path, vault: Path) -> NoteObservation:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = normalize_text(raw)
    blocks = split_blocks(text)
    title = extract_title(path, raw)
    concepts = extract_concepts(text)
    links = sorted(set(link.strip() for link in WIKILINK_RE.findall(raw) if link.strip()))[:20]
    tags = sorted(set(TAG_RE.findall(raw)))[:20]
    actions = extract_lines_with_words(blocks, ACTION_WORDS)
    blockers = extract_lines_with_words(blocks, BLOCKER_WORDS)
    completions = extract_lines_with_words(blocks, COMPLETION_WORDS)
    summary_source = next((block for block in blocks if extract_concepts(block)), blocks[0] if blocks else title)

    day = note_date(path)
    return NoteObservation(
        note_id=note_id_for_path(path, vault),
        path=relative_path(path, vault),
        day=day.isoformat() if day else None,
        title=title,
        summary=short_text(summary_source, 160),
        concepts=concepts,
        wikilinks=links,
        tags=tags,
        actions=actions,
        blockers=blockers,
        completions=completions,
    )


def candidate_notes(vault: Path, cutoff: date, days: int, max_notes: int) -> list[Path]:
    paths = [path for path in vault.rglob("*.md") if path.is_file() and not ignored_path(path)]
    filtered: list[Path] = []
    for path in paths:
        day = note_date(path)
        if days > 0 and day is not None:
            if day > cutoff or (cutoff - day).days >= days:
                continue
        filtered.append(path)

    filtered.sort(key=lambda p: (note_date(p) or date.min, p.stat().st_mtime), reverse=True)
    return filtered[:max_notes]


def add_source_once(node: GraphNode, source: str) -> None:
    if source not in node.sources:
        node.sources.append(source)


def build_graph(observations: list[NoteObservation]) -> MazeGraph:
    graph = MazeGraph(observations=observations)

    for obs in observations:
        note_node = graph.node(obs.note_id, obs.title, "note")
        note_node.evidence_count += 1
        note_node.weight += 0.5 + len(obs.concepts) * 0.2
        add_source_once(note_node, obs.path)

        if obs.blockers:
            note_node.blocker_count += len(obs.blockers)
            note_node.weight += 0.4 * len(obs.blockers)
        if obs.actions:
            note_node.action_count += len(obs.actions)
            note_node.weight += 0.35 * len(obs.actions)
        if obs.completions:
            note_node.completion_count += len(obs.completions)
            note_node.weight += 0.2 * len(obs.completions)

        for concept in obs.concepts:
            node = graph.node(concept_id(concept), concept, "concept")
            node.evidence_count += 1
            node.blocker_count += len(obs.blockers)
            node.action_count += len(obs.actions)
            node.completion_count += len(obs.completions)
            node.weight += 1.0 + 0.25 * len(obs.actions) + 0.35 * len(obs.blockers) + 0.15 * len(obs.completions)
            add_source_once(node, obs.path)
            graph.add_edge(obs.note_id, node.node_id, "mentions", obs.path, weight=1.0)

        for left_index, left in enumerate(obs.concepts):
            for right in obs.concepts[left_index + 1 :]:
                graph.add_edge(concept_id(left), concept_id(right), "co_occurs", obs.path, weight=0.8)
                graph.add_edge(concept_id(right), concept_id(left), "co_occurs", obs.path, weight=0.8)

        for link in obs.wikilinks:
            link_node_id = "link:" + link
            link_node = graph.node(link_node_id, link, "wikilink")
            link_node.evidence_count += 1
            link_node.weight += 0.4
            add_source_once(link_node, obs.path)
            graph.add_edge(obs.note_id, link_node_id, "links_to", obs.path, weight=0.7)

    return graph


def incoming_counts(graph: MazeGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for edge in graph.edges.values():
        counts[edge.target] += 1
    return counts


def outgoing_counts(graph: MazeGraph) -> Counter[str]:
    counts: Counter[str] = Counter()
    for edge in graph.edges.values():
        counts[edge.source] += 1
    return counts


def node_voltage(node: GraphNode, in_degree: int, out_degree: int) -> float:
    link_gap = 1.0 if in_degree <= 1 or out_degree <= 1 else 0.0
    completion_damping = min(node.completion_count, 4) * 0.35
    return max(
        0.0,
        node.weight
        + node.evidence_count * 0.55
        + node.blocker_count * 0.75
        + node.action_count * 0.55
        + link_gap
        - completion_damping,
    )


def schedule_graph_spikes(graph: MazeGraph, theta: float, top_k: int) -> list[dict[str, object]]:
    indeg = incoming_counts(graph)
    outdeg = outgoing_counts(graph)
    spikes: list[dict[str, object]] = []

    for node in graph.nodes.values():
        if node.kind != "concept":
            continue
        voltage = node_voltage(node, indeg[node.node_id], outdeg[node.node_id])
        if voltage < theta and len(spikes) >= top_k:
            continue

        neighbors = []
        for edge in graph.edges.values():
            if edge.source == node.node_id and edge.target.startswith("concept:"):
                neighbors.append(edge.target.removeprefix("concept:"))
            elif edge.target == node.node_id and edge.source.startswith("concept:"):
                neighbors.append(edge.source.removeprefix("concept:"))

        spikes.append(
            {
                "topic": node.label,
                "voltage": round(voltage, 3),
                "theta": theta,
                "evidence_count": node.evidence_count,
                "blocker_count": node.blocker_count,
                "action_count": node.action_count,
                "completion_count": node.completion_count,
                "sources": node.sources[:6],
                "neighbors": sorted(set(neighbors))[:8],
                "suggested_action": CONCEPT_ACTION_TARGETS.get(node.label, "把这个主题压成一张观察卡：事实、判断、下一步。"),
                "spike": voltage >= theta,
            }
        )

    spikes.sort(key=lambda item: float(item["voltage"]), reverse=True)
    return spikes[:top_k]


def frontier_nodes(graph: MazeGraph, top_k: int) -> list[dict[str, object]]:
    indeg = incoming_counts(graph)
    outdeg = outgoing_counts(graph)
    frontier: list[dict[str, object]] = []
    for node in graph.nodes.values():
        if node.kind not in {"concept", "wikilink"}:
            continue
        degree = indeg[node.node_id] + outdeg[node.node_id]
        if degree <= 1 or (node.kind == "wikilink" and indeg[node.node_id] == 1):
            frontier.append(
                {
                    "label": node.label,
                    "kind": node.kind,
                    "degree": degree,
                    "sources": node.sources[:3],
                    "reason": "孤岛/弱连接节点，容易造成局部输入链接不起来。",
                }
            )
    frontier.sort(key=lambda item: (int(item["degree"]), str(item["label"])))
    return frontier[:top_k]


def render_markdown(graph: MazeGraph, spikes: list[dict[str, object]], frontier: list[dict[str, object]]) -> str:
    stats = graph.to_packet()["stats"]
    lines: list[str] = [
        "# MazeGraph-LIF 报告",
        "",
        f"- Version: `{VERSION}`",
        f"- Observations: {stats['observations']}",
        f"- Nodes: {stats['nodes']}",
        f"- Edges: {stats['edges']}",
        "",
        "## 核心判断",
        "",
        "这个报告把 Obsidian 先还原成等效迷宫图，再在图节点上做 LIF 式注意力触发。LIF 不直接承担完整记忆压缩，只负责提示哪些节点现在最该被处理。",
        "",
        "## Top Graph Spikes",
        "",
        "| Topic | V | θ | Evidence | Blocker | Action | Completion | Suggested action |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for spike in spikes:
        lines.append(
            f"| {spike['topic']} | {spike['voltage']} | {spike['theta']} | "
            f"{spike['evidence_count']} | {spike['blocker_count']} | {spike['action_count']} | "
            f"{spike['completion_count']} | {spike['suggested_action']} |"
        )

    lines.extend(["", "## Spike Cards", ""])
    for spike in spikes:
        status = "TRIGGERED" if spike["spike"] else "watch"
        lines.extend(
            [
                f"### {spike['topic']} · {status}",
                "",
                f"- Voltage: `{spike['voltage']}` / theta `{spike['theta']}`",
                f"- Sources: {', '.join(str(item) for item in spike['sources']) or '无'}",
                f"- Neighbors: {', '.join(str(item) for item in spike['neighbors']) or '暂无'}",
                f"- Next action: {spike['suggested_action']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Frontier / Weak Links",
            "",
            "| Node | Kind | Degree | Reason | Sources |",
            "|---|---|---:|---|---|",
        ]
    )
    for item in frontier:
        lines.append(
            f"| {item['label']} | {item['kind']} | {item['degree']} | {item['reason']} | {', '.join(item['sources'])} |"
        )

    concept_nodes = [node for node in graph.nodes.values() if node.kind == "concept"]
    concept_nodes.sort(key=lambda node: (-node.weight, node.label))
    lines.extend(["", "## Concept Map Top Nodes", "", "| Concept | Weight | Evidence | Sources |", "|---|---:|---:|---|"])
    for node in concept_nodes[:12]:
        lines.append(f"| {node.label} | {node.weight:.2f} | {node.evidence_count} | {', '.join(node.sources[:4])} |")

    lines.extend(
        [
            "",
            "## 如何接入现有 LIF-Memory",
            "",
            "推荐流程：",
            "",
            "```text",
            "Raw Note -> Observation -> MazeGraph update -> Graph LIF spike -> lif_memory daily task",
            "```",
            "",
            "也就是说，`maze_graph_lif.py` 负责保留路径关系；`lif_memory.py` 继续负责日卡、反馈闭环和行动压力。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    vault = args.vault.resolve()
    cutoff = parse_cutoff(args.today)
    paths = candidate_notes(vault, cutoff, args.days, args.max_notes)
    observations = [observe_note(path, vault) for path in paths]
    graph = build_graph(observations)
    spikes = schedule_graph_spikes(graph, theta=args.theta, top_k=args.top_k)
    frontier = frontier_nodes(graph, top_k=args.top_k)
    markdown = render_markdown(graph, spikes, frontier)

    packet = graph.to_packet()
    packet["spikes"] = spikes
    packet["frontier"] = frontier

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
