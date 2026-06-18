from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import lif_memory as core

VERSION = "0.8.0"
WIKILINK_RE = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-/\u4e00-\u9fff]+)")
CLAIM_WORDS = ["结论", "判断", "说明", "证明", "意味着", "不是", "应该", "不能", "可以", "本质", "核心", "主线", "补充"]
GOAL_HINTS = {
    "thesis": ["论文", "证据", "主线", "闭环", "实验", "LIF", "后向散射", "EEG", "负阻", "KS1092"],
    "experiment": ["实验", "测试", "波形", "数据", "设备", "USRP", "KS1092", "负阻", "LIF"],
    "career": ["简历", "求职", "实习", "项目", "大模型", "嵌入式", "GitHub", "岗位"],
    "ai_memory": ["AI", "智能体", "Obsidian", "记忆", "LIF-Memory", "图谱", "agent", "RAG"],
}
GENERATED_NAMES = {"LIF-Memory 回放结果.md", "Obsidian-LIF 知识图谱报告.md", "MazeGraph-LIF 等效迷宫图.md"}
GENERATED_DIRS = {"AI_Compression", ".lif_memory", "outputs", "lif_output"}


@dataclass
class Observation:
    note_id: str
    path: Path
    title: str
    folder: str
    summary: str
    links: list[str] = field(default_factory=list)
    resolved_links: list[Path] = field(default_factory=list)
    unresolved_links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    concepts: list[dict[str, object]] = field(default_factory=list)
    claims: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    state_scores: dict[str, float] = field(default_factory=dict)
    blockers: int = 0
    completions: int = 0


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    weight: float
    evidence: str = ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a MazeGraph layer for Obsidian before LIF attention triggering.")
    p.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    p.add_argument("--output", type=Path, default=Path("MazeGraph-LIF 等效迷宫图.md"), help="Markdown report path.")
    p.add_argument("--json-output", type=Path, default=None, help="Optional JSON graph output.")
    p.add_argument("--observations-dir", type=Path, default=None, help="Optional directory for per-note observation cards.")
    p.add_argument("--goal", choices=["all", *GOAL_HINTS.keys()], default="all", help="Ranking lens for the report.")
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--max-files", type=int, default=0)
    p.add_argument("--version", action="version", version=f"MazeGraph-LIF {VERSION}")
    return p.parse_args()


def md_link(path: Path, vault: Path) -> str:
    try:
        path = path.relative_to(vault)
    except ValueError:
        pass
    return str(path).replace("\\", "/")


def stable_id(prefix: str, value: str) -> str:
    if prefix == "note":
        return "note::" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    safe = re.sub(r"[^\w\-./\u4e00-\u9fff]+", "_", value.strip()).strip("_") or "unknown"
    return f"{prefix}::{safe}"


def should_scan(path: Path) -> bool:
    return path.suffix.lower() == ".md" and path.name not in GENERATED_NAMES and not any(p in GENERATED_DIRS for p in path.parts) and not core.ignored_path(path)


def markdown_files(vault: Path, max_files: int) -> list[Path]:
    files = [p for p in vault.rglob("*.md") if should_scan(p)]
    files.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return files[:max_files] if max_files > 0 else files


def extract_links(text: str) -> list[str]:
    seen: list[str] = []
    for m in WIKILINK_RE.finditer(text):
        target = m.group(1).strip()
        if target and target not in seen:
            seen.append(target)
    return seen


def extract_tags(text: str) -> list[str]:
    tags: list[str] = []
    for m in TAG_RE.finditer(text):
        tag = m.group(1).strip()
        if tag and not tag[0].isdigit() and tag not in tags:
            tags.append(tag)
    return tags


def alias_index(paths: list[Path], vault: Path) -> dict[str, list[Path]]:
    idx: dict[str, list[Path]] = defaultdict(list)
    for p in paths:
        rel = md_link(p, vault)
        for key in {p.stem, p.name, rel, str(Path(rel).with_suffix(""))}:
            idx[key].append(p)
    return idx


def resolve_link(target: str, idx: dict[str, list[Path]]) -> Path | None:
    clean = target.replace("\\", "/").strip()
    for key in (clean, str(Path(clean).with_suffix("")), Path(clean).stem):
        if idx.get(key):
            return sorted(idx[key], key=lambda p: len(p.parts))[0]
    return None


def units(text: str) -> list[str]:
    text = core.normalize_text(text)
    parts = re.split(r"(?<=[。！？!?])\s+|\n+", text)
    return [re.sub(r"\s+", " ", p).strip(" -\t") for p in parts if len(p.strip()) >= 8]


def trim(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def folder_for(path: Path, vault: Path) -> str:
    try:
        folder = str(path.parent.relative_to(vault)).replace("\\", "/")
    except ValueError:
        folder = str(path.parent).replace("\\", "/")
    return "." if folder in {"", "."} else folder


def concepts_for(text: str, tags: list[str], folder: str, state_scores: dict[str, float]) -> list[dict[str, object]]:
    concepts: list[dict[str, object]] = []
    seen: set[str] = set()
    for topic, words in core.TOPIC_RULES.items():
        hits = core.matched_words(text, words)
        if hits:
            cid = stable_id("topic", topic)
            concepts.append({"id": cid, "label": topic, "source": "topic_rule", "score": round(min(1.0, 0.25 + 0.18 * len(hits)), 3), "hits": hits[:8]})
            seen.add(cid)
    for state, score in state_scores.items():
        cid = stable_id("state", state)
        if cid not in seen and score > 0:
            concepts.append({"id": cid, "label": state, "source": "lif_state", "score": round(min(1.0, score / max(core.NEURONS[state].evidence_cap, 1.0)), 3), "hits": [state]})
            seen.add(cid)
    for tag in tags:
        cid = stable_id("tag", tag)
        if cid not in seen:
            concepts.append({"id": cid, "label": tag, "source": "tag", "score": 0.45, "hits": ["#" + tag]})
            seen.add(cid)
    if folder != ".":
        cid = stable_id("folder", folder)
        concepts.append({"id": cid, "label": folder.split("/")[-1], "source": "folder", "score": 0.25, "hits": [folder]})
    concepts.sort(key=lambda c: float(c["score"]), reverse=True)
    return concepts


def state_view(day: date, path: Path, text: str) -> tuple[dict[str, float], int, int]:
    daily = core.extract_daily_evidence(day, path, text, core.NEURONS)
    scores = {name: round(e.evidence, 3) for name, e in daily.items() if e.evidence > 0}
    blockers = sum(1 for e in daily.values() for item in e.items if core.item_has_blocker(item))
    completions = sum(1 for e in daily.values() for item in e.items if core.item_has_completion(item))
    return scores, blockers, completions


def build_observations(vault: Path, max_files: int) -> list[Observation]:
    paths = markdown_files(vault, max_files)
    idx = alias_index(paths, vault)
    observations: list[Observation] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        links = extract_links(text)
        resolved: list[Path] = []
        unresolved: list[str] = []
        for link in links:
            target = resolve_link(link, idx)
            resolved.append(target) if target else unresolved.append(link)
        day = core.note_date(path) or date.today()
        states, blockers, completions = state_view(day, path, text)
        tags = extract_tags(text)
        folder = folder_for(path, vault)
        obs_units = units(text)
        preferred = [u for u in obs_units if core.contains_any(u, core.ACTION_WORDS + core.BLOCKER_WORDS + CLAIM_WORDS)]
        observations.append(Observation(
            note_id=stable_id("note", md_link(path, vault)),
            path=path,
            title=path.stem,
            folder=folder,
            summary=trim((preferred or obs_units or ["空笔记或仅包含无法解析的内容。"])[0], 180),
            links=links,
            resolved_links=resolved,
            unresolved_links=unresolved,
            tags=tags,
            concepts=concepts_for(text, tags, folder, states),
            claims=[trim(u, 180) for u in obs_units if core.contains_any(u, CLAIM_WORDS)][:5],
            actions=[trim(u, 160) for u in obs_units if core.contains_any(u, core.ACTION_WORDS)][:5],
            state_scores=states,
            blockers=blockers,
            completions=completions,
        ))
    return observations


def build_edges(observations: list[Observation], vault: Path) -> list[Edge]:
    by_path = {o.path: o for o in observations}
    edges: dict[tuple[str, str, str], Edge] = {}

    def add(edge: Edge) -> None:
        key = (edge.source, edge.target, edge.relation)
        old = edges.get(key)
        edges[key] = Edge(edge.source, edge.target, edge.relation, round((old.weight if old else 0) + edge.weight, 3), old.evidence if old else edge.evidence)

    for obs in observations:
        for target_path in obs.resolved_links:
            target = by_path.get(target_path)
            if target:
                add(Edge(obs.note_id, target.note_id, "explicit_link", 1.0, md_link(target_path, vault)))
        for c in obs.concepts:
            add(Edge(obs.note_id, str(c["id"]), "mentions", float(c["score"]), ", ".join(str(x) for x in c.get("hits", []))))
        for i, claim in enumerate(obs.claims):
            claim_id = f"claim::{obs.note_id.split('::', 1)[1]}::{i + 1}"
            add(Edge(obs.note_id, claim_id, "has_claim", 1.0, claim))
            for c in obs.concepts[:5]:
                add(Edge(claim_id, str(c["id"]), "about", 0.5 * float(c["score"]), claim))
        ids = [str(c["id"]) for c in obs.concepts[:8]]
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                add(Edge(min(a, b), max(a, b), "co_occurs", 0.2, obs.title))
    return list(edges.values())


def concept_stats(observations: list[Observation], edges: list[Edge]) -> dict[str, dict[str, object]]:
    degree = Counter()
    for e in edges:
        if not e.source.startswith("note::") and not e.source.startswith("claim::"):
            degree[e.source] += 1
        if not e.target.startswith("note::") and not e.target.startswith("claim::"):
            degree[e.target] += 1
    stats: dict[str, dict[str, object]] = {}
    for obs in observations:
        for c in obs.concepts:
            cid = str(c["id"])
            s = stats.setdefault(cid, {"label": c["label"], "sources": Counter(), "note_count": 0, "score": 0.0, "blockers": 0, "completions": 0, "state_scores": Counter(), "evidence": []})
            s["sources"][str(c["source"])] += 1
            s["note_count"] += 1
            s["score"] += float(c["score"])
            s["blockers"] += obs.blockers
            s["completions"] += obs.completions
            for state, value in obs.state_scores.items():
                s["state_scores"][state] += value
            if len(s["evidence"]) < 6:
                s["evidence"].append(obs.title)
    for cid, s in stats.items():
        pressure = sum(s["state_scores"].values()) / max(int(s["note_count"]), 1)
        tension = max(0.0, pressure + 0.35 * int(s["blockers"]) - 0.28 * int(s["completions"]) + 0.15 * degree[cid])
        s["sources"] = dict(s["sources"])
        s["state_scores"] = dict(s["state_scores"].most_common())
        s["score"] = round(float(s["score"]), 3)
        s["degree"] = int(degree[cid])
        s["lif_tension"] = round(tension, 3)
    return stats


def graph_dict(vault: Path, observations: list[Observation], edges: list[Edge], stats: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_count": len(observations),
        "edge_count": len(edges),
        "concept_count": len(stats),
        "observations": [{"note_id": o.note_id, "path": md_link(o.path, vault), "title": o.title, "summary": o.summary, "links": [md_link(p, vault) for p in o.resolved_links], "unresolved_links": o.unresolved_links, "tags": o.tags, "concepts": o.concepts, "claims": o.claims, "actions": o.actions, "state_scores": o.state_scores, "blockers": o.blockers, "completions": o.completions} for o in observations],
        "edges": [e.__dict__ for e in edges],
        "concepts": stats,
    }


def goal_score(item: tuple[str, dict[str, object]], goal: str) -> float:
    _, s = item
    score = float(s.get("lif_tension", 0.0))
    if goal == "all":
        return score
    text = " ".join([str(s.get("label", "")), *[str(x) for x in s.get("evidence", [])]]).lower()
    score += sum(0.8 for h in GOAL_HINTS[goal] if h.lower() in text)
    states = s.get("state_scores", {})
    if isinstance(states, dict):
        state_name = {"thesis": "Thesis", "experiment": "Experiment", "career": "Career", "ai_memory": "AI_Memory"}[goal]
        score += float(states.get(state_name, 0.0)) * 0.3
    return score


def render_report(vault: Path, observations: list[Observation], edges: list[Edge], stats: dict[str, dict[str, object]], goal: str, topn: int) -> str:
    concepts = sorted(stats.items(), key=lambda it: goal_score(it, goal), reverse=True)[:topn]
    in_deg = Counter(e.target for e in edges)
    out_deg = Counter(e.source for e in edges)
    frontiers = sorted(observations, key=lambda o: sum(o.state_scores.values()) + o.blockers * 0.9 + len(o.unresolved_links) * 0.6 + (2 if in_deg[o.note_id] + out_deg[o.note_id] <= 1 else 0), reverse=True)[:topn]
    lines = [
        "# MazeGraph-LIF 等效迷宫图", "", f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", f"版本：{VERSION}", f"目标视角：`{goal}`", "",
        "## 核心判断", "", "LIF 不直接保存知识库。原始笔记保留完整信息，MazeGraph 保存节点、边、观察和证据来源，LIF 只在图谱状态上触发注意力。", "",
        "```text", "Raw Note -> Observation -> Concept/Claim/Edge -> MazeGraph -> LIF attention -> Task", "```", "",
        "## 全库统计", "", f"- 观察卡数量：{len(observations)}", f"- 概念数量：{len(stats)}", f"- 边数量：{len(edges)}", f"- 未解析链接：{sum(len(o.unresolved_links) for o in observations)}", "",
        "## 当前最值得注意的概念", "", "| 概念 | LIF张力 | 笔记数 | 连接度 | 阻塞/完成 | 主导状态 | 证据示例 |", "|---|---:|---:|---:|---:|---|---|",
    ]
    for cid, s in concepts:
        states = s.get("state_scores", {})
        dominant = ", ".join(f"{k}:{v:.1f}" for k, v in list(states.items())[:3]) if isinstance(states, dict) else "-"
        evidence = "; ".join(str(x) for x in s.get("evidence", [])[:3])
        lines.append(f"| `{cid}` {s.get('label', '')} | {float(s.get('lif_tension', 0.0)):.2f} | {s.get('note_count', 0)} | {s.get('degree', 0)} | {s.get('blockers', 0)}/{s.get('completions', 0)} | {dominant or '-'} | {evidence or '-'} |")
    lines += ["", "## 迷宫前沿：最需要重新接线的笔记", "", "这些笔记通常同时具备高张力、阻塞信号、未解析链接或低连接度，适合作为智能体下一轮主动探索入口。", ""]
    for o in frontiers:
        concepts_text = ", ".join(str(c["label"]) for c in o.concepts[:4]) or "-"
        states = ", ".join(f"{k}:{v:.1f}" for k, v in sorted(o.state_scores.items(), key=lambda it: it[1], reverse=True)[:3]) or "-"
        lines.append(f"- [[{md_link(o.path, vault)}]]｜状态 {states}｜概念 {concepts_text}｜{o.summary}")
    lines += ["", "## 推荐命令", "", "```powershell", 'python .\\maze_graph_builder.py --vault "..\\.." --goal thesis --output "AI_Compression\\MazeGraph-LIF 等效迷宫图.md" --json-output "AI_Compression\\maze_graph.json" --observations-dir "AI_Compression\\observations"', "", 'python .\\lif_memory.py --mode daily --llm-review --top-k 1', "```", "", "## 下一步", "", "1. 让 `lif_memory.py` 在 spike 后读取 `maze_graph.json`，把 linked evidence / backlink evidence 放进 spike packet。", "2. 增加 `graph_current = Σ neighbor_evidence * edge_weight * graph_decay`，把图邻居作为 secondary current。", "3. 增加 `frontier mode`，让智能体优先探索孤立但高张力的笔记。"]
    return "\n".join(lines)


def observation_card(o: Observation, vault: Path) -> str:
    lines = [f"# Observation: {o.title}", "", f"note_id: `{o.note_id}`", f"source: [[{md_link(o.path, vault)}]]", f"folder: `{o.folder}`", "", "## 一句话观察", "", o.summary, "", "## 概念对齐"]
    lines += [f"- `{c['id']}`｜{c['label']}｜{c['source']}｜score={float(c['score']):.2f}" for c in o.concepts[:12]] or ["- 暂无稳定概念命中。"]
    lines += ["", "## 关键判断"] + ([f"- {x}" for x in o.claims] or ["- 暂无明确判断句。"])
    lines += ["", "## 可执行动作"] + ([f"- {x}" for x in o.actions] or ["- 暂无明确动作句。"])
    lines += ["", "## 图谱连接", "", f"- resolved_links: {len(o.resolved_links)}", f"- unresolved_links: {len(o.unresolved_links)}", f"- state_scores: `{json.dumps(o.state_scores, ensure_ascii=False)}`"]
    return "\n".join(lines)


def resolve_output(vault: Path, path: Path) -> Path:
    return path if path.is_absolute() else vault / path


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    observations = build_observations(vault, args.max_files)
    edges = build_edges(observations, vault)
    stats = concept_stats(observations, edges)
    output = resolve_output(vault, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(vault, observations, edges, stats, args.goal, args.topn), encoding="utf-8")
    if args.json_output:
        json_path = resolve_output(vault, args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(graph_dict(vault, observations, edges, stats), ensure_ascii=False, indent=2), encoding="utf-8")
    if args.observations_dir:
        obs_dir = resolve_output(vault, args.observations_dir)
        obs_dir.mkdir(parents=True, exist_ok=True)
        for o in observations:
            name = re.sub(r"[^\w\-\u4e00-\u9fff]+", "_", md_link(o.path, vault)).strip("_")
            (obs_dir / f"{o.note_id.split('::', 1)[1]}_{name}.md").write_text(observation_card(o, vault), encoding="utf-8")
    print(f"Scanned {len(observations)} notes.")
    print(f"Built {len(stats)} concepts and {len(edges)} edges.")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
