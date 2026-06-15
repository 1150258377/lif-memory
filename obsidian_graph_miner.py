from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import lif_memory as core


VERSION = "0.1.0"

WIKILINK_RE = re.compile(r"!?\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-/\u4e00-\u9fff]+)")
DAILY_NOTE_RE = re.compile(r"^20\d{2}-\d{2}-\d{2}$")

GENERATED_REPORT_NAMES = {
    "LIF-Memory 回放结果.md",
    "LIF-Memory 洞察整合.md",
    "LIF-Memory 状态版回放.md",
    "Obsidian-LIF 知识图谱报告.md",
}


@dataclass
class NoteNode:
    path: Path
    title: str
    folder: str
    text: str
    links: list[str] = field(default_factory=list)
    resolved_links: list[Path] = field(default_factory=list)
    unresolved_links: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    state_scores: dict[str, float] = field(default_factory=dict)
    top_fragments: dict[str, list[core.EvidenceItem]] = field(default_factory=dict)

    @property
    def is_daily(self) -> bool:
        return DAILY_NOTE_RE.match(self.path.stem) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine an Obsidian vault as a graph and project it into LIF-Memory states.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    parser.add_argument("--output", type=Path, default=Path("Obsidian-LIF 知识图谱报告.md"), help="Markdown report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON graph summary output.")
    parser.add_argument("--topn", type=int, default=12, help="Top notes per section.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional maximum number of markdown files to scan.")
    parser.add_argument("--version", action="version", version=f"Obsidian Graph Miner {VERSION}")
    return parser.parse_args()


def should_scan(path: Path) -> bool:
    if path.name in GENERATED_REPORT_NAMES:
        return False
    if core.ignored_path(path):
        return False
    return path.suffix.lower() == ".md"


def md_files(vault: Path, max_files: int = 0) -> list[Path]:
    files = [path for path in vault.rglob("*.md") if should_scan(path)]
    files.sort(key=lambda item: (len(item.parts), str(item).lower()))
    return files[:max_files] if max_files > 0 else files


def md_link(path: Path, vault: Path) -> str:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def extract_links(text: str) -> list[str]:
    links: list[str] = []
    for match in WIKILINK_RE.finditer(text):
        target = match.group(1).strip()
        if target and target not in links:
            links.append(target)
    return links


def extract_tags(text: str) -> list[str]:
    tags = []
    for match in TAG_RE.finditer(text):
        tag = match.group(1).strip()
        if tag and not tag[0].isdigit() and tag not in tags:
            tags.append(tag)
    return tags


def title_for(path: Path) -> str:
    return path.stem


def folder_for(path: Path, vault: Path) -> str:
    try:
        rel_parent = path.parent.relative_to(vault)
    except ValueError:
        rel_parent = path.parent
    value = str(rel_parent).replace("\\", "/")
    return "." if value in {"", "."} else value


def build_alias_index(paths: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[path.stem].append(path)
        index[path.name].append(path)
        index[str(path.with_suffix("")).replace("\\", "/")].append(path)
    return index


def resolve_link(target: str, alias_index: dict[str, list[Path]]) -> Path | None:
    clean = target.strip().replace("\\", "/")
    candidates = alias_index.get(clean)
    if candidates:
        return sorted(candidates, key=lambda item: len(item.parts))[0]
    stem = Path(clean).stem
    candidates = alias_index.get(stem)
    if candidates:
        return sorted(candidates, key=lambda item: len(item.parts))[0]
    return None


def score_note_states(node: NoteNode) -> None:
    day = core.note_date(node.path) or date.today()
    daily = core.extract_daily_evidence(day, node.path, node.text, core.NEURONS)
    node.state_scores = {name: evidence.evidence for name, evidence in daily.items()}
    node.top_fragments = {name: evidence.items[:3] for name, evidence in daily.items() if evidence.items}


def load_graph(vault: Path, max_files: int = 0) -> dict[Path, NoteNode]:
    paths = md_files(vault, max_files=max_files)
    alias_index = build_alias_index(paths)
    nodes: dict[Path, NoteNode] = {}

    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        links = extract_links(text)
        node = NoteNode(
            path=path,
            title=title_for(path),
            folder=folder_for(path, vault),
            text=text,
            links=links,
            tags=extract_tags(text),
        )
        for link in links:
            resolved = resolve_link(link, alias_index)
            if resolved is None:
                node.unresolved_links.append(link)
            else:
                node.resolved_links.append(resolved)
        score_note_states(node)
        nodes[path] = node

    return nodes


def inbound_map(nodes: dict[Path, NoteNode]) -> dict[Path, set[Path]]:
    inbound: dict[Path, set[Path]] = defaultdict(set)
    for path, node in nodes.items():
        for target in node.resolved_links:
            inbound[target].add(path)
    return inbound


def top_by(counter: Counter[Path], topn: int) -> list[tuple[Path, int]]:
    return [(path, value) for path, value in counter.most_common(topn) if value > 0]


def state_top_notes(nodes: dict[Path, NoteNode], topn: int) -> dict[str, list[tuple[Path, float]]]:
    result: dict[str, list[tuple[Path, float]]] = {}
    for state in core.NEURONS:
        scored = [(path, node.state_scores.get(state, 0.0)) for path, node in nodes.items()]
        scored = [(path, score) for path, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        result[state] = scored[:topn]
    return result


def folder_state_summary(nodes: dict[Path, NoteNode]) -> dict[str, Counter[str]]:
    summary: dict[str, Counter[str]] = defaultdict(Counter)
    for node in nodes.values():
        for state, score in node.state_scores.items():
            if score > 0:
                summary[node.folder][state] += round(score, 2)
    return summary


def render_note_link(path: Path, vault: Path) -> str:
    return f"[[{md_link(path, vault)}]]"


def render_report(vault: Path, nodes: dict[Path, NoteNode], topn: int) -> str:
    inbound = inbound_map(nodes)
    out_counter = Counter({path: len(node.resolved_links) for path, node in nodes.items()})
    in_counter = Counter({path: len(inbound.get(path, set())) for path in nodes})
    bridge_counter = Counter({path: out_counter[path] + in_counter[path] for path in nodes})
    unresolved = [(path, link) for path, node in nodes.items() for link in node.unresolved_links]
    folders = Counter(node.folder for node in nodes.values())
    tags = Counter(tag for node in nodes.values() for tag in node.tags)
    daily_count = sum(1 for node in nodes.values() if node.is_daily)
    state_notes = state_top_notes(nodes, topn)
    folder_summary = folder_state_summary(nodes)

    lines: list[str] = []
    lines.append("# Obsidian-LIF 知识图谱报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append("")
    lines.append("## 核心判断")
    lines.append("")
    lines.append("Obsidian 不是一堆文本文件，而是一个带边的记忆图。")
    lines.append("")
    lines.append("```text")
    lines.append("日记 = 时间入口")
    lines.append("wikilink = 显式记忆边")
    lines.append("文件夹 = 项目/领域边")
    lines.append("tag = 人工分类边")
    lines.append("共同 LIF 状态 = 隐式语义边")
    lines.append("LIF 电压 = 图上某个目标方向的未释放压力")
    lines.append("```")
    lines.append("")
    lines.append("因此 LIF-Memory 不应该只读当天日记，而应该在触发 spike 后沿着图回查相关证据。")
    lines.append("")

    lines.append("## 全库统计")
    lines.append("")
    lines.append(f"- Markdown 笔记数：{len(nodes)}")
    lines.append(f"- 日记笔记数：{daily_count}")
    lines.append(f"- 显式 wikilink 边数：{sum(len(node.resolved_links) for node in nodes.values())}")
    lines.append(f"- 未解析链接数：{len(unresolved)}")
    lines.append("")

    lines.append("## 文件夹分布")
    lines.append("")
    lines.append("| 文件夹 | 笔记数 | 主导 LIF 状态 |")
    lines.append("|---|---:|---|")
    for folder, count in folders.most_common(topn):
        state_counter = folder_summary.get(folder, Counter())
        top_states = ", ".join(f"{state}:{score:.1f}" for state, score in state_counter.most_common(3)) or "-"
        lines.append(f"| {folder} | {count} | {top_states} |")
    lines.append("")

    lines.append("## 图谱枢纽")
    lines.append("")
    lines.append("### 入链最多")
    lines.append("")
    for path, count in top_by(in_counter, topn):
        lines.append(f"- {render_note_link(path, vault)}：{count} 条反链")
    lines.append("")
    lines.append("### 出链最多")
    lines.append("")
    for path, count in top_by(out_counter, topn):
        lines.append(f"- {render_note_link(path, vault)}：{count} 条外链")
    lines.append("")
    lines.append("### 桥接笔记")
    lines.append("")
    for path, count in top_by(bridge_counter, topn):
        lines.append(f"- {render_note_link(path, vault)}：入链+出链 {count}")
    lines.append("")

    lines.append("## LIF 状态证据池")
    lines.append("")
    for state, scored in state_notes.items():
        lines.append(f"### {state}")
        lines.append("")
        if not scored:
            lines.append("- 暂无明显证据。")
            lines.append("")
            continue
        for path, score in scored:
            node = nodes[path]
            fragments = node.top_fragments.get(state, [])
            reason = fragments[0].snippet if fragments else node.title
            lines.append(f"- {render_note_link(path, vault)}：{score:.2f} ｜ {reason}")
        lines.append("")

    lines.append("## 未解析链接")
    lines.append("")
    if not unresolved:
        lines.append("- 暂无。")
    else:
        for path, link in unresolved[:topn]:
            lines.append(f"- {render_note_link(path, vault)} -> `[[{link}]]`")
    lines.append("")

    lines.append("## 与 LIF-Memory 的结合方式")
    lines.append("")
    lines.append("下一步应该把 LIF-Memory 从线性日记回放升级为图传播：")
    lines.append("")
    lines.append("```text")
    lines.append("日记片段触发 evidence vector")
    lines.append("        ↓")
    lines.append("沿 wikilink / 反链 / 同文件夹 / 同状态证据池 扩展上下文")
    lines.append("        ↓")
    lines.append("把相关笔记的证据作为 secondary current")
    lines.append("        ↓")
    lines.append("LIF 状态电位更新")
    lines.append("        ↓")
    lines.append("spike 事件包同时携带原始片段 + 图谱邻居证据")
    lines.append("```")
    lines.append("")
    lines.append("这会解决一个关键问题：日记里只写了“第四章缺数据”，但真正解释它的证据可能在项目笔记、实验笔记、论文审查笔记和负阻笔记里。")
    lines.append("")

    lines.append("## 建议的 v0.4 方向")
    lines.append("")
    lines.append("1. `primary evidence`：当天日记中的直接证据。")
    lines.append("2. `linked evidence`：日记显式链接到的笔记。")
    lines.append("3. `backlink evidence`：反向指向该笔记的内容。")
    lines.append("4. `folder evidence`：同项目目录下的证据。")
    lines.append("5. `state-neighbor evidence`：同一个 LIF 状态下高分笔记。")
    lines.append("6. `graph_decay`：图距离越远，电流越弱。")
    lines.append("")
    lines.append("公式可以变成：")
    lines.append("")
    lines.append("```text")
    lines.append("V_new = leak(V_old) + primary_current + graph_current - completion_inhibition")
    lines.append("graph_current = Σ neighbor_evidence * edge_weight * graph_decay")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, vault: Path, nodes: dict[Path, NoteNode]) -> None:
    inbound = inbound_map(nodes)
    payload = {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "note_count": len(nodes),
        "nodes": [
            {
                "path": md_link(path, vault),
                "title": node.title,
                "folder": node.folder,
                "is_daily": node.is_daily,
                "links": [md_link(link, vault) for link in node.resolved_links],
                "unresolved_links": node.unresolved_links,
                "inbound_count": len(inbound.get(path, set())),
                "state_scores": {state: round(score, 2) for state, score in node.state_scores.items() if score > 0},
                "tags": node.tags,
            }
            for path, node in nodes.items()
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_output(vault: Path, output: Path) -> Path:
    return output if output.is_absolute() else vault / output


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    nodes = load_graph(vault, max_files=args.max_files)
    report = render_report(vault, nodes, args.topn)
    output = resolve_output(vault, args.output)
    output.write_text(report, encoding="utf-8")
    if args.json_output:
        write_json(resolve_output(vault, args.json_output), vault, nodes)
    print(f"Scanned {len(nodes)} notes.")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
