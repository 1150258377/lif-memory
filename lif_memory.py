from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

VERSION = "0.2.0"

DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)

ACTION_WORDS = [
    "今天",
    "目标",
    "接下来",
    "下一步",
    "需要",
    "必须",
    "应该",
    "测试",
    "修改",
    "写",
    "投",
    "整理",
    "量化",
    "记录",
    "推进",
    "验证",
    "闭环",
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
    "延毕",
    "焦虑",
    "混乱",
    "撕扯",
    "难受",
    "动不了",
]

COMPLETION_WORDS = [
    "完成",
    "做完",
    "已经",
    "搞定",
    "跑通",
    "出结果",
    "保存",
    "写完",
    "提交",
    "测出来",
]

INCOMPLETE_WORDS = [
    "未完成",
    "没有完成",
    "还没",
    "尚未",
    "缺",
    "不够",
    "没封住",
    "没有数据",
]

TIME_PRESSURE_WORDS = [
    "今天",
    "明天",
    "截止",
    "提交",
    "答辩",
    "最近",
    "现在",
    "当前",
    "下一阶段",
    "晚上",
    "早上",
]


@dataclass(frozen=True)
class NeuronConfig:
    theta: float
    decay: float
    reset_ratio: float
    cooldown_days: int
    evidence_cap: float
    keywords: list[str]
    suggestion: str


@dataclass
class EvidenceItem:
    day: date
    path: Path
    snippet: str
    score: float
    keywords: list[str]
    modifiers: list[str] = field(default_factory=list)

    def to_packet(self, vault: Path) -> dict[str, object]:
        return {
            "note": self.day.isoformat(),
            "path": md_link(self.path, vault),
            "snippet": self.snippet,
            "score": round(self.score, 2),
            "matched_keywords": self.keywords,
            "modifiers": self.modifiers,
        }


@dataclass
class DailyEvidence:
    evidence: float = 0.0
    completion: float = 0.0
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class NeuronState:
    v: float = 0.0
    last_spike_date: date | None = None


@dataclass
class Spike:
    day: date
    neuron: str
    voltage: float
    threshold: float
    evidence_items: list[EvidenceItem]
    suggestion: str


NEURONS: dict[str, NeuronConfig] = {
    "Experiment": NeuronConfig(
        theta=7.5,
        decay=0.82,
        reset_ratio=0.35,
        cooldown_days=1,
        evidence_cap=6.5,
        keywords=[
            "负阻",
            "测试",
            "波形",
            "KS1092",
            "LIF",
            "SSVEP",
            "EEG",
            "事件率",
            "RMS",
            "后向散射",
            "USRP",
            "实验",
            "数据",
            "ADC",
            "前端",
            "阈值",
            "压缩比",
            "恢复",
        ],
        suggestion="接下来 30 分钟只做一个可记录实验动作：测一组关键波形或阈值数据，并把截图/数值写回实验笔记。",
    ),
    "Thesis": NeuronConfig(
        theta=7.0,
        decay=0.84,
        reset_ratio=0.38,
        cooldown_days=1,
        evidence_cap=6.2,
        keywords=[
            "论文",
            "第三章",
            "第四章",
            "盲审",
            "答辩",
            "提交",
            "证据",
            "图",
            "章节",
            "导师",
            "逻辑",
            "写进论文",
            "创新点",
            "摘要",
        ],
        suggestion="打开论文主文档，只处理一个证据块：把现有数据变成一句结论、一张图说明或一个可答辩的限制条件。",
    ),
    "Career": NeuronConfig(
        theta=6.5,
        decay=0.80,
        reset_ratio=0.35,
        cooldown_days=1,
        evidence_cap=5.8,
        keywords=[
            "简历",
            "实习",
            "工作",
            "求职",
            "面试",
            "大模型",
            "国企",
            "央企",
            "应届",
            "岗位",
            "投递",
            "机会",
            "华为",
            "GitHub",
        ],
        suggestion="只推进求职链路里的一个小动作：改一个简历项目条目，或投递/收藏一个和 AI+硬件相关的岗位。",
    ),
    "AI_Memory": NeuronConfig(
        theta=7.2,
        decay=0.83,
        reset_ratio=0.36,
        cooldown_days=1,
        evidence_cap=6.0,
        keywords=[
            "AI",
            "大模型",
            "agent",
            "智能体",
            "记忆",
            "主动",
            "RAG",
            "Obsidian",
            "skill",
            "LIF-Memory",
            "snn",
            "SNN",
            "脉冲",
            "Codex",
        ],
        suggestion="把 AI 记忆想法压成一个最小可验证实验：补一条规则、跑一次回放、记录一次触发是否合理。",
    ),
    "Health": NeuronConfig(
        theta=5.8,
        decay=0.78,
        reset_ratio=0.40,
        cooldown_days=1,
        evidence_cap=5.2,
        keywords=[
            "焦虑",
            "累",
            "崩",
            "睡",
            "身体",
            "情绪",
            "宿舍",
            "运动",
            "恢复",
            "压力",
            "疲惫",
            "害怕",
            "撕扯",
            "难受",
            "色情",
            "吃",
        ],
        suggestion="先做一个恢复动作：离开屏幕 10 分钟，喝水或走动，然后只回到一个最小任务，不重新规划整个人生。",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Obsidian daily notes as LIF memory states.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to the nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=7, help="Number of latest daily notes to replay.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON event packet output path.")
    parser.add_argument("--daily-spike-budget", type=int, default=2, help="Maximum spike cards emitted per replay day.")
    parser.add_argument(
        "--states",
        type=str,
        default=",".join(NEURONS.keys()),
        help="Comma-separated states to replay, for example: Experiment,Thesis,Career.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Markdown report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF-Memory {VERSION}")
    return parser.parse_args()


def vault_root_from_script() -> Path:
    script_path = Path(__file__).resolve()
    search_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents, script_path.parent, *script_path.parents]
    seen: set[Path] = set()
    for candidate in search_roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / ".obsidian").exists():
            return candidate
    return Path.cwd().resolve()


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


def ignored_path(path: Path) -> bool:
    ignored_parts = {".git", ".obsidian", ".trash", "__pycache__", ".venv", "node_modules"}
    return any(part in ignored_parts for part in path.parts)


def find_daily_notes(vault: Path, cutoff: date, days: int) -> list[tuple[date, Path]]:
    if days <= 0:
        return []

    candidates: dict[date, Path] = {}
    for path in vault.rglob("20??-??-??.md"):
        if ignored_path(path):
            continue
        day = note_date(path)
        if day is None or day > cutoff:
            continue
        # Prefer shallower daily notes when duplicate dates exist.
        current = candidates.get(day)
        if current is None or len(path.parts) < len(current.parts):
            candidates[day] = path

    return sorted(candidates.items())[-days:]


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
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


def short_reason(block: str, limit: int = 96) -> str:
    block = re.sub(r"\s+", " ", block).strip()
    if len(block) <= limit:
        return block
    return block[: limit - 1] + "…"


def block_score(block: str, keyword_hits: list[str]) -> tuple[float, list[str], float]:
    score = 0.85 + min(len(keyword_hits), 5) * 0.35
    modifiers: list[str] = []

    if contains_any(block, ACTION_WORDS):
        score += 0.50
        modifiers.append("action")
    if contains_any(block, BLOCKER_WORDS):
        score += 0.65
        modifiers.append("blocker")
    if contains_any(block, TIME_PRESSURE_WORDS):
        score += 0.30
        modifiers.append("time_pressure")

    completion = 0.0
    if contains_any(block, COMPLETION_WORDS) and not contains_any(block, INCOMPLETE_WORDS):
        completion = 0.70
        modifiers.append("completion_inhibition")

    return score, modifiers, completion


def extract_daily_evidence(day: date, path: Path, text: str, active_neurons: dict[str, NeuronConfig]) -> dict[str, DailyEvidence]:
    result = {name: DailyEvidence() for name in active_neurons}

    for block in split_blocks(text):
        for name, config in active_neurons.items():
            keyword_hits = matched_words(block, config.keywords)
            if not keyword_hits:
                continue

            score, modifiers, completion = block_score(block, keyword_hits)
            item = EvidenceItem(
                day=day,
                path=path,
                snippet=short_reason(block),
                score=score,
                keywords=keyword_hits[:8],
                modifiers=modifiers,
            )

            bucket = result[name]
            bucket.evidence += score
            bucket.completion += completion
            bucket.items.append(item)

    for name, bucket in result.items():
        bucket.evidence = min(bucket.evidence, active_neurons[name].evidence_cap)
        bucket.completion = min(bucket.completion, 2.2)
        bucket.items.sort(key=lambda item: item.score, reverse=True)
        bucket.items = bucket.items[:6]

    return result


def can_spike(state: NeuronState, day: date, config: NeuronConfig) -> bool:
    if state.last_spike_date is None:
        return True
    delta = (day - state.last_spike_date).days
    return delta >= config.cooldown_days


def replay(
    notes: list[tuple[date, Path]],
    daily_spike_budget: int,
    active_neurons: dict[str, NeuronConfig],
) -> tuple[list[Spike], list[dict[str, object]], dict[str, NeuronState]]:
    states = {name: NeuronState() for name in active_neurons}
    spikes: list[Spike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = extract_daily_evidence(day, path, text, active_neurons)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day

        row: dict[str, object] = {"date": day.isoformat(), "path": path, "delta_days": delta_days}
        candidates: list[tuple[float, str, NeuronConfig, NeuronState, DailyEvidence]] = []

        for name, config in active_neurons.items():
            state = states[name]
            evidence = daily[name]
            old_v = state.v
            leak_factor = config.decay ** delta_days
            new_v = max(0.0, leak_factor * old_v + evidence.evidence - evidence.completion)
            state.v = new_v

            row[name] = {
                "old_v": old_v,
                "new_v": new_v,
                "input": evidence.evidence,
                "completion": evidence.completion,
                "spike": False,
                "evidence_count": len(evidence.items),
            }

            if new_v >= config.theta and can_spike(state, day, config):
                candidates.append((new_v / config.theta, name, config, state, evidence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, config, state, evidence in candidates[: max(0, daily_spike_budget)]:
            spike = Spike(
                day=day,
                neuron=name,
                voltage=state.v,
                threshold=config.theta,
                evidence_items=evidence.items[:4],
                suggestion=config.suggestion,
            )
            spikes.append(spike)
            state.last_spike_date = day
            state.v = config.theta * config.reset_ratio

            item = row[name]
            assert isinstance(item, dict)
            item["new_v"] = state.v
            item["spike"] = True

        timeline.append(row)

    return spikes, timeline, states


def md_link(path: Path, vault: Path) -> str:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def trigger_reason(spike: Spike) -> str:
    names = {
        "Experiment": "实验闭环证据持续积累，且当前任务具有可行动性。",
        "Thesis": "论文证据、章节逻辑或提交压力持续积累，需要收束成一个可写入的证据块。",
        "Career": "求职与机会信号持续积累，需要把长期焦虑转成一次具体推进。",
        "AI_Memory": "AI 记忆系统相关想法持续积累，需要压成一次可验证实验。",
        "Health": "身体或情绪压力信号持续积累，需要先降低系统负荷再回到最小任务。",
    }
    return names.get(spike.neuron, "状态电位超过阈值，需要回查证据并生成下一步行动。")


def spike_packet(spike: Spike, vault: Path) -> dict[str, object]:
    return {
        "spike_type": spike.neuron,
        "time": spike.day.isoformat(),
        "V": round(spike.voltage, 2),
        "threshold": round(spike.threshold, 2),
        "evidence_notes": [item.to_packet(vault) for item in spike.evidence_items],
        "trigger_reason": trigger_reason(spike),
        "suggested_action": spike.suggestion,
    }


def render_summary(
    lines: list[str],
    active_neurons: dict[str, NeuronConfig],
    spikes: list[Spike],
    states: dict[str, NeuronState],
) -> None:
    spike_counter = Counter(spike.neuron for spike in spikes)

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 状态 | 最终电位 | 阈值 | Spike 数 | 解释 |")
    lines.append("|---|---:|---:|---:|---|")
    for name, config in active_neurons.items():
        final_v = states[name].v
        ratio = final_v / config.theta if config.theta else 0.0
        if ratio >= 0.9:
            hint = "接近触发，需要近期处理"
        elif ratio >= 0.55:
            hint = "有趋势，但还没到必须行动"
        else:
            hint = "当前负荷较低"
        lines.append(f"| {name} | {final_v:.2f} | {config.theta:.2f} | {spike_counter[name]} | {hint} |")
    lines.append("")


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[Spike],
    timeline: list[dict[str, object]],
    states: dict[str, NeuronState],
    active_neurons: dict[str, NeuronConfig],
) -> str:
    lines: list[str] = []
    lines.append("# LIF-Memory 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append(f"状态集合：{', '.join(active_neurons.keys())}")
    lines.append("")

    render_summary(lines, active_neurons, spikes, states)

    lines.append("## 触发卡片")
    lines.append("")

    if not spikes:
        lines.append("本次回放没有状态变量超过阈值。可以降低 theta，或增加更明确的任务/阻塞关键词。")
        lines.append("")

    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Spike {index}: {spike.neuron} / {spike.day.isoformat()}")
        lines.append("")
        lines.append(f"- 电位：{spike.voltage:.2f}")
        lines.append(f"- 阈值：{spike.threshold:.2f}")
        lines.append("- 事件包：")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(spike_packet(spike, vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("- 触发原因：")
        if spike.evidence_items:
            for item in spike.evidence_items:
                lines.append(f"  - [[{md_link(item.path, vault)}]]：{item.snippet}")
        else:
            lines.append("  - 当日日志中出现了相关目标信号。")
        lines.append("- 建议动作：")
        lines.append(f"  - {spike.suggestion}")
        lines.append("")

    lines.append("## 状态轨迹")
    lines.append("")
    header = "| 日期 | 日志 | " + " | ".join(active_neurons.keys()) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(active_neurons)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in timeline:
        path = row["path"]
        assert isinstance(path, Path)
        values = []
        for name in active_neurons:
            item = row[name]
            assert isinstance(item, dict)
            mark = " *" if item.get("spike") else ""
            values.append(
                f"{float(item['new_v']):.2f}{mark} "
                f"<br><sub>in {float(item['input']):.1f}, done {float(item['completion']):.1f}</sub>"
            )
        lines.append(
            f"| {row['date']} | [[{md_link(path, vault)}]] | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    lines.append("说明：带 `*` 的电位表示当天该状态神经元触发过 spike；`in` 是证据输入，`done` 是完成抑制。")
    lines.append("")

    lines.append("## 调参规则")
    lines.append("")
    lines.append("- 误触发太多：提高对应状态的 `theta`，或降低关键词覆盖范围。")
    lines.append("- 总是太晚触发：降低 `theta`，或提高关键证据词权重。")
    lines.append("- 每天重复提醒：增加 `cooldown_days` 或降低 `evidence_cap`。")
    lines.append("- 完成后还一直触发：增加 `COMPLETION_WORDS`，并检查是否被 `INCOMPLETE_WORDS` 抵消。")
    lines.append("")

    lines.append("## 系统边界")
    lines.append("")
    lines.append("LIF-Memory 不尝试从 spike 中恢复完整原始语言。")
    lines.append("")
    lines.append("```text")
    lines.append("原始记忆层：Obsidian 保存完整语言，负责保真、追溯、重新检索")
    lines.append("状态触发层：LIF 电位保存趋势，负责积累、泄漏、抑制、阈值判断")
    lines.append("事件层：spike 携带证据索引，负责唤醒相关原文并生成行动建议")
    lines.append("```")
    lines.append("")
    lines.append("所以本实验验证的不是 `spike -> 恢复原文`，而是 `spike -> 召回证据 -> 触发行动`。")
    lines.append("")

    lines.append("## 人工评价")
    lines.append("")
    lines.append("逐条标注：")
    lines.append("")
    lines.append("```text")
    lines.append("合理 / 太早 / 太晚 / 无用 / 应该触发但没触发")
    lines.append("```")
    lines.append("")
    lines.append("下一轮根据人工评价调整：关键词、theta、decay、completion 抑制、cooldown 和 evidence_cap。")
    lines.append("")
    return "\n".join(lines)


def parse_states(value: str) -> dict[str, NeuronConfig]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in NEURONS]
    if unknown:
        raise SystemExit(f"Unknown states: {', '.join(unknown)}. Available: {', '.join(NEURONS)}")
    return {name: NEURONS[name] for name in requested}


def resolve_output_path(vault: Path, output: Path | None) -> Path | None:
    if output is None:
        return None
    return output if output.is_absolute() else vault / output


def write_json_output(path: Path, vault: Path, spikes: list[Spike]) -> None:
    packets = [spike_packet(spike, vault) for spike in spikes]
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault = (args.vault or vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    active_neurons = parse_states(args.states)
    notes = find_daily_notes(vault, cutoff, args.days)

    spikes, timeline, states = replay(notes, args.daily_spike_budget, active_neurons)
    report = render_markdown(vault, notes, spikes, timeline, states, active_neurons)

    if args.dry_run:
        print(report)
    else:
        output = resolve_output_path(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Replayed {len(notes)} notes.")
        print(f"Generated {len(spikes)} spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_output_path(vault, args.json_output)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
