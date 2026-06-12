from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path


DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")


ACTION_WORDS = [
    "今天",
    "目标",
    "接下来",
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
]

INCOMPLETE_WORDS = [
    "未完成",
    "没有完成",
    "还没",
    "尚未",
    "缺",
    "不够",
    "没封住",
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
]


@dataclass
class NeuronConfig:
    theta: float
    decay: float
    reset_ratio: float
    cooldown_days: int
    keywords: list[str]
    suggestion: str


@dataclass
class DailyEvidence:
    evidence: float = 0.0
    completion: float = 0.0
    reasons: list[str] = field(default_factory=list)


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
    reasons: list[str]
    suggestion: str


NEURONS: dict[str, NeuronConfig] = {
    "Experiment": NeuronConfig(
        theta=7.5,
        decay=0.82,
        reset_ratio=0.35,
        cooldown_days=1,
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
        ],
        suggestion="接下来 30 分钟只做一个可记录实验动作：测一组关键波形或阈值数据，并把截图/数值写回实验笔记。",
    ),
    "Thesis": NeuronConfig(
        theta=7.0,
        decay=0.84,
        reset_ratio=0.38,
        cooldown_days=1,
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
        ],
        suggestion="打开论文主文档，只处理一个证据块：把现有数据变成一句结论、一张图说明或一个可答辩的限制条件。",
    ),
    "Career": NeuronConfig(
        theta=6.5,
        decay=0.80,
        reset_ratio=0.35,
        cooldown_days=1,
        keywords=[
            "简历",
            "实习",
            "工作",
            "求职",
            "面试",
            "大模型",
            "国企",
            "应届",
            "岗位",
            "投递",
            "机会",
        ],
        suggestion="只推进求职链路里的一个小动作：改一个简历项目条目，或投递/收藏一个和 AI+硬件相关的岗位。",
    ),
    "AI_Memory": NeuronConfig(
        theta=7.2,
        decay=0.83,
        reset_ratio=0.36,
        cooldown_days=1,
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
        ],
        suggestion="把 AI 记忆想法压成一个最小可验证实验：补一条规则、跑一次回放、记录一次触发是否合理。",
    ),
    "Health": NeuronConfig(
        theta=5.8,
        decay=0.78,
        reset_ratio=0.40,
        cooldown_days=1,
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
        ],
        suggestion="先做一个恢复动作：离开屏幕 10 分钟，喝水或走动，然后只回到一个最小任务，不重新规划整个人生。",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Obsidian daily notes as LIF memory states.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to this script's vault root.")
    parser.add_argument("--days", type=int, default=7, help="Number of latest daily notes to replay.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--daily-spike-budget", type=int, default=2, help="Maximum spike cards emitted per replay day.")
    return parser.parse_args()


def vault_root_from_script() -> Path:
    script_path = Path(__file__).resolve()
    for candidate in [script_path.parent, *script_path.parents, Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if (candidate / ".obsidian").exists():
            return candidate
    return Path.cwd().resolve()


def note_date(path: Path) -> date | None:
    match = DATE_RE.match(path.stem)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def find_daily_notes(vault: Path, cutoff: date, days: int) -> list[tuple[date, Path]]:
    candidates: dict[date, Path] = {}
    search_dirs = [vault, vault / "06 日志复盘" / str(cutoff.year)]
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        for path in search_dir.glob("20??-??-??.md"):
            day = note_date(path)
            if day and day <= cutoff:
                candidates.setdefault(day, path)
    return sorted(candidates.items())[-days:]


def contains_any(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def count_matches(text: str, words: list[str]) -> int:
    lower = text.lower()
    return sum(1 for word in words if word.lower() in lower)


def split_blocks(text: str) -> list[str]:
    raw_blocks = re.split(r"[\n。！？!?；;]+", text)
    return [block.strip() for block in raw_blocks if block.strip()]


def short_reason(block: str, limit: int = 92) -> str:
    block = re.sub(r"\s+", " ", block).strip()
    if len(block) <= limit:
        return block
    return block[: limit - 1] + "…"


def extract_daily_evidence(text: str) -> dict[str, DailyEvidence]:
    result = {name: DailyEvidence() for name in NEURONS}
    for block in split_blocks(text):
        for name, config in NEURONS.items():
            keyword_hits = count_matches(block, config.keywords)
            if keyword_hits == 0:
                continue

            score = 0.9 + min(keyword_hits, 5) * 0.38
            if contains_any(block, ACTION_WORDS):
                score += 0.60
            if contains_any(block, BLOCKER_WORDS):
                score += 0.75
            if contains_any(block, TIME_PRESSURE_WORDS):
                score += 0.45

            completion = 0.0
            if contains_any(block, COMPLETION_WORDS) and not contains_any(block, INCOMPLETE_WORDS):
                completion = 0.70

            result[name].evidence += score
            result[name].completion += completion
            if len(result[name].reasons) < 6:
                result[name].reasons.append(short_reason(block))

    for name in result:
        result[name].evidence = min(result[name].evidence, 6.5)
        result[name].completion = min(result[name].completion, 2.2)
    return result


def can_spike(state: NeuronState, day: date, config: NeuronConfig) -> bool:
    if state.last_spike_date is None:
        return True
    delta = (day - state.last_spike_date).days
    return delta > config.cooldown_days


def replay(notes: list[tuple[date, Path]], daily_spike_budget: int) -> tuple[list[Spike], list[dict[str, object]]]:
    states = {name: NeuronState() for name in NEURONS}
    spikes: list[Spike] = []
    timeline: list[dict[str, object]] = []

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = extract_daily_evidence(text)
        row: dict[str, object] = {"date": day.isoformat(), "path": path}
        candidates: list[tuple[float, str, NeuronConfig, NeuronState, DailyEvidence]] = []

        for name, config in NEURONS.items():
            state = states[name]
            evidence = daily[name]
            old_v = state.v
            new_v = max(0.0, config.decay * old_v + evidence.evidence - evidence.completion)
            state.v = new_v
            row[name] = {
                "old_v": old_v,
                "new_v": new_v,
                "evidence": evidence.evidence,
                "completion": evidence.completion,
            }

            if new_v >= config.theta and can_spike(state, day, config):
                candidates.append((new_v / config.theta, name, config, state, evidence))
            else:
                row[name]["spike"] = False

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, config, state, evidence in candidates[: max(0, daily_spike_budget)]:
            spikes.append(
                Spike(
                    day=day,
                    neuron=name,
                    voltage=state.v,
                    threshold=config.theta,
                    reasons=evidence.reasons[:4],
                    suggestion=config.suggestion,
                )
            )
            state.last_spike_date = day
            state.v = config.theta * config.reset_ratio
            item = row[name]
            assert isinstance(item, dict)
            item["new_v"] = state.v
            item["spike"] = True

        timeline.append(row)

    return spikes, timeline


def md_link(path: Path, vault: Path) -> str:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def render_markdown(vault: Path, notes: list[tuple[date, Path]], spikes: list[Spike], timeline: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# LIF-Memory 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")
    lines.append("## 触发卡片")
    lines.append("")

    if not spikes:
        lines.append("本次回放没有状态变量超过阈值。可以降低 theta，或增加更明确的任务/阻塞关键词。")
    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Spike {index}: {spike.neuron} / {spike.day.isoformat()}")
        lines.append("")
        lines.append(f"- 电位：{spike.voltage:.2f}")
        lines.append(f"- 阈值：{spike.threshold:.2f}")
        lines.append("- 事件包：")
        event_packet = {
            "spike_type": spike.neuron,
            "time": spike.day.isoformat(),
            "V": round(spike.voltage, 2),
            "threshold": round(spike.threshold, 2),
            "evidence_notes": [
                {
                    "note": spike.day.isoformat(),
                    "snippet": reason,
                }
                for reason in spike.reasons
            ],
            "trigger_reason": trigger_reason(spike),
            "suggested_action": spike.suggestion,
        }
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(event_packet, ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("- 触发原因：")
        if spike.reasons:
            for reason in spike.reasons:
                lines.append(f"  - {reason}")
        else:
            lines.append("  - 当日日志中出现了相关目标信号。")
        lines.append("- 建议动作：")
        lines.append(f"  - {spike.suggestion}")
        lines.append("")

    lines.append("## 状态轨迹")
    lines.append("")
    header = "| 日期 | 日志 | Experiment | Thesis | Career | AI_Memory | Health |"
    sep = "|---|---|---:|---:|---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for row in timeline:
        path = row["path"]
        assert isinstance(path, Path)
        values = []
        for name in NEURONS:
            item = row[name]
            assert isinstance(item, dict)
            mark = " *" if item.get("spike") else ""
            values.append(f"{float(item['new_v']):.2f}{mark}")
        lines.append(
            f"| {row['date']} | [[{md_link(path, vault)}]] | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    lines.append("说明：带 `*` 的电位表示当天该状态神经元触发过 spike，表格中的数值是触发复位后的电位。")
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
    lines.append("下一轮根据人工评价调整：关键词、theta、decay、completion 抑制和 cooldown。")
    lines.append("")
    return "\n".join(lines)


def trigger_reason(spike: Spike) -> str:
    names = {
        "Experiment": "实验闭环证据持续积累，且当前任务具有可行动性。",
        "Thesis": "论文证据、章节逻辑或提交压力持续积累，需要收束成一个可写入的证据块。",
        "Career": "求职与机会信号持续积累，需要把长期焦虑转成一次具体推进。",
        "AI_Memory": "AI 记忆系统相关想法持续积累，需要压成一次可验证实验。",
        "Health": "身体或情绪压力信号持续积累，需要先降低系统负荷再回到最小任务。",
    }
    return names.get(spike.neuron, "状态电位超过阈值，需要回查证据并生成下一步行动。")


def main() -> None:
    args = parse_args()
    vault = (args.vault or vault_root_from_script()).resolve()
    cutoff = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()
    notes = find_daily_notes(vault, cutoff, args.days)

    spikes, timeline = replay(notes, args.daily_spike_budget)
    output = args.output
    if not output.is_absolute():
        output = vault / output
    output.write_text(render_markdown(vault, notes, spikes, timeline), encoding="utf-8")

    print(f"Replayed {len(notes)} notes.")
    print(f"Generated {len(spikes)} spikes.")
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
