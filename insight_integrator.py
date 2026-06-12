from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import lif_memory as core

VERSION = "0.4.0"


@dataclass(frozen=True)
class LatentQuestion:
    theta: float
    decay: float
    reset_ratio: float
    cooldown_days: int
    evidence_cap: float
    keywords: list[str]
    conflict_words: list[str]
    completion_words: list[str]
    emergent_insight: str
    next_validation_action: str


@dataclass
class Fragment:
    day: date
    path: Path
    snippet: str
    score: float
    matched_keywords: list[str]
    role: str = "evidence"

    def packet(self, vault: Path) -> dict[str, object]:
        return {
            "note": self.day.isoformat(),
            "path": core.md_link(self.path, vault),
            "snippet": self.snippet,
            "score": round(self.score, 2),
            "role": self.role,
            "matched_keywords": self.matched_keywords,
        }


@dataclass
class DailyIntegration:
    evidence: float = 0.0
    inhibition: float = 0.0
    fragments: list[Fragment] = field(default_factory=list)


@dataclass
class InsightState:
    v: float = 0.0
    last_spike_date: date | None = None
    fragments: list[Fragment] = field(default_factory=list)


@dataclass
class InsightSpike:
    day: date
    question: str
    voltage: float
    threshold: float
    fragments: list[Fragment]
    emergent_insight: str
    next_validation_action: str


DEFAULT_QUESTIONS: dict[str, LatentQuestion] = {
    "Innovation_Claim": LatentQuestion(
        theta=8.0,
        decay=0.86,
        reset_ratio=0.30,
        cooldown_days=2,
        evidence_cap=6.8,
        keywords=[
            "创新点",
            "拼凑",
            "事件",
            "LIF",
            "后向散射",
            "脑机",
            "BCI",
            "EEG",
            "SSVEP",
            "节律",
            "物理层",
            "反射系数",
            "ADC",
            "无线",
            "恢复",
        ],
        conflict_words=["拼凑", "怀疑", "不知道", "不够", "别人做过", "已有", "难道", "没意义"],
        completion_words=["明白", "确认", "定义", "结论", "可以写成", "我知道了"],
        emergent_insight=(
            "创新点不应表述为“LIF + 后向散射”的简单拼接，而应表述为："
            "把 EEG 的任务相关节律先压缩成稀疏事件，再让事件直接调制后向散射链路，"
            "最终证明无线物理层中仍能保留可检测的节律证据。"
        ),
        next_validation_action=(
            "做一张三链对齐图：原始 EEG/SSVEP 节律峰、LIF 事件节律或事件率、"
            "后向散射恢复后的边带能量/PSD 峰，证明三者指向同一个任务节律。"
        ),
    ),
    "Experimental_Closure": LatentQuestion(
        theta=7.6,
        decay=0.84,
        reset_ratio=0.35,
        cooldown_days=1,
        evidence_cap=6.5,
        keywords=[
            "测试",
            "实验",
            "数据",
            "阈值",
            "170",
            "RMS",
            "KS1092",
            "比较器",
            "USRP",
            "波形",
            "压缩比",
            "事件率",
            "sigma",
            "高斯",
            "恢复",
            "闭环",
        ],
        conflict_words=["还没", "缺", "不够", "问题", "不稳", "没封住", "卡"],
        completion_words=["完成", "跑通", "测出来", "保存", "已经", "出结果"],
        emergent_insight=(
            "当前实验的核心不是继续增加模块，而是封住最小闭环证据："
            "输入阈值、事件压缩、节律保留、无线恢复四者要共同指向同一个结论。"
        ),
        next_validation_action=(
            "补一个最小闭环表：输入幅值/前端增益、事件数、事件率、压缩比、"
            "无线恢复节律峰、是否可检测。"
        ),
    ),
    "Thesis_Closure": LatentQuestion(
        theta=7.4,
        decay=0.85,
        reset_ratio=0.34,
        cooldown_days=1,
        evidence_cap=6.2,
        keywords=[
            "论文",
            "第四章",
            "第三章",
            "摘要",
            "创新点",
            "盲审",
            "答辩",
            "证据",
            "图",
            "逻辑",
            "章节",
            "写进论文",
            "导师",
        ],
        conflict_words=["缺", "不够", "不会写", "不知道", "质疑", "换题", "没封住", "混乱"],
        completion_words=["写完", "完成", "定稿", "提交", "整理好了"],
        emergent_insight=(
            "论文不应该按器件和脚本堆砌，而应该按“信息链”组织："
            "EEG 节律输入、LIF 事件化、后向散射承载、接收端节律检测。"
        ),
        next_validation_action=(
            "把第四章重排成一条证据链：每节只回答一个问题，最后落到"
            "“事件化后仍可无线检测任务节律”。"
        ),
    ),
    "Action_Bottleneck": LatentQuestion(
        theta=7.0,
        decay=0.80,
        reset_ratio=0.45,
        cooldown_days=1,
        evidence_cap=5.8,
        keywords=[
            "焦虑",
            "动不了",
            "宿舍",
            "图书馆",
            "运动",
            "吃",
            "睡",
            "简历",
            "投递",
            "毕业",
            "延毕",
            "害怕",
            "压力",
            "接下来",
        ],
        conflict_words=["不知道", "纠结", "害怕", "难受", "失败", "动不了", "混乱"],
        completion_words=["完成", "做完", "已经", "去了", "保存"],
        emergent_insight=(
            "当前瓶颈往往不是缺少计划，而是多个未闭环任务同时占用工作记忆。"
            "系统应该把任务压缩成一个下一步，而不是继续生成更多提醒。"
        ),
        next_validation_action=(
            "只选择一个 30 分钟动作：实验记录、论文证据块、简历投递或身体恢复；"
            "完成后再重新评估。"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Integrate weak note fragments into LIF-style insight spikes.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to integrate.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--questions", type=str, default=None, help="Comma-separated latent questions to run.")
    parser.add_argument("--min-fragments", type=int, default=3, help="Minimum non-completion fragments required for an insight spike.")
    parser.add_argument("--daily-insight-budget", type=int, default=2, help="Maximum insight spikes emitted per day.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Insight 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON insight packet output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF Insight Integrator {VERSION}")
    return parser.parse_args()


def parse_cutoff(value: str | None) -> date:
    return date.today() if value is None else datetime.strptime(value, "%Y-%m-%d").date()


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


def block_to_fragment(day: date, path: Path, block: str, question: LatentQuestion) -> tuple[Fragment | None, float]:
    hits = matched_words(block, question.keywords)
    if not hits:
        return None, 0.0

    role = "evidence"
    score = 0.75 + min(len(hits), 6) * 0.28
    inhibition = 0.0

    if contains_any(block, question.conflict_words):
        score += 0.75
        role = "conflict"
    if contains_any(block, core.ACTION_WORDS):
        score += 0.35
    if contains_any(block, core.TIME_PRESSURE_WORDS):
        score += 0.25
    if contains_any(block, question.completion_words):
        inhibition = 0.85
        if role == "evidence":
            role = "completion"

    return (
        Fragment(
            day=day,
            path=path,
            snippet=core.short_reason(block, limit=110),
            score=score,
            matched_keywords=hits[:8],
            role=role,
        ),
        inhibition,
    )


def extract_daily_integrations(
    day: date,
    path: Path,
    text: str,
    questions: Mapping[str, LatentQuestion],
) -> dict[str, DailyIntegration]:
    result = {name: DailyIntegration() for name in questions}
    for block in core.split_blocks(text):
        for name, question in questions.items():
            fragment, inhibition = block_to_fragment(day, path, block, question)
            if fragment is None:
                continue
            bucket = result[name]
            bucket.evidence += fragment.score
            bucket.inhibition += inhibition
            bucket.fragments.append(fragment)

    for name, bucket in result.items():
        bucket.evidence = min(bucket.evidence, questions[name].evidence_cap)
        bucket.inhibition = min(bucket.inhibition, 2.5)
        bucket.fragments.sort(key=lambda item: item.score, reverse=True)
        bucket.fragments = bucket.fragments[:8]
    return result


def enough_fragments(state: InsightState, min_fragments: int) -> bool:
    non_completion = [item for item in state.fragments if item.role != "completion"]
    distinct_days = {item.day for item in non_completion}
    return len(non_completion) >= min_fragments and len(distinct_days) >= 1


def can_spike(state: InsightState, day: date, question: LatentQuestion, min_fragments: int) -> bool:
    if state.v < question.theta:
        return False
    if not enough_fragments(state, min_fragments):
        return False
    if state.last_spike_date is None:
        return True
    return (day - state.last_spike_date).days >= question.cooldown_days


def select_fragments(fragments: list[Fragment], limit: int = 8) -> list[Fragment]:
    seen: set[str] = set()
    selected: list[Fragment] = []
    ranked = sorted(
        fragments,
        key=lambda item: (item.role == "conflict", item.role == "evidence", item.score, item.day),
        reverse=True,
    )
    for fragment in ranked:
        key = fragment.snippet
        if key in seen:
            continue
        seen.add(key)
        selected.append(fragment)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: item.day)


def replay_insights(
    notes: list[tuple[date, Path]],
    questions: Mapping[str, LatentQuestion],
    daily_insight_budget: int = 2,
    min_fragments: int = 3,
) -> tuple[list[InsightSpike], list[dict[str, object]], dict[str, InsightState]]:
    states = {name: InsightState() for name in questions}
    spikes: list[InsightSpike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = extract_daily_integrations(day, path, text, questions)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day
        row: dict[str, object] = {"date": day.isoformat(), "path": path, "delta_days": delta_days}
        candidates: list[tuple[float, str, LatentQuestion, InsightState]] = []

        for name, question in questions.items():
            state = states[name]
            bucket = daily[name]
            old_v = state.v
            state.v = max(0.0, (question.decay ** delta_days) * old_v + bucket.evidence - bucket.inhibition)
            if bucket.fragments:
                state.fragments.extend(bucket.fragments)
                state.fragments = select_fragments(state.fragments, limit=16)

            row[name] = {
                "old_v": old_v,
                "new_v": state.v,
                "input": bucket.evidence,
                "inhibition": bucket.inhibition,
                "fragments": len(state.fragments),
                "spike": False,
            }

            if can_spike(state, day, question, min_fragments):
                candidates.append((state.v / question.theta, name, question, state))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, question, state in candidates[: max(0, daily_insight_budget)]:
            fragments = select_fragments(state.fragments, limit=8)
            spikes.append(
                InsightSpike(
                    day=day,
                    question=name,
                    voltage=state.v,
                    threshold=question.theta,
                    fragments=fragments,
                    emergent_insight=question.emergent_insight,
                    next_validation_action=question.next_validation_action,
                )
            )
            state.last_spike_date = day
            state.v = question.theta * question.reset_ratio
            state.fragments = state.fragments[-2:]
            item = row[name]
            assert isinstance(item, dict)
            item["new_v"] = state.v
            item["spike"] = True

        timeline.append(row)

    return spikes, timeline, states


def insight_packet(spike: InsightSpike, vault: Path) -> dict[str, object]:
    return {
        "spike_type": "Insight",
        "latent_question": spike.question,
        "time": spike.day.isoformat(),
        "V": round(spike.voltage, 2),
        "threshold": round(spike.threshold, 2),
        "integrated_fragments": [fragment.packet(vault) for fragment in spike.fragments],
        "emergent_insight": spike.emergent_insight,
        "next_validation_action": spike.next_validation_action,
    }


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[InsightSpike],
    timeline: list[dict[str, object]],
    states: Mapping[str, InsightState],
    questions: Mapping[str, LatentQuestion],
) -> str:
    lines: list[str] = []
    lines.append("# LIF Insight Integrator 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")
    lines.append("## 这次验证的不是提醒，而是整合")
    lines.append("")
    lines.append("LIF 的作用不是看到一个关键词就触发，而是把多个分散、弱小、互相拉扯的证据片段积累到阈值，形成一次 insight spike。")
    lines.append("")
    lines.append("```text")
    lines.append("weak fragments -> leaky integration -> threshold crossing -> emergent insight")
    lines.append("```")
    lines.append("")

    lines.append("## 状态汇总")
    lines.append("")
    lines.append("| Latent question | Final V | Threshold | Stored fragments |")
    lines.append("|---|---:|---:|---:|")
    for name, question in questions.items():
        state = states[name]
        lines.append(f"| {name} | {state.v:.2f} | {question.theta:.2f} | {len(state.fragments)} |")
    lines.append("")

    lines.append("## Insight spikes")
    lines.append("")
    if not spikes:
        lines.append("本次没有 insight spike。可能是证据还不够分散/不够重复，或者阈值过高。")
        lines.append("")

    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Insight {index}: {spike.question} / {spike.day.isoformat()}")
        lines.append("")
        lines.append(f"- 电位：{spike.voltage:.2f}")
        lines.append(f"- 阈值：{spike.threshold:.2f}")
        lines.append("- 被整合的碎片：")
        for fragment in spike.fragments:
            lines.append(
                f"  - `{fragment.role}` [[{core.md_link(fragment.path, vault)}]] "
                f"{fragment.day.isoformat()}：{fragment.snippet}"
            )
        lines.append("- 涌现判断：")
        lines.append(f"  - {spike.emergent_insight}")
        lines.append("- 最小验证动作：")
        lines.append(f"  - {spike.next_validation_action}")
        lines.append("- JSON packet:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(insight_packet(spike, vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## 轨迹")
    lines.append("")
    header = "| 日期 | 日志 | " + " | ".join(questions.keys()) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(questions)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in timeline:
        path = row["path"]
        assert isinstance(path, Path)
        values = []
        for name in questions:
            item = row[name]
            assert isinstance(item, dict)
            mark = " *" if item.get("spike") else ""
            values.append(
                f"{float(item['new_v']):.2f}{mark}"
                f"<br><sub>in {float(item['input']):.1f}, inhibit {float(item['inhibition']):.1f}, frag {int(item['fragments'])}</sub>"
            )
        lines.append(f"| {row['date']} | [[{core.md_link(path, vault)}]] | " + " | ".join(values) + " |")
    lines.append("")
    lines.append("说明：`*` 表示当天产生 insight spike；frag 是该 latent question 已整合的片段数。")
    lines.append("")
    return "\n".join(lines)


def select_questions(value: str | None) -> dict[str, LatentQuestion]:
    if value is None or not value.strip():
        return dict(DEFAULT_QUESTIONS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in DEFAULT_QUESTIONS]
    if unknown:
        raise SystemExit(f"Unknown questions: {', '.join(unknown)}. Available: {', '.join(DEFAULT_QUESTIONS)}")
    return {name: DEFAULT_QUESTIONS[name] for name in requested}


def resolve_output(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def write_json_output(path: Path, vault: Path, spikes: list[InsightSpike]) -> None:
    packets = [insight_packet(spike, vault) for spike in spikes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    questions = select_questions(args.questions)
    notes = core.find_daily_notes(vault, cutoff, args.days)
    spikes, timeline, states = replay_insights(
        notes,
        questions,
        daily_insight_budget=args.daily_insight_budget,
        min_fragments=args.min_fragments,
    )
    report = render_markdown(vault, notes, spikes, timeline, states, questions)

    if args.dry_run:
        print(report)
    else:
        output = resolve_output(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Integrated {len(notes)} notes.")
        print(f"Generated {len(spikes)} insight spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_output(vault, args.json_output)
    if json_output is not None:
        write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
