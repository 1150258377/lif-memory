from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import lif_memory
import predictive_lif_memory

VERSION = "0.1.0"

NEGATIVE_CONFLICT_EDGES: dict[tuple[str, str], float] = {
    ("Experiment", "Career"): 0.34,
    ("Career", "Experiment"): 0.34,
    ("Experiment", "Health"): 0.42,
    ("Health", "Experiment"): 0.42,
    ("Thesis", "Career"): 0.30,
    ("Career", "Thesis"): 0.30,
    ("Thesis", "Health"): 0.36,
    ("Health", "Thesis"): 0.36,
    ("AI_Memory", "Thesis"): 0.20,
    ("Thesis", "AI_Memory"): 0.20,
}

FIRST_PRINCIPLE_PROMPTS: dict[str, str] = {
    "Experiment": "真正限制实验推进的最小变量是什么：可信数据、仪器链路、参数扫描、还是复现环境？",
    "Thesis": "论文真正缺的不是字数，而是哪一个最小证据块、图表或逻辑闭环？",
    "Career": "求职真正缺的是岗位数量、简历表达、作品证明，还是方向定位？",
    "AI_Memory": "AI 记忆系统真正要证明的不是功能多，而是哪一个不可替代的记忆动力学机制？",
    "Health": "当前身体/情绪状态到底是在提示需要休息，还是提示系统任务配置已经错误？",
}


@dataclass
class MetaSpikeConfig:
    """Parameters for first-principle / coordinate-reset spikes."""

    theta_meta: float = 2.15
    prediction_error_weight: float = 0.55
    conflict_weight: float = 0.85
    global_voltage_weight: float = 0.35
    low_completion_weight: float = 0.45
    micro_density_weight: float = 0.25
    completion_floor: float = 0.25
    max_cards: int = 5


@dataclass
class MetaSpike:
    day: str
    state: str
    meta_energy: float
    prediction_error: float
    conflict: float
    global_voltage: float
    low_completion: float
    micro_density: float
    voltage_ratio: float
    effective_ratio: float
    topic: str
    first_principle_question: str
    coordinate_reset: str
    delete_or_downgrade: list[str] = field(default_factory=list)
    next_probe: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "state": self.state,
            "topic": self.topic,
            "meta_energy": round(self.meta_energy, 4),
            "prediction_error": round(self.prediction_error, 4),
            "conflict": round(self.conflict, 4),
            "global_voltage": round(self.global_voltage, 4),
            "low_completion": round(self.low_completion, 4),
            "micro_density": round(self.micro_density, 4),
            "voltage_ratio": round(self.voltage_ratio, 4),
            "effective_ratio": round(self.effective_ratio, 4),
            "first_principle_question": self.first_principle_question,
            "coordinate_reset": self.coordinate_reset,
            "delete_or_downgrade": self.delete_or_downgrade,
            "next_probe": self.next_probe,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect first-principle / coordinate-reset meta-spikes from Predictive LIF-Memory traces."
    )
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--states", type=str, default=",".join(lif_memory.NEURONS.keys()))
    parser.add_argument("--output", type=Path, default=Path("Meta-Spike Memory Report.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--predictive-state-file", type=Path, default=Path("predictive_lif_state.json"))
    parser.add_argument("--theta-meta", type=float, default=2.15)
    parser.add_argument("--max-cards", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--version", action="version", version=f"Meta-Spike Memory {VERSION}")
    return parser.parse_args()


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def resolve_vault(value: Path | None) -> Path:
    return (value or lif_memory.vault_root_from_script()).resolve()


def resolve_output_path(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def parse_cutoff(value: str | None) -> date:
    return lif_memory.parse_cutoff(value)


def points_by_day(points: list[predictive_lif_memory.LatentPoint]) -> dict[str, list[predictive_lif_memory.LatentPoint]]:
    grouped: dict[str, list[predictive_lif_memory.LatentPoint]] = {}
    for point in points:
        grouped.setdefault(point.day, []).append(point)
    return grouped


def point_map(points: list[predictive_lif_memory.LatentPoint]) -> dict[tuple[str, str], predictive_lif_memory.LatentPoint]:
    return {(point.day, point.state): point for point in points}


def conflict_for_state(point: predictive_lif_memory.LatentPoint, same_day: list[predictive_lif_memory.LatentPoint]) -> float:
    total = 0.0
    by_state = {item.state: item for item in same_day}
    for (src, dst), weight in NEGATIVE_CONFLICT_EDGES.items():
        if src != point.state:
            continue
        other = by_state.get(dst)
        if other is None:
            continue
        total += weight * max(point.effective_ratio, 0.0) * max(other.effective_ratio, 0.0)
    return total


def global_voltage(same_day: list[predictive_lif_memory.LatentPoint]) -> float:
    if not same_day:
        return 0.0
    return sum(max(point.ratio, 0.0) for point in same_day) / len(same_day)


def micro_density(same_day: list[predictive_lif_memory.LatentPoint]) -> float:
    if not same_day:
        return 0.0
    return sum(1.0 for point in same_day if point.micro_spike) / len(same_day)


def low_completion(point: predictive_lif_memory.LatentPoint, config: MetaSpikeConfig) -> float:
    if point.threshold <= 0:
        return 0.0
    completion_ratio = point.completion / point.threshold
    return clamp((config.completion_floor - completion_ratio) / max(config.completion_floor, 1e-9), 0.0, 1.0)


def first_principle_question(point: predictive_lif_memory.LatentPoint) -> str:
    topic = point.topic or "当前主题"
    state_question = FIRST_PRINCIPLE_PROMPTS.get(point.state, "真正生成这个困境的最小变量是什么？")
    return f"{state_question} 当前 topic 是「{topic}」，先不要继续加任务，而是找出一条最小生成变量。"


def coordinate_reset(point: predictive_lif_memory.LatentPoint) -> str:
    if point.state == "Experiment":
        return "从“继续做实验”重置为“识别最小可信证据链”：输入、参数、波形/截图、判据、可复现性。"
    if point.state == "Thesis":
        return "从“继续写论文”重置为“一个证据块能否支撑一个论断”：结论、图、限制条件、答辩说法。"
    if point.state == "Career":
        return "从“继续海投”重置为“一个项目表达能否证明岗位能力”：目标岗位、能力关键词、项目证据。"
    if point.state == "AI_Memory":
        return "从“继续堆功能”重置为“证明记忆何时主动激活”：预测误差、micro trace、macro action。"
    if point.state == "Health":
        return "从“硬撑推进任务”重置为“恢复系统执行带宽”：睡眠、饮食、移动、一个最小任务。"
    return "从局部动作重置为底层变量：删掉表层任务，找到能同时降低多个状态电位的变量。"


def deletion_candidates(point: predictive_lif_memory.LatentPoint, same_day: list[predictive_lif_memory.LatentPoint]) -> list[str]:
    candidates: list[str] = []
    if point.state in {"Experiment", "Thesis"}:
        candidates.append("暂时删除不能产生证据块的泛泛阅读/泛泛规划。")
    if point.state == "Career":
        candidates.append("暂时删除和目标岗位无关的海量信息流。")
    if point.state == "AI_Memory":
        candidates.append("暂时删除没有评估指标的新功能冲动。")
    if point.state == "Health":
        candidates.append("暂时删除需要高认知负荷但不能立刻闭环的任务。")

    high_states = [item.state for item in same_day if item.state != point.state and item.effective_ratio >= 0.75]
    if high_states:
        candidates.append("把这些高压状态降级为背景：" + ", ".join(high_states[:3]))
    return candidates[:3]


def next_probe(point: predictive_lif_memory.LatentPoint) -> str:
    if point.state == "Experiment":
        return "写一行最小实验判据：今天只要得到哪一个数值/截图，就算这条链路前进。"
    if point.state == "Thesis":
        return "写一句可答辩论断，并标注它需要哪一张图或哪一个数据支撑。"
    if point.state == "Career":
        return "选一个岗位 JD，把项目经历改成它能看懂的一条能力证明。"
    if point.state == "AI_Memory":
        return "跑一次 predictive/meta report，观察 micro-spike 是否比 macro-spike 更早暴露问题。"
    if point.state == "Health":
        return "先做一个 10 分钟恢复动作，然后只保留一个最小下一步。"
    return "写下一个能同时降低两个以上状态电位的最小动作。"


def meta_energy_for_point(
    point: predictive_lif_memory.LatentPoint,
    same_day: list[predictive_lif_memory.LatentPoint],
    config: MetaSpikeConfig,
) -> tuple[float, dict[str, float]]:
    pred = point.error_ema
    conflict = conflict_for_state(point, same_day)
    global_v = global_voltage(same_day)
    completion_gap = low_completion(point, config)
    micro = micro_density(same_day)
    energy = (
        config.prediction_error_weight * pred
        + config.conflict_weight * conflict
        + config.global_voltage_weight * global_v
        + config.low_completion_weight * completion_gap
        + config.micro_density_weight * micro
    )
    terms = {
        "prediction_error": pred,
        "conflict": conflict,
        "global_voltage": global_v,
        "low_completion": completion_gap,
        "micro_density": micro,
    }
    return energy, terms


def detect_meta_spikes(
    predictive_report: predictive_lif_memory.PredictiveReport,
    config: MetaSpikeConfig,
) -> list[MetaSpike]:
    grouped = points_by_day(predictive_report.points)
    cards: list[MetaSpike] = []
    for day, same_day in grouped.items():
        for point in same_day:
            energy, terms = meta_energy_for_point(point, same_day, config)
            if energy < config.theta_meta:
                continue
            cards.append(
                MetaSpike(
                    day=day,
                    state=point.state,
                    meta_energy=energy,
                    prediction_error=terms["prediction_error"],
                    conflict=terms["conflict"],
                    global_voltage=terms["global_voltage"],
                    low_completion=terms["low_completion"],
                    micro_density=terms["micro_density"],
                    voltage_ratio=point.ratio,
                    effective_ratio=point.effective_ratio,
                    topic=point.topic or "",
                    first_principle_question=first_principle_question(point),
                    coordinate_reset=coordinate_reset(point),
                    delete_or_downgrade=deletion_candidates(point, same_day),
                    next_probe=next_probe(point),
                )
            )
    cards.sort(key=lambda item: (item.meta_energy, item.conflict, item.prediction_error), reverse=True)
    return cards[: config.max_cards]


def render_markdown(
    notes: list[tuple[date, Path]],
    meta_spikes: list[MetaSpike],
    config: MetaSpikeConfig,
) -> str:
    lines: list[str] = []
    lines.append("# Meta-Spike Memory Report")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")
    lines.append("## 核心思想")
    lines.append("")
    lines.append("普通 spike 回答“下一步做什么”；meta-spike 回答“当前问题空间是否已经错了”。")
    lines.append("")
    lines.append("```text")
    lines.append("高预测误差 + 高状态冲突 + 高全局电位 + 低完成信号")
    lines.append("-> 不继续局部搜索")
    lines.append("-> 触发 first-principle / coordinate-reset spike")
    lines.append("```")
    lines.append("")
    lines.append("## 数学形式")
    lines.append("")
    lines.append("```text")
    lines.append("M_i(t) =")
    lines.append("  alpha * EMA(prediction_error_i)")
    lines.append("+ beta  * conflict_i(t)")
    lines.append("+ gamma * global_voltage(t)")
    lines.append("+ delta * low_completion_i(t)")
    lines.append("+ rho   * micro_density(t)")
    lines.append("")
    lines.append("S_meta_i(t) = 1[M_i(t) > Theta_meta]")
    lines.append("```")
    lines.append("")
    lines.append(f"当前 `Theta_meta = {config.theta_meta:.2f}`。")
    lines.append("")
    if not meta_spikes:
        lines.append("## 结果")
        lines.append("")
        lines.append("本次没有触发 meta-spike。说明当前更适合继续普通 LIF spike / predictive micro-spike，而不是重构问题空间。")
        lines.append("")
        return "\n".join(lines)

    lines.append("## Meta-Spike 卡片")
    lines.append("")
    for index, spike in enumerate(meta_spikes, start=1):
        lines.append(f"### Meta-Spike {index}: {spike.state} / {spike.day}")
        lines.append("")
        lines.append(f"- Topic：{spike.topic or 'unknown'}")
        lines.append(f"- Meta energy：{spike.meta_energy:.2f}")
        lines.append(f"- Prediction error：{spike.prediction_error:.2f}")
        lines.append(f"- Conflict：{spike.conflict:.2f}")
        lines.append(f"- Global voltage：{spike.global_voltage:.2f}")
        lines.append(f"- Low completion：{spike.low_completion:.2f}")
        lines.append(f"- Micro density：{spike.micro_density:.2f}")
        lines.append(f"- Voltage ratio：{spike.voltage_ratio:.2f}")
        lines.append(f"- Effective ratio：{spike.effective_ratio:.2f}")
        lines.append("")
        lines.append("#### First-principle question")
        lines.append("")
        lines.append(spike.first_principle_question)
        lines.append("")
        lines.append("#### Coordinate reset")
        lines.append("")
        lines.append(spike.coordinate_reset)
        lines.append("")
        if spike.delete_or_downgrade:
            lines.append("#### Delete / downgrade")
            lines.append("")
            for item in spike.delete_or_downgrade:
                lines.append(f"- {item}")
            lines.append("")
        lines.append("#### Next probe")
        lines.append("")
        lines.append(spike.next_probe)
        lines.append("")
    lines.append("## 使用边界")
    lines.append("")
    lines.append("- meta-spike 不是更多任务，而是任务生成机制的重置。")
    lines.append("- 如果 meta-spike 频繁出现，说明系统可能长期处于高冲突、低完成状态，需要降低任务数量或重新定义主线。")
    lines.append("- 如果完全不出现，说明当前局部搜索仍可工作，不需要强行第一性原理化。")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[list[MetaSpike], str, Path | None, Path | None]:
    vault = resolve_vault(args.vault)
    cutoff = parse_cutoff(args.today)
    active_neurons = lif_memory.parse_states(args.states)
    notes = lif_memory.find_daily_notes(vault, cutoff, args.days)

    predictive_state_path = resolve_output_path(vault, args.predictive_state_file)
    controller_state = predictive_lif_memory.load_controller_state(predictive_state_path)
    spikes, timeline, _states = lif_memory.replay(
        notes=notes,
        daily_spike_budget=0,
        active_neurons=active_neurons,
        topic_policies=None,
        completion_signals=None,
    )
    _ = spikes

    predictive_config = predictive_lif_memory.PredictiveConfig()
    predictive_report = predictive_lif_memory.build_predictive_report(
        timeline=timeline,
        active_states=list(active_neurons.keys()),
        controller_state=controller_state,
        config=predictive_config,
    )

    config = MetaSpikeConfig(theta_meta=args.theta_meta, max_cards=args.max_cards)
    meta_spikes = detect_meta_spikes(predictive_report, config)
    markdown = render_markdown(notes, meta_spikes, config)

    output_path = resolve_output_path(vault, args.output)
    json_path = resolve_output_path(vault, args.json_output)
    if not args.dry_run:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(
                json.dumps(
                    {
                        "version": VERSION,
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "theta_meta": config.theta_meta,
                        "meta_spikes": [spike.to_json() for spike in meta_spikes],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
    return meta_spikes, markdown, output_path, json_path


def main() -> None:
    args = parse_args()
    meta_spikes, markdown, output_path, json_path = run(args)
    if args.dry_run:
        print(markdown)
        return
    print(f"Generated meta-spikes: {len(meta_spikes)}")
    if output_path:
        print(f"Wrote: {output_path}")
    if json_path:
        print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
