from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import lif_memory

VERSION = "0.1.0"

STATE_GRAPH: dict[str, dict[str, float]] = {
    "Experiment": {"Thesis": 0.42, "Career": 0.18, "AI_Memory": 0.16, "Health": 0.14},
    "Thesis": {"Experiment": 0.38, "Career": 0.22, "AI_Memory": 0.14, "Health": 0.16},
    "Career": {"Thesis": 0.26, "Experiment": 0.16, "AI_Memory": 0.30, "Health": 0.18},
    "AI_Memory": {"Career": 0.28, "Experiment": 0.18, "Thesis": 0.16},
    "Health": {"Experiment": 0.24, "Thesis": 0.24, "Career": 0.18},
}


@dataclass
class PredictiveConfig:
    """Controller parameters for the predictive LIF layer.

    The layer is intentionally small and auditable. It does not train an
    embedding model and does not upload notes. It only reads the latent voltage
    trace produced by lif_memory.replay and adds a JEPA-style prediction loop.
    """

    micro_ratio: float = 0.55
    macro_ratio: float = 1.00
    target_micro_rate: float = 0.35
    homeostasis_eta: float = 0.08
    prediction_error_weight: float = 0.45
    diffusion_alpha: float = 0.15
    min_micro_ratio: float = 0.35
    max_micro_ratio: float = 0.85
    prediction_decay: float = 0.86
    input_gain: float = 0.16
    completion_gain: float = 0.12
    persistent_error_gain: float = 0.28


@dataclass
class PredictiveControllerState:
    micro_threshold_delta: dict[str, float] = field(default_factory=dict)
    prediction_error_ema: dict[str, float] = field(default_factory=dict)
    micro_streak: dict[str, int] = field(default_factory=dict)
    runs: int = 0
    updated_at: str = ""

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "PredictiveControllerState":
        return cls(
            micro_threshold_delta={str(k): float(v) for k, v in dict(data.get("micro_threshold_delta", {})).items()},
            prediction_error_ema={str(k): float(v) for k, v in dict(data.get("prediction_error_ema", {})).items()},
            micro_streak={str(k): int(v) for k, v in dict(data.get("micro_streak", {})).items()},
            runs=int(data.get("runs", 0)),
            updated_at=str(data.get("updated_at", "")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "runs": self.runs,
            "updated_at": self.updated_at,
            "micro_threshold_delta": {k: round(v, 6) for k, v in sorted(self.micro_threshold_delta.items())},
            "prediction_error_ema": {k: round(v, 6) for k, v in sorted(self.prediction_error_ema.items())},
            "micro_streak": {k: int(v) for k, v in sorted(self.micro_streak.items())},
        }


@dataclass
class LatentPoint:
    day: str
    state: str
    voltage: float
    v_fast: float
    v_slow: float
    evidence_input: float
    completion: float
    threshold: float
    ratio: float
    diffused_ratio: float
    predicted_ratio: float
    prediction_error: float
    error_ema: float
    micro_threshold_ratio: float
    effective_ratio: float
    micro_spike: bool
    macro_candidate: bool
    official_spike: bool
    topic: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "day": self.day,
            "state": self.state,
            "topic": self.topic,
            "voltage": round(self.voltage, 4),
            "v_fast": round(self.v_fast, 4),
            "v_slow": round(self.v_slow, 4),
            "evidence_input": round(self.evidence_input, 4),
            "completion": round(self.completion, 4),
            "threshold": round(self.threshold, 4),
            "ratio": round(self.ratio, 4),
            "diffused_ratio": round(self.diffused_ratio, 4),
            "predicted_ratio": round(self.predicted_ratio, 4),
            "prediction_error": round(self.prediction_error, 4),
            "error_ema": round(self.error_ema, 4),
            "micro_threshold_ratio": round(self.micro_threshold_ratio, 4),
            "effective_ratio": round(self.effective_ratio, 4),
            "micro_spike": self.micro_spike,
            "macro_candidate": self.macro_candidate,
            "official_spike": self.official_spike,
        }


@dataclass
class PredictiveReport:
    points: list[LatentPoint]
    micro_counts: dict[str, int]
    macro_counts: dict[str, int]
    final_error_ema: dict[str, float]
    controller_state: PredictiveControllerState

    def to_json(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "micro_counts": self.micro_counts,
            "macro_counts": self.macro_counts,
            "final_error_ema": {k: round(v, 6) for k, v in sorted(self.final_error_ema.items())},
            "controller_state": self.controller_state.to_json(),
            "points": [point.to_json() for point in self.points],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a JEPA-style predictive layer over the LIF-Memory latent voltage trace."
    )
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to replay.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--states", type=str, default=",".join(lif_memory.NEURONS.keys()))
    parser.add_argument("--state-file", type=Path, default=Path("predictive_lif_state.json"))
    parser.add_argument("--output", type=Path, default=Path("Predictive LIF-Memory Report.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--micro-ratio", type=float, default=0.55)
    parser.add_argument("--macro-ratio", type=float, default=1.00)
    parser.add_argument("--target-micro-rate", type=float, default=0.35)
    parser.add_argument("--prediction-error-weight", type=float, default=0.45)
    parser.add_argument("--diffusion-alpha", type=float, default=0.15)
    parser.add_argument("--daily-spike-budget", type=int, default=0, help="Use 0 to observe subthreshold voltage without reset.")
    parser.add_argument("--version", action="version", version=f"Predictive LIF-Memory {VERSION}")
    return parser.parse_args()


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def load_controller_state(path: Path | None) -> PredictiveControllerState:
    if path is None or not path.exists():
        return PredictiveControllerState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return PredictiveControllerState()
    if not isinstance(data, dict):
        return PredictiveControllerState()
    return PredictiveControllerState.from_json(data)


def save_controller_state(path: Path | None, state: PredictiveControllerState) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    state.updated_at = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(state.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_vault(value: Path | None) -> Path:
    return (value or lif_memory.vault_root_from_script()).resolve()


def resolve_output_path(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def parse_cutoff(value: str | None) -> date:
    return lif_memory.parse_cutoff(value)


def row_state(row: dict[str, Any], state_name: str) -> dict[str, Any]:
    item = row.get(state_name)
    if not isinstance(item, dict):
        return {}
    return item


def normalized_input(raw_input: float, threshold: float) -> float:
    if threshold <= 0:
        return 0.0
    return clamp(raw_input / max(threshold, 1e-9), 0.0, 1.5)


def build_base_ratios(row: dict[str, Any], active_states: list[str]) -> dict[str, float]:
    ratios: dict[str, float] = {}
    for state_name in active_states:
        item = row_state(row, state_name)
        threshold = float(item.get("effective_threshold") or 1.0)
        voltage = float(item.get("new_v") or 0.0)
        ratios[state_name] = voltage / threshold if threshold else 0.0
    return ratios


def diffuse_ratios(base: dict[str, float], diffusion_alpha: float) -> dict[str, float]:
    diffused: dict[str, float] = {}
    for state_name, value in base.items():
        neighbors = STATE_GRAPH.get(state_name, {})
        if not neighbors:
            diffused[state_name] = value
            continue
        neighbor_pressure = sum(base.get(other, 0.0) * weight for other, weight in neighbors.items())
        weight_sum = sum(abs(weight) for weight in neighbors.values()) or 1.0
        diffused[state_name] = value + diffusion_alpha * (neighbor_pressure / weight_sum)
    return diffused


def predict_ratio(prev: LatentPoint | None, config: PredictiveConfig) -> float:
    if prev is None:
        return 0.0
    predicted = (
        prev.diffused_ratio * config.prediction_decay
        + normalized_input(prev.evidence_input, prev.threshold) * config.input_gain
        - normalized_input(prev.completion, prev.threshold) * config.completion_gain
        + prev.error_ema * config.persistent_error_gain
    )
    return clamp(predicted, 0.0, 2.5)


def micro_threshold_ratio(state_name: str, controller_state: PredictiveControllerState, config: PredictiveConfig) -> float:
    delta = controller_state.micro_threshold_delta.get(state_name, 0.0)
    return clamp(config.micro_ratio + delta, config.min_micro_ratio, config.max_micro_ratio)


def is_macro_candidate(
    ratio: float,
    effective_ratio: float,
    micro_spike: bool,
    micro_streak: int,
    config: PredictiveConfig,
) -> bool:
    if ratio >= config.macro_ratio:
        return True
    if micro_spike and micro_streak >= 3 and effective_ratio >= max(config.macro_ratio * 0.82, config.micro_ratio):
        return True
    return False


def build_predictive_report(
    timeline: list[dict[str, Any]],
    active_states: list[str],
    controller_state: PredictiveControllerState,
    config: PredictiveConfig,
) -> PredictiveReport:
    points: list[LatentPoint] = []
    previous_by_state: dict[str, LatentPoint] = {}
    micro_counts = {name: 0 for name in active_states}
    macro_counts = {name: 0 for name in active_states}

    for row in timeline:
        day = str(row.get("date", ""))
        base_ratios = build_base_ratios(row, active_states)
        diffused_ratios = diffuse_ratios(base_ratios, config.diffusion_alpha)

        for state_name in active_states:
            item = row_state(row, state_name)
            threshold = float(item.get("effective_threshold") or 1.0)
            voltage = float(item.get("new_v") or 0.0)
            ratio = base_ratios.get(state_name, 0.0)
            diffused_ratio = diffused_ratios.get(state_name, ratio)
            prev = previous_by_state.get(state_name)
            predicted = predict_ratio(prev, config)
            error = abs(diffused_ratio - predicted)
            old_ema = controller_state.prediction_error_ema.get(state_name, 0.0)
            error_ema = 0.72 * old_ema + 0.28 * error
            controller_state.prediction_error_ema[state_name] = error_ema

            threshold_ratio = micro_threshold_ratio(state_name, controller_state, config)
            effective_ratio = diffused_ratio + config.prediction_error_weight * error_ema
            micro_spike = effective_ratio >= threshold_ratio
            if micro_spike:
                controller_state.micro_streak[state_name] = controller_state.micro_streak.get(state_name, 0) + 1
                micro_counts[state_name] += 1
            else:
                controller_state.micro_streak[state_name] = 0

            macro_candidate = is_macro_candidate(
                ratio=ratio,
                effective_ratio=effective_ratio,
                micro_spike=micro_spike,
                micro_streak=controller_state.micro_streak.get(state_name, 0),
                config=config,
            )
            if macro_candidate:
                macro_counts[state_name] += 1

            point = LatentPoint(
                day=day,
                state=state_name,
                voltage=voltage,
                v_fast=float(item.get("new_fast") or 0.0),
                v_slow=float(item.get("new_slow") or 0.0),
                evidence_input=float(item.get("input") or 0.0),
                completion=float(item.get("completion") or 0.0),
                threshold=threshold,
                ratio=ratio,
                diffused_ratio=diffused_ratio,
                predicted_ratio=predicted,
                prediction_error=error,
                error_ema=error_ema,
                micro_threshold_ratio=threshold_ratio,
                effective_ratio=effective_ratio,
                micro_spike=micro_spike,
                macro_candidate=macro_candidate,
                official_spike=bool(item.get("spike")),
                topic=str(item.get("topic") or ""),
            )
            points.append(point)
            previous_by_state[state_name] = point

    update_homeostasis(controller_state, micro_counts, max(len(timeline), 1), active_states, config)
    controller_state.runs += 1
    return PredictiveReport(
        points=points,
        micro_counts=micro_counts,
        macro_counts=macro_counts,
        final_error_ema=dict(controller_state.prediction_error_ema),
        controller_state=controller_state,
    )


def update_homeostasis(
    controller_state: PredictiveControllerState,
    micro_counts: dict[str, int],
    n_days: int,
    active_states: list[str],
    config: PredictiveConfig,
) -> None:
    for state_name in active_states:
        rate = micro_counts.get(state_name, 0) / max(n_days, 1)
        delta = controller_state.micro_threshold_delta.get(state_name, 0.0)
        delta += config.homeostasis_eta * (rate - config.target_micro_rate)
        controller_state.micro_threshold_delta[state_name] = clamp(
            delta,
            config.min_micro_ratio - config.micro_ratio,
            config.max_micro_ratio - config.micro_ratio,
        )


def summarize_final_points(points: list[LatentPoint]) -> dict[str, LatentPoint]:
    final: dict[str, LatentPoint] = {}
    for point in points:
        final[point.state] = point
    return final


def top_macro_candidates(points: list[LatentPoint], limit: int = 8) -> list[LatentPoint]:
    candidates = [point for point in points if point.macro_candidate]
    candidates.sort(key=lambda point: (point.effective_ratio, point.prediction_error, point.ratio), reverse=True)
    return candidates[:limit]


def top_prediction_errors(points: list[LatentPoint], limit: int = 8) -> list[LatentPoint]:
    ordered = sorted(points, key=lambda point: point.prediction_error, reverse=True)
    return ordered[:limit]


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    report: PredictiveReport,
    config: PredictiveConfig,
) -> str:
    lines: list[str] = []
    lines.append("# Predictive LIF-Memory Report")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")
    lines.append("## 核心升级")
    lines.append("")
    lines.append("这一层不替代 `lif_memory.py`，而是在经典 LIF 回放得到的亚阈值电位轨迹上增加预测闭环：")
    lines.append("")
    lines.append("```text")
    lines.append("LIF latent state z_t -> predictor -> predicted z_(t+1)")
    lines.append("actual z_(t+1) - predicted z_(t+1) -> prediction_error")
    lines.append("prediction_error + graph diffusion + voltage ratio -> micro/macro activation")
    lines.append("homeostasis -> adjust micro threshold")
    lines.append("```")
    lines.append("")
    lines.append("这样即使没有正式 spike，系统也可以保留 micro-spike 和 prediction-error trace，避免“太稀疏导致什么都没学到”。")
    lines.append("")
    lines.append("## 数学形式")
    lines.append("")
    lines.append("```text")
    lines.append("r_i(t) = V_i(t) / theta_i")
    lines.append("r'_i(t) = r_i(t) + diffusion_alpha * sum_j A_ij r_j(t)")
    lines.append("r_hat_i(t) = decay * r'_i(t-1) + input_gain * input_i(t-1) - completion_gain * completion_i(t-1)")
    lines.append("error_i(t) = |r'_i(t) - r_hat_i(t)|")
    lines.append("effective_i(t) = r'_i(t) + prediction_error_weight * EMA(error_i)")
    lines.append("micro_spike_i(t) = effective_i(t) >= theta_micro_i")
    lines.append("theta_micro_i <- theta_micro_i + eta * (firing_rate_i - target_rate)")
    lines.append("```")
    lines.append("")
    lines.append("## 汇总")
    lines.append("")
    final = summarize_final_points(report.points)
    lines.append("| 状态 | 最终 ratio | micro 次数 | macro 候选 | error EMA | micro 阈值 | 解释 |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for state_name, point in sorted(final.items()):
        if point.effective_ratio >= config.macro_ratio:
            hint = "需要显式处理"
        elif report.micro_counts.get(state_name, 0) > 0:
            hint = "已有亚阈值激活"
        elif point.error_ema > 0.25:
            hint = "预测误差在积累"
        else:
            hint = "当前负荷较低"
        lines.append(
            f"| {state_name} | {point.ratio:.2f} | {report.micro_counts.get(state_name, 0)} | "
            f"{report.macro_counts.get(state_name, 0)} | {point.error_ema:.2f} | {point.micro_threshold_ratio:.2f} | {hint} |"
        )
    lines.append("")
    lines.append("## Macro 候选")
    lines.append("")
    candidates = top_macro_candidates(report.points)
    if not candidates:
        lines.append("本次没有 predictive macro candidate。系统仍然保存了 micro trace 和 prediction-error trace。")
        lines.append("")
    else:
        for point in candidates:
            lines.append(f"### {point.day} / {point.state}")
            lines.append("")
            lines.append(f"- Topic：{point.topic or 'unknown'}")
            lines.append(f"- Voltage ratio：{point.ratio:.2f}")
            lines.append(f"- Diffused ratio：{point.diffused_ratio:.2f}")
            lines.append(f"- Predicted ratio：{point.predicted_ratio:.2f}")
            lines.append(f"- Prediction error：{point.prediction_error:.2f}")
            lines.append(f"- Effective ratio：{point.effective_ratio:.2f}")
            lines.append(f"- Official spike：{point.official_spike}")
            lines.append("")
    lines.append("## 最大预测误差")
    lines.append("")
    lines.append("| 日期 | 状态 | topic | predicted | actual diffused | error | effective |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for point in top_prediction_errors(report.points):
        lines.append(
            f"| {point.day} | {point.state} | {point.topic or ''} | {point.predicted_ratio:.2f} | "
            f"{point.diffused_ratio:.2f} | {point.prediction_error:.2f} | {point.effective_ratio:.2f} |"
        )
    lines.append("")
    lines.append("## Micro trace")
    lines.append("")
    lines.append("| 日期 | 状态 | ratio | diffused | error EMA | micro threshold | micro | macro candidate |")
    lines.append("|---|---|---:|---:|---:|---:|---|---|")
    for point in report.points:
        if not point.micro_spike and point.prediction_error < 0.12:
            continue
        micro = "yes" if point.micro_spike else ""
        macro = "yes" if point.macro_candidate else ""
        lines.append(
            f"| {point.day} | {point.state} | {point.ratio:.2f} | {point.diffused_ratio:.2f} | "
            f"{point.error_ema:.2f} | {point.micro_threshold_ratio:.2f} | {micro} | {macro} |"
        )
    lines.append("")
    lines.append("## 设计边界")
    lines.append("")
    lines.append("- 这一层不是端到端神经网络训练，也不会微调 embedding。")
    lines.append("- `prediction_error` 不是客观真理，只表示当前 latent trajectory 与上一轮预测不一致。")
    lines.append("- micro-spike 默认不直接打扰用户；macro candidate 才适合转成行动卡片或 AhaEngine 输入。")
    lines.append("- 原始笔记仍然是 source of truth；本层只保存派生的 controller state。")
    lines.append("")
    lines.append("## 本地状态文件")
    lines.append("")
    lines.append("默认写入：")
    lines.append("")
    lines.append("```text")
    lines.append("predictive_lif_state.json")
    lines.append("```")
    lines.append("")
    lines.append("该文件包含 micro 阈值调节、prediction-error EMA 和 micro streak，属于私人派生状态，不应提交。")
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[PredictiveReport, str, Path | None, Path | None]:
    vault = resolve_vault(args.vault)
    cutoff = parse_cutoff(args.today)
    active_neurons = lif_memory.parse_states(args.states)
    active_states = list(active_neurons.keys())
    notes = lif_memory.find_daily_notes(vault, cutoff, args.days)
    controller_state_path = resolve_output_path(vault, args.state_file)
    controller_state = load_controller_state(controller_state_path)

    spikes, timeline, _states = lif_memory.replay(
        notes=notes,
        daily_spike_budget=args.daily_spike_budget,
        active_neurons=active_neurons,
        topic_policies=None,
        completion_signals=None,
    )

    # The official spike list is intentionally not rendered here. This layer
    # focuses on the latent trajectory, including subthreshold states.
    _ = spikes

    config = PredictiveConfig(
        micro_ratio=args.micro_ratio,
        macro_ratio=args.macro_ratio,
        target_micro_rate=args.target_micro_rate,
        prediction_error_weight=args.prediction_error_weight,
        diffusion_alpha=args.diffusion_alpha,
    )
    report = build_predictive_report(timeline, active_states, controller_state, config)
    markdown = render_markdown(vault, notes, report, config)

    output_path = resolve_output_path(vault, args.output)
    json_path = resolve_output_path(vault, args.json_output)

    if not args.dry_run:
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown, encoding="utf-8")
        if json_path is not None:
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        save_controller_state(controller_state_path, report.controller_state)

    return report, markdown, output_path, json_path


def main() -> None:
    args = parse_args()
    report, markdown, output_path, json_path = run(args)
    if args.dry_run:
        print(markdown)
        return
    print(f"Generated predictive points: {len(report.points)}")
    if output_path:
        print(f"Wrote: {output_path}")
    if json_path:
        print(f"Wrote JSON: {json_path}")
    if args.state_file:
        print(f"Updated state: {args.state_file}")


if __name__ == "__main__":
    main()
