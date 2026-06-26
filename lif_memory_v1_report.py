from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import insight_integrator as insights
import lif_memory as core
import llm_adapter

VERSION = "1.0.0-convergence"


@dataclass(frozen=True)
class ArchitectureLayer:
    name: str
    purpose: str
    modules: list[str]
    output: str


@dataclass(frozen=True)
class VersionPhase:
    phase: str
    versions: str
    problem_solved: str
    unified_layer: str
    result: str


ARCHITECTURE = [
    ArchitectureLayer(
        name="Memory Source",
        purpose="Keep original notes as the lossless memory substrate.",
        modules=["Obsidian daily notes", "project notes", "wikilinks", "folders", "tags"],
        output="evidence candidates",
    ),
    ArchitectureLayer(
        name="Evidence Sensor",
        purpose="Turn note fragments into auditable input current instead of raw keyword counts.",
        modules=["lif_memory.EvidenceVector", "topic rules", "completion scan", "LLM reviewer"],
        output="scored evidence vectors",
    ),
    ArchitectureLayer(
        name="LIF Dynamics",
        purpose="Accumulate, leak, inhibit, threshold, reset, and cool down topic states.",
        modules=["V_fast", "V_slow", "theta", "decay", "completion_inhibition", "topic_policies"],
        output="state trajectory and spike events",
    ),
    ArchitectureLayer(
        name="Spike Interface",
        purpose="Return the smallest useful card: evidence chain, priority, blocker type, action policy, and completion target.",
        modules=["daily mode", "replay report", "insight profile", "graph report", "feedback memory"],
        output="action spikes and insight spikes",
    ),
]


VERSION_STORY = [
    VersionPhase(
        phase="1. Evidence replay",
        versions="v0.2",
        problem_solved="Obsidian notes were scattered and needed traceable evidence packets.",
        unified_layer="Memory Source",
        result="Recursive discovery, evidence snippets, date-gap leakage, JSON/Markdown output.",
    ),
    VersionPhase(
        phase="2. Semantic current",
        versions="v0.3",
        problem_solved="Keyword hits were too crude to represent action pressure.",
        unified_layer="Evidence Sensor",
        result="EvidenceVector adds actionability, urgency, blocker, completion, specificity, novelty, confidence.",
    ),
    VersionPhase(
        phase="3. Action decision",
        versions="v0.4",
        problem_solved="A spike needed to say what to do, not only what crossed threshold.",
        unified_layer="Spike Interface",
        result="priority, blocker_type, action_policy, completion_target.",
    ),
    VersionPhase(
        phase="4. Long-running memory",
        versions="v0.5",
        problem_solved="A one-window replay forgot repeated loops.",
        unified_layer="LIF Dynamics",
        result="topic history persists days_seen, blocker_count, completion_count, last_action_policy.",
    ),
    VersionPhase(
        phase="5. Dual-timescale state",
        versions="v0.6",
        problem_solved="Acute pressure and background pressure should not decay at the same speed.",
        unified_layer="LIF Dynamics",
        result="V_fast and V_slow produce a combined spike voltage.",
    ),
    VersionPhase(
        phase="6. Human feedback loop",
        versions="v0.7.0-v0.7.3",
        problem_solved="The system needed to learn which reminders were useful, done, early, or muted.",
        unified_layer="Spike Interface",
        result="Markdown closures and lif_memory_feedback.json persist topic policies.",
    ),
    VersionPhase(
        phase="7. LLM semantic review",
        versions="v0.7.4",
        problem_solved="Rule-based semantics needed calibration without handing control to an LLM.",
        unified_layer="Evidence Sensor",
        result="LLM is a reviewer only; it cannot update voltage, threshold, cooldown, or policy.",
    ),
    VersionPhase(
        phase="8. Graph and insight expansion",
        versions="graph miner + insight integrator",
        problem_solved="Memory is not only time-series notes; it also has graph structure and explanatory tension.",
        unified_layer="Memory Source / Spike Interface",
        result="Obsidian graph report and domain insight profiles extend action spikes into thought spikes.",
    ),
]


HARVEST = [
    "LIF-Memory is not a summary tool; it is an event-driven memory dynamics prototype.",
    "Original notes preserve full information; voltage preserves unresolved pressure or explanatory tension.",
    "EvidenceVector is the semantic sensor layer between text and LIF current.",
    "V_fast/V_slow separate acute pressure from long-running background pressure.",
    "Spike packets provide evidence, priority, blocker type, action policy, and completion target.",
    "Feedback closures turn human judgement into future thresholds, cooldown, priority, and mute policies.",
    "LLM review calibrates semantics but does not control the dynamical state.",
    "The same LIF pattern supports both action profiles and insight profiles.",
]


def state_summary() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, config in core.NEURONS.items():
        rows.append(
            {
                "state": name,
                "theta": config.theta,
                "fast_decay": config.decay,
                "slow_decay": config.slow_decay,
                "evidence_cap": config.evidence_cap,
                "cooldown_days": config.cooldown_days,
                "suggestion": config.suggestion,
            }
        )
    return rows


def topic_policy_summary() -> list[dict[str, object]]:
    return [
        {
            "topic": topic,
            "priority": priority,
            "completion_target": core.COMPLETION_TARGETS.get(topic, "完成一个可判定的小结果。"),
        }
        for topic, priority in core.TOPIC_PRIORITY_OVERRIDES.items()
    ]


def build_payload() -> dict[str, object]:
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "report_version": VERSION,
        "core_version": core.VERSION,
        "insight_integrator_version": insights.VERSION,
        "llm_providers": sorted(llm_adapter.PROVIDER_PRESETS.keys()),
        "architecture": [asdict(item) for item in ARCHITECTURE],
        "version_story": [asdict(item) for item in VERSION_STORY],
        "states": state_summary(),
        "topic_policies": topic_policy_summary(),
        "harvest": HARVEST,
    }


def render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# LIF-Memory v1.0 Convergence Report",
        "",
        f"生成时间：{payload['generated_at']}",
        f"报告版本：{payload['report_version']}",
        f"核心版本：lif_memory={payload['core_version']}, insight_integrator={payload['insight_integrator_version']}",
        "",
        "## 一句话收束",
        "",
        "LIF-Memory 是一个面向 Obsidian 的事件驱动个人记忆动力系统：原始笔记保留完整信息，EvidenceVector 把片段转成输入电流，快慢 LIF 状态追踪未闭环压力或解释张力，spike 卡片返回证据链、行动策略和完成目标。",
        "",
        "## 统一链路",
        "",
        "```text",
        "Obsidian 原始笔记 -> 证据片段 -> EvidenceVector -> V_fast / V_slow -> threshold spike -> 行动/洞察卡 -> 人工反馈 -> topic policy",
        "```",
        "",
        "## 架构层",
        "",
    ]

    for item in payload["architecture"]:
        lines.extend(
            [
                f"### {item['name']}",
                "",
                f"- 目的：{item['purpose']}",
                f"- 模块：{', '.join(item['modules'])}",
                f"- 输出：{item['output']}",
                "",
            ]
        )

    lines.extend(["## 版本断层如何融合", ""])
    for item in payload["version_story"]:
        lines.extend(
            [
                f"### {item['phase']}（{item['versions']}）",
                "",
                f"- 当时解决的问题：{item['problem_solved']}",
                f"- 现在归入层级：{item['unified_layer']}",
                f"- 形成的结果：{item['result']}",
                "",
            ]
        )

    lines.extend(
        [
            "## 当前状态神经元",
            "",
            "| State | theta | fast_decay | slow_decay | evidence_cap | cooldown |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for state in payload["states"]:
        lines.append(
            f"| {state['state']} | {state['theta']} | {state['fast_decay']} | {state['slow_decay']} | {state['evidence_cap']} | {state['cooldown_days']} |"
        )

    lines.extend(["", "## 主线 topic 与完成目标", ""])
    for topic in payload["topic_policies"]:
        lines.extend([f"### {topic['topic']} / {topic['priority']}", "", str(topic["completion_target"]), ""])

    lines.extend(["## 可观收获", ""])
    for index, item in enumerate(payload["harvest"], start=1):
        lines.append(f"{index}. {item}")

    lines.extend(
        [
            "",
            "## 建议的 v1.0 边界",
            "",
            "v1.0 不再继续堆新功能。它的边界是：能稳定生成一张今日主卡、能输出一次全量 replay、能把 spike 关闭反馈写回持久记忆、能用 graph/insight 报告解释长期证据链。",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a v1.0 convergence report for LIF-Memory.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory v1.0 收束报告.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload()
    markdown = render_markdown(payload)
    if args.json_output:
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(markdown)
    else:
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
