from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

VERSION = "1.0.0"


RELATION_ALIASES = {
    "supports": "support",
    "supporting": "support",
    "supported_by": "support",
    "blocks": "inhibition",
    "block": "inhibition",
    "inhibits": "inhibition",
    "inhibit": "inhibition",
    "downgrades": "downgrade",
    "downgraded": "downgrade",
    "reframes": "reframe",
    "reframing": "reframe",
    "causes": "causal",
    "cause": "causal",
    "co_activation": "bridge",
    "coactivation": "bridge",
}


@dataclass
class TopicPolicyDraft:
    topic: str
    threshold_delta: float = 0.0
    priority: str | None = None
    action_policy: str | None = None
    cooldown_days: int = 0
    muted: bool = False
    feedback: str = "adaptive_schema"
    reasons: list[str] = field(default_factory=list)
    relation_sources: list[dict[str, Any]] = field(default_factory=list)

    def merge(self, other: "TopicPolicyDraft") -> None:
        self.threshold_delta += other.threshold_delta
        self.cooldown_days = max(self.cooldown_days, other.cooldown_days)
        self.muted = self.muted or other.muted
        self.reasons.extend(other.reasons)
        self.relation_sources.extend(other.relation_sources)
        self.priority = strongest_priority(self.priority, other.priority)
        self.action_policy = strongest_action_policy(self.action_policy, other.action_policy)

    def clamp(self) -> None:
        self.threshold_delta = max(-2.0, min(2.0, self.threshold_delta))
        self.cooldown_days = max(0, min(30, self.cooldown_days))
        self.reasons = dedupe(self.reasons)[:8]

    def to_feedback_item(self) -> dict[str, Any]:
        self.clamp()
        item: dict[str, Any] = {
            "topic": self.topic,
            "feedback": self.feedback,
            "threshold_delta": round(self.threshold_delta, 3),
            "cooldown_days": self.cooldown_days,
            "muted": self.muted,
            "reason": "; ".join(self.reasons),
            "relation_sources": self.relation_sources[:8],
        }
        if self.priority:
            item["priority"] = self.priority
        if self.action_policy:
            item["action_policy"] = self.action_policy
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert adaptive relation schema into lif_memory feedback policy.")
    parser.add_argument("--schema", type=Path, default=Path("lif_relation_schema.generated.json"), help="Adaptive schema JSON from adaptive_relation_spike.py.")
    parser.add_argument("--output", type=Path, default=Path("lif_memory_adaptive_policy.json"), help="Output feedback-policy JSON readable by lif_memory.py --feedback-file.")
    parser.add_argument("--report", type=Path, default=Path("LIF-Schema-Policy-Adapter 报告.md"), help="Markdown report path.")
    parser.add_argument("--min-confidence", type=float, default=0.0, help="Ignore relations below this confidence if confidence is present.")
    parser.add_argument("--only-confirmed", action="store_true", help="Use only relations confirmed by human calibration.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF schema policy adapter {VERSION}")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Schema file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit("Schema file must contain a JSON object.")
    return data


def dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    for item in items:
        if item and item not in result:
            result.append(item)
    return result


def normalize_relation_type(value: Any) -> str:
    relation_type = str(value or "unknown").strip().lower().replace(" ", "_").replace("-", "_")
    return RELATION_ALIASES.get(relation_type, relation_type or "unknown")


def confidence_of(relation: Mapping[str, Any]) -> float:
    raw = relation.get("confidence", relation.get("score", 1.0))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0
    if value > 1.0:
        return min(value / 10.0, 1.0)
    return max(0.0, value)


def relation_name(relation: Mapping[str, Any]) -> str:
    name = str(relation.get("name") or "").strip()
    if name:
        return name
    source = str(relation.get("source") or "").strip()
    target = str(relation.get("target") or "").strip()
    relation_type = normalize_relation_type(relation.get("relation_type"))
    return f"{source} -[{relation_type}]-> {target}"


def strongest_priority(left: str | None, right: str | None) -> str | None:
    rank = {None: 0, "P2": 1, "P1": 2, "P0": 3}
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def strongest_action_policy(left: str | None, right: str | None) -> str | None:
    rank = {
        None: 0,
        "continue": 1,
        "downgrade": 2,
        "isolate": 3,
        "recover_first": 4,
        "stop": 5,
    }
    return left if rank.get(left, 0) >= rank.get(right, 0) else right


def make_policy(
    topic: str,
    relation: Mapping[str, Any],
    threshold_delta: float,
    priority: str | None,
    action_policy: str | None,
    cooldown_days: int = 0,
    muted: bool = False,
) -> TopicPolicyDraft:
    relation_type = normalize_relation_type(relation.get("relation_type"))
    reason = str(relation.get("hypothesis") or relation.get("reason") or relation_name(relation)).strip()
    source_record = {
        "relation": relation_name(relation),
        "relation_type": relation_type,
        "source": relation.get("source"),
        "target": relation.get("target"),
        "confidence": round(confidence_of(relation), 3),
    }
    return TopicPolicyDraft(
        topic=topic,
        threshold_delta=threshold_delta,
        priority=priority,
        action_policy=action_policy,
        cooldown_days=cooldown_days,
        muted=muted,
        reasons=[reason],
        relation_sources=[source_record],
    )


def relation_to_policy_drafts(relation: Mapping[str, Any]) -> list[TopicPolicyDraft]:
    source = str(relation.get("source") or "").strip()
    target = str(relation.get("target") or "").strip()
    relation_type = normalize_relation_type(relation.get("relation_type"))
    if not source or not target:
        return []

    drafts: list[TopicPolicyDraft] = []

    if relation_type == "support":
        drafts.append(make_policy(target, relation, threshold_delta=-0.45, priority="P0", action_policy="continue"))
        drafts.append(make_policy(source, relation, threshold_delta=-0.15, priority="P1", action_policy="continue"))
    elif relation_type == "conflict":
        drafts.append(make_policy(source, relation, threshold_delta=0.35, priority="P1", action_policy="isolate", cooldown_days=1))
        drafts.append(make_policy(target, relation, threshold_delta=0.25, priority="P1", action_policy="isolate", cooldown_days=1))
    elif relation_type == "inhibition":
        drafts.append(make_policy(target, relation, threshold_delta=0.75, priority="P0", action_policy="recover_first", cooldown_days=1))
        drafts.append(make_policy(source, relation, threshold_delta=-0.10, priority="P1", action_policy="continue"))
    elif relation_type == "reframe":
        drafts.append(make_policy(target, relation, threshold_delta=-0.25, priority="P1", action_policy="continue"))
        drafts.append(make_policy(source, relation, threshold_delta=0.15, priority="P2", action_policy="downgrade", cooldown_days=2))
    elif relation_type == "downgrade":
        drafts.append(make_policy(source, relation, threshold_delta=1.00, priority="P2", action_policy="downgrade", cooldown_days=7))
        drafts.append(make_policy(target, relation, threshold_delta=-0.25, priority="P0", action_policy="continue"))
    elif relation_type == "causal":
        drafts.append(make_policy(source, relation, threshold_delta=-0.10, priority="P1", action_policy="continue"))
        drafts.append(make_policy(target, relation, threshold_delta=-0.25, priority="P1", action_policy="continue"))
    elif relation_type == "bridge":
        drafts.append(make_policy(source, relation, threshold_delta=-0.10, priority="P1", action_policy="continue"))
        drafts.append(make_policy(target, relation, threshold_delta=-0.10, priority="P1", action_policy="continue"))
    elif relation_type == "unknown":
        return []
    else:
        drafts.append(make_policy(source, relation, threshold_delta=0.0, priority="P1", action_policy="continue"))
        drafts.append(make_policy(target, relation, threshold_delta=0.0, priority="P1", action_policy="continue"))

    return drafts


def iter_relations(schema: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = schema.get("relations", [])
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def convert_schema_to_policies(
    schema: Mapping[str, Any],
    min_confidence: float = 0.0,
    only_confirmed: bool = False,
) -> tuple[list[TopicPolicyDraft], list[dict[str, Any]]]:
    merged: dict[str, TopicPolicyDraft] = {}
    skipped: list[dict[str, Any]] = []

    for relation in iter_relations(schema):
        conf = confidence_of(relation)
        confirmed = bool(relation.get("confirmed_by_human", False))
        relation_type = normalize_relation_type(relation.get("relation_type"))

        if relation_type == "unknown":
            skipped.append({"relation": relation_name(relation), "reason": "unknown relation_type"})
            continue
        if conf < min_confidence:
            skipped.append({"relation": relation_name(relation), "reason": f"confidence {conf:.3f} < {min_confidence:.3f}"})
            continue
        if only_confirmed and not confirmed:
            skipped.append({"relation": relation_name(relation), "reason": "not confirmed_by_human"})
            continue

        for draft in relation_to_policy_drafts(relation):
            if draft.topic in merged:
                merged[draft.topic].merge(draft)
            else:
                merged[draft.topic] = draft

    policies = list(merged.values())
    policies.sort(key=lambda item: (item.priority or "P2", -item.threshold_delta, item.topic))
    for policy in policies:
        policy.clamp()
    return policies, skipped


def output_packet(schema: Mapping[str, Any], policies: list[TopicPolicyDraft], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "version": VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_schema_version": schema.get("version"),
        "source_schema_generator": schema.get("generator"),
        "adapter": "schema_policy_adapter.py",
        "feedback": [policy.to_feedback_item() for policy in policies],
        "skipped_relations": skipped,
        "usage": {
            "lif_memory": "python lif_memory.py --vault . --days 14 --mode daily --feedback-file lif_memory_adaptive_policy.json",
            "note": "This output intentionally uses the existing feedback-file schema so lif_memory.py does not need to be modified first.",
        },
    }


def render_report(schema: Mapping[str, Any], policies: list[TopicPolicyDraft], skipped: list[dict[str, Any]], output_path: Path) -> str:
    lines: list[str] = []
    lines.append("# LIF Schema Policy Adapter 报告")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"Schema generator：{schema.get('generator', 'unknown')}")
    lines.append(f"Relations：{len(iter_relations(schema))}")
    lines.append(f"Policies：{len(policies)}")
    lines.append(f"Skipped：{len(skipped)}")
    lines.append("")
    lines.append("## 核心作用")
    lines.append("")
    lines.append("这个适配器把自动归纳的 relation schema 转成 `lif_memory.py --feedback-file` 可读取的策略文件。")
    lines.append("")
    lines.append("```text")
    lines.append("adaptive relation schema -> topic policy -> threshold / priority / action_policy")
    lines.append("```")
    lines.append("")
    lines.append("也就是说，关系不再只出现在报告里，而是可以反过来改变 LIF-Memory 的触发方式。")
    lines.append("")
    lines.append("## 生成的 topic policies")
    lines.append("")
    lines.append("| Topic | Priority | Action policy | Threshold delta | Cooldown | Reason |")
    lines.append("|---|---|---|---:|---:|---|")
    for policy in policies:
        policy.clamp()
        lines.append(
            f"| {policy.topic} | {policy.priority or ''} | {policy.action_policy or ''} | "
            f"{policy.threshold_delta:.2f} | {policy.cooldown_days} | {'; '.join(policy.reasons[:2])} |"
        )
    lines.append("")
    if skipped:
        lines.append("## 跳过的 relations")
        lines.append("")
        for item in skipped:
            lines.append(f"- {item.get('relation')}: {item.get('reason')}")
        lines.append("")
    lines.append("## 下一步运行")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"python \"04 项目库\\P2_LIF-Memory\\lif_memory.py\" --vault \".\" --days 14 --mode daily --feedback-file \"{output_path.name}\" --output \"今日 LIF-Memory 主卡片.md\"")
    lines.append("```")
    lines.append("")
    lines.append("如果你只想使用人工确认过的 schema relation，重新运行适配器时加：")
    lines.append("")
    lines.append("```powershell")
    lines.append("python \"04 项目库\\P2_LIF-Memory\\schema_policy_adapter.py\" --schema \"lif_relation_schema.generated.json\" --only-confirmed")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    schema = load_json(args.schema)
    policies, skipped = convert_schema_to_policies(
        schema,
        min_confidence=args.min_confidence,
        only_confirmed=args.only_confirmed,
    )
    packet = output_packet(schema, policies, skipped)
    report = render_report(schema, policies, skipped, args.output)

    if args.dry_run:
        print(report)
        print(json.dumps(packet, ensure_ascii=False, indent=2))
        return

    write_json(args.output, packet)
    write_text(args.report, report)
    print(f"Loaded schema: {args.schema}")
    print(f"Generated policies: {len(policies)}")
    print(f"Skipped relations: {len(skipped)}")
    print(f"Wrote policy: {args.output}")
    print(f"Wrote report: {args.report}")


if __name__ == "__main__":
    main()
