from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import lif_memory as core

VERSION = "0.8.0"


ROLE_WORDS: dict[str, list[str]] = {
    "claim": [
        "我认为",
        "本质",
        "意味着",
        "创新",
        "创新点",
        "核心",
        "可以理解为",
        "应该表述为",
        "主张",
        "证明",
        "定义",
    ],
    "conflict": [
        "但是",
        "可是",
        "然而",
        "矛盾",
        "冲突",
        "不理解",
        "质疑",
        "难道",
        "问题",
        "不对",
        "不够",
        "说不通",
    ],
    "failure": [
        "失败",
        "测不到",
        "没有测出来",
        "没有放大",
        "不稳定",
        "不稳",
        "乱跳",
        "卡住",
        "做不出来",
        "推进不下去",
        "差点意思",
    ],
    "metric": [
        "Hz",
        "hz",
        "mV",
        "mv",
        "Vpp",
        "RMS",
        "uA",
        "μA",
        "PSD",
        "FFT",
        "事件率",
        "压缩比",
        "阈值",
        "sigma",
        "采样",
        "数据",
        "截图",
        "曲线",
    ],
    "decision": [
        "决定",
        "转向",
        "降级",
        "主线",
        "补充",
        "暂不",
        "接下来",
        "下一步",
        "只做",
        "收束",
        "隔离",
        "关闭",
        "保留",
    ],
    "emotion": [
        "焦虑",
        "害怕",
        "难受",
        "崩",
        "动不了",
        "失败了",
        "压力",
        "延毕",
        "丢脸",
        "撕扯",
    ],
    "externality": [
        "导师",
        "实验室",
        "宿舍",
        "毕业",
        "延毕",
        "简历",
        "投递",
        "岗位",
        "公司",
        "面试",
        "时间",
    ],
    "question": [
        "为什么",
        "如何",
        "到底",
        "怎么办",
        "能不能",
        "是不是",
        "该不该",
        "怎么证明",
    ],
    "novelty": [
        "新",
        "第一次",
        "突然",
        "发现",
        "洞察",
        "涌现",
        "重新理解",
        "关系",
        "连接",
    ],
}


EXTRA_TOPIC_RULES: dict[str, list[str]] = {
    "经济周期": ["经济周期", "周期", "衰退", "复苏", "就业", "库存", "危机", "需求", "供给"],
    "债务金融": ["债务", "杠杆", "信用", "融资", "资产负债表", "银行", "地产", "违约"],
    "激励制度": ["激励", "制度", "产权", "博弈", "监管", "寻租", "道德风险", "委托代理"],
}


@dataclass(frozen=True)
class RelationRule:
    name: str
    relation_type: str
    source_topics: tuple[str, ...]
    threshold: float
    required_roles: tuple[str, ...]
    boost_roles: tuple[str, ...]
    emergent_claim: str
    next_validation_action: str
    priority: str = "P1"


@dataclass
class RelationEvidence:
    day: date
    path: Path
    snippet: str
    topics: list[str]
    roles: list[str]
    matched_keywords: list[str]
    score: float

    def packet(self, vault: Path) -> dict[str, object]:
        return {
            "note": self.day.isoformat(),
            "path": core.md_link(self.path, vault),
            "snippet": self.snippet,
            "topics": self.topics,
            "roles": self.roles,
            "matched_keywords": self.matched_keywords,
            "score": round(self.score, 2),
        }


@dataclass
class RelationSpike:
    name: str
    relation_type: str
    source_topics: list[str]
    day: date
    score: float
    threshold: float
    priority: str
    emergent_claim: str
    next_validation_action: str
    evidence_chain: list[RelationEvidence]
    role_counts: dict[str, int]
    topic_scores: dict[str, float]
    bridge_count: int = 0
    status: str = "open"

    def spike_id(self) -> str:
        safe_name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "-", self.name).strip("-")
        return f"{self.day.isoformat()}-Relation-{safe_name}"


RELATION_RULES: list[RelationRule] = [
    RelationRule(
        name="负阻降级与主线收束",
        relation_type="downgrade_or_reframe",
        source_topics=("负阻", "论文闭环", "LIF链路"),
        threshold=6.8,
        required_roles=("failure", "conflict", "decision"),
        boost_roles=("metric", "claim", "novelty"),
        priority="P0",
        emergent_claim=(
            "负阻更适合被降级为非线性前端探索或补充实验；论文主线应收束到 "
            "EEG→LIF 事件化→后向散射承载→接收端节律检测。"
        ),
        next_validation_action=(
            "写一页主线判定：为什么负阻不再承担主创新，为什么 LIF+后向散射可以作为最小可证明链路。"
        ),
    ),
    RelationRule(
        name="实验数据到论文证据链",
        relation_type="support_chain",
        source_topics=("实验数据模板", "LIF链路", "论文闭环"),
        threshold=6.4,
        required_roles=("metric", "claim"),
        boost_roles=("decision", "novelty"),
        priority="P0",
        emergent_claim=(
            "当前最有价值的不是继续增加模块，而是把已有阈值、事件率、PSD、边带能量等数据 "
            "整理成可进入论文的证据链。"
        ),
        next_validation_action=(
            "生成一张最小闭环表：输入/阈值、事件数、事件率、压缩比、无线恢复峰、结论。"
        ),
    ),
    RelationRule(
        name="AI记忆项目转向求职表达",
        relation_type="career_reframe",
        source_topics=("AI求职转向", "AI记忆", "LIF链路"),
        threshold=6.2,
        required_roles=("claim", "decision"),
        boost_roles=("metric", "novelty", "externality"),
        priority="P1",
        emergent_claim=(
            "LIF-Memory 不只是个人工具，也可以被包装成 AI+嵌入式/事件驱动记忆系统项目，服务简历和作品集表达。"
        ),
        next_validation_action=(
            "写一个简历项目条目：问题、方法、LIF 状态变量、反馈闭环、可扩展关系层。"
        ),
    ),
    RelationRule(
        name="健康压力对实验执行的抑制",
        relation_type="inhibition",
        source_topics=("健康恢复", "实验数据模板", "论文闭环"),
        threshold=6.0,
        required_roles=("emotion", "externality"),
        boost_roles=("decision", "failure"),
        priority="P0",
        emergent_claim=(
            "如果情绪/身体压力持续和实验、论文同时出现，系统应优先输出 recover_first，"
            "否则会把真实执行瓶颈误判为技术问题。"
        ),
        next_validation_action=(
            "先做 10 分钟恢复动作，再只选择一个 30 分钟可完成任务；不要在高压状态重新规划全部人生。"
        ),
    ),
    RelationRule(
        name="经济现象机制化",
        relation_type="mechanism_link",
        source_topics=("经济周期", "债务金融", "激励制度"),
        threshold=6.4,
        required_roles=("claim", "conflict"),
        boost_roles=("question", "novelty"),
        priority="P2",
        emergent_claim=(
            "经济学洞察不应停留在单个事件判断，而应把周期、债务和激励约束连成机制链。"
        ),
        next_validation_action=(
            "写一张机制卡：现象、参与者、约束、反馈循环、反例。"
        ),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect cross-topic relation spikes from Obsidian notes.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=30, help="Number of latest daily notes to scan.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Relation-Spike 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON relation-spike packet output.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Extra minimum score override. 0 uses each rule threshold.")
    parser.add_argument("--top-k-evidence", type=int, default=10, help="Evidence items kept per relation spike.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF Relation Spike {VERSION}")
    return parser.parse_args()


def parse_cutoff(value: str | None) -> date:
    return date.today() if value is None else datetime.strptime(value, "%Y-%m-%d").date()


def topic_rules() -> dict[str, list[str]]:
    rules: dict[str, list[str]] = {}
    if hasattr(core, "TOPIC_RULES"):
        rules.update(core.TOPIC_RULES)
    rules.update(EXTRA_TOPIC_RULES)
    return rules


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


def detect_topics(block: str, rules: Mapping[str, list[str]]) -> tuple[list[str], list[str]]:
    topics: list[str] = []
    keywords: list[str] = []
    for topic, words in rules.items():
        hits = matched_words(block, words)
        if hits:
            topics.append(topic)
            keywords.extend(hits)
    return topics, list(dict.fromkeys(keywords))


def detect_roles(block: str) -> list[str]:
    roles: list[str] = []
    for role, words in ROLE_WORDS.items():
        if contains_any(block, words):
            roles.append(role)

    if re.search(r"\d+(\.\d+)?\s*(Hz|hz|mV|mv|V|Vpp|RMS|uA|μA|ms|s)\b", block):
        if "metric" not in roles:
            roles.append("metric")
    return roles


def evidence_score(topics: list[str], roles: list[str], matched_keywords: list[str]) -> float:
    score = 0.6
    score += min(len(topics), 4) * 0.55
    score += min(len(matched_keywords), 8) * 0.12
    score += min(len(roles), 5) * 0.35
    if "conflict" in roles or "failure" in roles:
        score += 0.35
    if "metric" in roles:
        score += 0.25
    if "decision" in roles:
        score += 0.25
    if len(topics) >= 2:
        score += 0.55
    return score


def extract_relation_evidence(notes: list[tuple[date, Path]], rules: Mapping[str, list[str]]) -> list[RelationEvidence]:
    evidence: list[RelationEvidence] = []
    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for block in core.split_blocks(text):
            topics, keywords = detect_topics(block, rules)
            roles = detect_roles(block)
            if not topics:
                continue
            if not roles:
                roles = ["evidence"]
            item = RelationEvidence(
                day=day,
                path=path,
                snippet=core.short_reason(block, limit=128),
                topics=topics,
                roles=roles,
                matched_keywords=keywords[:12],
                score=evidence_score(topics, roles, keywords),
            )
            evidence.append(item)
    return evidence


def relation_score(rule: RelationRule, evidence: list[RelationEvidence]) -> tuple[float, dict[str, int], dict[str, float], int]:
    topic_scores: dict[str, float] = {topic: 0.0 for topic in rule.source_topics}
    role_counts: Counter[str] = Counter()
    bridge_count = 0

    for item in evidence:
        topic_overlap = [topic for topic in rule.source_topics if topic in item.topics]
        if not topic_overlap:
            continue
        if len(topic_overlap) >= 2:
            bridge_count += 1
        for topic in topic_overlap:
            topic_scores[topic] += item.score
        for role in item.roles:
            role_counts[role] += 1

    covered_topics = sum(1 for value in topic_scores.values() if value > 0)
    coverage_ratio = covered_topics / max(len(rule.source_topics), 1)
    score = coverage_ratio * 4.0
    score += min(bridge_count, 4) * 0.65

    for role in rule.required_roles:
        if role_counts.get(role, 0) > 0:
            score += 0.85

    for role in rule.boost_roles:
        if role_counts.get(role, 0) > 0:
            score += 0.35

    if role_counts.get("conflict", 0) and role_counts.get("decision", 0):
        score += 0.65
    if role_counts.get("failure", 0) and role_counts.get("claim", 0):
        score += 0.55
    if role_counts.get("metric", 0) and role_counts.get("claim", 0):
        score += 0.45

    return score, dict(role_counts), topic_scores, bridge_count


def select_evidence(rule: RelationRule, evidence: list[RelationEvidence], limit: int) -> list[RelationEvidence]:
    candidates: list[RelationEvidence] = []
    for item in evidence:
        if any(topic in item.topics for topic in rule.source_topics):
            candidates.append(item)

    def rank(item: RelationEvidence) -> tuple[float, int, int, date]:
        topic_overlap = len([topic for topic in rule.source_topics if topic in item.topics])
        role_overlap = len([role for role in item.roles if role in rule.required_roles or role in rule.boost_roles])
        return (item.score, topic_overlap, role_overlap, item.day)

    seen: set[str] = set()
    selected: list[RelationEvidence] = []
    for item in sorted(candidates, key=rank, reverse=True):
        key = item.snippet
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: item.day)


def detect_relation_spikes(
    evidence: list[RelationEvidence],
    rules: list[RelationRule],
    min_score: float = 0.0,
    top_k_evidence: int = 10,
) -> list[RelationSpike]:
    spikes: list[RelationSpike] = []
    if not evidence:
        return spikes

    latest_day = max(item.day for item in evidence)

    for rule in rules:
        score, role_counts, topic_scores, bridge_count = relation_score(rule, evidence)
        threshold = max(rule.threshold, min_score) if min_score > 0 else rule.threshold
        covered_topics = [topic for topic, value in topic_scores.items() if value > 0]
        has_required_role = any(role_counts.get(role, 0) > 0 for role in rule.required_roles)
        if score < threshold:
            continue
        if len(covered_topics) < 2:
            continue
        if not has_required_role:
            continue

        chain = select_evidence(rule, evidence, limit=top_k_evidence)
        spikes.append(
            RelationSpike(
                name=rule.name,
                relation_type=rule.relation_type,
                source_topics=list(rule.source_topics),
                day=latest_day,
                score=score,
                threshold=threshold,
                priority=rule.priority,
                emergent_claim=rule.emergent_claim,
                next_validation_action=rule.next_validation_action,
                evidence_chain=chain,
                role_counts=role_counts,
                topic_scores=topic_scores,
                bridge_count=bridge_count,
            )
        )

    spikes.sort(key=lambda item: (item.priority == "P0", item.score), reverse=True)
    return spikes


def relation_packet(spike: RelationSpike, vault: Path) -> dict[str, object]:
    return {
        "spike_type": "Relation",
        "spike_id": spike.spike_id(),
        "name": spike.name,
        "relation_type": spike.relation_type,
        "time": spike.day.isoformat(),
        "score": round(spike.score, 2),
        "threshold": round(spike.threshold, 2),
        "priority": spike.priority,
        "source_topics": spike.source_topics,
        "role_counts": spike.role_counts,
        "topic_scores": {key: round(value, 2) for key, value in spike.topic_scores.items()},
        "bridge_count": spike.bridge_count,
        "emergent_claim": spike.emergent_claim,
        "next_validation_action": spike.next_validation_action,
        "evidence_chain": [item.packet(vault) for item in spike.evidence_chain],
        "status": spike.status,
    }


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    evidence: list[RelationEvidence],
    spikes: list[RelationSpike],
) -> str:
    lines: list[str] = []
    lines.append("# LIF Relation Spike 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append(f"关系证据片段数：{len(evidence)}")
    lines.append("")
    lines.append("## 这次验证的不是状态，而是关系")
    lines.append("")
    lines.append("旧的 LIF-Memory 主要判断：哪个状态电位超过阈值。")
    lines.append("")
    lines.append("Relation Spike 判断：多个主题虽然单独看不一定足够强，但它们之间是否形成了新的支持、抑制、降级、转向或机制关系。")
    lines.append("")
    lines.append("```text")
    lines.append("fragments -> topic/role evidence -> cross-topic relation score -> relation spike")
    lines.append("```")
    lines.append("")

    topic_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    bridge_counter = 0
    for item in evidence:
        topic_counter.update(item.topics)
        role_counter.update(item.roles)
        if len(item.topics) >= 2:
            bridge_counter += 1

    lines.append("## 证据概览")
    lines.append("")
    lines.append("| 项目 | 数值 |")
    lines.append("|---|---:|")
    lines.append(f"| Evidence fragments | {len(evidence)} |")
    lines.append(f"| Bridge fragments | {bridge_counter} |")
    lines.append(f"| Topic types | {len(topic_counter)} |")
    lines.append(f"| Role types | {len(role_counter)} |")
    lines.append("")
    if topic_counter:
        lines.append("Top topics：" + "，".join(f"{topic}({count})" for topic, count in topic_counter.most_common(8)))
        lines.append("")
    if role_counter:
        lines.append("Top roles：" + "，".join(f"{role}({count})" for role, count in role_counter.most_common(8)))
        lines.append("")

    lines.append("## Relation spikes")
    lines.append("")
    if not spikes:
        lines.append("本次没有 relation spike。可能原因：跨主题证据不足、缺少 conflict/decision/metric 等角色，或关系阈值过高。")
        lines.append("")
        lines.append("可尝试：增加 `--days 90`，或把相关日记中“失败/决策/数据/主线变化”写得更具体。")
        lines.append("")

    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Relation {index}: {spike.name}")
        lines.append("")
        lines.append(f"- Spike ID：`{spike.spike_id()}`")
        lines.append(f"- Relation type：{spike.relation_type}")
        lines.append(f"- Priority：{spike.priority}")
        lines.append(f"- Score：{spike.score:.2f} / {spike.threshold:.2f}")
        lines.append(f"- Source topics：{', '.join(spike.source_topics)}")
        lines.append(f"- Bridge fragments：{spike.bridge_count}")
        lines.append(f"- Role counts：{json.dumps(spike.role_counts, ensure_ascii=False)}")
        lines.append("- 涌现判断：")
        lines.append(f"  - {spike.emergent_claim}")
        lines.append("- 最小验证动作：")
        lines.append(f"  - {spike.next_validation_action}")
        lines.append("- 证据链：")
        for item in spike.evidence_chain:
            lines.append(
                f"  - `{item.day.isoformat()}` [[{core.md_link(item.path, vault)}]] "
                f"topics={item.topics} roles={item.roles}：{item.snippet}"
            )
        lines.append("- JSON packet:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(relation_packet(spike, vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## 使用建议")
    lines.append("")
    lines.append("- `relation_spike.py` 不替代 `lif_memory.py`，它是关系层。")
    lines.append("- 先用 `lif_memory.py --mode daily` 找今天最该处理的一件事。")
    lines.append("- 再用 `relation_spike.py --days 30` 看最近问题之间有没有出现主线转向、降级、支持链或抑制关系。")
    lines.append("- 真正值得保存的不是提醒，而是能改变下一步策略的 relation spike。")
    lines.append("")

    return "\n".join(lines)


def resolve_output(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def write_json_output(path: Path, vault: Path, spikes: list[RelationSpike]) -> None:
    packets = [relation_packet(spike, vault) for spike in spikes]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    notes = core.find_daily_notes(vault, cutoff, args.days)
    rules = topic_rules()
    evidence = extract_relation_evidence(notes, rules)
    spikes = detect_relation_spikes(
        evidence,
        RELATION_RULES,
        min_score=args.min_score,
        top_k_evidence=args.top_k_evidence,
    )
    report = render_markdown(vault, notes, evidence, spikes)

    if args.dry_run:
        print(report)
    else:
        output = resolve_output(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Scanned {len(notes)} notes.")
        print(f"Extracted {len(evidence)} relation evidence fragments.")
        print(f"Generated {len(spikes)} relation spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_output(vault, args.json_output)
    if json_output is not None:
        write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
