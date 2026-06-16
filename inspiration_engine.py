from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import lif_memory as core

VERSION = "0.1.1"


@dataclass(frozen=True)
class InspirationConfig:
    theta: float = 8.0
    slow_decay: float = 0.94
    tension_decay: float = 0.86
    bridge_decay: float = 0.90
    spark_decay: float = 0.40
    reset_ratio: float = 0.35
    cooldown_days: int = 1
    evidence_cap: float = 7.2
    min_fragments: int = 4
    min_roles: int = 2
    min_distinct_days: int = 1


@dataclass
class InspirationFragment:
    day: date
    path: Path
    snippet: str
    score: float
    roles: list[str]
    domains: list[str]
    matched_keywords: list[str]

    def packet(self, vault: Path) -> dict[str, object]:
        return {
            "note": self.day.isoformat(),
            "path": core.md_link(self.path, vault),
            "snippet": self.snippet,
            "score": round(self.score, 2),
            "roles": self.roles,
            "domains": self.domains,
            "matched_keywords": self.matched_keywords,
        }


@dataclass
class DailyInspiration:
    input_total: float = 0.0
    tension_input: float = 0.0
    bridge_input: float = 0.0
    spark_input: float = 0.0
    validation_input: float = 0.0
    fragments: list[InspirationFragment] = field(default_factory=list)


@dataclass
class InspirationState:
    v: float = 0.0
    v_incubation: float = 0.0
    v_tension: float = 0.0
    v_bridge: float = 0.0
    v_spark: float = 0.0
    last_spike_date: date | None = None
    fragments: list[InspirationFragment] = field(default_factory=list)


@dataclass
class InspirationSpike:
    day: date
    voltage: float
    threshold: float
    fragments: list[InspirationFragment]
    v_incubation: float
    v_tension: float
    v_bridge: float
    v_spark: float
    dominant_domains: list[str]
    dominant_roles: list[str]
    claim_seed: str
    validation_action: str


ROLE_KEYWORDS: dict[str, list[str]] = {
    "tension": [
        "为什么",
        "难道",
        "不理解",
        "矛盾",
        "悖论",
        "冲突",
        "卡住",
        "差点意思",
        "困惑",
        "到底",
        "反而",
        "但是",
        "问题是",
    ],
    "bridge": [
        "类似",
        "像",
        "等价",
        "映射",
        "类比",
        "迁移",
        "联系",
        "可以理解为",
        "本质上",
        "机制",
    ],
    "compression": [
        "本质",
        "核心",
        "一句话",
        "压缩",
        "定义",
        "命名",
        "主线",
        "闭环",
        "归结",
        "收束",
        "抽象",
    ],
    "spark": [
        "灵感",
        "洞察",
        "突然",
        "意识到",
        "发现",
        "想到",
        "新",
        "突破",
        "第一次",
        "原来",
        "是不是",
        "能不能",
    ],
    "validation": [
        "验证",
        "测试",
        "实验",
        "数据",
        "图",
        "对比",
        "判据",
        "指标",
        "证明",
        "写成",
        "跑一次",
    ],
}

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "LIF": ["LIF", "脉冲", "spike", "事件", "阈值", "泄漏", "双时间尺度"],
    "Backscatter": ["后向散射", "反射系数", "USRP", "边带", "915", "无线", "标签"],
    "EEG": ["脑电", "EEG", "SSVEP", "节律", "事件率", "高斯", "恢复"],
    "NegativeResistance": ["负阻", "NDR", "斜率", "抵消", "偏置", "放大"],
    "Thesis": ["论文", "章节", "第三章", "第四章", "创新点", "证据链", "盲审"],
    "AI_Memory": ["AI", "大模型", "agent", "智能体", "记忆", "Obsidian", "RAG"],
    "Career": ["简历", "求职", "实习", "岗位", "投递", "项目经历"],
    "Economics": ["经济", "周期", "利率", "通胀", "债务", "金融危机", "就业"],
    "Health": ["焦虑", "难受", "睡", "身体", "压力", "害怕", "恢复"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect inspiration spikes from weak Obsidian note fragments.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path.")
    parser.add_argument("--days", type=int, default=30, help="Number of latest daily notes to scan.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--theta", type=float, default=8.0, help="Inspiration spike threshold.")
    parser.add_argument("--min-fragments", type=int, default=4, help="Minimum fragments required before emitting a spike.")
    parser.add_argument("--min-roles", type=int, default=2, help="Minimum distinct fragment roles required.")
    parser.add_argument("--daily-budget", type=int, default=1, help="Maximum inspiration spikes per day.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Inspiration 灵感回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF Inspiration Engine {VERSION}")
    return parser.parse_args()


def parse_cutoff(value: str | None) -> date:
    return date.today() if value is None else datetime.strptime(value, "%Y-%m-%d").date()


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def matched_words(text: str, words: Iterable[str]) -> list[str]:
    return core.matched_words(text, words)


def classify_roles(block: str) -> tuple[list[str], list[str]]:
    roles: list[str] = []
    hits: list[str] = []
    for role, words in ROLE_KEYWORDS.items():
        role_hits = matched_words(block, words)
        if role_hits:
            roles.append(role)
            hits.extend(role_hits)
    return roles, hits


def classify_domains(block: str) -> tuple[list[str], list[str]]:
    domains: list[str] = []
    hits: list[str] = []
    for domain, words in DOMAIN_KEYWORDS.items():
        domain_hits = matched_words(block, words)
        if domain_hits:
            domains.append(domain)
            hits.extend(domain_hits)
    return domains, hits


def fragment_score(block: str, roles: list[str], domains: list[str], role_hits: list[str], domain_hits: list[str]) -> float:
    if not roles and not domains:
        return 0.0

    score = 0.30
    role_weights = {
        "tension": 0.95,
        "bridge": 0.90,
        "compression": 0.80,
        "spark": 1.10,
        "validation": 0.55,
    }
    score += sum(role_weights.get(role, 0.0) for role in roles)
    score += min(len(domain_hits), 8) * 0.12
    score += max(0, len(domains) - 1) * 0.55
    score += min(len(role_hits), 8) * 0.08
    score += core.specificity_score(block) * 0.55

    if "tension" in roles and "bridge" in roles:
        score += 0.65
    if "compression" in roles and "validation" in roles:
        score += 0.40
    if "spark" in roles and ("tension" in roles or "bridge" in roles):
        score += 0.55

    return score


def block_to_fragment(day: date, path: Path, block: str) -> InspirationFragment | None:
    roles, role_hits = classify_roles(block)
    domains, domain_hits = classify_domains(block)
    score = fragment_score(block, roles, domains, role_hits, domain_hits)

    if score < 0.75:
        return None
    if not roles:
        roles = ["domain_evidence"]

    return InspirationFragment(
        day=day,
        path=path,
        snippet=core.short_reason(block, limit=120),
        score=score,
        roles=roles,
        domains=domains,
        matched_keywords=[*role_hits, *domain_hits][:12],
    )


def extract_daily_inspiration(day: date, path: Path, text: str, config: InspirationConfig) -> DailyInspiration:
    daily = DailyInspiration()
    for block in core.split_blocks(text):
        fragment = block_to_fragment(day, path, block)
        if fragment is None:
            continue

        daily.fragments.append(fragment)
        daily.input_total += fragment.score
        if "tension" in fragment.roles:
            daily.tension_input += fragment.score
        if "bridge" in fragment.roles or len(fragment.domains) >= 2:
            daily.bridge_input += fragment.score
        if "spark" in fragment.roles:
            daily.spark_input += fragment.score
        if "validation" in fragment.roles:
            daily.validation_input += fragment.score

    daily.input_total = min(daily.input_total, config.evidence_cap)
    daily.tension_input = min(daily.tension_input, config.evidence_cap * 0.80)
    daily.bridge_input = min(daily.bridge_input, config.evidence_cap * 0.80)
    daily.spark_input = min(daily.spark_input, config.evidence_cap * 0.65)
    daily.validation_input = min(daily.validation_input, config.evidence_cap * 0.60)
    daily.fragments.sort(key=lambda item: item.score, reverse=True)
    daily.fragments = daily.fragments[:10]
    return daily


def select_fragments(fragments: list[InspirationFragment], limit: int = 10) -> list[InspirationFragment]:
    seen: set[str] = set()
    ranked = sorted(
        fragments,
        key=lambda item: (
            "spark" in item.roles,
            "bridge" in item.roles,
            "tension" in item.roles,
            len(item.domains),
            item.score,
            item.day,
        ),
        reverse=True,
    )
    selected: list[InspirationFragment] = []
    for fragment in ranked:
        key = fragment.snippet
        if key in seen:
            continue
        seen.add(key)
        selected.append(fragment)
        if len(selected) >= limit:
            break
    return sorted(selected, key=lambda item: item.day)


def dominant_values(fragments: list[InspirationFragment], field_name: str, limit: int = 5) -> list[str]:
    counter: Counter[str] = Counter()
    for fragment in fragments:
        values = getattr(fragment, field_name)
        for value in values:
            counter[value] += 1
    return [value for value, _ in counter.most_common(limit)]


def enough_material(state: InspirationState, config: InspirationConfig) -> bool:
    fragments = select_fragments(state.fragments, limit=16)
    roles = {role for fragment in fragments for role in fragment.roles}
    days = {fragment.day for fragment in fragments}
    has_tension_or_bridge = bool({"tension", "bridge", "compression"} & roles)
    return (
        len(fragments) >= config.min_fragments
        and len(roles) >= config.min_roles
        and len(days) >= config.min_distinct_days
        and has_tension_or_bridge
    )


def can_spike(state: InspirationState, day: date, config: InspirationConfig) -> bool:
    if state.v < config.theta:
        return False
    if not enough_material(state, config):
        return False
    if state.last_spike_date is None:
        return True
    return (day - state.last_spike_date).days >= config.cooldown_days


def combine_voltage(state: InspirationState) -> float:
    return (
        0.35 * state.v_incubation
        + 0.25 * state.v_tension
        + 0.25 * state.v_bridge
        + 0.15 * state.v_spark
    )


def claim_seed_for(fragments: list[InspirationFragment], domains: list[str], roles: list[str]) -> str:
    domain_text = " / ".join(domains[:3]) if domains else "当前主题"
    role_text = " + ".join(roles[:3]) if roles else "弱证据积累"

    if "bridge" in roles and len(domains) >= 2:
        return f"把 {domain_text} 看成同一个机制链路，而不是孤立问题；当前灵感来自跨域映射：{role_text}。"
    if "tension" in roles and "compression" in roles:
        return f"把反复出现的张力压缩成一个可验证命题：{domain_text} 的核心问题需要被重新定义。"
    if "spark" in roles:
        return f"这个灵感不是凭空出现，而是旧片段长期孵化后被一个小线索点燃：{domain_text}。"
    return f"围绕 {domain_text} 的弱片段已经足够多，应该写成一张洞察卡，而不是继续散落在日记里。"


def validation_action_for(domains: list[str], roles: list[str]) -> str:
    domain_set = set(domains)
    if domain_set & {"LIF", "Backscatter", "EEG", "NegativeResistance", "Thesis"}:
        return "写一张“命题—证据—反例—验证动作”卡：一句核心命题，一组已有数据/现象，一个可能反例，一个 30 分钟内能做的验证。"
    if "AI_Memory" in domain_set:
        return "把灵感落成一次最小代码实验：新增一条规则或一个输出字段，跑一次回放，记录触发是否更合理。"
    if "Career" in domain_set:
        return "把灵感转成一个简历/项目表达：问题、方法、指标、结果、可展示产物各写一句。"
    if "Economics" in domain_set:
        return "写一张经济学对照卡：主流解释、反向解释、机制链条、下一次观察指标。"
    if "Health" in domain_set:
        return "先把灵感降载成一个恢复动作和一个最小任务，避免在高压状态下继续扩大规划。"
    if "validation" in roles:
        return "立刻补一个最小验证动作：一个判据、一条数据、一张图或一个反例。"
    return "写一张 150 字洞察卡：旧问题、触发线索、可能机制、下一步验证。"


def replay_inspiration(
    notes: list[tuple[date, Path]],
    config: InspirationConfig,
    daily_budget: int = 1,
) -> tuple[list[InspirationSpike], list[dict[str, object]], InspirationState]:
    state = InspirationState()
    spikes: list[InspirationSpike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = extract_daily_inspiration(day, path, text, config)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day

        old_v = state.v
        state.v_incubation = max(
            0.0,
            (config.slow_decay ** delta_days) * state.v_incubation
            + 0.35 * daily.input_total
            + 0.20 * daily.validation_input,
        )
        state.v_tension = max(0.0, (config.tension_decay ** delta_days) * state.v_tension + daily.tension_input)
        state.v_bridge = max(0.0, (config.bridge_decay ** delta_days) * state.v_bridge + daily.bridge_input)
        state.v_spark = max(0.0, (config.spark_decay ** delta_days) * state.v_spark + daily.spark_input)
        state.v = combine_voltage(state)

        if daily.fragments:
            state.fragments.extend(daily.fragments)
            state.fragments = select_fragments(state.fragments, limit=24)

        row: dict[str, object] = {
            "date": day.isoformat(),
            "path": path,
            "delta_days": delta_days,
            "old_v": old_v,
            "new_v": state.v,
            "v_incubation": state.v_incubation,
            "v_tension": state.v_tension,
            "v_bridge": state.v_bridge,
            "v_spark": state.v_spark,
            "input": daily.input_total,
            "tension_input": daily.tension_input,
            "bridge_input": daily.bridge_input,
            "spark_input": daily.spark_input,
            "fragments": len(state.fragments),
            "spike": False,
        }

        emitted_today = 0
        while emitted_today < max(0, daily_budget) and can_spike(state, day, config):
            fragments = select_fragments(state.fragments, limit=10)
            domains = dominant_values(fragments, "domains", limit=5)
            roles = dominant_values(fragments, "roles", limit=5)
            spike = InspirationSpike(
                day=day,
                voltage=state.v,
                threshold=config.theta,
                fragments=fragments,
                v_incubation=state.v_incubation,
                v_tension=state.v_tension,
                v_bridge=state.v_bridge,
                v_spark=state.v_spark,
                dominant_domains=domains,
                dominant_roles=roles,
                claim_seed=claim_seed_for(fragments, domains, roles),
                validation_action=validation_action_for(domains, roles),
            )
            spikes.append(spike)
            state.last_spike_date = day
            state.v_incubation *= config.reset_ratio
            state.v_tension *= config.reset_ratio
            state.v_bridge *= config.reset_ratio
            state.v_spark = 0.0
            state.v = combine_voltage(state)
            state.fragments = state.fragments[-4:]
            row["spike"] = True
            row["new_v"] = state.v
            emitted_today += 1

        timeline.append(row)

    return spikes, timeline, state


def inspiration_packet(spike: InspirationSpike, vault: Path) -> dict[str, object]:
    return {
        "spike_type": "Inspiration",
        "time": spike.day.isoformat(),
        "V": round(spike.voltage, 2),
        "threshold": round(spike.threshold, 2),
        "mechanism": {
            "meaning": "Inspiration is modeled as slow incubation plus unresolved tension plus cross-domain bridge plus a short spark cue.",
            "formula": "V = 0.35*V_incubation + 0.25*V_tension + 0.25*V_bridge + 0.15*V_spark",
            "V_incubation": round(spike.v_incubation, 2),
            "V_tension": round(spike.v_tension, 2),
            "V_bridge": round(spike.v_bridge, 2),
            "V_spark": round(spike.v_spark, 2),
        },
        "dominant_domains": spike.dominant_domains,
        "dominant_roles": spike.dominant_roles,
        "claim_seed": spike.claim_seed,
        "validation_action": spike.validation_action,
        "fragments": [fragment.packet(vault) for fragment in spike.fragments],
    }


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[InspirationSpike],
    timeline: list[dict[str, object]],
    state: InspirationState,
    config: InspirationConfig,
) -> str:
    lines: list[str] = []
    lines.append("# LIF-Inspiration 灵感机制回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")
    lines.append("## 灵感机制")
    lines.append("")
    lines.append("灵感不是随机生成的新句子，而是长期孵化的弱片段，在“张力、跨域桥接、压缩定义、突然线索”同时出现时越过阈值。")
    lines.append("")
    lines.append("```text")
    lines.append("weak fragments -> incubation")
    lines.append("unresolved tension -> V_tension")
    lines.append("cross-domain analogy/bridge -> V_bridge")
    lines.append("small cue / sudden formulation -> V_spark")
    lines.append("V = 0.35*incubation + 0.25*tension + 0.25*bridge + 0.15*spark")
    lines.append("```")
    lines.append("")
    lines.append(f"当前最终电位：{state.v:.2f} / 阈值：{config.theta:.2f}")
    lines.append("")
    lines.append("## Inspiration spikes")
    lines.append("")
    if not spikes:
        lines.append("本次没有灵感 spike。说明弱片段还没有同时形成足够的张力、桥接或压缩线索。")
        lines.append("")
    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Inspiration {index}: {spike.day.isoformat()}")
        lines.append("")
        lines.append(f"- 电位：{spike.voltage:.2f}")
        lines.append(f"- 阈值：{spike.threshold:.2f}")
        lines.append(f"- 主导领域：{', '.join(spike.dominant_domains) or '未识别'}")
        lines.append(f"- 主导角色：{', '.join(spike.dominant_roles) or '未识别'}")
        lines.append(f"- Claim seed：{spike.claim_seed}")
        lines.append(f"- 最小验证动作：{spike.validation_action}")
        lines.append("- 片段证据：")
        for fragment in spike.fragments:
            lines.append(
                f"  - `{'+'.join(fragment.roles)}` `{','.join(fragment.domains) or 'general'}` "
                f"[[{core.md_link(fragment.path, vault)}]] {fragment.day.isoformat()}：{fragment.snippet}"
            )
        lines.append("- JSON packet:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(inspiration_packet(spike, vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    lines.append("## 轨迹")
    lines.append("")
    lines.append("| 日期 | 日志 | V | incubation | tension | bridge | spark | input | frag |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in timeline:
        path = row["path"]
        assert isinstance(path, Path)
        mark = " *" if row.get("spike") else ""
        lines.append(
            f"| {row['date']} | [[{core.md_link(path, vault)}]] | "
            f"{float(row['new_v']):.2f}{mark} | "
            f"{float(row['v_incubation']):.2f} | "
            f"{float(row['v_tension']):.2f} | "
            f"{float(row['v_bridge']):.2f} | "
            f"{float(row['v_spark']):.2f} | "
            f"{float(row['input']):.2f} | "
            f"{int(row['fragments'])} |"
        )
    lines.append("")
    lines.append("说明：带 `*` 的日期表示当天生成过 inspiration spike。灵感卡必须落到一个最小验证动作，否则只算想法噪声。")
    lines.append("")
    return "\n".join(lines)


def resolve_output(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def write_json_output(path: Path, vault: Path, spikes: list[InspirationSpike]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packets = [inspiration_packet(spike, vault) for spike in spikes]
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    notes = core.find_daily_notes(vault, cutoff, args.days)
    config = InspirationConfig(
        theta=args.theta,
        min_fragments=args.min_fragments,
        min_roles=args.min_roles,
    )
    spikes, timeline, state = replay_inspiration(notes, config, daily_budget=args.daily_budget)
    report = render_markdown(vault, notes, spikes, timeline, state, config)

    if args.dry_run:
        print(report)
    else:
        output = resolve_output(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Scanned {len(notes)} notes.")
        print(f"Generated {len(spikes)} inspiration spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_output(vault, args.json_output)
    if json_output is not None:
        write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
