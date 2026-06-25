from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import llm_adapter
except Exception:  # pragma: no cover - optional local adapter
    llm_adapter = None  # type: ignore[assignment]


VERSION = "0.10.0-aha-engine"

BLOCKER_WORDS = [
    "卡", "不知道", "缺", "没有", "不够", "失败", "问题", "害怕", "拖延", "延毕", "焦虑", "混乱", "难受", "动不了",
    "矛盾", "冲突", "质疑", "漏洞", "解释不了", "说不通", "不连续", "漂移",
]
COMPLETION_WORDS = ["完成", "做完", "已经", "搞定", "跑通", "出结果", "保存", "写完", "提交", "测出来"]
ACTION_WORDS = ["今天", "今晚", "下一步", "需要", "必须", "应该", "测试", "修改", "写", "整理", "推进", "验证", "闭环"]
AHA_TERMS = ["顿悟", "灵光一闪", "Aha", "aha", "重构", "旧模型", "新模型", "解释", "belief", "update"]
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}|[\u4e00-\u9fff]{2,}")
STOP_TERMS = {
    "今天", "现在", "这个", "那个", "就是", "然后", "需要", "应该", "感觉", "什么", "一个", "自己",
    "进行", "目前", "接下来", "问题", "系统", "输出", "the", "and", "with", "that", "this", "from",
}


@dataclass(frozen=True)
class EvidenceSlice:
    source: str
    text: str
    score: float = 0.0
    kind: str = "evidence"

    def to_packet(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AhaInputs:
    query: str
    spike: Mapping[str, Any]
    evidence: Sequence[EvidenceSlice]
    high_loss: Sequence[EvidenceSlice]
    field_context: Mapping[str, Any] = field(default_factory=dict)

    def compact_text(self, limit: int = 3600) -> str:
        parts: list[str] = []
        if self.query:
            parts.append(f"QUERY: {self.query}")
        topic = self.spike.get("topic") or self.field_context.get("topic") or ""
        if topic:
            parts.append(f"TOPIC: {topic}")
        for item in list(self.evidence)[:6]:
            parts.append(f"EVIDENCE[{item.kind}] {item.source}: {item.text}")
        for item in list(self.high_loss)[:6]:
            parts.append(f"HIGH_LOSS {item.source}: {item.text}")
        text = "\n".join(parts)
        return text if len(text) <= limit else text[: limit - 1] + "…"


@dataclass
class AhaCard:
    aha_id: str
    query: str
    topic: str
    trigger_type: str
    pressure_ratio: float
    reconstruction_pressure: float
    old_model: str
    contradiction: str
    new_model: str
    essence: str
    action_delta: str
    falsification_test: str
    evidence: list[EvidenceSlice]
    quality_score: float
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    version: str = VERSION

    def to_packet(self) -> dict[str, Any]:
        packet = asdict(self)
        packet["evidence"] = [item.to_packet() for item in self.evidence]
        return packet


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, Mapping):
        return " ".join(normalize_text(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(normalize_text(item) for item in value)
    return str(value)


def short_text(value: Any, limit: int = 220) -> str:
    text = normalize_text(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def load_json(path: Path | None) -> Any:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_from_packet(data: Any, preferred_keys: Sequence[str]) -> list[Any]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, Mapping):
        for key in preferred_keys:
            value = data.get(key)
            if isinstance(value, list):
                return value
        return [data]
    return []


def normalize_spikes(data: Any, field_data: Any = None) -> list[dict[str, Any]]:
    spikes = [item for item in list_from_packet(data, ["lif_spikes", "spikes", "cards", "items"]) if isinstance(item, Mapping)]
    if spikes:
        return [dict(item) for item in spikes]
    if isinstance(field_data, Mapping):
        if field_data.get("spike") or field_data.get("top_hits"):
            return [
                {
                    "spike_id": f"field::{field_data.get('topic', 'custom')}",
                    "topic": field_data.get("topic", "自定义问题场"),
                    "spike_type": "Field",
                    "primary_state": "AI_Memory",
                    "V": field_data.get("reconstruction_pressure", field_data.get("field_energy", 0.0)),
                    "threshold": 1.0,
                    "evidence_notes": field_data.get("top_hits", []),
                    "trigger_reason": field_data.get("insight_card", "连续问题场已经形成可重构压力。"),
                }
            ]
    return []


def evidence_from_spike(spike: Mapping[str, Any]) -> list[EvidenceSlice]:
    raw_items = spike.get("evidence_notes") or spike.get("top_hits") or spike.get("evidence") or []
    items: list[EvidenceSlice] = []
    if isinstance(raw_items, list):
        for raw in raw_items:
            if isinstance(raw, Mapping):
                source = str(raw.get("path") or raw.get("source") or raw.get("title") or raw.get("note") or "evidence")
                text = short_text(raw.get("snippet") or raw.get("text") or raw.get("reason") or raw)
                score = float(raw.get("score", raw.get("semantic", 0.0)) or 0.0)
            else:
                source = "evidence"
                text = short_text(raw)
                score = 0.0
            if text:
                items.append(EvidenceSlice(source=source, text=text, score=score, kind="spike"))
    trigger = short_text(spike.get("trigger_reason") or spike.get("suggested_action") or "")
    if trigger:
        items.append(EvidenceSlice(source="spike.trigger", text=trigger, kind="trigger"))
    return items[:8]


def high_loss_from_reconstruction(data: Any) -> list[EvidenceSlice]:
    raw_items = list_from_packet(data, ["high_loss_observations", "high_loss", "high_reconstruction_loss"])
    items: list[EvidenceSlice] = []
    for raw in raw_items:
        if isinstance(raw, Mapping):
            observation = raw.get("observation", raw)
            source = str(raw.get("source") or normalize_text(observation.get("source") if isinstance(observation, Mapping) else "") or raw.get("slot_id") or "high_loss")
            text = short_text((observation.get("text") if isinstance(observation, Mapping) else observation) or raw)
            score = float(raw.get("reconstruction_loss", raw.get("loss", raw.get("score", 0.0))) or 0.0)
        else:
            source = "high_loss"
            text = short_text(raw)
            score = 0.0
        if text:
            items.append(EvidenceSlice(source=source, text=text, score=score, kind="high_loss"))
    return items[:8]


def field_hits_from_packet(data: Any) -> list[EvidenceSlice]:
    hits = list_from_packet(data, ["top_hits", "hits"])
    items: list[EvidenceSlice] = []
    for raw in hits:
        if not isinstance(raw, Mapping):
            continue
        source = str(raw.get("path") or raw.get("title") or "field_hit")
        text = short_text(raw.get("snippet") or raw)
        score = float(raw.get("score", 0.0) or 0.0)
        if text:
            items.append(EvidenceSlice(source=source, text=text, score=score, kind="field"))
    return items[:8]


def pressure_ratio(spike: Mapping[str, Any]) -> float:
    v = spike.get("V", spike.get("voltage", 0.0))
    theta = spike.get("threshold", 1.0)
    try:
        v_f = float(v)
        theta_f = float(theta)
    except (TypeError, ValueError):
        return 0.0
    if theta_f <= 0:
        return 0.0
    return round(v_f / theta_f, 3)


def reconstruction_pressure(high_loss: Sequence[EvidenceSlice], field_data: Mapping[str, Any] | None = None) -> float:
    values = [max(0.0, float(item.score)) for item in high_loss if item.score]
    if field_data:
        for key in ["reconstruction_pressure", "field_energy"]:
            raw = field_data.get(key)
            if isinstance(raw, (int, float)):
                values.append(float(raw))
    if not values:
        return 0.0
    return round(min(1.0, sum(values) / max(len(values), 1)), 3)


def infer_topic(spike: Mapping[str, Any], query: str, field_data: Mapping[str, Any] | None = None) -> str:
    for value in [spike.get("topic"), spike.get("primary_state"), field_data.get("topic") if field_data else None]:
        if value:
            return str(value)
    if contains_any(query, ["论文", "盲审", "创新点", "第四章"]):
        return "论文闭环"
    if contains_any(query, ["LIF", "脑电", "EEG", "后向散射"]):
        return "LIF链路"
    if contains_any(query, ["记忆", "顿悟", "灵光一闪", "Aha", "agent"]):
        return "AI记忆"
    return "自定义问题"


def top_terms(slices: Sequence[EvidenceSlice], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for item in slices:
        for token in TOKEN_RE.findall(item.text):
            token = token.lower()
            if token not in STOP_TERMS and len(token) >= 2:
                counter[token] += 1 + item.score
    return [term for term, _ in counter.most_common(limit)]


def infer_old_model(topic: str, query: str, evidence_text: str) -> str:
    combined = " ".join([topic, query, evidence_text])
    if contains_any(combined, ["顿悟", "灵光一闪", "Aha", "记忆", "agent", "LIF-Memory"]):
        return "我原来以为只要增加记忆数量、检索强度、角色数量或 LIF 神经元，系统就会自然产生人的灵光一闪。"
    if contains_any(combined, ["论文", "创新点", "盲审", "第四章"]):
        return "我原来以为论文卡住主要是因为实验还不够多，或者某个模块还没有完全做完。"
    if contains_any(combined, ["实验", "负阻", "USRP", "波形", "数据"]):
        return "我原来以为继续做更多实验、测更多波形，系统就会自然闭环。"
    if contains_any(combined, ["焦虑", "难受", "延毕", "压力", "身体"]):
        return "我原来以为只要继续想清楚、继续规划，就能从当前失控状态里恢复行动。"
    return "我原来把当前问题理解成一个需要更多信息、更多努力或更多模块的问题。"


def infer_contradiction(inputs: AhaInputs) -> str:
    slices = list(inputs.high_loss) + list(inputs.evidence)
    conflict_items = [item for item in slices if contains_any(item.text, BLOCKER_WORDS)]
    if inputs.high_loss:
        selected = list(inputs.high_loss)[:3]
        joined = "；".join(f"{item.source}: {item.text}" for item in selected)
        return f"当前证据中存在高重建误差或旧 slot 难以解释的片段：{joined}"
    if conflict_items:
        selected = conflict_items[:3]
        joined = "；".join(f"{item.source}: {item.text}" for item in selected)
        return f"这些片段反复出现阻塞/矛盾信号，说明旧解释不能直接导出行动：{joined}"
    if inputs.evidence:
        selected = inputs.evidence[:3]
        joined = "；".join(f"{item.source}: {item.text}" for item in selected)
        return f"证据已经足够触发 spike，但它们仍停留在提醒和总结层面：{joined}"
    return "系统已经触发 spike，但缺少能说明旧解释为什么失败的显式证据链。"


def infer_new_model(topic: str, query: str, evidence_text: str) -> str:
    combined = " ".join([topic, query, evidence_text])
    if contains_any(combined, ["顿悟", "灵光一闪", "Aha", "记忆", "LIF-Memory", "agent"]):
        return "顿悟不是检索增强，也不是角色辩论变多；顿悟是 spike 之后用冲突证据推翻旧模型，并生成一个更高压缩率的新解释。"
    if contains_any(combined, ["论文", "创新点", "盲审", "第四章"]):
        return "论文真正需要的不是继续扩大系统，而是把最小证明链路压缩成导师和盲审能接受的一个主张、一组证据和一个边界条件。"
    if contains_any(combined, ["实验", "负阻", "USRP", "波形", "数据"]):
        return "实验闭环不是做完所有可能实验，而是隔离出一个可写入论文的判据：可用、不可用、降级或作为补充。"
    if contains_any(combined, ["焦虑", "难受", "延毕", "压力", "身体"]):
        return "当前最优策略不是继续解释人生，而是先降低生理和情绪负荷，再恢复一个最小任务闭环。"
    return "当前问题的关键不是增加材料，而是显式更新解释模型：从“更多信息”转向“更强压缩、更少动作、更可验证”。"


def infer_essence(topic: str, query: str, new_model: str) -> str:
    combined = " ".join([topic, query, new_model])
    if contains_any(combined, ["顿悟", "灵光一闪", "Aha", "LIF-Memory"]):
        return "Spike 不是顿悟；spike 只是打开解释重构的门。"
    if contains_any(combined, ["论文", "创新点"]):
        return "论文的本质不是系统很大，而是最小主张能被证据闭环支撑。"
    if contains_any(combined, ["实验", "数据", "负阻"]):
        return "实验的本质不是继续测，而是得到一个能改变论文决策的判据。"
    return "真正的洞察必须把旧解释压缩成一个能改变行动的新解释。"


def infer_action_delta(spike: Mapping[str, Any], topic: str, query: str, new_model: str) -> str:
    suggestion = normalize_text(spike.get("suggested_action") or spike.get("completion_target") or "")
    combined = " ".join([topic, query, new_model])
    if contains_any(combined, ["顿悟", "灵光一闪", "Aha", "LIF-Memory"]):
        return "下一版不要继续增加角色或神经元；先把 AhaEngine 接到 spike 后面，强制每次输出 old_model、contradiction、new_model、action_delta 和 falsification_test。"
    if suggestion:
        return f"不要把 spike 只当提醒；按新模型把下一步缩成一个会改变状态的动作：{suggestion}"
    return "下一步只做一个能验证新模型的动作，并记录它是否真的改变了后续决策。"


def infer_falsification_test(topic: str, query: str) -> str:
    combined = " ".join([topic, query])
    if contains_any(combined, ["顿悟", "灵光一闪", "Aha", "LIF-Memory"]):
        return "连续运行 3 次同类 spike：如果输出仍只是总结/提醒而不能稳定给出“我原来以为 X，其实是 Y”，则这次升级失败。"
    if contains_any(combined, ["论文", "创新点"]):
        return "把新模型写成一句论文主张给导师或自己审稿：如果不能推出一张图/一个实验/一个限制条件，则该洞察失败。"
    return "如果这个新解释不能让下一步行动比原来更小、更明确、更可验证，则该洞察失败。"


def quality_score(card: Mapping[str, Any]) -> float:
    score = 0.0
    for key in ["old_model", "contradiction", "new_model", "essence", "action_delta", "falsification_test"]:
        value = normalize_text(card.get(key))
        if len(value) >= 12:
            score += 1.0
    action_delta = normalize_text(card.get("action_delta"))
    falsification = normalize_text(card.get("falsification_test"))
    if contains_any(action_delta, ACTION_WORDS):
        score += 0.7
    if contains_any(falsification, ["如果", "失败", "验证", "不能", "连续", "则"]):
        score += 0.7
    return round(min(1.0, score / 7.4), 3)


def heuristic_card(inputs: AhaInputs) -> AhaCard:
    topic = infer_topic(inputs.spike, inputs.query, inputs.field_context)
    pressure = pressure_ratio(inputs.spike)
    recon = reconstruction_pressure(inputs.high_loss, inputs.field_context)
    evidence_text = inputs.compact_text()
    old_model = infer_old_model(topic, inputs.query, evidence_text)
    contradiction = infer_contradiction(inputs)
    new_model = infer_new_model(topic, inputs.query, evidence_text)
    essence = infer_essence(topic, inputs.query, new_model)
    action_delta = infer_action_delta(inputs.spike, topic, inputs.query, new_model)
    falsification = infer_falsification_test(topic, inputs.query)
    trigger_type = "high_loss+spike" if inputs.high_loss and pressure >= 1.0 else "high_loss" if inputs.high_loss else "spike"
    aha_id = normalize_text(inputs.spike.get("spike_id")) or f"{datetime.now().date().isoformat()}-{topic}-aha"
    card = AhaCard(
        aha_id=aha_id,
        query=inputs.query,
        topic=topic,
        trigger_type=trigger_type,
        pressure_ratio=pressure,
        reconstruction_pressure=recon,
        old_model=old_model,
        contradiction=contradiction,
        new_model=new_model,
        essence=essence,
        action_delta=action_delta,
        falsification_test=falsification,
        evidence=list(inputs.evidence[:5]) + list(inputs.high_loss[:3]),
        quality_score=0.0,
    )
    card.quality_score = quality_score(card.to_packet())
    return card


def llm_prompt(inputs: AhaInputs, draft: AhaCard) -> list[dict[str, str]]:
    system = (
        "你是 LIF-Memory 的 AhaEngine。你的任务不是总结，也不是安慰，而是把 spike 转换成顿悟卡。"
        "顿悟必须包含：旧模型、冲突证据、新模型、行动改变、可证伪测试。"
        "必须只输出 JSON object，不要输出 Markdown。"
    )
    user = {
        "task": "Upgrade this draft AhaCard using only the supplied evidence. Preserve the schema.",
        "rule": "If there is no action_delta or falsification_test, the card is invalid.",
        "input": {
            "query": inputs.query,
            "spike": dict(inputs.spike),
            "evidence": [item.to_packet() for item in inputs.evidence],
            "high_loss": [item.to_packet() for item in inputs.high_loss],
            "field_context": dict(inputs.field_context),
        },
        "draft": draft.to_packet(),
        "required_schema": {
            "old_model": "string",
            "contradiction": "string",
            "new_model": "string",
            "essence": "string",
            "action_delta": "string",
            "falsification_test": "string",
        },
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def maybe_refine_with_llm(inputs: AhaInputs, draft: AhaCard, args: argparse.Namespace) -> AhaCard:
    if not getattr(args, "llm_synthesize", False) or llm_adapter is None:
        return draft
    try:
        config = llm_adapter.config_from_args(args)
        content = llm_adapter.call_chat_completions(config, llm_prompt(inputs, draft))
        data = llm_adapter.extract_json_object(content)
    except Exception:
        return draft

    packet = draft.to_packet()
    for key in ["old_model", "contradiction", "new_model", "essence", "action_delta", "falsification_test"]:
        value = normalize_text(data.get(key))
        if value:
            packet[key] = value
    packet["quality_score"] = quality_score(packet)
    return AhaCard(
        aha_id=str(packet["aha_id"]),
        query=str(packet["query"]),
        topic=str(packet["topic"]),
        trigger_type=str(packet["trigger_type"]),
        pressure_ratio=float(packet["pressure_ratio"]),
        reconstruction_pressure=float(packet["reconstruction_pressure"]),
        old_model=str(packet["old_model"]),
        contradiction=str(packet["contradiction"]),
        new_model=str(packet["new_model"]),
        essence=str(packet["essence"]),
        action_delta=str(packet["action_delta"]),
        falsification_test=str(packet["falsification_test"]),
        evidence=draft.evidence,
        quality_score=float(packet["quality_score"]),
        created_at=str(packet["created_at"]),
        version=str(packet["version"]),
    )


def build_aha_cards(
    query: str,
    spike_data: Any,
    reconstruction_data: Any = None,
    field_data: Any = None,
    top_k: int = 3,
    args: argparse.Namespace | None = None,
) -> list[AhaCard]:
    field_context = field_data if isinstance(field_data, Mapping) else {}
    spikes = normalize_spikes(spike_data, field_data=field_context)
    high_loss = high_loss_from_reconstruction(reconstruction_data)
    field_evidence = field_hits_from_packet(field_context)
    cards: list[AhaCard] = []

    for spike in spikes[: max(1, top_k)]:
        evidence = evidence_from_spike(spike)
        if not evidence:
            evidence = list(field_evidence)
        inputs = AhaInputs(query=query, spike=spike, evidence=evidence, high_loss=high_loss, field_context=field_context)
        draft = heuristic_card(inputs)
        cards.append(maybe_refine_with_llm(inputs, draft, args) if args else draft)

    if not cards and (high_loss or field_evidence):
        pseudo_spike = {
            "spike_id": f"{datetime.now().date().isoformat()}-AhaEngine-field-only",
            "topic": infer_topic({}, query, field_context),
            "V": field_context.get("reconstruction_pressure", 1.0) if field_context else 1.0,
            "threshold": 1.0,
            "evidence_notes": [item.to_packet() for item in field_evidence],
            "trigger_reason": "没有显式 spike，但重建误差或连续问题场已经足够形成 Aha 候选。",
        }
        inputs = AhaInputs(query=query, spike=pseudo_spike, evidence=field_evidence, high_loss=high_loss, field_context=field_context)
        draft = heuristic_card(inputs)
        cards.append(maybe_refine_with_llm(inputs, draft, args) if args else draft)

    cards.sort(key=lambda card: (card.quality_score, card.pressure_ratio, card.reconstruction_pressure), reverse=True)
    return cards[:top_k]


def demo_spikes() -> list[dict[str, Any]]:
    return [
        {
            "spike_id": "2026-06-25-AI_Memory-AhaEngine",
            "topic": "AI记忆",
            "primary_state": "AI_Memory",
            "V": 8.7,
            "threshold": 7.2,
            "trigger_reason": "LIF-Memory 已经能触发提醒，但用户仍觉得没有人的灵光一闪。",
            "suggested_action": "补一条规则、跑一次回放、记录触发是否合理。",
            "evidence_notes": [
                {
                    "path": "2026-06-25.md",
                    "snippet": "系统已经有 LIF 电压、连续问题场、多角色辩论，但输出仍然像总结和提醒。",
                    "score": 2.1,
                }
            ],
        }
    ]


def demo_reconstruction() -> dict[str, Any]:
    return {
        "high_loss_observations": [
            {
                "slot_id": "slot_0",
                "reconstruction_loss": 0.86,
                "observation": {
                    "source": "2026-06-25.md",
                    "text": "我希望 LIF-Memory 有人的灵光一闪，但它现在只是把已有材料综合得很深刻。",
                },
            }
        ]
    }


def render_markdown(cards: Sequence[AhaCard]) -> str:
    lines: list[str] = [
        "# LIF-Memory AhaEngine 顿悟卡",
        "",
        f"- Version: `{VERSION}`",
        f"- Cards: `{len(cards)}`",
        "",
        "> Spike 不是顿悟；Spike 只是打开解释重构的门。",
        "",
    ]
    if not cards:
        lines.extend(["本次没有生成 AhaCard。", ""])
        return "\n".join(lines)

    for index, card in enumerate(cards, 1):
        lines.extend(
            [
                f"## {index}. {card.topic}",
                "",
                f"- Aha ID: `{card.aha_id}`",
                f"- Trigger: `{card.trigger_type}`",
                f"- Pressure ratio: `{card.pressure_ratio:.2f}`",
                f"- Reconstruction pressure: `{card.reconstruction_pressure:.2f}`",
                f"- Quality: `{card.quality_score:.2f}`",
                "",
                "### 旧模型",
                "",
                card.old_model,
                "",
                "### 冲突证据",
                "",
                card.contradiction,
                "",
                "### 新模型",
                "",
                card.new_model,
                "",
                "### 一句话本质",
                "",
                f"**{card.essence}**",
                "",
                "### 行动改变",
                "",
                card.action_delta,
                "",
                "### 可证伪测试",
                "",
                card.falsification_test,
                "",
                "### 证据切片",
                "",
            ]
        )
        for item in card.evidence:
            lines.append(f"- `{item.kind}` `{item.source}`：{item.text}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LIF spikes into Aha/BELIEF-UPDATE cards.")
    parser.add_argument("--spikes", type=Path, default=None, help="JSON output from lif_memory.py --json-output.")
    parser.add_argument("--reconstruction", type=Path, default=None, help="Optional JSON output from unsupervised_memory_field.py.")
    parser.add_argument("--field", type=Path, default=None, help="Optional JSON output from continuous_problem_field.py.")
    parser.add_argument("--query", type=str, default="", help="Original user question / coordinate for this Aha run.")
    parser.add_argument("--top-k", type=int, default=3, help="Maximum Aha cards to generate.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory AhaCards.md"), help="Markdown report output path.")
    parser.add_argument("--json-output", type=Path, default=Path("lif_aha_cards.json"), help="JSON AhaCard output path.")
    parser.add_argument("--llm-synthesize", action="store_true", help="Use configured LLM to refine the deterministic AhaCard.")
    parser.add_argument("--demo", action="store_true", help="Run built-in demo input.")
    parser.add_argument("--version", action="version", version=f"AhaEngine {VERSION}")
    if llm_adapter is not None:
        llm_adapter.add_cli_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo or (args.spikes is None and args.reconstruction is None and args.field is None):
        spike_data = demo_spikes()
        reconstruction_data = demo_reconstruction()
        field_data = None
        query = args.query or "为什么 LIF-Memory 还没有人的灵光一闪？"
    else:
        spike_data = load_json(args.spikes)
        reconstruction_data = load_json(args.reconstruction)
        field_data = load_json(args.field)
        query = args.query

    cards = build_aha_cards(
        query=query,
        spike_data=spike_data,
        reconstruction_data=reconstruction_data,
        field_data=field_data,
        top_k=max(1, args.top_k),
        args=args,
    )
    markdown = render_markdown(cards)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote: {args.output}")
    else:
        print(markdown)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([card.to_packet() for card in cards], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Wrote JSON: {args.json_output}")


if __name__ == "__main__":
    main()
