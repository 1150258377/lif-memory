from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

VERSION = "0.10.0"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]")
DATE_RE = re.compile(r"(20\d{2})[-_/年](\d{1,2})[-_/月](\d{1,2})")

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "Thesis": ["论文", "毕业", "延毕", "导师", "创新", "答辩", "盲审", "系统", "实验闭环", "chapter", "thesis"],
    "LIF_Memory": ["lif memory", "记忆", "memory", "engram", "灵光一闪", "洞察", "连续", "离散", "向量库", "embedding"],
    "Experiment": ["实验", "电路", "pcb", "负阻", "ks1092", "示波器", "uspr", "usrp", "后向散射", "lif", "脑电", "心电"],
    "Career": ["工作", "简历", "实习", "岗位", "求职", "国企", "央企", "offer", "career"],
    "Health": ["减肥", "饮食", "健身", "焦虑", "难受", "睡眠", "腰痛", "health"],
    "System": ["代码", "github", "接口", "api", "配置", "运行", "bug", "python", "json"],
}

EMOTION_KEYWORDS: dict[str, list[str]] = {
    "anxious": ["焦虑", "慌", "难受", "害怕", "迷茫", "压力", "崩", "担心"],
    "blocked": ["卡", "不理解", "不会", "说不通", "矛盾", "混乱", "不知道"],
    "decisive": ["确定", "决定", "必须", "今晚", "今天", "马上", "推进", "下单"],
    "curious": ["为什么", "怎么做到", "原理", "本质", "洞察", "启发"],
    "stable": ["完成", "闭环", "可以", "没问题", "成功", "验证"],
}

CONFLICT_GROUPS: list[tuple[str, str, list[str], list[str]]] = [
    ("EEG_vs_ECG", "EEG 脑电主线与 ECG 心电降难度路线之间存在方向冲突", ["脑电", "eeg", "bci"], ["心电", "ecg"]),
    ("Sparse_vs_Async", "稀疏性证明与异步性证明之间存在论证焦点冲突", ["稀疏", "压缩", "compression"], ["异步", "事件", "时间化", "async"]),
    ("Build_vs_Insight", "做实验/写代码与提炼本质洞察之间存在行动冲突", ["实验", "pcb", "代码", "调试", "下单"], ["本质", "洞察", "创新", "意义"]),
    ("Recall_vs_Reconstruct", "静态召回与动态重建之间存在记忆系统路线冲突", ["检索", "向量库", "摘要", "保存"], ["重建", "激活", "连续", "灵光一闪"]),
]


@dataclass
class MemoryEvent:
    """One compressed memory event extracted from raw notes, sessions or JSON packets."""

    event_id: str
    topic: str
    claim: str
    problem: str = ""
    decision: str = ""
    evidence: list[str] = field(default_factory=list)
    emotion: str = "neutral"
    importance: int = 3
    created_at: str = ""
    updated_at: str = ""
    source: str = "unknown"
    status: str = "active"  # active / uncertain / outdated / contradicted
    confidence: float = 0.65
    tags: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join([self.topic, self.claim, self.problem, self.decision, " ".join(self.evidence), " ".join(self.tags)])


@dataclass
class ActivatedMemory:
    event_id: str
    topic: str
    score: float
    current: float
    voltage: float
    fired: bool
    reason: str
    event: MemoryEvent


@dataclass
class ConflictPacket:
    conflict_id: str
    kind: str
    topic: str
    description: str
    event_ids: list[str]
    severity: float


@dataclass
class InsightCard:
    topic: str
    pattern: str
    tension: str
    next_step: str
    supporting_event_ids: list[str]
    confidence: float


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(normalize_text(v) for v in value.values())
    return str(value)


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;\n])\s*", text)
    return [part.strip(" -\t\n") for part in parts if len(part.strip()) >= 4]


def tokens(text: str) -> set[str]:
    raw = [match.group(0).lower() for match in WORD_RE.finditer(text)]
    out: set[str] = set()
    cjk_buffer: list[str] = []
    for tok in raw:
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            cjk_buffer.append(tok)
            out.add(tok)
            continue
        if cjk_buffer:
            out.update("".join(cjk_buffer[i : i + 2]) for i in range(max(0, len(cjk_buffer) - 1)))
            out.update("".join(cjk_buffer[i : i + 3]) for i in range(max(0, len(cjk_buffer) - 2)))
            cjk_buffer = []
        out.add(tok)
    if cjk_buffer:
        out.update("".join(cjk_buffer[i : i + 2]) for i in range(max(0, len(cjk_buffer) - 1)))
        out.update("".join(cjk_buffer[i : i + 3]) for i in range(max(0, len(cjk_buffer) - 2)))
    return {tok for tok in out if tok}


def keyword_hits(text: str, words: Iterable[str]) -> int:
    low = text.lower()
    return sum(low.count(word.lower()) for word in words)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def stable_id(*parts: str) -> str:
    raw = "\n".join(part for part in parts if part)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        match = DATE_RE.search(value)
        if not match:
            return None
        year, month, day = (int(group) for group in match.groups())
        return datetime(year, month, day, tzinfo=timezone.utc)


def infer_topic(text: str) -> str:
    scores = {topic: keyword_hits(text, words) for topic, words in TOPIC_KEYWORDS.items()}
    topic, score = max(scores.items(), key=lambda kv: kv[1])
    return topic if score > 0 else "General"


def infer_emotion(text: str) -> str:
    scores = {name: keyword_hits(text, words) for name, words in EMOTION_KEYWORDS.items()}
    emotion, score = max(scores.items(), key=lambda kv: kv[1])
    return emotion if score > 0 else "neutral"


def infer_importance(text: str, topic: str, emotion: str) -> int:
    score = 2.0
    score += min(keyword_hits(text, ["必须", "今天", "今晚", "截止", "毕业", "论文", "核心", "闭环", "下单", "github"]), 5) * 0.45
    if topic in {"Thesis", "Experiment", "LIF_Memory"}:
        score += 0.5
    if emotion in {"anxious", "blocked", "decisive"}:
        score += 0.5
    if len(text) > 280:
        score += 0.35
    return int(round(clamp(score, 1, 5)))


def pick_sentence(sentences: list[str], keywords: Iterable[str], fallback: str = "") -> str:
    scored = []
    for sentence in sentences:
        scored.append((keyword_hits(sentence, keywords), min(len(sentence), 160), sentence))
    scored = [item for item in scored if item[0] > 0]
    if scored:
        return max(scored, key=lambda item: (item[0], item[1]))[2][:220]
    return fallback[:220]


def extract_created_at(text: str, fallback: str = "") -> str:
    dt = parse_datetime(fallback) or parse_datetime(text)
    return dt.isoformat(timespec="seconds") if dt else now_iso()


def memory_event_from_text(text: str, source: str = "unknown", metadata: dict[str, Any] | None = None) -> MemoryEvent | None:
    text = normalize_text(text)
    if len(text) < 8:
        return None
    metadata = metadata or {}
    sentences = split_sentences(text)
    if not sentences:
        sentences = [text[:240]]

    topic = normalize_text(metadata.get("topic")) or infer_topic(text)
    claim = normalize_text(metadata.get("claim")) or pick_sentence(
        sentences,
        ["认为", "核心", "本质", "主线", "创新", "目标", "方案", "结论", "应该", "需要"],
        fallback=sentences[0],
    )
    problem = normalize_text(metadata.get("problem")) or pick_sentence(
        sentences,
        ["问题", "卡", "不理解", "疑惑", "担心", "困难", "矛盾", "为什么", "怎么", "质疑", "说不通"],
    )
    decision = normalize_text(metadata.get("decision")) or pick_sentence(
        sentences,
        ["决定", "下一步", "今天", "今晚", "必须", "马上", "需要", "应该", "闭环", "下单", "提交"],
    )
    evidence = []
    for key in ["evidence", "sources", "support", "context"]:
        value = metadata.get(key)
        if isinstance(value, list):
            evidence.extend(normalize_text(item)[:180] for item in value if normalize_text(item))
        elif isinstance(value, str):
            evidence.append(value[:180])
    if not evidence:
        evidence = sentences[:2]

    created = normalize_text(metadata.get("created_at") or metadata.get("timestamp") or metadata.get("time") or metadata.get("date"))
    created_at = extract_created_at(text, created)
    emotion = normalize_text(metadata.get("emotion")) or infer_emotion(text)
    importance_raw = metadata.get("importance")
    if isinstance(importance_raw, (int, float)):
        importance = int(clamp(float(importance_raw), 1, 5))
    else:
        importance = infer_importance(text, topic, emotion)

    event_id = normalize_text(metadata.get("event_id") or metadata.get("id")) or stable_id(source, created_at, topic, claim, problem)
    confidence = float(clamp(float(metadata.get("confidence", 0.65)) if isinstance(metadata.get("confidence", 0.65), (int, float)) else 0.65))
    status = normalize_text(metadata.get("status")) or "active"
    tags = [topic, emotion]
    if status != "active":
        tags.append(status)

    return MemoryEvent(
        event_id=event_id,
        topic=topic,
        claim=claim,
        problem=problem,
        decision=decision,
        evidence=evidence[:5],
        emotion=emotion,
        importance=importance,
        created_at=created_at,
        updated_at=now_iso(),
        source=source,
        status=status,
        confidence=confidence,
        tags=sorted(set(tags)),
    )


def iter_json_records(value: Any, source: str) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, list):
        for i, item in enumerate(value):
            if isinstance(item, dict):
                yield from iter_json_records(item, f"{source}[{i}]")
            else:
                yield f"{source}[{i}]", {"text": normalize_text(item)}
        return
    if not isinstance(value, dict):
        yield source, {"text": normalize_text(value)}
        return

    text_keys = ["text", "content", "body", "message", "query", "answer", "summary", "claim", "problem", "decision"]
    has_text = any(isinstance(value.get(key), (str, int, float, bool)) for key in text_keys)
    if has_text:
        yield source, value

    for key in ["events", "messages", "sessions", "conversations", "items", "records", "selected_actions", "selected_salience", "observations", "nodes", "tasks", "claims", "tensions"]:
        child = value.get(key)
        if isinstance(child, (list, dict)):
            yield from iter_json_records(child, f"{source}.{key}")


def load_markdown_chunks(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    heading_chunks = re.split(r"\n(?=#{1,6}\s+)", text)
    chunks = []
    for chunk in heading_chunks:
        chunk = chunk.strip()
        if len(chunk) >= 20:
            chunks.append(chunk)
    if chunks:
        return chunks
    return [part.strip() for part in re.split(r"\n\s*\n", text) if len(part.strip()) >= 20]


def load_events_from_path(path: Path) -> list[MemoryEvent]:
    events: list[MemoryEvent] = []
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        packet = json.loads(path.read_text(encoding="utf-8"))
        for record_source, record in iter_json_records(packet, path.name):
            text = normalize_text(record.get("text") or record.get("content") or record.get("body") or record)
            event = memory_event_from_text(text, source=record_source, metadata=record)
            if event:
                events.append(event)
    else:
        for i, chunk in enumerate(load_markdown_chunks(path)):
            event = memory_event_from_text(chunk, source=f"{path.name}#{i}")
            if event:
                events.append(event)
    return dedupe_events(events)


def dedupe_events(events: Iterable[MemoryEvent]) -> list[MemoryEvent]:
    by_id: dict[str, MemoryEvent] = {}
    for event in events:
        old = by_id.get(event.event_id)
        if old is None or (event.importance, event.confidence) >= (old.importance, old.confidence):
            by_id[event.event_id] = event
    return list(by_id.values())


def load_state(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"version": VERSION, "events": [], "voltages": {}, "conflicts": []}
    return json.loads(path.read_text(encoding="utf-8"))


def events_from_state(state: dict[str, Any]) -> list[MemoryEvent]:
    events = []
    for item in state.get("events", []):
        if isinstance(item, dict):
            try:
                events.append(MemoryEvent(**{**asdict(MemoryEvent(event_id="", topic="", claim="")), **item}))
            except TypeError:
                event = memory_event_from_text(normalize_text(item), source="state", metadata=item)
                if event:
                    events.append(event)
    return events


def merge_events(existing: Iterable[MemoryEvent], new_events: Iterable[MemoryEvent]) -> list[MemoryEvent]:
    by_id = {event.event_id: event for event in existing}
    for event in new_events:
        old = by_id.get(event.event_id)
        if not old:
            by_id[event.event_id] = event
            continue
        merged = MemoryEvent(
            **{
                **asdict(old),
                "claim": event.claim or old.claim,
                "problem": event.problem or old.problem,
                "decision": event.decision or old.decision,
                "evidence": list(dict.fromkeys(old.evidence + event.evidence))[:5],
                "importance": max(old.importance, event.importance),
                "updated_at": now_iso(),
                "confidence": max(old.confidence, event.confidence),
                "tags": sorted(set(old.tags + event.tags)),
            }
        )
        by_id[event.event_id] = merged
    return list(by_id.values())


class MemoryActivationEngine:
    def __init__(self, leak: float = 0.82, threshold: float = 0.72, reset_ratio: float = 0.35) -> None:
        self.leak = leak
        self.threshold = threshold
        self.reset_ratio = reset_ratio

    def score_current(self, query: str, event: MemoryEvent) -> tuple[float, str]:
        q_tokens = tokens(query)
        e_tokens = tokens(event.text)
        overlap = len(q_tokens & e_tokens) / max(1, min(len(q_tokens), 48))
        semantic = clamp(overlap * 1.7)
        importance = event.importance / 5.0
        confidence = event.confidence
        topic_bonus = 0.12 if infer_topic(query) == event.topic else 0.0
        emotion_bonus = 0.10 if event.emotion in {"anxious", "blocked", "decisive"} else 0.0
        status_penalty = {"active": 0.0, "uncertain": 0.08, "outdated": 0.18, "contradicted": 0.22}.get(event.status, 0.05)
        age_bonus = self._recency_bonus(event.created_at)
        current = clamp(0.46 * semantic + 0.20 * importance + 0.12 * confidence + topic_bonus + emotion_bonus + 0.10 * age_bonus - status_penalty)
        reason = f"semantic={semantic:.2f}, importance={importance:.2f}, confidence={confidence:.2f}, recency={age_bonus:.2f}, status={event.status}"
        return round(current, 4), reason

    def _recency_bonus(self, created_at: str) -> float:
        dt = parse_datetime(created_at)
        if not dt:
            return 0.25
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        days = max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds() / 86400.0)
        return clamp(math.exp(-days / 30.0))

    def activate(self, query: str, events: list[MemoryEvent], voltages: dict[str, float] | None = None, top_k: int = 12) -> tuple[list[ActivatedMemory], dict[str, float]]:
        voltages = dict(voltages or {})
        activated: list[ActivatedMemory] = []
        for event in events:
            current, reason = self.score_current(query, event)
            previous = float(voltages.get(event.event_id, 0.0))
            voltage = previous * self.leak + current
            dynamic_threshold = self.threshold - 0.035 * (event.importance - 3)
            fired = voltage >= dynamic_threshold
            stored_voltage = voltage * self.reset_ratio if fired else voltage
            voltages[event.event_id] = round(stored_voltage, 4)
            score = round(0.72 * current + 0.28 * min(voltage, 1.2), 4)
            activated.append(ActivatedMemory(event.event_id, event.topic, score, current, round(voltage, 4), fired, reason, event))
        activated.sort(key=lambda item: (item.fired, item.score, item.event.importance), reverse=True)
        return activated[:top_k], voltages


def detect_conflicts(events: list[MemoryEvent]) -> list[ConflictPacket]:
    conflicts: list[ConflictPacket] = []
    for kind, description, left_words, right_words in CONFLICT_GROUPS:
        left = [event for event in events if keyword_hits(event.text, left_words) > 0]
        right = [event for event in events if keyword_hits(event.text, right_words) > 0]
        if not left or not right:
            continue
        topic_counter = Counter(event.topic for event in left + right)
        topic = topic_counter.most_common(1)[0][0]
        ids = [event.event_id for event in sorted(left + right, key=lambda item: item.importance, reverse=True)[:6]]
        severity = clamp((len(left) + len(right)) / max(4, len(events)) + 0.25)
        conflicts.append(ConflictPacket(stable_id(kind, " ".join(ids)), kind, topic, description, ids, round(severity, 4)))
    return conflicts


def generate_insights(query: str, activated: list[ActivatedMemory], conflicts: list[ConflictPacket]) -> list[InsightCard]:
    cards: list[InsightCard] = []
    if not activated:
        return cards
    topic_groups: dict[str, list[ActivatedMemory]] = defaultdict(list)
    for item in activated:
        topic_groups[item.topic].append(item)

    for topic, group in sorted(topic_groups.items(), key=lambda kv: sum(item.score for item in kv[1]), reverse=True)[:3]:
        top = sorted(group, key=lambda item: item.score, reverse=True)[:4]
        problems = [item.event.problem for item in top if item.event.problem]
        decisions = [item.event.decision for item in top if item.event.decision]
        pattern = build_pattern(topic, top, problems)
        conflict = next((conf for conf in conflicts if conf.topic == topic or any(event.event_id in conf.event_ids for event in top)), None)
        tension = conflict.description if conflict else build_tension(topic, top)
        next_step = build_next_step(topic, query, decisions)
        confidence = round(clamp(sum(item.score for item in top) / max(1, len(top)) + 0.08 * len(top)), 4)
        cards.append(InsightCard(topic, pattern, tension, next_step, [item.event_id for item in top], confidence))
    return cards


def build_pattern(topic: str, group: list[ActivatedMemory], problems: list[str]) -> str:
    fired_count = sum(1 for item in group if item.fired)
    if problems:
        return f"{topic} 中反复被激活的不是孤立事实，而是同一类阻塞：{problems[0]}"
    return f"{topic} 中有 {len(group)} 条相关记忆被激活，其中 {fired_count} 条达到 spike 阈值。"


def build_tension(topic: str, group: list[ActivatedMemory]) -> str:
    emotions = Counter(item.event.emotion for item in group)
    top_emotion = emotions.most_common(1)[0][0]
    if top_emotion in {"anxious", "blocked"}:
        return f"当前张力主要不是信息不足，而是 {topic} 已经积累到需要决策或降维处理。"
    return f"当前张力来自多个旧判断在新问题下重新组合，需要把可执行结论从历史材料里压缩出来。"


def build_next_step(topic: str, query: str, decisions: list[str]) -> str:
    if decisions:
        return f"先复用最近已经形成的决策：{decisions[0]}"
    if topic == "Thesis":
        return "把当前问题压成一页：核心主张、三条证据、一个缺口、今晚最小动作。"
    if topic == "LIF_Memory":
        return "把本轮高分记忆固化为 MemoryEvent，并标注 active / uncertain / contradicted，避免只做相似检索。"
    if topic == "Experiment":
        return "先跑一个最小闭环验证，不扩展新功能；只保留输入、事件、输出、可复现实验记录。"
    if topic == "Career":
        return "把项目经历改写成岗位可读的三段：硬件能力、系统闭环、AI/嵌入式迁移价值。"
    return "把被激活的记忆写成一个可执行问题，然后只推进最小下一步。"


def build_packet(query: str, events: list[MemoryEvent], state: dict[str, Any], top_k: int = 12) -> tuple[dict[str, Any], dict[str, Any]]:
    conflicts = detect_conflicts(events)
    engine = MemoryActivationEngine()
    activated, voltages = engine.activate(query, events, state.get("voltages", {}), top_k=top_k)
    insights = generate_insights(query, activated, conflicts)
    packet = {
        "version": VERSION,
        "created_at": now_iso(),
        "query": query,
        "event_count": len(events),
        "activated": [serialize_activation(item) for item in activated],
        "conflicts": [asdict(item) for item in conflicts],
        "insights": [asdict(item) for item in insights],
        "state_summary": {
            "active_events": sum(1 for event in events if event.status == "active"),
            "uncertain_events": sum(1 for event in events if event.status == "uncertain"),
            "contradicted_events": sum(1 for event in events if event.status == "contradicted"),
            "topics": dict(Counter(event.topic for event in events)),
        },
    }
    new_state = {
        "version": VERSION,
        "updated_at": now_iso(),
        "events": [asdict(event) for event in events],
        "voltages": voltages,
        "conflicts": [asdict(item) for item in conflicts],
    }
    return packet, new_state


def serialize_activation(item: ActivatedMemory) -> dict[str, Any]:
    data = asdict(item)
    data["event"] = asdict(item.event)
    return data


def render_markdown(packet: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# LIF Dynamic Memory Event Report v{packet['version']}")
    lines.append("")
    lines.append("```text")
    lines.append("raw notes / sessions -> MemoryEvent -> LIF-style activation -> conflict state -> insight cards")
    lines.append("```")
    lines.append("")
    lines.append(f"- query: `{packet['query']}`")
    lines.append(f"- created_at: `{packet['created_at']}`")
    lines.append(f"- event_count: `{packet['event_count']}`")
    lines.append("")
    lines.append("## Insight cards")
    lines.append("")
    if not packet["insights"]:
        lines.append("No insight card was generated. Add more events or lower the activation threshold.")
    for i, card in enumerate(packet["insights"], 1):
        lines.append(f"### {i}. {card['topic']}")
        lines.append("")
        lines.append(f"- pattern: {card['pattern']}")
        lines.append(f"- tension: {card['tension']}")
        lines.append(f"- next_step: {card['next_step']}")
        lines.append(f"- confidence: `{card['confidence']}`")
        lines.append(f"- supporting_event_ids: `{', '.join(card['supporting_event_ids'])}`")
        lines.append("")
    lines.append("## Activated memories")
    lines.append("")
    for i, item in enumerate(packet["activated"], 1):
        event = item["event"]
        spike = "YES" if item["fired"] else "no"
        lines.append(f"### {i}. {event['topic']} / spike={spike}")
        lines.append("")
        lines.append(f"- score: `{item['score']}`")
        lines.append(f"- voltage: `{item['voltage']}`")
        lines.append(f"- claim: {event['claim']}")
        if event.get("problem"):
            lines.append(f"- problem: {event['problem']}")
        if event.get("decision"):
            lines.append(f"- decision: {event['decision']}")
        lines.append(f"- reason: {item['reason']}")
        lines.append(f"- source: `{event['source']}`")
        lines.append("")
    if packet["conflicts"]:
        lines.append("## Conflict state")
        lines.append("")
        for conflict in packet["conflicts"]:
            lines.append(f"- `{conflict['kind']}` severity={conflict['severity']}: {conflict['description']}")
        lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("This module does not train a model. It is a local, auditable controller layer that compresses raw history into typed MemoryEvents, then uses LIF-style activation and conflict detection to reconstruct the current context.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic MemoryEvent layer for LIF-Memory.")
    parser.add_argument("--input", type=Path, nargs="*", default=[], help="JSON or Markdown files to extract MemoryEvents from.")
    parser.add_argument("--query", type=str, required=True, help="Current question used to activate memory events.")
    parser.add_argument("--state", type=Path, default=Path("lif_memory_events.json"), help="Persistent local MemoryEvent state file.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Dynamic-Memory-Report.md"), help="Markdown report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON packet output path.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of activated memories to show.")
    parser.add_argument("--no-state-update", action="store_true", help="Do not write the merged MemoryEvent state file.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF Dynamic Memory Event Engine {VERSION}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = load_state(args.state)
    existing = events_from_state(state)
    extracted: list[MemoryEvent] = []
    for path in args.input:
        extracted.extend(load_events_from_path(path))
    events = merge_events(existing, extracted)
    packet, new_state = build_packet(args.query, events, state, top_k=args.top_k)
    report = render_markdown(packet)

    if args.dry_run:
        print(report)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_state_update:
        args.state.parent.mkdir(parents=True, exist_ok=True)
        args.state.write_text(json.dumps(new_state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.json_output:
        print(f"wrote {args.json_output}")
    if not args.no_state_update:
        print(f"updated {args.state}")


if __name__ == "__main__":
    main()
