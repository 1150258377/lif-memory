from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VERSION = "0.9.0-unsupervised-memory-field"

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
EN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{1,}")
ZH_RE = re.compile(r"[\u4e00-\u9fff]+")

STOP_TERMS = {
    "今天",
    "现在",
    "这个",
    "那个",
    "就是",
    "然后",
    "需要",
    "应该",
    "感觉",
    "什么",
    "一个",
    "自己",
    "进行",
    "目前",
    "接下来",
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
}

ACTION_WORDS = ["今天", "接下来", "下一步", "需要", "必须", "应该", "测试", "写", "整理", "投递", "记录", "验证"]
BLOCKER_WORDS = ["卡", "不知道", "缺", "没有", "失败", "问题", "害怕", "焦虑", "难受", "动不了", "延毕"]
COMPLETION_WORDS = ["完成", "已经", "跑通", "出结果", "保存", "写完", "测出来", "提交", "搞定"]
INCOMPLETE_WORDS = ["没完成", "未完成", "还没", "尚未", "没有完成", "缺"]


@dataclass(frozen=True)
class PrimitiveFeatures:
    actionability: float = 0.0
    blocker: float = 0.0
    completion: float = 0.0
    specificity: float = 0.0

    def to_packet(self) -> dict[str, float]:
        return {
            "actionability": round(self.actionability, 3),
            "blocker": round(self.blocker, 3),
            "completion": round(self.completion, 3),
            "specificity": round(self.specificity, 3),
        }


@dataclass(frozen=True)
class Observation:
    day: date
    source: str
    text: str
    embedding: tuple[float, ...]
    terms: tuple[str, ...]
    features: PrimitiveFeatures
    intensity: float

    def to_packet(self) -> dict[str, object]:
        return {
            "day": self.day.isoformat(),
            "source": self.source,
            "text": self.text,
            "terms": list(self.terms[:12]),
            "features": self.features.to_packet(),
            "intensity": round(self.intensity, 3),
        }


@dataclass
class MemorySlot:
    slot_id: str
    prototype: list[float]
    v_fast: float = 0.0
    v_slow: float = 0.0
    v: float = 0.0
    assigned_count: int = 0
    reconstruction_loss: float = 1.0
    evidence: list[Observation] = field(default_factory=list)
    term_counts: Counter[str] = field(default_factory=Counter)

    @property
    def label(self) -> str:
        terms = [term for term, _ in self.term_counts.most_common(4) if term not in STOP_TERMS]
        if not terms:
            return self.slot_id
        return f"{self.slot_id}: " + " / ".join(terms)

    def pressure_ratio(self, threshold: float) -> float:
        return self.v / threshold if threshold > 0 else 0.0

    def to_packet(self, threshold: float) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "label": self.label,
            "v_fast": round(self.v_fast, 3),
            "v_slow": round(self.v_slow, 3),
            "v": round(self.v, 3),
            "pressure_ratio": round(self.pressure_ratio(threshold), 3),
            "assigned_count": self.assigned_count,
            "reconstruction_loss": round(self.reconstruction_loss, 3),
            "top_terms": self.term_counts.most_common(12),
            "top_evidence": [item.to_packet() for item in self.evidence[:5]],
        }


@dataclass(frozen=True)
class Assignment:
    observation: Observation
    slot_id: str
    similarity: float
    reconstruction_loss: float

    def to_packet(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "similarity": round(self.similarity, 3),
            "reconstruction_loss": round(self.reconstruction_loss, 3),
            "observation": self.observation.to_packet(),
        }


@dataclass(frozen=True)
class UnsupervisedMemoryFieldResult:
    version: str
    slots: Sequence[MemorySlot]
    assignments: Sequence[Assignment]
    global_reconstruction_loss: float
    high_loss_observations: Sequence[Assignment]
    threshold: float

    @property
    def active_slots(self) -> list[MemorySlot]:
        return [slot for slot in self.slots if slot.assigned_count > 0]

    def to_packet(self) -> dict[str, object]:
        return {
            "version": self.version,
            "global_reconstruction_loss": round(self.global_reconstruction_loss, 3),
            "threshold": self.threshold,
            "slots": [slot.to_packet(self.threshold) for slot in self.slots],
            "assignments": [item.to_packet() for item in self.assignments],
            "high_loss_observations": [item.to_packet() for item in self.high_loss_observations],
        }


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def short_text(text: str, limit: int = 140) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def split_blocks(text: str) -> list[str]:
    text = normalize_text(text)
    raw_blocks = re.split(r"[\n。！？!?；;]+", text)
    blocks = [re.sub(r"\s+", " ", block).strip(" -\t") for block in raw_blocks]
    return [block for block in blocks if len(block) >= 6]


def infer_day(source: str, fallback: date | None = None) -> date:
    match = DATE_RE.search(source)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return fallback or date.today()


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []

    for match in EN_WORD_RE.finditer(text):
        tokens.append(match.group(0).lower())

    for segment in ZH_RE.findall(text):
        if len(segment) <= 4:
            tokens.append(segment)
        for n in (2, 3, 4):
            if len(segment) >= n:
                tokens.extend(segment[i : i + n] for i in range(0, len(segment) - n + 1))

    unit_hits = re.findall(r"\d+(?:\.\d+)?\s*(?:mV|V|Hz|kHz|MHz|Ω|ohm|欧姆|倍|%)", text, flags=re.I)
    tokens.extend(hit.replace(" ", "").lower() for hit in unit_hits)

    return [token for token in tokens if token and token not in STOP_TERMS]


def stable_hash(text: str) -> int:
    value = 2166136261
    for char in text:
        value ^= ord(char)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def embed_terms(terms: Sequence[str], dim: int = 96) -> tuple[float, ...]:
    vector = [0.0] * dim
    if not terms:
        return tuple(vector)

    counts = Counter(terms)
    for term, count in counts.items():
        idx = stable_hash(term) % dim
        sign = 1.0 if (stable_hash(term + "#sign") % 2 == 0) else -1.0
        vector[idx] += sign * (1.0 + math.log1p(count))

    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot(a, b) / (norm_a * norm_b)


def normalize_vector(values: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def blend_prototype(old: Sequence[float], new: Sequence[float], learning_rate: float) -> list[float]:
    mixed = [(1.0 - learning_rate) * a + learning_rate * b for a, b in zip(old, new)]
    return normalize_vector(mixed)


def primitive_features(text: str) -> PrimitiveFeatures:
    specificity = min(
        1.0,
        0.10 * len(re.findall(r"\d+(?:\.\d+)?", text))
        + 0.12 * sum(token in text for token in ["->", "→", "/", "%", "mV", "Hz", "Ω"])
        + 0.08 * len(re.findall(r"\b[A-Z]{2,}\b", text)),
    )
    actionability = 1.0 if contains_any(text, ACTION_WORDS) else 0.0
    blocker = 1.0 if contains_any(text, BLOCKER_WORDS) else 0.0
    completion = 1.0 if contains_any(text, COMPLETION_WORDS) and not contains_any(text, INCOMPLETE_WORDS) else 0.0
    return PrimitiveFeatures(
        actionability=actionability,
        blocker=blocker,
        completion=completion,
        specificity=specificity,
    )


def observation_intensity(features: PrimitiveFeatures, term_count: int) -> float:
    base = 0.55 + 0.04 * min(term_count, 25)
    base += 0.28 * features.actionability
    base += 0.35 * features.blocker
    base += 0.22 * features.specificity
    base -= 0.18 * features.completion
    return max(0.15, min(2.2, base))


def extract_observations(notes: Mapping[str, str], fallback_day: date | None = None, dim: int = 96) -> list[Observation]:
    observations: list[Observation] = []
    for source, text in notes.items():
        day = infer_day(source, fallback=fallback_day)
        for block in split_blocks(text):
            terms = tokenize(block)
            if not terms:
                continue
            features = primitive_features(block)
            observations.append(
                Observation(
                    day=day,
                    source=source,
                    text=short_text(block),
                    embedding=embed_terms(terms, dim=dim),
                    terms=tuple(terms),
                    features=features,
                    intensity=observation_intensity(features, len(terms)),
                )
            )
    observations.sort(key=lambda item: (item.day, item.source, item.text))
    return observations


def choose_slot(
    observation: Observation,
    slots: Sequence[MemorySlot],
    min_similarity_for_existing: float,
) -> tuple[MemorySlot, float]:
    occupied = [slot for slot in slots if slot.assigned_count > 0]
    empty = [slot for slot in slots if slot.assigned_count == 0]

    if not occupied:
        return empty[0] if empty else slots[0], 0.0

    scored = [(slot, cosine(observation.embedding, slot.prototype)) for slot in occupied]
    best_slot, best_similarity = max(scored, key=lambda pair: pair[1])

    if best_similarity < min_similarity_for_existing and empty:
        return empty[0], 0.0

    return best_slot, best_similarity


def apply_time_decay(slots: Sequence[MemorySlot], delta_days: int, fast_decay: float, slow_decay: float) -> None:
    if delta_days <= 0:
        return
    for slot in slots:
        slot.v_fast *= fast_decay ** delta_days
        slot.v_slow *= slow_decay ** delta_days
        slot.v = 0.72 * slot.v_fast + 0.28 * slot.v_slow


def update_slot(slot: MemorySlot, observation: Observation, similarity: float, learning_rate: float) -> Assignment:
    if slot.assigned_count == 0:
        slot.prototype = list(observation.embedding)
        similarity = 1.0

    reconstruction_loss = max(0.0, 1.0 - similarity)
    completion_inhibition = 0.55 * observation.features.completion

    slot.v_fast = max(0.0, slot.v_fast + observation.intensity - completion_inhibition)
    slot.v_slow = max(0.0, slot.v_slow + 0.45 * observation.intensity - 0.25 * completion_inhibition)
    slot.v = 0.72 * slot.v_fast + 0.28 * slot.v_slow
    slot.assigned_count += 1

    slot.prototype = blend_prototype(slot.prototype, observation.embedding, learning_rate)
    slot.reconstruction_loss = (
        reconstruction_loss
        if slot.assigned_count == 1
        else 0.78 * slot.reconstruction_loss + 0.22 * reconstruction_loss
    )
    slot.evidence.append(observation)
    slot.evidence.sort(key=lambda item: (item.intensity, item.day.toordinal()), reverse=True)
    slot.evidence = slot.evidence[:12]
    slot.term_counts.update(term for term in observation.terms if term not in STOP_TERMS)

    return Assignment(
        observation=observation,
        slot_id=slot.slot_id,
        similarity=similarity,
        reconstruction_loss=reconstruction_loss,
    )


def reconstruct_unsupervised_memory_field(
    notes: Mapping[str, str],
    slot_count: int = 8,
    epochs: int = 2,
    dim: int = 96,
    min_similarity_for_existing: float = 0.42,
    learning_rate: float = 0.22,
    fast_decay: float = 0.84,
    slow_decay: float = 0.94,
    threshold: float = 3.6,
    fallback_day: date | None = None,
) -> UnsupervisedMemoryFieldResult:
    observations = extract_observations(notes, fallback_day=fallback_day, dim=dim)
    slots = [
        MemorySlot(slot_id=f"slot_{idx}", prototype=[0.0] * dim)
        for idx in range(max(1, slot_count))
    ]
    if not observations:
        return UnsupervisedMemoryFieldResult(
            version=VERSION,
            slots=slots,
            assignments=[],
            global_reconstruction_loss=1.0,
            high_loss_observations=[],
            threshold=threshold,
        )

    assignments: list[Assignment] = []
    for _epoch in range(max(1, epochs)):
        last_day = observations[0].day
        epoch_assignments: list[Assignment] = []
        for observation in observations:
            apply_time_decay(slots, max(0, (observation.day - last_day).days), fast_decay, slow_decay)
            slot, similarity = choose_slot(observation, slots, min_similarity_for_existing)
            epoch_assignments.append(update_slot(slot, observation, similarity, learning_rate))
            last_day = observation.day
        assignments = epoch_assignments

    global_loss = sum(item.reconstruction_loss for item in assignments) / len(assignments)
    high_loss = sorted(assignments, key=lambda item: item.reconstruction_loss, reverse=True)[: min(8, len(assignments))]
    slots.sort(key=lambda slot: (slot.assigned_count > 0, slot.v), reverse=True)

    return UnsupervisedMemoryFieldResult(
        version=VERSION,
        slots=slots,
        assignments=assignments,
        global_reconstruction_loss=global_loss,
        high_loss_observations=high_loss,
        threshold=threshold,
    )


def find_daily_notes(vault: Path, days: int, today: date | None = None) -> dict[str, str]:
    today = today or date.today()
    candidates: list[tuple[date, Path]] = []
    for path in vault.rglob("20??-??-??.md"):
        if any(part in {".git", ".obsidian", ".venv", "node_modules", "__pycache__", "tests"} for part in path.parts):
            continue
        day = infer_day(path.name, fallback=today)
        if day <= today:
            candidates.append((day, path))
    candidates.sort(key=lambda pair: pair[0])
    selected = candidates[-max(days, 0):]
    return {str(path): path.read_text(encoding="utf-8", errors="ignore") for _, path in selected}


def demo_notes() -> dict[str, str]:
    return {
        "2026-06-20.md": "今天完成了 SSVEP 频谱图和事件率整理，但是 LIF 后向散射链路还缺一组可写进论文第四章的数据。",
        "2026-06-21.md": "接下来需要固定 KS1092 后级输入条件，测试 50 欧姆负载、偏置和 LIF 输出事件率，并保存波形截图。",
        "2026-06-22.md": "我现在延毕压力很大，感觉难受、焦虑、动不了，需要先恢复。",
        "2026-06-23.md": "LIF-Memory 需要借鉴 NeRF 的重建误差和多观测一致性，不要再手工定义论文实验情绪这些视角。",
        "2026-06-24.md": "求职方面要把 AI 加嵌入式项目写进简历，投递岗位时突出低功耗事件驱动系统。",
    }


def render_markdown(result: UnsupervisedMemoryFieldResult) -> str:
    lines: list[str] = [
        "# LIF-Memory 自动潜在状态场",
        "",
        f"- Version: `{result.version}`",
        f"- Global reconstruction loss: `{result.global_reconstruction_loss:.3f}`",
        f"- Active slots: `{len(result.active_slots)}`",
        "",
        "## 自动形成的 latent slots",
        "",
        "| Slot | V | Ratio | Count | Recon loss | Auto label |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for slot in result.slots:
        if slot.assigned_count <= 0:
            continue
        lines.append(
            f"| {slot.slot_id} | {slot.v:.2f} | {slot.pressure_ratio(result.threshold):.2f} | "
            f"{slot.assigned_count} | {slot.reconstruction_loss:.2f} | {slot.label} |"
        )

    lines.extend(["", "## Slot evidence", ""])
    for slot in result.slots:
        if slot.assigned_count <= 0:
            continue
        lines.extend([f"### {slot.label}", ""])
        lines.append(f"- 电位：`{slot.v:.2f}`，重建误差：`{slot.reconstruction_loss:.2f}`")
        lines.append("- 证据：")
        for item in slot.evidence[:5]:
            lines.append(f"  - `{item.day.isoformat()}` {item.text}")
        lines.append("")

    lines.extend(["## 高重建误差观测", ""])
    if not result.high_loss_observations:
        lines.append("- 暂无。")
    for item in result.high_loss_observations:
        lines.append(
            f"- loss=`{item.reconstruction_loss:.2f}` slot=`{item.slot_id}` "
            f"`{item.observation.day.isoformat()}` {item.observation.text}"
        )

    lines.extend(
        [
            "",
            "## 使用含义",
            "",
            "这里的 slot 不是预设的论文/实验/情绪分类，而是由观测文本自动聚合出来的潜在状态。",
            "重建误差高的片段表示当前 slot 无法很好解释，应优先作为新状态、拆分状态或补充观测来处理。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unsupervised NeRF-like latent memory field for LIF-Memory.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault root. If omitted, built-in demo notes are used.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to read.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--slots", type=int, default=8, help="Number of latent slots.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of online clustering passes.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()

    notes = find_daily_notes(args.vault, args.days, today=today) if args.vault else demo_notes()
    result = reconstruct_unsupervised_memory_field(
        notes,
        slot_count=args.slots,
        epochs=args.epochs,
        fallback_day=today,
    )
    markdown = render_markdown(result)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result.to_packet(), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
