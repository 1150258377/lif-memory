from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from unsupervised_memory_field import (
    Observation,
    cosine,
    embed_terms,
    extract_observations,
    stable_hash,
    tokenize,
)


VERSION = "0.10.0-hippocampal-lif-network"


@dataclass(frozen=True)
class HippocampalConfig:
    """Small engineering hippocampus built from LIF-style units.

    The model is deliberately small and deterministic:
    EC  = input observation embedding.
    DG  = sparse pattern separation code.
    CA3 = recurrent associative LIF field.
    CA1 = read/write gate against long-term traces.
    Cortex = stable semantic trace store.
    """

    dg_units: int = 384
    ca3_units: int = 256
    ca1_units: int = 96
    dg_top_k: int = 12
    ca3_top_k: int = 18
    dg_fanout: int = 4
    ca3_threshold: float = 0.88
    ca3_leak: float = 0.62
    recurrent_gain: float = 0.72
    inhibition_strength: float = 0.18
    hebbian_lr: float = 0.09
    weight_decay: float = 0.995
    max_weight: float = 1.0
    ca1_match_threshold: float = 0.22
    dim: int = 96


@dataclass
class LIFUnit:
    unit_id: str
    v: float = 0.0
    threshold: float = 1.0
    leak: float = 0.62
    spike_count: int = 0

    def step(self, current: float) -> bool:
        self.v = max(0.0, self.v * self.leak + current)
        if self.v >= self.threshold:
            self.spike_count += 1
            self.v = 0.0
            return True
        return False

    def to_packet(self) -> dict[str, object]:
        return {"unit_id": self.unit_id, "v": round(self.v, 3), "spike_count": self.spike_count}


@dataclass(frozen=True)
class DGCode:
    active_units: tuple[int, ...]
    activations: tuple[float, ...]
    terms: tuple[str, ...]

    @property
    def sparsity(self) -> float:
        return len(self.active_units)

    def overlap(self, other: "DGCode") -> float:
        a = set(self.active_units)
        b = set(other.active_units)
        return len(a & b) / max(1, len(a | b))

    def to_packet(self) -> dict[str, object]:
        return {
            "active_units": list(self.active_units),
            "activations": [round(x, 3) for x in self.activations],
            "terms": list(self.terms[:12]),
        }


@dataclass(frozen=True)
class CA3State:
    spikes: tuple[int, ...]
    feedforward_units: tuple[int, ...]
    completed_units: tuple[int, ...]
    mean_recurrent_current: float

    @property
    def completion_gain(self) -> int:
        return len(set(self.completed_units) - set(self.feedforward_units))

    def overlap(self, other: "CA3State") -> float:
        a = set(self.spikes)
        b = set(other.spikes)
        return len(a & b) / max(1, len(a | b))

    def to_packet(self) -> dict[str, object]:
        return {
            "spikes": list(self.spikes),
            "feedforward_units": list(self.feedforward_units),
            "completed_units": list(self.completed_units),
            "completion_gain": self.completion_gain,
            "mean_recurrent_current": round(self.mean_recurrent_current, 3),
        }


@dataclass
class CorticalTrace:
    trace_id: str
    signature: set[int]
    term_counts: Counter[str] = field(default_factory=Counter)
    evidence: list[Observation] = field(default_factory=list)
    write_count: int = 0
    last_similarity: float = 0.0

    @property
    def label(self) -> str:
        terms = [term for term, _ in self.term_counts.most_common(5)]
        return self.trace_id if not terms else f"{self.trace_id}: " + " / ".join(terms)

    def match(self, spikes: Iterable[int]) -> float:
        current = set(spikes)
        if not current or not self.signature:
            return 0.0
        return len(current & self.signature) / len(current)

    def term_match(self, terms: Iterable[str]) -> float:
        current = Counter(terms)
        if not current or not self.term_counts:
            return 0.0
        overlap = sum(min(count, self.term_counts.get(term, 0)) for term, count in current.items())
        return overlap / max(1, sum(current.values()))

    def combined_match(self, spikes: Iterable[int], terms: Iterable[str]) -> float:
        ca3_similarity = self.match(spikes)
        semantic_similarity = self.term_match(terms)
        if semantic_similarity < 0.06:
            return 0.0
        return 0.62 * ca3_similarity + 0.38 * semantic_similarity

    def update(self, observation: Observation, spikes: Iterable[int], similarity: float) -> None:
        self.signature.update(spikes)
        self.term_counts.update(observation.terms)
        self.evidence.append(observation)
        self.evidence.sort(key=lambda item: (item.intensity, item.day.toordinal()), reverse=True)
        self.evidence = self.evidence[:10]
        self.write_count += 1
        self.last_similarity = similarity

    def to_packet(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "label": self.label,
            "signature_size": len(self.signature),
            "write_count": self.write_count,
            "last_similarity": round(self.last_similarity, 3),
            "top_terms": self.term_counts.most_common(12),
            "top_evidence": [item.to_packet() for item in self.evidence[:5]],
        }


@dataclass(frozen=True)
class CA1Decision:
    mode: str
    trace_id: str
    similarity: float
    reason: str

    def to_packet(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "trace_id": self.trace_id,
            "similarity": round(self.similarity, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HippocampalStep:
    observation: Observation
    dg: DGCode
    ca3: CA3State
    ca1: CA1Decision

    def to_packet(self) -> dict[str, object]:
        return {
            "observation": self.observation.to_packet(),
            "dg": self.dg.to_packet(),
            "ca3": self.ca3.to_packet(),
            "ca1": self.ca1.to_packet(),
        }


@dataclass(frozen=True)
class ProbeResult:
    text: str
    dg: DGCode
    ca3: CA3State
    best_trace_id: str | None
    similarity: float
    recalled_terms: tuple[str, ...]
    recalled_evidence: tuple[str, ...]
    completed: bool

    def to_packet(self) -> dict[str, object]:
        return {
            "text": self.text,
            "dg": self.dg.to_packet(),
            "ca3": self.ca3.to_packet(),
            "best_trace_id": self.best_trace_id,
            "similarity": round(self.similarity, 3),
            "completed": self.completed,
            "recalled_terms": list(self.recalled_terms),
            "recalled_evidence": list(self.recalled_evidence),
        }


@dataclass(frozen=True)
class HippocampalMetrics:
    dg_mean_sparsity: float
    dg_mean_pairwise_overlap: float
    ca3_mean_pairwise_overlap: float
    trace_count: int
    write_count: int
    update_count: int

    def to_packet(self) -> dict[str, object]:
        return {
            "dg_mean_sparsity": round(self.dg_mean_sparsity, 3),
            "dg_mean_pairwise_overlap": round(self.dg_mean_pairwise_overlap, 3),
            "ca3_mean_pairwise_overlap": round(self.ca3_mean_pairwise_overlap, 3),
            "trace_count": self.trace_count,
            "write_count": self.write_count,
            "update_count": self.update_count,
        }


@dataclass
class HippocampalResult:
    version: str
    config: HippocampalConfig
    steps: list[HippocampalStep]
    cortex: list[CorticalTrace]
    metrics: HippocampalMetrics
    probe_result: ProbeResult | None = None

    def to_packet(self) -> dict[str, object]:
        return {
            "version": self.version,
            "config": self.config.__dict__,
            "metrics": self.metrics.to_packet(),
            "steps": [step.to_packet() for step in self.steps],
            "cortex": [trace.to_packet() for trace in self.cortex],
            "probe": None if self.probe_result is None else self.probe_result.to_packet(),
        }


def _pairwise_mean_overlap(items: Sequence[object]) -> float:
    values: list[float] = []
    for idx, item in enumerate(items):
        for other in items[idx + 1 :]:
            if hasattr(item, "overlap"):
                values.append(item.overlap(other))  # type: ignore[attr-defined]
    return sum(values) / len(values) if values else 0.0


def _unit_score(unit_idx: int, vector: Sequence[float], terms: Sequence[str], salt: str, modulo: int) -> float:
    score = 0.0
    for dim_idx, value in enumerate(vector):
        if value == 0.0:
            continue
        h = stable_hash(f"{salt}:{unit_idx}:{dim_idx}")
        sign = 1.0 if h % 2 == 0 else -1.0
        score += sign * value / math.sqrt(max(1, len(vector)))
    for term in terms:
        if stable_hash(f"{salt}:term:{term}") % modulo == unit_idx % modulo:
            score += 0.35 + 0.06 * min(len(term), 8)
    return score


def _top_k(scores: Sequence[float], k: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
    ranked = sorted(enumerate(scores), key=lambda item: (item[1], -item[0]), reverse=True)
    selected = [(idx, max(0.0, value)) for idx, value in ranked[: max(1, k)] if value > 0.0]
    if not selected:
        selected = [(ranked[0][0], 0.0)]
    return tuple(idx for idx, _ in selected), tuple(value for _, value in selected)


class HippocampalLIFNetwork:
    """DG-CA3-CA1-Cortex memory network for LIF-Memory.

    This is not a biological simulation. It is an engineering hippocampus:
    - DG produces sparse separated codes from semantic observations.
    - CA3 is a recurrent LIF associative field with Hebbian weights.
    - CA1 decides whether a pattern should update an old trace or create a new one.
    - Cortex stores stable trace labels, evidence and signatures.
    """

    def __init__(self, config: HippocampalConfig | None = None):
        self.config = config or HippocampalConfig()
        self.ca3_units = [
            LIFUnit(f"ca3_{idx}", threshold=self.config.ca3_threshold, leak=self.config.ca3_leak)
            for idx in range(self.config.ca3_units)
        ]
        self.weights: dict[tuple[int, int], float] = {}
        self.cortex: list[CorticalTrace] = []
        self.steps: list[HippocampalStep] = []

    def dentate_gyrus(self, observation: Observation | str) -> DGCode:
        if isinstance(observation, str):
            terms = tuple(tokenize(observation))
            vector = embed_terms(terms, dim=self.config.dim)
        else:
            terms = observation.terms
            vector = observation.embedding
        scores = [
            _unit_score(idx, vector, terms, "DG", self.config.dg_units)
            for idx in range(self.config.dg_units)
        ]
        active, activations = _top_k(scores, self.config.dg_top_k)
        return DGCode(active_units=active, activations=activations, terms=tuple(terms))

    def _dg_to_ca3_current(self, dg: DGCode) -> tuple[list[float], tuple[int, ...]]:
        currents = [0.0] * self.config.ca3_units
        for dg_idx, activation in zip(dg.active_units, dg.activations):
            for fan_idx in range(self.config.dg_fanout):
                ca3_idx = stable_hash(f"DG2CA3:{dg_idx}:{fan_idx}") % self.config.ca3_units
                currents[ca3_idx] += max(0.05, activation) / self.config.dg_fanout
        active, _ = _top_k(currents, self.config.ca3_top_k)
        return currents, active

    def _recurrent_current(self, active: Iterable[int]) -> list[float]:
        currents = [0.0] * self.config.ca3_units
        for src in active:
            for (left, right), weight in self.weights.items():
                if left == src:
                    currents[right] += self.config.recurrent_gain * weight
                elif right == src:
                    currents[left] += self.config.recurrent_gain * weight
        return currents

    def ca3_attractor(self, dg: DGCode, learn: bool = False) -> CA3State:
        feedforward, feedforward_units = self._dg_to_ca3_current(dg)
        recurrent = self._recurrent_current(feedforward_units)
        combined = [
            max(0.0, feedforward[idx] + recurrent[idx] - self.config.inhibition_strength)
            for idx in range(self.config.ca3_units)
        ]
        ranked = sorted(enumerate(combined), key=lambda item: (item[1], -item[0]), reverse=True)
        candidate_indices = [idx for idx, current in ranked[: self.config.ca3_top_k * 2] if current > 0.0]
        spikes: list[int] = []
        for idx in candidate_indices:
            # Strong feedforward/recurrent currents should produce a deterministic spike
            # even when a previous probe reset the LIF voltage.
            if combined[idx] >= self.config.ca3_threshold or self.ca3_units[idx].step(combined[idx]):
                spikes.append(idx)
            if len(spikes) >= self.config.ca3_top_k:
                break
        if not spikes:
            spikes = list(feedforward_units[: self.config.ca3_top_k])
        completed = tuple(idx for idx in spikes if idx not in set(feedforward_units))
        mean_rec = sum(recurrent[idx] for idx in spikes) / max(1, len(spikes))
        if learn:
            self._hebbian_update(spikes)
        return CA3State(
            spikes=tuple(spikes),
            feedforward_units=tuple(feedforward_units),
            completed_units=completed,
            mean_recurrent_current=mean_rec,
        )

    def _hebbian_update(self, spikes: Sequence[int]) -> None:
        # Decay old relations so the memory graph does not freeze permanently.
        for key in list(self.weights):
            self.weights[key] *= self.config.weight_decay
            if self.weights[key] < 0.005:
                del self.weights[key]
        unique = sorted(set(spikes))
        for left_idx, left in enumerate(unique):
            for right in unique[left_idx + 1 :]:
                key = (left, right)
                self.weights[key] = min(
                    self.config.max_weight,
                    self.weights.get(key, 0.0) + self.config.hebbian_lr,
                )

    def ca1_gate(self, observation: Observation, ca3: CA3State) -> CA1Decision:
        best_trace: CorticalTrace | None = None
        best_similarity = 0.0
        for trace in self.cortex:
            similarity = trace.combined_match(ca3.spikes, observation.terms)
            if similarity > best_similarity:
                best_trace = trace
                best_similarity = similarity

        if best_trace is None or best_similarity < self.config.ca1_match_threshold:
            trace = CorticalTrace(trace_id=f"trace_{len(self.cortex)}", signature=set(ca3.spikes))
            trace.update(observation, ca3.spikes, similarity=1.0)
            self.cortex.append(trace)
            return CA1Decision("write_new", trace.trace_id, best_similarity, "CA1 mismatch: create a new cortical trace")

        best_trace.update(observation, ca3.spikes, similarity=best_similarity)
        return CA1Decision("update_existing", best_trace.trace_id, best_similarity, "CA1 matched an existing cortical trace")

    def observe(self, observation: Observation) -> HippocampalStep:
        dg = self.dentate_gyrus(observation)
        ca3 = self.ca3_attractor(dg, learn=True)
        ca1 = self.ca1_gate(observation, ca3)
        step = HippocampalStep(observation=observation, dg=dg, ca3=ca3, ca1=ca1)
        self.steps.append(step)
        return step

    def train(self, observations: Iterable[Observation]) -> None:
        for observation in observations:
            self.observe(observation)

    def probe(self, text: str) -> ProbeResult:
        dg = self.dentate_gyrus(text)
        ca3 = self.ca3_attractor(dg, learn=False)
        best_trace: CorticalTrace | None = None
        best_similarity = 0.0
        probe_terms = tuple(tokenize(text))
        for trace in self.cortex:
            similarity = trace.combined_match(ca3.spikes, probe_terms)
            if similarity > best_similarity:
                best_trace = trace
                best_similarity = similarity
        return ProbeResult(
            text=text,
            dg=dg,
            ca3=ca3,
            best_trace_id=None if best_trace is None else best_trace.trace_id,
            similarity=best_similarity,
            recalled_terms=tuple(term for term, _ in (best_trace.term_counts.most_common(8) if best_trace else [])),
            recalled_evidence=tuple(item.text for item in (best_trace.evidence[:3] if best_trace else [])),
            completed=best_similarity >= self.config.ca1_match_threshold,
        )

    def metrics(self) -> HippocampalMetrics:
        dg_codes = [step.dg for step in self.steps]
        ca3_states = [step.ca3 for step in self.steps]
        write_count = sum(1 for step in self.steps if step.ca1.mode == "write_new")
        update_count = sum(1 for step in self.steps if step.ca1.mode == "update_existing")
        return HippocampalMetrics(
            dg_mean_sparsity=sum(len(code.active_units) for code in dg_codes) / max(1, len(dg_codes)),
            dg_mean_pairwise_overlap=_pairwise_mean_overlap(dg_codes),
            ca3_mean_pairwise_overlap=_pairwise_mean_overlap(ca3_states),
            trace_count=len(self.cortex),
            write_count=write_count,
            update_count=update_count,
        )

    def result(self, probe_text: str | None = None) -> HippocampalResult:
        probe = self.probe(probe_text) if probe_text else None
        return HippocampalResult(
            version=VERSION,
            config=self.config,
            steps=list(self.steps),
            cortex=list(self.cortex),
            metrics=self.metrics(),
            probe_result=probe,
        )


def build_hippocampal_lif_memory(
    notes: Mapping[str, str],
    config: HippocampalConfig | None = None,
    fallback_day: date | None = None,
    probe_text: str | None = None,
) -> HippocampalResult:
    network = HippocampalLIFNetwork(config)
    observations = extract_observations(notes, fallback_day=fallback_day, dim=network.config.dim)
    network.train(observations)
    return network.result(probe_text=probe_text)


def demo_notes() -> dict[str, str]:
    return {
        "2026-06-20.md": "今天完成了 SSVEP 频谱图，但 LIF 后向散射链路还缺论文第四章的数据。",
        "2026-06-21.md": "导师质疑创新点，需要把 LIF 后向散射、事件化和最小闭环实验讲清楚。",
        "2026-06-22.md": "PCB 需要继续检查 KS1092 输入、比较器阈值和 MOS 后向散射开关。",
        "2026-06-23.md": "LIF-Memory 不能只是触发器，需要海马体 DG CA3 CA1 的递归联想记忆网络。",
        "2026-06-24.md": "求职方面要把 AI 加嵌入式项目写进简历，突出低功耗事件驱动系统。",
    }


def find_daily_notes(vault: Path, days: int, today: date | None = None) -> dict[str, str]:
    today = today or date.today()
    candidates: list[tuple[date, Path]] = []
    for path in vault.rglob("20??-??-??.md"):
        if any(part in {".git", ".obsidian", ".venv", "node_modules", "__pycache__", "tests"} for part in path.parts):
            continue
        try:
            day = datetime.strptime(path.stem, "%Y-%m-%d").date()
        except ValueError:
            day = today
        if day <= today:
            candidates.append((day, path))
    candidates.sort(key=lambda pair: pair[0])
    selected = candidates[-max(days, 0):]
    return {str(path): path.read_text(encoding="utf-8", errors="ignore") for _, path in selected}


def render_markdown(result: HippocampalResult) -> str:
    metrics = result.metrics
    lines = [
        "# LIF-Memory 海马体 LIF 网络测试",
        "",
        f"- Version: `{result.version}`",
        f"- DG mean sparsity: `{metrics.dg_mean_sparsity:.2f}`",
        f"- DG mean pairwise overlap: `{metrics.dg_mean_pairwise_overlap:.3f}`",
        f"- CA3 mean pairwise overlap: `{metrics.ca3_mean_pairwise_overlap:.3f}`",
        f"- Cortical traces: `{metrics.trace_count}`",
        f"- CA1 writes / updates: `{metrics.write_count}` / `{metrics.update_count}`",
        "",
        "## Cortex 长期记忆痕迹",
        "",
        "| Trace | Writes | Signature | Last sim | Label |",
        "|---|---:|---:|---:|---|",
    ]
    for trace in result.cortex:
        lines.append(
            f"| {trace.trace_id} | {trace.write_count} | {len(trace.signature)} | "
            f"{trace.last_similarity:.2f} | {trace.label} |"
        )
    lines.extend(["", "## DG/CA3/CA1 步进记录", ""])
    for idx, step in enumerate(result.steps[:12], start=1):
        lines.append(
            f"- Step {idx}: DG active=`{len(step.dg.active_units)}`, "
            f"CA3 spikes=`{len(step.ca3.spikes)}`, "
            f"completion_gain=`{step.ca3.completion_gain}`, "
            f"CA1=`{step.ca1.mode}` -> `{step.ca1.trace_id}`；"
            f"证据：{step.observation.text}"
        )
    if result.probe_result:
        probe = result.probe_result
        lines.extend([
            "",
            "## Probe 回忆测试",
            "",
            f"- Cue: `{probe.text}`",
            f"- Completed: `{probe.completed}`",
            f"- Best trace: `{probe.best_trace_id}`",
            f"- Similarity: `{probe.similarity:.3f}`",
            f"- Recalled terms: `{', '.join(probe.recalled_terms)}`",
            "- Recalled evidence:",
        ])
        for item in probe.recalled_evidence:
            lines.append(f"  - {item}")
    lines.extend([
        "",
        "## 解释",
        "",
        "- DG overlap 低，说明相似输入被稀疏分离。",
        "- CA3 通过递归权重把共同出现的 spike 绑定成联想簇。",
        "- CA1 的 write/update 体现读写判断。",
        "- Cortex trace 保存稳定语义标签、证据和 CA3 signature。",
    ])
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hippocampus-inspired LIF memory network for LIF-Memory.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault root. If omitted, built-in demo notes are used.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to read.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--probe", type=str, default="老师又问 LIF 后向散射创新点怎么证明", help="Partial cue used to test CA3 completion.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()
    notes = find_daily_notes(args.vault, args.days, today=today) if args.vault else demo_notes()
    result = build_hippocampal_lif_memory(notes, fallback_day=today, probe_text=args.probe)
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
