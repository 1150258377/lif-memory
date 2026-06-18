from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

VERSION = "0.9.0"

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}|[\u4e00-\u9fff]{2,}")

SIGNAL_WORDS: dict[str, list[str]] = {
    "novelty": ["新", "引入", "更新", "第一次", "突破", "发现", "重构", "新增", "v0", "agent", "智能体"],
    "conflict": ["矛盾", "冲突", "不理解", "卡", "失败", "不稳", "质疑", "缺", "漏洞", "反驳", "链接不起来"],
    "risk": ["风险", "危险", "延毕", "截止", "不会", "不能", "丢失", "崩", "错误", "幻觉", "不可复现"],
    "value": ["论文", "闭环", "证据", "求职", "项目", "简历", "核心", "主线", "实验", "代码", "可执行"],
    "urgency": ["今天", "现在", "今晚", "明天", "马上", "最近", "截止", "提交", "答辩", "版本"],
    "fatigue": ["重复", "旧", "已经", "太多", "空转", "低价值", "无效", "泛泛", "废话"],
}

ACTION_NAMES = [
    "fine_inspect",
    "coarse_scan",
    "backtrack",
    "summarize",
    "test_code",
    "write_memory",
    "delegate_agent",
    "drop_path",
]


@dataclass
class LIFNeuron:
    """A minimal leaky-integrate-and-fire state variable.

    This is controller dynamics for symbolic/LLM-agent routing, not a biological simulation.
    """

    name: str
    threshold: float = 1.0
    leak: float = 0.82
    reset: float = 0.0
    voltage: float = 0.0
    spikes: int = 0

    def step(self, current: float) -> bool:
        self.voltage = self.voltage * self.leak + current
        fired = self.voltage >= self.threshold
        if fired:
            self.spikes += 1
            self.voltage = self.reset
        return fired

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "threshold": self.threshold,
            "leak": self.leak,
            "voltage": round(self.voltage, 4),
            "spikes": self.spikes,
        }


@dataclass
class LIFPopulation:
    name: str
    neurons: dict[str, LIFNeuron]

    def step(self, currents: dict[str, float]) -> dict[str, bool]:
        events: dict[str, bool] = {}
        for key, neuron in self.neurons.items():
            events[key] = neuron.step(float(currents.get(key, 0.0)))
        return events

    def snapshots(self) -> dict[str, dict[str, Any]]:
        return {key: neuron.snapshot() for key, neuron in self.neurons.items()}


@dataclass
class Candidate:
    candidate_id: str
    title: str
    text: str
    source: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SaliencePacket:
    candidate_id: str
    title: str
    source: str
    signals: dict[str, float]
    amygdala_spikes: dict[str, bool]
    salience_score: float
    suggested_mode: str


@dataclass
class ActionPacket:
    candidate_id: str
    title: str
    selected_action: str
    action_scores: dict[str, float]
    striatum_spikes: dict[str, bool]
    salience_score: float
    reason: str


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


def token_counter(text: str) -> Counter[str]:
    return Counter(tok.lower() for tok in TOKEN_RE.findall(text))


def count_words(text: str, words: Iterable[str]) -> int:
    return sum(text.lower().count(word.lower()) for word in words)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def candidate_from_packet(item: dict[str, Any], fallback_index: int = 0, source: str = "packet") -> Candidate:
    candidate_id = normalize_text(
        item.get("id") or item.get("candidate_id") or item.get("node_id") or item.get("path") or f"candidate::{fallback_index}"
    )
    title = normalize_text(item.get("title") or item.get("label") or item.get("name") or item.get("summary") or candidate_id)
    text = normalize_text(item)
    return Candidate(candidate_id=candidate_id, title=title[:120], text=text, source=source, metadata=item)


def load_candidates(path: Path) -> list[Candidate]:
    packet = json.loads(path.read_text(encoding="utf-8"))
    candidates: list[Candidate] = []

    if isinstance(packet, list):
        for i, item in enumerate(packet):
            if isinstance(item, dict):
                candidates.append(candidate_from_packet(item, i, path.name))
        return candidates

    if not isinstance(packet, dict):
        return []

    # Supports llm_maze_graph.json, maze_graph_lif.json, maze_graph.json, or generic JSON.
    for key in ["lif_spikes", "spikes", "tasks", "tensions", "claims", "nodes", "observations", "frontier"]:
        values = packet.get(key)
        if isinstance(values, list):
            for item in values:
                if isinstance(item, dict):
                    candidates.append(candidate_from_packet(item, len(candidates), key))

    # If no explicit candidate arrays exist, treat top-level children as candidates.
    if not candidates:
        for key, value in packet.items():
            if isinstance(value, dict):
                candidates.append(candidate_from_packet({"id": key, **value}, len(candidates), "top_level"))
            elif isinstance(value, list):
                candidates.append(Candidate(candidate_id=key, title=key, text=normalize_text(value), source="top_level"))

    seen: set[str] = set()
    deduped: list[Candidate] = []
    for candidate in candidates:
        if candidate.candidate_id in seen:
            continue
        seen.add(candidate.candidate_id)
        deduped.append(candidate)
    return deduped


class AmygdalaLikeSalience:
    """Salience population: novelty/conflict/risk/value/urgency/fatigue."""

    def __init__(self) -> None:
        self.population = LIFPopulation(
            name="amygdala_like_salience",
            neurons={
                "novelty": LIFNeuron("novelty", threshold=0.72, leak=0.78),
                "conflict": LIFNeuron("conflict", threshold=0.68, leak=0.84),
                "risk": LIFNeuron("risk", threshold=0.70, leak=0.86),
                "value": LIFNeuron("value", threshold=0.75, leak=0.82),
                "urgency": LIFNeuron("urgency", threshold=0.66, leak=0.80),
                "fatigue": LIFNeuron("fatigue", threshold=0.80, leak=0.88),
            },
        )

    def signals_for(self, candidate: Candidate, global_terms: Counter[str]) -> dict[str, float]:
        text = candidate.text
        tokens = token_counter(text)
        length_factor = clamp(math.log(len(text) + 1, 500), 0.15, 1.0)
        signals: dict[str, float] = {}
        for name, words in SIGNAL_WORDS.items():
            hits = count_words(text, words)
            signals[name] = clamp((hits / 6.0) + 0.06 * length_factor)

        rare_terms = [term for term, count in tokens.items() if count > 0 and global_terms.get(term, 0) <= 1]
        signals["novelty"] = clamp(signals["novelty"] + min(len(rare_terms), 12) / 30.0)

        for key in ["lif_score", "score", "priority", "urgency", "voltage", "weight"]:
            raw = candidate.metadata.get(key)
            if isinstance(raw, (int, float)):
                signals["value"] = clamp(signals["value"] + min(abs(float(raw)), 10.0) / 20.0)

        if len(tokens) < 8:
            signals["fatigue"] = clamp(signals["fatigue"] + 0.25)
        if signals["urgency"] > 0.35 or signals["risk"] > 0.35:
            signals["fatigue"] = max(0.0, signals["fatigue"] - 0.15)

        return {key: round(value, 4) for key, value in signals.items()}

    def step(self, candidate: Candidate, global_terms: Counter[str]) -> SaliencePacket:
        signals = self.signals_for(candidate, global_terms)
        spikes = self.population.step(signals)
        salience = (
            1.15 * signals["value"]
            + 1.05 * signals["conflict"]
            + 1.00 * signals["risk"]
            + 0.90 * signals["novelty"]
            + 0.80 * signals["urgency"]
            - 0.55 * signals["fatigue"]
        )
        salience_score = round(clamp(salience / 3.1), 4)
        suggested_mode = "fine" if salience_score >= 0.46 or any(spikes[k] for k in ["conflict", "risk", "value"]) else "coarse"
        return SaliencePacket(
            candidate_id=candidate.candidate_id,
            title=candidate.title,
            source=candidate.source,
            signals=signals,
            amygdala_spikes=spikes,
            salience_score=salience_score,
            suggested_mode=suggested_mode,
        )


class StriatumLikeActionSelector:
    """Action population: competing actions selected from salience packets."""

    def __init__(self) -> None:
        self.population = LIFPopulation(
            name="striatum_like_action_selection",
            neurons={
                "fine_inspect": LIFNeuron("fine_inspect", threshold=0.80, leak=0.76),
                "coarse_scan": LIFNeuron("coarse_scan", threshold=0.72, leak=0.74),
                "backtrack": LIFNeuron("backtrack", threshold=0.78, leak=0.82),
                "summarize": LIFNeuron("summarize", threshold=0.76, leak=0.78),
                "test_code": LIFNeuron("test_code", threshold=0.76, leak=0.80),
                "write_memory": LIFNeuron("write_memory", threshold=0.74, leak=0.80),
                "delegate_agent": LIFNeuron("delegate_agent", threshold=0.82, leak=0.78),
                "drop_path": LIFNeuron("drop_path", threshold=0.76, leak=0.86),
            },
        )

    def action_scores(self, packet: SaliencePacket, candidate: Candidate) -> dict[str, float]:
        signals = packet.signals
        text = candidate.text.lower()
        code_hits = sum(text.count(word) for word in ["python", "def ", "class ", "json", "api", "bug", "测试", "代码", "程序"])
        agent_hits = sum(text.count(word) for word in ["agent", "智能体", "多视角", "llm", "角色", "分派"])
        scores = {
            "fine_inspect": 0.42 * packet.salience_score + 0.25 * signals["conflict"] + 0.20 * signals["risk"] + 0.16 * signals["value"],
            "coarse_scan": 0.26 + 0.23 * signals["novelty"] + 0.10 * (1.0 - packet.salience_score),
            "backtrack": 0.18 + 0.35 * signals["conflict"] + 0.20 * signals["fatigue"],
            "summarize": 0.20 + 0.25 * signals["value"] + 0.18 * signals["urgency"],
            "test_code": 0.12 + min(code_hits, 6) / 12.0 + 0.15 * signals["risk"],
            "write_memory": 0.16 + 0.22 * signals["value"] + 0.22 * signals["novelty"] + 0.10 * signals["urgency"],
            "delegate_agent": 0.08 + min(agent_hits, 5) / 10.0 + 0.22 * signals["novelty"],
            "drop_path": 0.10 + 0.45 * signals["fatigue"] - 0.18 * signals["urgency"] - 0.18 * signals["risk"],
        }
        return {key: round(clamp(value), 4) for key, value in scores.items()}

    def step(self, packet: SaliencePacket, candidate: Candidate) -> ActionPacket:
        scores = self.action_scores(packet, candidate)
        spikes = self.population.step(scores)
        fired = [name for name, has_spike in spikes.items() if has_spike]
        selected = max(fired, key=lambda name: scores[name]) if fired else max(scores, key=scores.get)
        reason = explain_action(selected, packet, scores)
        return ActionPacket(
            candidate_id=packet.candidate_id,
            title=packet.title,
            selected_action=selected,
            action_scores=scores,
            striatum_spikes=spikes,
            salience_score=packet.salience_score,
            reason=reason,
        )


def explain_action(action: str, packet: SaliencePacket, scores: dict[str, float]) -> str:
    signals = sorted(packet.signals.items(), key=lambda kv: kv[1], reverse=True)[:3]
    signal_text = ", ".join(f"{key}={value:.2f}" for key, value in signals)
    return f"{action} wins with score={scores[action]:.2f}; salience={packet.salience_score:.2f}; top signals: {signal_text}."


def run_controller(candidates: list[Candidate], top_k: int = 12) -> dict[str, Any]:
    global_terms = Counter()
    for candidate in candidates:
        global_terms.update(token_counter(candidate.text))

    amygdala = AmygdalaLikeSalience()
    striatum = StriatumLikeActionSelector()

    salience_packets: list[SaliencePacket] = []
    action_packets: list[ActionPacket] = []
    for candidate in candidates:
        salience = amygdala.step(candidate, global_terms)
        action = striatum.step(salience, candidate)
        salience_packets.append(salience)
        action_packets.append(action)

    ranked = sorted(action_packets, key=lambda packet: (packet.salience_score, max(packet.action_scores.values())), reverse=True)[:top_k]
    selected_ids = {packet.candidate_id for packet in ranked}
    selected_salience = [packet for packet in salience_packets if packet.candidate_id in selected_ids]

    return {
        "version": VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "controller": "amygdala_striatum_lif",
        "input_count": len(candidates),
        "top_k": top_k,
        "selected_actions": [asdict(packet) for packet in ranked],
        "selected_salience": [asdict(packet) for packet in selected_salience],
        "population_state": {
            "amygdala": amygdala.population.snapshots(),
            "striatum": striatum.population.snapshots(),
        },
        "action_histogram": dict(Counter(packet.selected_action for packet in action_packets)),
    }


def render_markdown(packet: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Amygdala-Striatum LIF Report v{packet['version']}")
    lines.append("")
    lines.append("This report adds a coarse-to-fine action-selection layer on top of MazeGraph / LLM-MazeGraph outputs.")
    lines.append("")
    lines.append("```text")
    lines.append("candidate nodes -> amygdala-like salience LIF -> striatum-like action LIF -> next action")
    lines.append("```")
    lines.append("")
    lines.append(f"- created_at: `{packet['created_at']}`")
    lines.append(f"- input_count: `{packet['input_count']}`")
    lines.append(f"- top_k: `{packet['top_k']}`")
    lines.append("")
    lines.append("## Action histogram")
    lines.append("")
    for action, count in sorted(packet["action_histogram"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- `{action}`: {count}")
    lines.append("")
    lines.append("## Selected actions")
    lines.append("")
    for i, action in enumerate(packet["selected_actions"], 1):
        lines.append(f"### {i}. {action['title']}")
        lines.append("")
        lines.append(f"- candidate_id: `{action['candidate_id']}`")
        lines.append(f"- selected_action: `{action['selected_action']}`")
        lines.append(f"- salience_score: `{action['salience_score']}`")
        lines.append(f"- reason: {action['reason']}")
        lines.append("- action_scores:")
        for name, score in sorted(action["action_scores"].items(), key=lambda kv: -kv[1])[:5]:
            lines.append(f"  - `{name}`: {score}")
        lines.append("")
    lines.append("## Boundary")
    lines.append("")
    lines.append("This module is not a biological simulation. It is a symbolic, event-driven controller inspired by amygdala-like salience and striatum-like action selection.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Amygdala-Striatum LIF controller for LIF-Memory JSON packets.")
    parser.add_argument("--input", type=Path, required=True, help="Input JSON from llm_maze_explorer.py / maze_graph_lif.py / maze_graph_builder.py.")
    parser.add_argument("--output", type=Path, default=Path("Amygdala-Striatum LIF Report.md"), help="Markdown report path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--top-k", type=int, default=12, help="Number of selected actions to show.")
    parser.add_argument("--dry-run", action="store_true", help="Print the Markdown report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"Amygdala-Striatum LIF {VERSION}")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(args.input)
    packet = run_controller(candidates, top_k=args.top_k)
    report = render_markdown(packet)

    if args.dry_run:
        print(report)
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.output}")
    if args.json_output:
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
