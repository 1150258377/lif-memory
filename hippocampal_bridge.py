from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping

from hippocampal_lif_memory import (
    HippocampalConfig,
    HippocampalResult,
    build_hippocampal_lif_memory,
    demo_notes,
    find_daily_notes,
)

VERSION = "0.1.0-hippocampal-bridge"


@dataclass(frozen=True)
class HippocampalRecall:
    """Compact bridge packet used by the web UI, field LIF and AhaEngine.

    The hippocampal network should not directly answer the user. It contributes
    a recall signal: which trace was recalled, what evidence was recovered, and
    how much current should be added to the downstream LIF decision.
    """

    enabled: bool
    query: str
    trace_id: str | None = None
    similarity: float = 0.0
    completed: bool = False
    completion_gain: int = 0
    current_boost: float = 0.0
    recalled_terms: list[str] = field(default_factory=list)
    recalled_evidence: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    cortex: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_packet(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "enabled": self.enabled,
            "query": self.query,
            "trace_id": self.trace_id,
            "similarity": round(self.similarity, 4),
            "completed": self.completed,
            "completion_gain": self.completion_gain,
            "current_boost": round(self.current_boost, 4),
            "recalled_terms": self.recalled_terms,
            "recalled_evidence": self.recalled_evidence,
            "metrics": self.metrics,
            "cortex": self.cortex,
            "reason": self.reason,
        }


def parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def calculate_current_boost(recall: HippocampalRecall) -> float:
    """Map hippocampal recall to an additive LIF input current.

    Similarity says the query matched an old trace. Completion gain says CA3
    produced associative units beyond the direct feedforward cue. Evidence and
    terms add a small bounded support signal so the hippocampus can push the
    downstream field toward a spike without dominating it.
    """

    boost = 0.0
    if recall.completed:
        boost += 0.8
    boost += 1.2 * max(0.0, recall.similarity)
    boost += 0.15 * min(len(recall.recalled_terms), 8)
    boost += 0.25 * min(len(recall.recalled_evidence), 3)
    boost += 0.08 * max(0, recall.completion_gain)
    return round(boost, 4)


def recall_from_result(query: str, result: HippocampalResult) -> HippocampalRecall:
    probe = result.probe_result
    metrics = result.metrics.to_packet()
    cortex = [trace.to_packet() for trace in result.cortex[:8]]
    if probe is None:
        return HippocampalRecall(
            enabled=True,
            query=query,
            metrics=metrics,
            cortex=cortex,
            reason="No probe was produced by the hippocampal network.",
        )

    draft = HippocampalRecall(
        enabled=True,
        query=query,
        trace_id=probe.best_trace_id,
        similarity=float(probe.similarity),
        completed=bool(probe.completed),
        completion_gain=int(probe.ca3.completion_gain),
        recalled_terms=list(probe.recalled_terms),
        recalled_evidence=list(probe.recalled_evidence),
        metrics=metrics,
        cortex=cortex,
        reason="CA3 completed the partial cue." if probe.completed else "CA3 did not cross the CA1 match threshold.",
    )
    return HippocampalRecall(
        **{**draft.__dict__, "current_boost": calculate_current_boost(draft)}
    )


def build_recall_from_notes(
    notes: Mapping[str, str],
    query: str,
    today: date | None = None,
    config: HippocampalConfig | None = None,
) -> HippocampalRecall:
    today = today or date.today()
    result = build_hippocampal_lif_memory(
        notes,
        config=config,
        fallback_day=today,
        probe_text=query,
    )
    return recall_from_result(query, result)


def build_recall_from_vault(
    vault: Path | str | None,
    query: str,
    days: int = 30,
    today: date | str | None = None,
    enabled: bool = True,
    config: HippocampalConfig | None = None,
    use_demo_when_missing: bool = False,
) -> HippocampalRecall:
    if not enabled:
        return HippocampalRecall(enabled=False, query=query, reason="Hippocampal recall is disabled.")

    today_date = parse_today(today) if isinstance(today, str) else (today or date.today())
    if vault is None or str(vault).strip() == "":
        if not use_demo_when_missing:
            return HippocampalRecall(enabled=True, query=query, reason="Vault path is empty.")
        notes = demo_notes()
    else:
        vault_path = Path(vault).expanduser()
        if not vault_path.exists():
            if not use_demo_when_missing:
                return HippocampalRecall(enabled=True, query=query, reason=f"Vault path does not exist: {vault_path}")
            notes = demo_notes()
        else:
            notes = find_daily_notes(vault_path, days=max(1, int(days)), today=today_date)

    if not notes:
        return HippocampalRecall(enabled=True, query=query, reason="No daily notes were found for hippocampal training.")
    return build_recall_from_notes(notes, query=query, today=today_date, config=config)


def hippocampal_field_hits(recall: HippocampalRecall) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for idx, text in enumerate(recall.recalled_evidence, start=1):
        hits.append(
            {
                "path": recall.trace_id or "hippocampal_recall",
                "title": f"hippocampal_recall_{idx}",
                "day": "hippocampus",
                "score": recall.current_boost,
                "semantic": recall.similarity,
                "time_weight": 1.0,
                "graph_bonus": 1.0,
                "modifiers": ["hippocampal_recall", "ca3_completion" if recall.completed else "ca3_probe"],
                "snippet": text,
            }
        )
    return hits


def hippocampal_aha_evidence(recall: HippocampalRecall) -> list[dict[str, Any]]:
    return [
        {
            "kind": "hippocampal_recall",
            "source": recall.trace_id or "hippocampal_recall",
            "text": text,
            "score": recall.similarity,
        }
        for text in recall.recalled_evidence
    ]


def inject_recall_current(daily_current: dict[date, float], today: date, recall: HippocampalRecall) -> None:
    if recall.enabled and recall.current_boost > 0:
        daily_current[today] = daily_current.get(today, 0.0) + recall.current_boost


def render_recall_markdown(recall: HippocampalRecall) -> str:
    packet = recall.to_packet()
    lines = [
        "# Hippocampal Recall",
        "",
        f"- Enabled: `{packet['enabled']}`",
        f"- Trace: `{packet['trace_id']}`",
        f"- Similarity: `{packet['similarity']}`",
        f"- Completed: `{packet['completed']}`",
        f"- CA3 completion gain: `{packet['completion_gain']}`",
        f"- LIF current boost: `{packet['current_boost']}`",
        f"- Reason: {packet['reason'] or 'n/a'}",
        "",
        "## Recalled terms",
        "",
    ]
    terms = ", ".join(recall.recalled_terms) if recall.recalled_terms else "None"
    lines.append(terms)
    lines.extend(["", "## Recalled evidence", ""])
    if not recall.recalled_evidence:
        lines.append("- None")
    for item in recall.recalled_evidence:
        lines.append(f"- {item}")
    lines.extend(["", "## Metrics", "", "```json", json.dumps(recall.metrics, ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines).rstrip() + "\n"
