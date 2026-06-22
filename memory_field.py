from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


VERSION = "0.8.0-memory-field"

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)


DIMENSIONS: dict[str, dict[str, object]] = {
    "experiment_loop": {
        "label": "实验闭环状态",
        "keywords": ["实验", "测试", "LIF", "KS1092", "SSVEP", "EEG", "后向散射", "USRP", "波形", "PSD", "事件率", "联调", "阈值"],
        "decay": 0.83,
        "slow_decay": 0.92,
        "threshold": 4.2,
        "completion_words": ["跑通", "测出来", "完成", "记录", "截图", "保存"],
    },
    "thesis_loop": {
        "label": "论文证据链状态",
        "keywords": ["论文", "章节", "第四章", "第三章", "盲审", "主线", "逻辑", "证据", "图", "创新点", "写作"],
        "decay": 0.86,
        "slow_decay": 0.94,
        "threshold": 4.0,
        "completion_words": ["写完", "放进论文", "改完", "整理", "形成"],
    },
    "career_transition": {
        "label": "求职表达状态",
        "keywords": ["简历", "求职", "岗位", "投递", "实习", "工作", "AI", "大模型", "嵌入式", "项目表达"],
        "decay": 0.88,
        "slow_decay": 0.96,
        "threshold": 3.6,
        "completion_words": ["投递", "写入简历", "改简历", "收藏岗位"],
    },
    "emotion_load": {
        "label": "情绪负荷状态",
        "keywords": ["焦虑", "难受", "害怕", "崩", "延毕", "动不了", "压力", "失败", "累", "迷茫", "羞耻"],
        "decay": 0.76,
        "slow_decay": 0.88,
        "threshold": 3.2,
        "completion_words": ["休息", "散步", "睡觉", "恢复", "吃饭", "运动"],
    },
    "ai_memory_design": {
        "label": "AI 记忆系统设计状态",
        "keywords": ["LIF-Memory", "记忆", "agent", "智能体", "NeRF", "重建", "一致性", "状态场", "洞察", "Obsidian"],
        "decay": 0.84,
        "slow_decay": 0.94,
        "threshold": 3.8,
        "completion_words": ["升级", "测试", "跑一次", "规则", "模块"],
    },
}


VIEWS: dict[str, dict[str, object]] = {
    "thesis": {
        "label": "论文视角",
        "weights": {"thesis_loop": 1.0, "experiment_loop": 0.65, "emotion_load": 0.25},
        "question": "这个状态如何影响论文主线和证据链？",
        "need": ["论文", "证据", "实验"],
    },
    "experiment": {
        "label": "实验视角",
        "weights": {"experiment_loop": 1.0, "thesis_loop": 0.35, "emotion_load": 0.25},
        "question": "这个状态下一步最小可验证实验是什么？",
        "need": ["实验", "测试", "波形"],
    },
    "career": {
        "label": "求职视角",
        "weights": {"career_transition": 1.0, "ai_memory_design": 0.45, "thesis_loop": 0.25},
        "question": "这个状态如何转化成项目表达或简历证据？",
        "need": ["简历", "求职", "项目"],
    },
    "emotion": {
        "label": "情绪视角",
        "weights": {"emotion_load": 1.0, "thesis_loop": 0.35, "experiment_loop": 0.35},
        "question": "这个状态是否正在把任务失败解释成自我失败？",
        "need": ["难受", "压力", "恢复"],
    },
    "memory": {
        "label": "记忆系统视角",
        "weights": {"ai_memory_design": 1.0, "thesis_loop": 0.35, "experiment_loop": 0.35},
        "question": "这个状态如何被重建、验证，并转成下一轮规则？",
        "need": ["记忆", "重建", "一致性"],
    },
}


@dataclass(frozen=True)
class Observation:
    day: date
    source: str
    text: str
    keywords: tuple[str, ...]
    dimensions: tuple[str, ...]
    intensity: float
    completion: float = 0.0

    def to_packet(self) -> dict[str, object]:
        return {
            "day": self.day.isoformat(),
            "source": self.source,
            "text": self.text,
            "keywords": list(self.keywords),
            "dimensions": list(self.dimensions),
            "intensity": round(self.intensity, 3),
            "completion": round(self.completion, 3),
        }


@dataclass
class LatentCell:
    name: str
    label: str
    v_fast: float = 0.0
    v_slow: float = 0.0
    v: float = 0.0
    threshold: float = 1.0
    evidence: list[Observation] = field(default_factory=list)

    @property
    def pressure_ratio(self) -> float:
        if self.threshold <= 0:
            return 0.0
        return self.v / self.threshold

    def to_packet(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "v_fast": round(self.v_fast, 3),
            "v_slow": round(self.v_slow, 3),
            "v": round(self.v, 3),
            "threshold": round(self.threshold, 3),
            "pressure_ratio": round(self.pressure_ratio, 3),
            "evidence_count": len(self.evidence),
            "top_evidence": [item.to_packet() for item in self.evidence[:3]],
        }


@dataclass(frozen=True)
class RenderedView:
    name: str
    label: str
    question: str
    score: float
    confidence: float
    explanation: str
    next_action: str
    support: tuple[Observation, ...]
    missing_observations: tuple[str, ...]

    def to_packet(self) -> dict[str, object]:
        return {
            "name": self.name,
            "label": self.label,
            "question": self.question,
            "score": round(self.score, 3),
            "confidence": round(self.confidence, 3),
            "explanation": self.explanation,
            "next_action": self.next_action,
            "support": [item.to_packet() for item in self.support],
            "missing_observations": list(self.missing_observations),
        }


@dataclass(frozen=True)
class MemoryFieldResult:
    version: str
    dominant_state: str
    latent_cells: Mapping[str, LatentCell]
    rendered_views: Mapping[str, RenderedView]
    consistency_loss: float
    observations: Sequence[Observation]

    def to_packet(self) -> dict[str, object]:
        return {
            "version": self.version,
            "dominant_state": self.dominant_state,
            "latent_cells": {name: cell.to_packet() for name, cell in self.latent_cells.items()},
            "rendered_views": {name: view.to_packet() for name, view in self.rendered_views.items()},
            "consistency_loss": round(self.consistency_loss, 3),
            "observations": [item.to_packet() for item in self.observations],
        }


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


def split_blocks(text: str) -> list[str]:
    text = normalize_text(text)
    raw_blocks = re.split(r"[\n。！？!?；;]+", text)
    blocks = [re.sub(r"\s+", " ", block).strip(" -\t") for block in raw_blocks]
    return [block for block in blocks if len(block) >= 6]


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


def infer_day(source: str, fallback: date | None = None) -> date:
    match = DATE_RE.search(source)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return fallback or date.today()


def short_text(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def block_specificity(block: str) -> float:
    number_hits = len(re.findall(r"\d+(?:\.\d+)?", block))
    symbol_hits = sum(token in block for token in ["->", "→", "/", "%", "mV", "Hz", "V", "kΩ", "Ω"])
    uppercase_hits = len(re.findall(r"\b[A-Z]{2,}\b", block))
    return min(1.0, 0.10 * number_hits + 0.12 * symbol_hits + 0.08 * uppercase_hits)


def extract_observations(notes: Mapping[str, str], fallback_day: date | None = None) -> list[Observation]:
    observations: list[Observation] = []
    for source, text in notes.items():
        day = infer_day(source, fallback=fallback_day)
        for block in split_blocks(text):
            dims: list[str] = []
            keywords: list[str] = []
            completion = 0.0
            for dim, config in DIMENSIONS.items():
                hits = matched_words(block, config["keywords"])  # type: ignore[arg-type]
                if not hits:
                    continue
                dims.append(dim)
                keywords.extend(hit for hit in hits if hit not in keywords)
                completion_hits = matched_words(block, config["completion_words"])  # type: ignore[arg-type]
                if completion_hits:
                    completion = max(completion, 0.75)
            if not dims:
                continue

            specificity = block_specificity(block)
            intensity = min(2.2, 0.55 + 0.28 * len(keywords) + 0.45 * specificity)
            if contains_any(block, ["今天", "现在", "必须", "接下来", "不知道", "卡", "失败", "难受"]):
                intensity += 0.35
            observations.append(
                Observation(
                    day=day,
                    source=source,
                    text=short_text(block),
                    keywords=tuple(keywords[:8]),
                    dimensions=tuple(dims),
                    intensity=round(min(intensity, 2.5), 4),
                    completion=completion,
                )
            )
    observations.sort(key=lambda item: (item.day, item.source, item.text))
    return observations


def update_latent_cells(observations: Sequence[Observation]) -> dict[str, LatentCell]:
    cells = {
        dim: LatentCell(
            name=dim,
            label=str(config["label"]),
            threshold=float(config["threshold"]),
        )
        for dim, config in DIMENSIONS.items()
    }
    if not observations:
        return cells

    last_day = min(item.day for item in observations)
    for obs in observations:
        delta_days = max(0, (obs.day - last_day).days)
        for dim, cell in cells.items():
            config = DIMENSIONS[dim]
            fast_decay = float(config["decay"]) ** delta_days
            slow_decay = float(config["slow_decay"]) ** delta_days
            cell.v_fast *= fast_decay
            cell.v_slow *= slow_decay

        for dim in obs.dimensions:
            cell = cells[dim]
            evidence_input = obs.intensity
            completion_inhibition = obs.completion * 0.65
            cell.v_fast = max(0.0, cell.v_fast + evidence_input - completion_inhibition)
            cell.v_slow = max(0.0, cell.v_slow + evidence_input * 0.45 - completion_inhibition * 0.25)
            cell.evidence.append(obs)

        for cell in cells.values():
            cell.v = 0.72 * cell.v_fast + 0.28 * cell.v_slow
            cell.evidence.sort(key=lambda item: item.intensity, reverse=True)
            cell.evidence = cell.evidence[:8]
        last_day = obs.day
    return cells


def render_explanation(view_name: str, cells: Mapping[str, LatentCell]) -> RenderedView:
    view = VIEWS[view_name]
    weights: Mapping[str, float] = view["weights"]  # type: ignore[assignment]
    score = sum(cells[dim].pressure_ratio * weight for dim, weight in weights.items() if dim in cells)
    support_pool: list[Observation] = []
    for dim in weights:
        support_pool.extend(cells[dim].evidence)
    seen: set[tuple[str, str]] = set()
    support: list[Observation] = []
    for item in sorted(support_pool, key=lambda obs: obs.intensity, reverse=True):
        key = (item.source, item.text)
        if key in seen:
            continue
        seen.add(key)
        support.append(item)
        if len(support) >= 4:
            break

    support_text = " ".join(item.text for item in support)
    missing = [word for word in view["need"] if word not in support_text]  # type: ignore[index]
    confidence = min(1.0, 0.18 + 0.18 * len(support) + 0.14 * max(0.0, score))
    if missing:
        confidence *= max(0.35, 1.0 - 0.16 * len(missing))

    top_dim = max(weights, key=lambda dim: cells[dim].pressure_ratio if dim in cells else 0.0)
    top_label = cells[top_dim].label if top_dim in cells else top_dim

    if view_name == "thesis":
        explanation = f"从论文视角看，主导潜在状态是「{top_label}」；它需要被翻译成证据链，而不是只停留在感受或灵感。"
        next_action = "输出一个论文证据块：结论句 + 支持证据 + 还缺的观测。"
    elif view_name == "experiment":
        explanation = f"从实验视角看，主导潜在状态是「{top_label}」；下一步应收缩为一个可判定的最小闭环。"
        next_action = "固定一个输入条件，记录输入、参数、输出波形/事件率和失败判据。"
    elif view_name == "career":
        explanation = f"从求职视角看，主导潜在状态是「{top_label}」；它只有转成项目表达才会产生价值。"
        next_action = "写一条简历项目 bullet：问题、方法、硬件/算法、指标。"
    elif view_name == "emotion":
        explanation = f"从情绪视角看，主导潜在状态是「{top_label}」；需要防止把任务阻塞误判为自我失败。"
        next_action = "先做一个 10 分钟恢复动作，再只保留一个最小任务。"
    else:
        explanation = f"从记忆系统视角看，主导潜在状态是「{top_label}」；洞察必须能回指原始证据并暴露缺失观测。"
        next_action = "把本次输出当作规则测试：证据是否足够、缺失项是否明确、行动是否可执行。"

    return RenderedView(
        name=view_name,
        label=str(view["label"]),
        question=str(view["question"]),
        score=score,
        confidence=confidence,
        explanation=explanation,
        next_action=next_action,
        support=tuple(support),
        missing_observations=tuple(missing),
    )


def reconstruct_memory_field(
    notes: Mapping[str, str],
    views: Sequence[str] | None = None,
    fallback_day: date | None = None,
) -> MemoryFieldResult:
    observations = extract_observations(notes, fallback_day=fallback_day)
    cells = update_latent_cells(observations)
    if views is None:
        views = tuple(VIEWS.keys())

    rendered = {name: render_explanation(name, cells) for name in views if name in VIEWS}
    if cells:
        dominant_state = max(cells.values(), key=lambda cell: cell.pressure_ratio).name
    else:
        dominant_state = ""

    if rendered:
        consistency_loss = sum(1.0 - view.confidence for view in rendered.values()) / len(rendered)
    else:
        consistency_loss = 1.0

    return MemoryFieldResult(
        version=VERSION,
        dominant_state=dominant_state,
        latent_cells=cells,
        rendered_views=rendered,
        consistency_loss=consistency_loss,
        observations=observations,
    )


def find_daily_notes(vault: Path, days: int, today: date | None = None) -> dict[str, str]:
    today = today or date.today()
    candidates: list[tuple[date, Path]] = []
    for path in vault.rglob("20??-??-??.md"):
        if any(part in {".git", ".obsidian", ".venv", "node_modules", "__pycache__"} for part in path.parts):
            continue
        day = infer_day(path.name, fallback=today)
        if day <= today:
            candidates.append((day, path))
    candidates.sort(key=lambda pair: pair[0])
    selected = candidates[-max(days, 0):]
    return {str(path): path.read_text(encoding="utf-8", errors="ignore") for _, path in selected}


def render_markdown(result: MemoryFieldResult) -> str:
    lines: list[str] = [
        "# LIF-Memory 潜在状态重建",
        "",
        f"- Version: `{result.version}`",
        f"- Dominant state: `{result.dominant_state}`",
        f"- Consistency loss: `{result.consistency_loss:.3f}`",
        "",
        "## 潜在状态场",
        "",
        "| State | V | Threshold | Ratio | Evidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in result.latent_cells.values():
        lines.append(
            f"| {cell.label} | {cell.v:.2f} | {cell.threshold:.2f} | {cell.pressure_ratio:.2f} | {len(cell.evidence)} |"
        )

    lines.extend(["", "## 多视角渲染", ""])
    for view in result.rendered_views.values():
        lines.extend(
            [
                f"### {view.label}",
                "",
                f"- 问题：{view.question}",
                f"- 分数：{view.score:.2f}",
                f"- 置信度：{view.confidence:.2f}",
                f"- 解释：{view.explanation}",
                f"- 下一步：{view.next_action}",
            ]
        )
        if view.missing_observations:
            lines.append(f"- 缺失观测：{', '.join(view.missing_observations)}")
        if view.support:
            lines.append("- 支持证据：")
            for item in view.support:
                lines.append(f"  - `{item.day.isoformat()}` {item.text}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def demo_notes() -> dict[str, str]:
    return {
        "2026-06-20.md": "今天完成了 SSVEP 频谱图和事件率整理，但是副组和 LIF 链路还没有完全联调上，论文证据链还缺一组可写进第四章的数据。",
        "2026-06-21.md": "接下来需要固定 KS1092 后级输入条件，测试 50 欧姆负载、偏置和 LIF 输出事件率，并保存波形截图。",
        "2026-06-22.md": "我感觉延毕压力很大，难受而且动不了，但也想到 LIF-Memory 可以借鉴 NeRF 的重建和多视角一致性。",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a NeRF-like latent memory field from Obsidian notes.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault root. If omitted, use built-in demo notes.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to read from the vault.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--views", type=str, default=",".join(VIEWS.keys()), help="Comma-separated views to render.")
    parser.add_argument("--demo", action="store_true", help="Run the built-in demo even if --vault is omitted.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    today = datetime.strptime(args.today, "%Y-%m-%d").date() if args.today else date.today()
    selected_views = [item.strip() for item in args.views.split(",") if item.strip()]

    if args.vault:
        notes = find_daily_notes(args.vault, args.days, today=today)
    else:
        notes = demo_notes()

    result = reconstruct_memory_field(notes, views=selected_views, fallback_day=today)
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
