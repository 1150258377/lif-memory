from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import llm_adapter

VERSION = "0.7.4"

DATE_RE = re.compile(r"^(20\d{2})-(\d{2})-(\d{2})$")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)

ACTION_WORDS = [
    "今天",
    "目标",
    "接下来",
    "下一步",
    "需要",
    "必须",
    "应该",
    "测试",
    "修改",
    "写",
    "投",
    "整理",
    "量化",
    "记录",
    "推进",
    "验证",
    "闭环",
]

BLOCKER_WORDS = [
    "卡",
    "不知道",
    "缺",
    "没有",
    "不够",
    "失败",
    "问题",
    "害怕",
    "拖延",
    "延毕",
    "焦虑",
    "混乱",
    "撕扯",
    "难受",
    "动不了",
]

COMPLETION_WORDS = [
    "完成",
    "做完",
    "已经",
    "搞定",
    "跑通",
    "出结果",
    "保存",
    "写完",
    "提交",
    "测出来",
]

INCOMPLETE_WORDS = [
    "未完成",
    "没有完成",
    "还没",
    "尚未",
    "缺",
    "不够",
    "没封住",
    "没有数据",
]

TIME_PRESSURE_WORDS = [
    "今天",
    "明天",
    "截止",
    "提交",
    "答辩",
    "最近",
    "现在",
    "当前",
    "下一阶段",
    "晚上",
    "早上",
]

SIGNAL_RECOVERY_WORDS = [
    "EEG",
    "脑电",
    "LIF",
    "SSVEP",
    "后向散射",
    "USRP",
    "波形",
    "频谱",
    "PSD",
    "事件率",
    "边带",
    "链路",
]

NOVELTY_WORDS = [
    "创新",
    "创新点",
    "突破",
    "第一次",
    "新",
    "定义",
    "提出",
]

STATE_CONTEXT_WORDS: dict[str, list[str]] = {
    "Experiment": ["链路", "设备", "仪器", "测量", "参数", "边带", "频谱", "功耗", "闭环", "数据集"],
    "Thesis": ["写作", "润色", "图表", "参考文献", "格式", "数据表达", "答辩故事", "主线"],
    "Career": ["招聘", "公司", "项目经历", "作品", "offer", "HR", "就业", "机会"],
    "AI_Memory": ["向量", "语义", "工具", "模型", "检索", "主动系统", "记忆系统", "状态变量"],
    "Health": ["休息", "睡眠", "眼睛", "腰", "胃", "搬宿舍", "毕业刺激", "自信", "安全感"],
}

TOPIC_RULES: dict[str, list[str]] = {
    "负阻": ["负阻", "negative resistance", "抵消", "斜率"],
    "LIF链路": ["LIF", "后向散射", "USRP", "EEG", "SSVEP", "事件率", "边带", "链路"],
    "实验数据模板": ["跑数据模板", "数据模板", "实验记录数据", "记录数据", "跑数据", "实验数据", "数据闭环"],
    "AI求职转向": ["AI求职", "AI 世界", "大模型机会", "all in AI", "实习", "简历", "大模型"],
    "论文闭环": ["论文", "第三章", "第四章", "摘要", "图", "逻辑", "证据", "主线"],
    "求职": ["简历", "实习", "找工作", "求职", "岗位", "投递", "国企", "应届", "面试"],
    "AI记忆": ["AI", "agent", "智能体", "Obsidian", "LIF-Memory", "记忆", "向量", "主动"],
    "健康恢复": ["焦虑", "累", "崩", "睡", "身体", "情绪", "宿舍", "压力", "害怕", "撕扯", "难受"],
}

TOPIC_PRIORITY_OVERRIDES = {
    "论文闭环": "P0",
    "LIF链路": "P0",
    "实验数据模板": "P0",
    "AI求职转向": "P0",
    "健康恢复": "P0",
    "负阻": "P1",
    "求职": "P1",
    "AI记忆": "P2",
}

COMPLETION_TARGETS = {
    "论文闭环": "写出一个可放进论文的证据块：一句结论、一张图说明或一个限制条件。",
    "实验数据模板": "建立一份实验数据记录模板：输入、参数、波形/截图、结论、失败判据。",
    "LIF链路": "形成一条 EEG→LIF→后向散射→USRP 的链路证据：波形、PSD、事件率或截图。",
    "负阻": "形成一页负阻隔离结论：可用/不可用/暂不作为主线。",
    "AI求职转向": "形成一版 AI+嵌入式 项目表达，并写入简历或求职材料。",
    "AI记忆": "完成一次最小系统实验：改一条规则、跑一次回放、记录触发是否合理。",
    "健康恢复": "完成一次 10 分钟恢复动作，并只保留一个下一步任务。",
    "求职": "完成一个求职链路动作：一个简历条目、一个岗位收藏或一次投递记录。",
}

CLOSURE_STATUSES = {"open", "done", "downgraded", "ignored", "postponed"}

COOLDOWN_RULES = {
    "done": 2,
    "downgraded": 7,
    "ignored": 3,
    "postponed": 1,
}

LLM_PROVIDER_PRESETS = llm_adapter.PROVIDER_PRESETS

STATE_TOPIC_PRIORS: dict[str, dict[str, float]] = {
    "Experiment": {"实验数据模板": 2.2, "LIF链路": 1.8, "负阻": 1.6, "求职": 0.1, "AI记忆": 0.4},
    "Thesis": {"论文闭环": 2.0, "LIF链路": 0.9, "实验数据模板": 0.8},
    "Career": {"AI求职转向": 2.1, "求职": 1.8, "AI记忆": 0.8},
    "AI_Memory": {"AI记忆": 1.7, "AI求职转向": 1.4, "LIF链路": 0.8},
    "Health": {"健康恢复": 2.0},
}

TOPIC_SECONDARY_STATES = {
    "AI求职转向": ["AI_Memory"],
    "LIF链路": ["Thesis"],
    "实验数据模板": ["Thesis"],
    "负阻": ["Thesis"],
    "论文闭环": ["Experiment"],
}

STATE_FALLBACK_TOPIC = {
    "Experiment": "实验闭环",
    "Thesis": "论文闭环",
    "Career": "求职",
    "AI_Memory": "AI记忆",
    "Health": "健康恢复",
}


@dataclass(frozen=True)
class EvidenceVector:
    target_weight: float
    actionability: float
    urgency: float
    blocker: float
    completion: float
    specificity: float
    novelty: float
    confidence: float
    disambiguation: str | None = None

    def to_packet(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "target_weight": round(self.target_weight, 3),
            "actionability": round(self.actionability, 3),
            "urgency": round(self.urgency, 3),
            "blocker": round(self.blocker, 3),
            "completion": round(self.completion, 3),
            "specificity": round(self.specificity, 3),
            "novelty": round(self.novelty, 3),
            "confidence": round(self.confidence, 3),
        }
        if self.disambiguation:
            payload["disambiguation"] = self.disambiguation
        return payload


@dataclass(frozen=True)
class NeuronConfig:
    theta: float
    decay: float
    reset_ratio: float
    cooldown_days: int
    evidence_cap: float
    keywords: list[str]
    suggestion: str
    slow_decay: float = 0.92
    fast_weight: float = 0.70
    slow_weight: float = 0.30
    slow_input_ratio: float = 0.45
    slow_completion_ratio: float = 0.35


@dataclass
class EvidenceItem:
    day: date
    path: Path
    snippet: str
    score: float
    keywords: list[str]
    modifiers: list[str] = field(default_factory=list)
    vector: EvidenceVector | None = None

    def to_packet(self, vault: Path) -> dict[str, object]:
        packet: dict[str, object] = {
            "note": self.day.isoformat(),
            "path": md_link(self.path, vault),
            "snippet": self.snippet,
            "score": round(self.score, 2),
            "matched_keywords": self.keywords,
            "modifiers": self.modifiers,
        }
        if self.vector:
            packet["evidence_vector"] = self.vector.to_packet()
        return packet


@dataclass
class DailyEvidence:
    evidence: float = 0.0
    completion: float = 0.0
    items: list[EvidenceItem] = field(default_factory=list)


@dataclass
class NeuronState:
    v: float = 0.0
    last_spike_date: date | None = None
    v_fast: float = 0.0
    v_slow: float = 0.0


@dataclass
class TopicHistory:
    days_seen: set[date] = field(default_factory=set)
    completion_count: int = 0
    blocker_count: int = 0
    evidence_count: int = 0
    last_action_policy: str | None = None


@dataclass
class TopicPolicy:
    threshold_delta: float = 0.0
    priority_override: str | None = None
    action_policy_override: str | None = None
    muted: bool = False
    cooldown_days: int = 0
    feedback_count: int = 0
    last_feedback: str | None = None

    def adjusted_threshold(self, base_threshold: float) -> float:
        cooldown_penalty = min(max(self.cooldown_days, 0), 7) * 0.15
        return max(0.5, base_threshold + self.threshold_delta + cooldown_penalty)

    def to_packet(self) -> dict[str, object]:
        return {
            "threshold_delta": round(self.threshold_delta, 3),
            "priority_override": self.priority_override,
            "action_policy_override": self.action_policy_override,
            "muted": self.muted,
            "cooldown_days": self.cooldown_days,
            "feedback_count": self.feedback_count,
            "last_feedback": self.last_feedback,
        }


@dataclass
class SpikeClosure:
    spike_id: str
    topic: str = ""
    primary_state: str = ""
    policy: str = ""
    status: str = "open"
    feedback: str = ""
    completion_evidence: str = ""
    closed_at: str = ""
    checked: bool = False


@dataclass
class Spike:
    day: date
    neuron: str
    voltage: float
    threshold: float
    evidence_items: list[EvidenceItem]
    suggestion: str
    previous_v: float = 0.0
    previous_v_fast: float = 0.0
    previous_v_slow: float = 0.0
    delta_days: int = 1
    leak_factor: float = 1.0
    slow_leak_factor: float = 1.0
    leaked_v: float = 0.0
    leaked_v_fast: float = 0.0
    leaked_v_slow: float = 0.0
    v_fast: float = 0.0
    v_slow: float = 0.0
    evidence_input: float = 0.0
    completion_inhibition: float = 0.0
    topic: str = "general"
    primary_state: str | None = None
    secondary_states: list[str] = field(default_factory=list)
    priority: str = "P2"
    blocker_type: str = "none"
    action_policy: str = "continue"
    completion_target: str = "完成一个可判定的小结果。"
    spike_id: str = ""
    decision_reason: str | None = None
    action_suggestion: str | None = None
    feedback_policy: TopicPolicy | None = None
    status: str = "open"
    feedback: str = ""
    completion_evidence: str = ""
    closed_at: str = ""


NEURONS: dict[str, NeuronConfig] = {
    "Experiment": NeuronConfig(
        theta=7.5,
        decay=0.82,
        reset_ratio=0.35,
        cooldown_days=1,
        evidence_cap=6.5,
        keywords=[
            "负阻",
            "测试",
            "波形",
            "KS1092",
            "LIF",
            "SSVEP",
            "EEG",
            "事件率",
            "RMS",
            "后向散射",
            "USRP",
            "实验",
            "数据",
            "ADC",
            "前端",
            "阈值",
            "压缩比",
            "恢复",
        ],
        suggestion="接下来 30 分钟只做一个可记录实验动作：测一组关键波形或阈值数据，并把截图/数值写回实验笔记。",
        slow_decay=0.88,
        slow_input_ratio=0.40,
        slow_completion_ratio=0.45,
    ),
    "Thesis": NeuronConfig(
        theta=7.0,
        decay=0.84,
        reset_ratio=0.38,
        cooldown_days=1,
        evidence_cap=6.2,
        keywords=[
            "论文",
            "第三章",
            "第四章",
            "盲审",
            "答辩",
            "提交",
            "证据",
            "图",
            "章节",
            "导师",
            "逻辑",
            "写进论文",
            "创新点",
            "摘要",
        ],
        suggestion="打开论文主文档，只处理一个证据块：把现有数据变成一句结论、一张图说明或一个可答辩的限制条件。",
        slow_decay=0.92,
        slow_input_ratio=0.50,
        slow_completion_ratio=0.40,
    ),
    "Career": NeuronConfig(
        theta=6.5,
        decay=0.80,
        reset_ratio=0.35,
        cooldown_days=1,
        evidence_cap=5.8,
        keywords=[
            "简历",
            "实习",
            "工作",
            "求职",
            "面试",
            "大模型",
            "国企",
            "央企",
            "应届",
            "岗位",
            "投递",
            "机会",
            "华为",
            "GitHub",
        ],
        suggestion="只推进求职链路里的一个小动作：改一个简历项目条目，或投递/收藏一个和 AI+硬件相关的岗位。",
        slow_decay=0.96,
        slow_input_ratio=0.55,
        slow_completion_ratio=0.25,
    ),
    "AI_Memory": NeuronConfig(
        theta=7.2,
        decay=0.83,
        reset_ratio=0.36,
        cooldown_days=1,
        evidence_cap=6.0,
        keywords=[
            "AI",
            "大模型",
            "agent",
            "智能体",
            "记忆",
            "主动",
            "RAG",
            "Obsidian",
            "skill",
            "LIF-Memory",
            "snn",
            "SNN",
            "脉冲",
            "Codex",
        ],
        suggestion="把 AI 记忆想法压成一个最小可验证实验：补一条规则、跑一次回放、记录一次触发是否合理。",
        slow_decay=0.94,
        slow_input_ratio=0.50,
        slow_completion_ratio=0.30,
    ),
    "Health": NeuronConfig(
        theta=5.8,
        decay=0.78,
        reset_ratio=0.40,
        cooldown_days=1,
        evidence_cap=5.2,
        keywords=[
            "焦虑",
            "累",
            "崩",
            "睡",
            "身体",
            "情绪",
            "宿舍",
            "运动",
            "恢复",
            "压力",
            "疲惫",
            "害怕",
            "撕扯",
            "难受",
            "色情",
            "吃",
        ],
        suggestion="先做一个恢复动作：离开屏幕 10 分钟，喝水或走动，然后只回到一个最小任务，不重新规划整个人生。",
        slow_decay=0.84,
        slow_input_ratio=0.30,
        slow_completion_ratio=0.65,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay Obsidian daily notes as LIF memory states.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to the nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=7, help="Number of latest daily notes to replay.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON event packet output path.")
    parser.add_argument("--feedback-file", type=Path, default=None, help="Optional JSON feedback file that updates topic policies.")
    parser.add_argument("--closure-file", type=Path, default=None, help="Optional Markdown report whose Spike feedback section closes old spikes.")
    parser.add_argument("--feedback-memory", type=Path, default=None, help="Persistent JSON feedback memory. Defaults to lif_memory_feedback.json beside this script.")
    parser.add_argument("--completion-scan", action="store_true", help="Scan local files for external completion signals.")
    parser.add_argument("--daily-spike-budget", type=int, default=2, help="Maximum spike cards emitted per replay day.")
    parser.add_argument("--mode", choices=["replay", "daily"], default="replay", help="Render full replay or daily top spike card.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top spikes to render in daily mode.")
    llm_adapter.add_cli_args(parser)
    parser.add_argument(
        "--states",
        type=str,
        default=",".join(NEURONS.keys()),
        help="Comma-separated states to replay, for example: Experiment,Thesis,Career.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Markdown report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF-Memory {VERSION}")
    return parser.parse_args()


def vault_root_from_script() -> Path:
    script_path = Path(__file__).resolve()
    search_roots = [Path.cwd().resolve(), *Path.cwd().resolve().parents, script_path.parent, *script_path.parents]
    seen: set[Path] = set()
    for candidate in search_roots:
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / ".obsidian").exists():
            return candidate
    return Path.cwd().resolve()


def parse_cutoff(value: str | None) -> date:
    if value is None:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def note_date(path: Path) -> date | None:
    match = DATE_RE.match(path.stem)
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def ignored_path(path: Path) -> bool:
    ignored_parts = {".git", ".obsidian", ".trash", "__pycache__", ".venv", "node_modules", "examples", "tests"}
    return any(part in ignored_parts for part in path.parts)


def find_daily_notes(vault: Path, cutoff: date, days: int) -> list[tuple[date, Path]]:
    if days <= 0:
        return []

    candidates: dict[date, Path] = {}
    for path in vault.rglob("20??-??-??.md"):
        if ignored_path(path):
            continue
        day = note_date(path)
        if day is None or day > cutoff:
            continue
        # Prefer shallower daily notes when duplicate dates exist.
        current = candidates.get(day)
        if current is None or len(path.parts) < len(current.parts):
            candidates[day] = path

    return sorted(candidates.items())[-days:]


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    return text


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


def split_blocks(text: str) -> list[str]:
    text = normalize_text(text)
    raw_blocks = re.split(r"[\n。！？!?；;]+", text)
    blocks = [re.sub(r"\s+", " ", block).strip(" -\t") for block in raw_blocks]
    return [block for block in blocks if len(block) >= 4]


def short_reason(block: str, limit: int = 96) -> str:
    block = re.sub(r"\s+", " ", block).strip()
    if len(block) <= limit:
        return block
    return block[: limit - 1] + "…"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(value, upper))


def specificity_score(block: str) -> float:
    number_hits = len(re.findall(r"\d+(?:\.\d+)?", block))
    uppercase_hits = len(re.findall(r"\b[A-Z]{2,}\b", block))
    symbol_hits = sum(token in block for token in ["->", "→", "|", "/", "%", "mV", "Hz"])
    return clamp((number_hits * 0.12) + (uppercase_hits * 0.10) + (symbol_hits * 0.12), upper=1.0)


def target_weight_for_state(
    state_name: str,
    block: str,
    keyword_hits: list[str],
    context_hits: list[str],
) -> tuple[float, str | None]:
    hit_strength = min(len(keyword_hits), 5) * 0.20
    context_strength = min(len(context_hits), 4) * 0.08
    target_weight = clamp(0.20 + hit_strength + context_strength, upper=1.0)
    disambiguation: str | None = None

    signal_recovery = "恢复" in keyword_hits and contains_any(block, SIGNAL_RECOVERY_WORDS)
    if state_name == "Health" and signal_recovery:
        strong_health_hits = matched_words(block, ["焦虑", "累", "崩", "睡", "身体", "情绪", "压力", "疲惫", "害怕", "撕扯", "难受"])
        if not strong_health_hits:
            target_weight *= 0.15
            disambiguation = "signal_recovery_not_health_recovery"

    if state_name == "Experiment" and signal_recovery:
        target_weight = min(1.0, target_weight + 0.20)
        disambiguation = "signal_recovery_as_experiment_evidence"

    if state_name == "Thesis" and keyword_hits == ["导师"]:
        target_weight *= 0.45
        disambiguation = "mentor_without_thesis_context"

    return target_weight, disambiguation


def evidence_vector_for_block(
    state_name: str,
    block: str,
    keyword_hits: list[str],
    context_hits: list[str],
) -> EvidenceVector:
    actionability = 1.0 if contains_any(block, ACTION_WORDS) else 0.0
    urgency = 1.0 if contains_any(block, TIME_PRESSURE_WORDS) else 0.0
    blocker = 1.0 if contains_any(block, BLOCKER_WORDS) else 0.0
    completion = 1.0 if contains_any(block, COMPLETION_WORDS) and not contains_any(block, INCOMPLETE_WORDS) else 0.0
    specificity = specificity_score(block)
    novelty = 1.0 if contains_any(block, NOVELTY_WORDS) else 0.0
    target_weight, disambiguation = target_weight_for_state(state_name, block, keyword_hits, context_hits)
    confidence = clamp(0.42 + len(keyword_hits) * 0.10 + len(context_hits) * 0.04 + specificity * 0.18)

    return EvidenceVector(
        target_weight=target_weight,
        actionability=actionability,
        urgency=urgency,
        blocker=blocker,
        completion=completion,
        specificity=specificity,
        novelty=novelty,
        confidence=confidence,
        disambiguation=disambiguation,
    )


def vector_score(vector: EvidenceVector) -> tuple[float, list[str], float]:
    modifiers: list[str] = ["vector_scored"]
    if vector.actionability:
        modifiers.append("action")
    if vector.blocker:
        modifiers.append("blocker")
    if vector.urgency:
        modifiers.append("time_pressure")
    if vector.specificity >= 0.25:
        modifiers.append("specific")
    if vector.novelty:
        modifiers.append("novelty")
    if vector.disambiguation:
        modifiers.append(vector.disambiguation)

    base = 0.35 + 2.45 * vector.target_weight
    multiplier = (
        0.72
        + 0.22 * vector.actionability
        + 0.18 * vector.urgency
        + 0.30 * vector.blocker
        + 0.16 * vector.specificity
        + 0.18 * vector.novelty
    )
    score = base * multiplier * vector.confidence

    completion = 0.0
    if vector.completion:
        completion = 0.85
        modifiers.append("completion_inhibition")

    return score, modifiers, completion


def extract_daily_evidence(day: date, path: Path, text: str, active_neurons: dict[str, NeuronConfig]) -> dict[str, DailyEvidence]:
    result = {name: DailyEvidence() for name in active_neurons}

    for block in split_blocks(text):
        for name, config in active_neurons.items():
            keyword_hits = matched_words(block, config.keywords)
            context_hits = matched_words(block, STATE_CONTEXT_WORDS.get(name, []))
            if not keyword_hits and len(context_hits) < 2:
                continue

            vector = evidence_vector_for_block(name, block, keyword_hits, context_hits)
            if vector.target_weight < 0.18:
                continue

            score, modifiers, completion = vector_score(vector)
            if score < 0.18:
                continue

            item = EvidenceItem(
                day=day,
                path=path,
                snippet=short_reason(block),
                score=score,
                keywords=[*keyword_hits, *[hit for hit in context_hits if hit not in keyword_hits]][:8],
                modifiers=modifiers,
                vector=vector,
            )

            bucket = result[name]
            bucket.evidence += score
            bucket.completion += completion
            bucket.items.append(item)

    for name, bucket in result.items():
        bucket.evidence = min(bucket.evidence, active_neurons[name].evidence_cap)
        bucket.completion = min(bucket.completion, 2.2)
        bucket.items.sort(key=lambda item: item.score, reverse=True)
        bucket.items = bucket.items[:6]

    return result


def completion_state_for_path(path: Path) -> str | None:
    text = str(path).lower()
    suffix = path.suffix.lower()
    if suffix in {".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".svg"} and any(
        word in text for word in ["实验", "数据", "wave", "figure", "plot", "result"]
    ):
        return "Experiment"
    if suffix in {".docx", ".tex", ".md", ".pdf"} and any(word in text for word in ["论文", "thesis", "章节", "盲审"]):
        return "Thesis"
    if suffix in {".pdf", ".docx"} and any(word in text for word in ["简历", "resume", "cv"]):
        return "Career"
    if any(word in text for word in ["p2_lif-memory", "lif-memory"]) and suffix in {".py", ".md", ".json"}:
        return "AI_Memory"
    return None


def scan_completion_signals(
    vault: Path,
    notes: list[tuple[date, Path]],
    active_neurons: dict[str, NeuronConfig],
) -> dict[date, dict[str, list[Path]]]:
    note_days = {day for day, _ in notes}
    if not note_days:
        return {}
    signals: dict[date, dict[str, list[Path]]] = {}
    for path in vault.rglob("*"):
        if not path.is_file() or ignored_path(path):
            continue
        state = completion_state_for_path(path)
        if state is None or state not in active_neurons:
            continue
        modified_day = datetime.fromtimestamp(path.stat().st_mtime).date()
        if modified_day not in note_days:
            continue
        signals.setdefault(modified_day, {}).setdefault(state, []).append(path)
    return signals


def apply_completion_signals(
    day: date,
    daily: dict[str, DailyEvidence],
    completion_signals: dict[date, dict[str, list[Path]]] | None,
) -> None:
    if not completion_signals:
        return
    for state, paths in completion_signals.get(day, {}).items():
        if state not in daily:
            continue
        amount = min(1.4, 0.7 * len(paths))
        daily[state].completion = min(2.2, daily[state].completion + amount)


def item_has_blocker(item: EvidenceItem) -> bool:
    return "blocker" in item.modifiers or bool(item.vector and item.vector.blocker)


def item_has_completion(item: EvidenceItem) -> bool:
    return "completion_inhibition" in item.modifiers or bool(item.vector and item.vector.completion)


def item_topics(item: EvidenceItem) -> list[str]:
    text = " ".join([item.snippet, *item.keywords])
    topics: list[str] = []
    for topic, words in TOPIC_RULES.items():
        if contains_any(text, words):
            topics.append(topic)
    return topics


def topic_match_score(topic: str, item: EvidenceItem) -> float:
    text = " ".join([item.snippet, *item.keywords])
    matches = matched_words(text, TOPIC_RULES.get(topic, []))
    if not matches:
        return 0.0
    score = item.score * (1.0 + min(len(matches), 4) * 0.25)
    if topic == "实验数据模板" and contains_any(text, ["实验", "跑数据", "记录数据", "数据模板", "数据闭环"]):
        score *= 1.8
    if topic == "求职" and not contains_any(text, ["简历", "实习", "找工作", "求职", "岗位", "投递", "国企", "应届", "面试"]):
        score *= 0.1
    if topic == "AI求职转向" and contains_any(text, ["简历", "实习", "岗位", "大模型", "all in AI"]):
        score *= 1.6
    return score


def infer_topic(neuron: str, items: list[EvidenceItem]) -> str:
    scores: dict[str, float] = {}
    for item in items:
        for topic in item_topics(item):
            prior = STATE_TOPIC_PRIORS.get(neuron, {}).get(topic, 1.0)
            scores[topic] = scores.get(topic, 0.0) + topic_match_score(topic, item) * prior
    if scores:
        return max(scores.items(), key=lambda pair: pair[1])[0]
    return STATE_FALLBACK_TOPIC.get(neuron, neuron)


def infer_secondary_states(primary_state: str, topic: str, items: list[EvidenceItem]) -> list[str]:
    text = " ".join([item.snippet for item in items])
    candidates = list(TOPIC_SECONDARY_STATES.get(topic, []))
    if topic == "AI求职转向":
        candidates.insert(0, "Career")
        if contains_any(text, ["简历", "实习", "岗位", "投递"]):
            candidates.append("AI_Memory")
    if contains_any(text, ["焦虑", "害怕", "难受", "压力", "崩", "撕扯"]):
        candidates.append("Health")
    if contains_any(text, ["论文", "写进论文", "第四章", "第三章", "主线"]):
        candidates.append("Thesis")
    if contains_any(text, ["实验", "数据", "LIF", "后向散射", "USRP", "负阻"]):
        candidates.append("Experiment")

    unique: list[str] = []
    for state in candidates:
        if state != primary_state and state not in unique:
            unique.append(state)
    return unique[:3]


def update_topic_history(history: dict[str, TopicHistory], daily: dict[str, DailyEvidence]) -> None:
    for bucket in daily.values():
        for item in bucket.items:
            topics = item_topics(item)
            if not topics:
                continue
            for topic in topics:
                record = history.setdefault(topic, TopicHistory())
                record.days_seen.add(item.day)
                record.evidence_count += 1
                if item_has_blocker(item):
                    record.blocker_count += 1
                if item_has_completion(item):
                    record.completion_count += 1


def default_completion_target(neuron: str, topic: str) -> str:
    if topic in COMPLETION_TARGETS:
        return COMPLETION_TARGETS[topic]
    targets = {
        "Experiment": "形成一条可写回实验记录的波形、数值或失败判据。",
        "Thesis": "写出一个可放进论文的证据块：一句结论、一张图说明或一个限制条件。",
        "Career": "完成一个求职链路动作：一个简历条目、一个岗位收藏或一次投递记录。",
        "AI_Memory": "完成一次最小系统实验：改一条规则、跑一次回放、记录触发是否合理。",
        "Health": "完成一次可记录的恢复动作，并把下一步缩小到一个最小任务。",
    }
    return targets.get(neuron, "完成一个可判定的小结果。")


def feedback_items_from_json(data: object) -> list[dict[str, object]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        raw_items = data.get("feedback", data.get("items", []))
        if isinstance(raw_items, list):
            return [item for item in raw_items if isinstance(item, dict)]
    return []


def normalize_feedback(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "_")


def apply_feedback_item(policy: TopicPolicy, item: dict[str, object]) -> None:
    label = normalize_feedback(item.get("feedback", item.get("label")))
    policy.feedback_count += 1
    policy.last_feedback = str(item.get("feedback", item.get("label", ""))).strip() or None

    if label in {"有用", "useful", "helpful"}:
        policy.threshold_delta -= 0.25
        policy.cooldown_days = max(0, policy.cooldown_days - 1)
    elif label in {"没用", "无用", "useless", "not_useful"}:
        policy.threshold_delta += 1.0
        policy.priority_override = "P2"
        policy.cooldown_days = max(policy.cooldown_days, 3)
    elif label in {"太早", "too_early"}:
        policy.threshold_delta += 0.75
        policy.cooldown_days = max(policy.cooldown_days, 2)
    elif label in {"太晚", "too_late"}:
        policy.threshold_delta -= 0.75
        policy.priority_override = "P1"
        policy.cooldown_days = max(0, policy.cooldown_days - 1)
    elif label in {"已完成", "完成", "done", "completed"}:
        policy.priority_override = "P2"
        policy.action_policy_override = "stop"
        policy.cooldown_days = max(policy.cooldown_days, 7)
    elif label in {"不要再提醒", "静音", "mute", "muted"}:
        policy.muted = True
        policy.priority_override = "P2"
        policy.action_policy_override = "stop"
        policy.cooldown_days = max(policy.cooldown_days, 30)
    elif label in {"升为p0", "p0", "raise_p0"}:
        policy.priority_override = "P0"
    elif label in {"降为p2", "p2", "lower_p2"}:
        policy.priority_override = "P2"

    if "threshold_delta" in item:
        policy.threshold_delta += float(item["threshold_delta"])
    if "priority" in item and item["priority"] is not None:
        policy.priority_override = str(item["priority"])
    if "action_policy" in item and item["action_policy"] is not None:
        policy.action_policy_override = str(item["action_policy"])
    if "cooldown_days" in item:
        policy.cooldown_days = max(policy.cooldown_days, int(item["cooldown_days"]))
    if "muted" in item:
        policy.muted = bool(item["muted"])


def apply_feedback(
    topic_policies: dict[str, TopicPolicy],
    feedback_items: list[dict[str, object]],
) -> dict[str, TopicPolicy]:
    for item in feedback_items:
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        policy = topic_policies.setdefault(topic, TopicPolicy())
        apply_feedback_item(policy, item)
    return topic_policies


def load_feedback_file(path: Path | None) -> dict[str, TopicPolicy]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return apply_feedback({}, feedback_items_from_json(data))


def default_feedback_memory_path() -> Path:
    return Path(__file__).resolve().with_name("lif_memory_feedback.json")


def parse_memory_date(value: object) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def make_spike_id(day: date, primary_state: str, topic: str) -> str:
    return f"{day.isoformat()}-{primary_state}-{topic}"


def normalize_closure_status(value: str, checked: bool = False) -> str:
    status = value.strip().lower().replace(" ", "_")
    aliases = {
        "已完成": "done",
        "完成": "done",
        "关闭": "done",
        "降级": "downgraded",
        "已降级": "downgraded",
        "忽略": "ignored",
        "推迟": "postponed",
        "延期": "postponed",
    }
    status = aliases.get(status, status)
    if status not in CLOSURE_STATUSES:
        status = "open"
    if checked and status == "open":
        return "done"
    return status


def closure_topic_from_spike_id(spike_id: str) -> tuple[str, str]:
    parts = spike_id.split("-", 4)
    if len(parts) == 5:
        return parts[4], parts[3]
    return "", ""


def load_spike_closures(path: Path | None) -> dict[str, SpikeClosure]:
    if path is None or not path.exists():
        return {}

    text = path.read_text(encoding="utf-8", errors="ignore")
    marker = "## Spike 反馈区"
    start = text.find(marker)
    if start < 0:
        return {}

    section = text[start + len(marker) :]
    next_section = re.search(r"\n##\s+", section)
    if next_section:
        section = section[: next_section.start()]

    closures: dict[str, SpikeClosure] = {}
    current: SpikeClosure | None = None
    item_re = re.compile(r"^\s*-\s+\[([ xX])\]\s+(.+?)\s*$")
    meta_re = re.compile(r"^\s*-\s*([^：:]+)[：:]\s*(.*?)\s*$")

    for raw_line in section.splitlines():
        item_match = item_re.match(raw_line)
        if item_match:
            checked = item_match.group(1).lower() == "x"
            spike_id = item_match.group(2).strip()
            topic, primary_state = closure_topic_from_spike_id(spike_id)
            current = SpikeClosure(
                spike_id=spike_id,
                topic=topic,
                primary_state=primary_state,
                checked=checked,
                status="done" if checked else "open",
            )
            closures[spike_id] = current
            continue

        if current is None:
            continue
        meta_match = meta_re.match(raw_line)
        if not meta_match:
            continue
        key = meta_match.group(1).strip().lower()
        value = meta_match.group(2).strip()
        if key in {"topic", "主题"}:
            current.topic = value
        elif key in {"primary", "primary state", "primary_state", "主状态"}:
            current.primary_state = value
        elif key in {"policy", "action policy", "action_policy", "策略"}:
            current.policy = value
        elif key in {"状态", "status"}:
            current.status = normalize_closure_status(value, current.checked)
        elif key in {"反馈", "feedback"}:
            current.feedback = value
        elif key in {"完成证据", "completion evidence", "completion_evidence"}:
            current.completion_evidence = value
        elif key in {"关闭时间", "closed at", "closed_at"}:
            current.closed_at = value

    for closure in closures.values():
        closure.status = normalize_closure_status(closure.status, closure.checked)
        if not closure.topic:
            closure.topic, closure.primary_state = closure_topic_from_spike_id(closure.spike_id)
    return closures


def policy_from_closure(closure: SpikeClosure) -> TopicPolicy | None:
    if not closure.topic or closure.status == "open":
        return None

    policy = TopicPolicy()
    if closure.feedback:
        apply_feedback_item(policy, {"topic": closure.topic, "feedback": closure.feedback})
    else:
        policy.feedback_count = 1
        policy.last_feedback = closure.status

    if closure.status == "done":
        policy.threshold_delta += 0.5
        policy.priority_override = policy.priority_override or "P2"
        policy.action_policy_override = policy.action_policy_override or "stop"
    elif closure.status == "downgraded":
        policy.threshold_delta += 1.0
        policy.priority_override = "P2"
        policy.action_policy_override = "downgrade"
    elif closure.status == "ignored":
        policy.threshold_delta += 0.75
        policy.priority_override = "P2"
        policy.action_policy_override = "stop"
    elif closure.status == "postponed":
        policy.threshold_delta += 0.25

    policy.cooldown_days = max(policy.cooldown_days, COOLDOWN_RULES.get(closure.status, 0))
    return policy


def policy_from_memory_record(topic: str, record: dict[str, object], today: date) -> TopicPolicy:
    status = normalize_closure_status(str(record.get("status", "open")))
    cooldown_until = parse_memory_date(record.get("cooldown_until"))
    remaining_cooldown = max((cooldown_until - today).days, 0) if cooldown_until else 0

    policy = TopicPolicy(
        threshold_delta=float(record.get("threshold_delta", 0.0)),
        priority_override=str(record["priority_override"]) if record.get("priority_override") else None,
        action_policy_override=str(record["action_policy_override"]) if record.get("action_policy_override") else None,
        muted=bool(record.get("muted", False)),
        cooldown_days=remaining_cooldown if cooldown_until else int(record.get("cooldown_days", 0)),
        feedback_count=int(record.get("feedback_count", 0)),
        last_feedback=str(record["last_feedback"]) if record.get("last_feedback") else None,
    )

    if status == "downgraded":
        policy.priority_override = "P2"
        policy.action_policy_override = "downgrade"
        policy.threshold_delta = max(policy.threshold_delta, 1.0)
    elif status == "ignored" and remaining_cooldown > 0:
        policy.priority_override = "P2"
        policy.action_policy_override = "stop"
        policy.threshold_delta = max(policy.threshold_delta, 0.75)
    elif status == "done" and remaining_cooldown > 0:
        policy.priority_override = policy.priority_override or "P2"
        policy.action_policy_override = policy.action_policy_override or "stop"
        policy.threshold_delta = max(policy.threshold_delta, 0.5)

    if not policy.last_feedback:
        policy.last_feedback = f"persistent:{status}"
    if policy.feedback_count <= 0:
        policy.feedback_count = 1
    return policy


def load_feedback_memory(path: Path | None, today: date) -> dict[str, TopicPolicy]:
    if path is None or not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_topics = data.get("topics", data)
    if not isinstance(raw_topics, dict):
        return {}

    policies: dict[str, TopicPolicy] = {}
    for topic, raw in raw_topics.items():
        if isinstance(topic, str) and isinstance(raw, dict):
            policies[topic] = policy_from_memory_record(topic, raw, today)
    return policies


def closure_memory_record(closure: SpikeClosure, today: date) -> dict[str, object] | None:
    if not closure.topic or closure.status == "open":
        return None
    policy = policy_from_closure(closure)
    if policy is None:
        return None
    closed_day = parse_memory_date(closure.closed_at) or today
    cooldown_days = COOLDOWN_RULES.get(closure.status, 0)
    cooldown_until = closed_day + timedelta(days=cooldown_days)
    return {
        "status": closure.status,
        "primary_state": closure.primary_state,
        "last_policy": closure.policy,
        "last_feedback": closure.feedback or closure.status,
        "completion_evidence": closure.completion_evidence,
        "closed_at": closed_day.isoformat(),
        "cooldown_until": cooldown_until.isoformat(),
        "threshold_delta": round(policy.threshold_delta, 6),
        "priority_override": policy.priority_override,
        "action_policy_override": policy.action_policy_override,
        "muted": policy.muted,
        "cooldown_days": policy.cooldown_days,
        "feedback_count": policy.feedback_count,
        "spike_ids": [closure.spike_id],
    }


def update_feedback_memory(path: Path | None, closures: dict[str, SpikeClosure], today: date) -> None:
    if path is None:
        return
    closed_records = [closure for closure in closures.values() if closure.status != "open" and closure.topic]
    if not closed_records:
        return

    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            data = {}
    else:
        data = {}

    raw_topics = data.setdefault("topics", {})
    if not isinstance(raw_topics, dict):
        raw_topics = {}
        data["topics"] = raw_topics

    for closure in closed_records:
        record = closure_memory_record(closure, today)
        if record is None:
            continue
        existing = raw_topics.get(closure.topic, {})
        if not isinstance(existing, dict):
            existing = {}
        spike_ids = list(existing.get("spike_ids", [])) if isinstance(existing.get("spike_ids"), list) else []
        if closure.spike_id not in spike_ids:
            spike_ids.append(closure.spike_id)
        existing.update(record)
        existing["spike_ids"] = spike_ids
        raw_topics[closure.topic] = existing

    data["version"] = VERSION
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def policies_from_closures(closures: dict[str, SpikeClosure]) -> dict[str, TopicPolicy]:
    policies: dict[str, TopicPolicy] = {}
    for closure in closures.values():
        new_policy = policy_from_closure(closure)
        if new_policy is None:
            continue
        stored = policies.setdefault(closure.topic, TopicPolicy())
        merge_topic_policy(stored, new_policy)
    return policies


def merge_topic_policy(target: TopicPolicy, source: TopicPolicy) -> TopicPolicy:
    target.threshold_delta += source.threshold_delta
    target.priority_override = source.priority_override or target.priority_override
    target.action_policy_override = source.action_policy_override or target.action_policy_override
    target.muted = target.muted or source.muted
    target.cooldown_days = max(target.cooldown_days, source.cooldown_days)
    target.feedback_count += source.feedback_count
    target.last_feedback = source.last_feedback or target.last_feedback
    return target


def merge_topic_policies(*policy_sets: dict[str, TopicPolicy]) -> dict[str, TopicPolicy]:
    merged: dict[str, TopicPolicy] = {}
    for policies in policy_sets:
        for topic, policy in policies.items():
            merge_topic_policy(merged.setdefault(topic, TopicPolicy()), policy)
    return merged


def apply_closures_to_spikes(spikes: list[Spike], closures: dict[str, SpikeClosure]) -> None:
    for spike in spikes:
        closure = closures.get(spike.spike_id)
        if closure is None:
            continue
        spike.status = closure.status
        spike.feedback = closure.feedback
        spike.completion_evidence = closure.completion_evidence
        spike.closed_at = closure.closed_at


def llm_config_from_args(args: argparse.Namespace) -> llm_adapter.LLMConfig:
    return llm_adapter.config_from_args(args)


def extract_json_object(text: str) -> dict[str, object]:
    return llm_adapter.extract_json_object(text)


def llm_review_prompt(spike: Spike, vault: Path) -> list[dict[str, str]]:
    return llm_adapter.review_prompt(spike_packet(spike, vault), list(NEURONS.keys()))


def call_openai_compatible_chat(config: llm_adapter.LLMConfig, messages: list[dict[str, str]]) -> str:
    return llm_adapter.call_chat_completions(config, messages)


def review_spike_with_llm(spike: Spike, vault: Path, config: llm_adapter.LLMConfig) -> dict[str, object]:
    return llm_adapter.review_spike(spike_packet(spike, vault), list(NEURONS.keys()), config)


def select_spikes_for_llm_review(spikes: list[Spike], mode: str, top_k: int) -> list[Spike]:
    if not spikes:
        return []
    if mode == "daily":
        latest_spike_day = max(spike.day for spike in spikes)
        candidates = [spike for spike in spikes if spike.day == latest_spike_day]
        return sorted(candidates, key=spike_rank, reverse=True)[: max(1, top_k)]
    return sorted(spikes, key=spike_rank, reverse=True)[: max(1, top_k)]


def run_llm_reviews(spikes: list[Spike], vault: Path, args: argparse.Namespace) -> dict[str, dict[str, object]]:
    if not args.llm_review:
        return {}
    config = llm_config_from_args(args)
    selected = select_spikes_for_llm_review(spikes, str(args.mode), int(args.top_k))
    return {spike.spike_id: review_spike_with_llm(spike, vault, config) for spike in selected}


def primary_state_for_spike(neuron: str, topic: str, items: list[EvidenceItem]) -> str:
    text = " ".join(item.snippet for item in items)
    if topic == "AI求职转向":
        return "Career"
    if topic in {"LIF链路", "实验数据模板", "负阻"}:
        return "Experiment"
    if topic == "论文闭环":
        return "Thesis"
    if topic == "健康恢复" or contains_any(text, ["崩", "压力", "害怕", "撕扯", "难受"]):
        return "Health" if neuron == "Health" else neuron
    return neuron


def decide_action(
    neuron: str,
    topic: str,
    voltage: float,
    threshold: float,
    evidence: DailyEvidence,
    topic_history: dict[str, TopicHistory],
    topic_policies: dict[str, TopicPolicy] | None = None,
) -> dict[str, str | None]:
    record = topic_history.get(topic, TopicHistory())
    ratio = voltage / threshold if threshold else 0.0
    text = " ".join(item.snippet for item in evidence.items)
    days_seen = len(record.days_seen)
    completion_count = record.completion_count
    blocker_count = record.blocker_count

    priority = "P0" if ratio >= 1.45 else "P1" if ratio >= 1.0 else "P2"
    blocker_type = "none"
    action_policy = "continue"
    completion_target = default_completion_target(neuron, topic)
    decision_reason: str | None = None
    action_suggestion: str | None = None
    policy = (topic_policies or {}).get(topic)

    has_current_blocker = any(item_has_blocker(item) for item in evidence.items)
    has_current_completion = evidence.completion > 0
    unclear_definition = contains_any(text, ["定义", "不清楚", "不知道", "是什么", "如何理解", "没给等效条件"])
    repeated_failure = (
        days_seen >= 3
        and blocker_count >= 2
        and completion_count <= 1
        and not has_current_completion
    )
    if topic == "负阻" and days_seen >= 2 and has_current_blocker and not has_current_completion:
        repeated_failure = True

    if neuron == "Health":
        blocker_type = "emotional_overload"
        action_policy = "recover_first"
        priority = "P0"
        completion_target = "完成一次 10 分钟恢复动作，并只保留一个下一步任务。"
        decision_reason = "身体或情绪负荷已经超过行动判断本身，先恢复再推进。"
        action_suggestion = "先做一个恢复动作：离开屏幕 10 分钟，喝水或走动，然后只回到一个最小任务。"
    elif repeated_failure:
        blocker_type = "repeated_failure"
        action_policy = "isolate"
        priority = "P1"
        if topic == "负阻":
            decision_reason = "负阻实验反复出现且完成信号不足，正在消耗论文主线推进能力。"
            action_suggestion = "只做一次隔离测试：I-V 小信号斜率、串联采样电阻、信号源/MCU 输出阻抗。若无法稳定复现，则负阻降级为论文补充模块。"
        else:
            decision_reason = f"{topic} 多次出现阻塞且缺少完成信号，需要先隔离最小失败原因。"
            action_suggestion = "只做一次最小隔离测试，得到可继续/降级/暂停的判据后再决定是否推进。"
    elif contains_any(text, ["崩", "压力", "害怕", "撕扯", "很难受", "焦虑"]):
        blocker_type = "emotional_overload"
        action_policy = "recover_first"
        priority = "P0" if ratio >= 1.25 else "P1"
        completion_target = "完成一次 10 分钟恢复动作，并只保留一个下一步任务。"
        decision_reason = "身体或情绪负荷已经超过行动判断本身，先恢复再推进。"
        action_suggestion = "先做一个恢复动作：离开屏幕 10 分钟，喝水或走动，然后只回到一个最小任务。"
    elif unclear_definition:
        blocker_type = "unclear_definition"
        action_policy = "isolate"
        priority = "P1"
        decision_reason = f"{topic} 的定义或判据仍不清楚，需要先把问题压成一个可验证条件。"
        action_suggestion = "先写出一个等效条件或判定标准，再做下一步实验或写作。"

    forced_priority = TOPIC_PRIORITY_OVERRIDES.get(topic)
    if forced_priority:
        priority = forced_priority

    if policy:
        if policy.priority_override:
            priority = policy.priority_override
        if policy.action_policy_override:
            action_policy = policy.action_policy_override
        if policy.last_feedback:
            decision_reason = decision_reason or f"{topic} 已应用用户反馈策略：{policy.last_feedback}。"

    decision = {
        "priority": priority,
        "blocker_type": blocker_type,
        "action_policy": action_policy,
        "completion_target": completion_target,
        "decision_reason": decision_reason,
        "action_suggestion": action_suggestion,
    }
    record.last_action_policy = action_policy
    return decision


def can_spike(state: NeuronState, day: date, config: NeuronConfig) -> bool:
    if state.last_spike_date is None:
        return True
    delta = (day - state.last_spike_date).days
    return delta >= config.cooldown_days


def combined_voltage(v_fast: float, v_slow: float, config: NeuronConfig) -> float:
    total_weight = config.fast_weight + config.slow_weight
    if total_weight <= 0:
        return v_fast
    return (config.fast_weight * v_fast + config.slow_weight * v_slow) / total_weight


def update_voltage_state(
    state: NeuronState,
    config: NeuronConfig,
    evidence: DailyEvidence,
    delta_days: int,
) -> dict[str, float]:
    has_timescale_state = state.v_fast > 0.0 or state.v_slow > 0.0
    old_v = state.v
    old_fast = state.v_fast if has_timescale_state else state.v
    old_slow = state.v_slow if has_timescale_state else state.v
    fast_leak_factor = config.decay ** delta_days
    slow_leak_factor = config.slow_decay ** delta_days
    leaked_fast = fast_leak_factor * old_fast
    leaked_slow = slow_leak_factor * old_slow
    new_fast = max(0.0, leaked_fast + evidence.evidence - evidence.completion)
    new_slow = max(
        0.0,
        leaked_slow
        + evidence.evidence * config.slow_input_ratio
        - evidence.completion * config.slow_completion_ratio,
    )
    new_v = combined_voltage(new_fast, new_slow, config)

    state.v_fast = new_fast
    state.v_slow = new_slow
    state.v = new_v

    return {
        "old_v": old_v,
        "old_fast": old_fast,
        "old_slow": old_slow,
        "new_v": new_v,
        "new_fast": new_fast,
        "new_slow": new_slow,
        "fast_leak_factor": fast_leak_factor,
        "slow_leak_factor": slow_leak_factor,
        "leaked_fast": leaked_fast,
        "leaked_slow": leaked_slow,
        "leaked_v": combined_voltage(leaked_fast, leaked_slow, config),
    }


def reset_after_spike(state: NeuronState, config: NeuronConfig) -> None:
    state.v_fast = config.theta * config.reset_ratio
    state.v_slow = min(state.v_slow, config.theta * 0.60)
    state.v = combined_voltage(state.v_fast, state.v_slow, config)


def replay(
    notes: list[tuple[date, Path]],
    daily_spike_budget: int,
    active_neurons: dict[str, NeuronConfig],
    topic_policies: dict[str, TopicPolicy] | None = None,
    completion_signals: dict[date, dict[str, list[Path]]] | None = None,
) -> tuple[list[Spike], list[dict[str, object]], dict[str, NeuronState]]:
    states = {name: NeuronState() for name in active_neurons}
    spikes: list[Spike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None
    topic_history: dict[str, TopicHistory] = {}
    if topic_policies is None:
        topic_policies = {}

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = extract_daily_evidence(day, path, text, active_neurons)
        apply_completion_signals(day, daily, completion_signals)
        update_topic_history(topic_history, daily)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day

        row: dict[str, object] = {"date": day.isoformat(), "path": path, "delta_days": delta_days}
        candidates: list[tuple[float, str, NeuronConfig, NeuronState, DailyEvidence]] = []

        for name, config in active_neurons.items():
            state = states[name]
            evidence = daily[name]
            voltage_update = update_voltage_state(state, config, evidence, delta_days)
            new_v = voltage_update["new_v"]

            row[name] = {
                "old_v": voltage_update["old_v"],
                "old_fast": voltage_update["old_fast"],
                "old_slow": voltage_update["old_slow"],
                "new_v": new_v,
                "new_fast": voltage_update["new_fast"],
                "new_slow": voltage_update["new_slow"],
                "fast_leak_factor": voltage_update["fast_leak_factor"],
                "slow_leak_factor": voltage_update["slow_leak_factor"],
                "leaked_v": voltage_update["leaked_v"],
                "leaked_fast": voltage_update["leaked_fast"],
                "leaked_slow": voltage_update["leaked_slow"],
                "input": evidence.evidence,
                "completion": evidence.completion,
                "spike": False,
                "evidence_count": len(evidence.items),
            }

            topic = infer_topic(name, evidence.items[:4])
            policy = topic_policies.get(topic)
            effective_threshold = policy.adjusted_threshold(config.theta) if policy else config.theta
            if policy and policy.muted:
                row[name]["muted_topic"] = topic
                row[name]["effective_threshold"] = effective_threshold
                continue
            row[name]["topic"] = topic
            row[name]["effective_threshold"] = effective_threshold

            if new_v >= effective_threshold and can_spike(state, day, config):
                candidates.append((new_v / effective_threshold, name, config, state, evidence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, config, state, evidence in candidates[: max(0, daily_spike_budget)]:
            item = row[name]
            assert isinstance(item, dict)
            topic = str(item.get("topic") or infer_topic(name, evidence.items[:4]))
            policy = topic_policies.get(topic)
            effective_threshold = float(item.get("effective_threshold", config.theta))
            decision = decide_action(name, topic, state.v, effective_threshold, evidence, topic_history, topic_policies)
            primary_state = primary_state_for_spike(name, topic, evidence.items[:4])
            secondary_states = infer_secondary_states(primary_state, topic, evidence.items[:4])
            spike = Spike(
                day=day,
                neuron=name,
                voltage=state.v,
                threshold=effective_threshold,
                evidence_items=evidence.items[:4],
                suggestion=str(decision.get("action_suggestion") or config.suggestion),
                previous_v=float(item["old_v"]),
                previous_v_fast=float(item["old_fast"]),
                previous_v_slow=float(item["old_slow"]),
                delta_days=delta_days,
                leak_factor=float(item["fast_leak_factor"]),
                slow_leak_factor=float(item["slow_leak_factor"]),
                leaked_v=float(item["leaked_v"]),
                leaked_v_fast=float(item["leaked_fast"]),
                leaked_v_slow=float(item["leaked_slow"]),
                v_fast=state.v_fast,
                v_slow=state.v_slow,
                evidence_input=evidence.evidence,
                completion_inhibition=evidence.completion,
                topic=topic,
                primary_state=primary_state,
                secondary_states=secondary_states,
                priority=str(decision["priority"]),
                blocker_type=str(decision["blocker_type"]),
                action_policy=str(decision["action_policy"]),
                completion_target=str(decision["completion_target"]),
                spike_id=make_spike_id(day, primary_state, topic),
                decision_reason=decision.get("decision_reason"),
                action_suggestion=decision.get("action_suggestion"),
                feedback_policy=policy,
            )
            spikes.append(spike)
            state.last_spike_date = day
            reset_after_spike(state, config)

            item["new_v"] = state.v
            item["new_fast"] = state.v_fast
            item["new_slow"] = state.v_slow
            item["spike"] = True

        timeline.append(row)

    return spikes, timeline, states


def md_link(path: Path, vault: Path) -> str:
    try:
        rel = path.relative_to(vault)
    except ValueError:
        rel = path
    return str(rel).replace("\\", "/")


def trigger_reason(spike: Spike) -> str:
    if spike.decision_reason:
        return spike.decision_reason
    names = {
        "Experiment": "实验闭环证据持续积累，且当前任务具有可行动性。",
        "Thesis": "论文证据、章节逻辑或提交压力持续积累，需要收束成一个可写入的证据块。",
        "Career": "求职与机会信号持续积累，需要把长期焦虑转成一次具体推进。",
        "AI_Memory": "AI 记忆系统相关想法持续积累，需要压成一次可验证实验。",
        "Health": "身体或情绪压力信号持续积累，需要先降低系统负荷再回到最小任务。",
    }
    return names.get(spike.neuron, "状态电位超过阈值，需要回查证据并生成下一步行动。")


def spike_packet(spike: Spike, vault: Path) -> dict[str, object]:
    packet: dict[str, object] = {
        "spike_id": spike.spike_id,
        "spike_type": spike.neuron,
        "primary_state": spike.primary_state or spike.neuron,
        "secondary_states": spike.secondary_states,
        "topic": spike.topic,
        "time": spike.day.isoformat(),
        "V": round(spike.voltage, 2),
        "threshold": round(spike.threshold, 2),
        "priority": spike.priority,
        "blocker_type": spike.blocker_type,
        "action_policy": spike.action_policy,
        "status": spike.status,
        "completion_evidence": spike.completion_evidence,
        "closed_at": spike.closed_at,
        "voltage_model": {
            "meaning": "V is accumulated actionable pressure for this state, not raw token count and not reconstructed language.",
            "formula": "V = weighted_sum(V_fast, V_slow); V_fast responds to recent pressure, V_slow preserves background pressure.",
            "previous_V": round(spike.previous_v, 2),
            "previous_V_fast": round(spike.previous_v_fast, 2),
            "previous_V_slow": round(spike.previous_v_slow, 2),
            "delta_days": spike.delta_days,
            "fast_leak_factor": round(spike.leak_factor, 4),
            "slow_leak_factor": round(spike.slow_leak_factor, 4),
            "leaked_V": round(spike.leaked_v, 2),
            "leaked_V_fast": round(spike.leaked_v_fast, 2),
            "leaked_V_slow": round(spike.leaked_v_slow, 2),
            "evidence_input": round(spike.evidence_input, 2),
            "completion_inhibition": round(spike.completion_inhibition, 2),
            "V_fast": round(spike.v_fast, 2),
            "V_slow": round(spike.v_slow, 2),
            "final_V_before_reset": round(spike.voltage, 2),
        },
        "evidence_notes": [item.to_packet(vault) for item in spike.evidence_items],
        "trigger_reason": trigger_reason(spike),
        "suggested_action": spike.suggestion,
        "completion_target": spike.completion_target,
    }
    if spike.feedback_policy:
        packet["feedback_policy"] = spike.feedback_policy.to_packet()
    return packet


def render_summary(
    lines: list[str],
    active_neurons: dict[str, NeuronConfig],
    spikes: list[Spike],
    states: dict[str, NeuronState],
) -> None:
    spike_counter = Counter(spike.neuron for spike in spikes)

    lines.append("## 汇总")
    lines.append("")
    lines.append("| 状态 | 最终电位 | 阈值 | Spike 数 | 解释 |")
    lines.append("|---|---:|---:|---:|---|")
    for name, config in active_neurons.items():
        final_v = states[name].v
        ratio = final_v / config.theta if config.theta else 0.0
        if ratio >= 0.9:
            hint = "接近触发，需要近期处理"
        elif ratio >= 0.55:
            hint = "有趋势，但还没到必须行动"
        else:
            hint = "当前负荷较低"
        lines.append(f"| {name} | {final_v:.2f} | {config.theta:.2f} | {spike_counter[name]} | {hint} |")
    lines.append("")


def render_spike_feedback_section(lines: list[str], spikes: list[Spike]) -> None:
    lines.append("## Spike 反馈区")
    lines.append("")
    lines.append("把 checkbox 改成 `[x]`，并把 `状态` 改成 `done / downgraded / ignored / postponed` 后，下次运行会读取这里并应用冷却。")
    lines.append("")
    if not spikes:
        lines.append("本次没有可反馈的 spike。")
        lines.append("")
        return
    for spike in spikes:
        checked = "x" if spike.status != "open" else " "
        lines.append(f"- [{checked}] {spike.spike_id}")
        lines.append(f"  - Topic：{spike.topic}")
        lines.append(f"  - Primary：{spike.primary_state or spike.neuron}")
        lines.append(f"  - Policy：{spike.action_policy}")
        lines.append(f"  - 状态：{spike.status}")
        lines.append(f"  - 反馈：{spike.feedback}")
        lines.append(f"  - 完成证据：{spike.completion_evidence}")
        lines.append(f"  - 关闭时间：{spike.closed_at}")
    lines.append("")


def render_llm_review_section(lines: list[str], llm_reviews: dict[str, dict[str, object]]) -> None:
    if not llm_reviews:
        return
    lines.append("## LLM Review")
    lines.append("")
    lines.append("LLM Review 只校准语义，不修改电压、阈值、cooldown 或最终 action_policy。")
    lines.append("")
    for spike_id, review in llm_reviews.items():
        lines.append(f"### {spike_id}")
        lines.append("")
        if "error" in review:
            lines.append(f"- Error：{review['error']}")
            lines.append(f"- Reason：{review.get('reason', '')}")
            lines.append("")
            continue
        lines.append(f"- 判断：{review.get('is_correct')}")
        if review.get("corrected_topic"):
            lines.append(f"- corrected_topic：{review['corrected_topic']}")
        if review.get("corrected_primary_state"):
            lines.append(f"- corrected_primary_state：{review['corrected_primary_state']}")
        if review.get("corrected_secondary_states"):
            lines.append(f"- corrected_secondary_states：{review['corrected_secondary_states']}")
        if review.get("better_completion_target"):
            lines.append(f"- better_completion_target：{review['better_completion_target']}")
        if review.get("risk"):
            lines.append(f"- risk：{review['risk']}")
        lines.append(f"- reason：{review.get('reason', '')}")
        lines.append("")


def spike_rank(spike: Spike) -> tuple[int, float, float]:
    priority_weight = {"P0": 3, "P1": 2, "P2": 1}.get(spike.priority, 0)
    ratio = spike.voltage / spike.threshold if spike.threshold else 0.0
    secondary_weight = min(len(spike.secondary_states), 3) * 0.1
    return (priority_weight, ratio + secondary_weight, spike.voltage)


def render_daily_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[Spike],
    top_k: int = 1,
    llm_reviews: dict[str, dict[str, object]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# 今日 LIF-Memory 主卡片")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append("")

    if not spikes:
        lines.append("今天没有超过阈值的主触发事件。")
        lines.append("")
        render_spike_feedback_section(lines, [])
        return "\n".join(lines)

    latest_spike_day = max(spike.day for spike in spikes)
    daily_candidates = [spike for spike in spikes if spike.day == latest_spike_day]
    selected = sorted(daily_candidates, key=spike_rank, reverse=True)[: max(1, top_k)]

    for index, spike in enumerate(selected, start=1):
        title = "今日只处理这一件事" if len(selected) == 1 else f"Top {index}"
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"Topic：{spike.topic}")
        lines.append(f"Primary：{spike.primary_state or spike.neuron}")
        if spike.secondary_states:
            lines.append(f"Secondary：{', '.join(spike.secondary_states)}")
        lines.append(f"Priority：{spike.priority}")
        lines.append(f"Policy：{spike.action_policy}")
        lines.append(f"Blocker：{spike.blocker_type}")
        lines.append(f"Completion target：{spike.completion_target}")
        lines.append("")
        lines.append("## 为什么是它")
        lines.append("")
        lines.append(trigger_reason(spike))
        if spike.evidence_items:
            lines.append("")
            for item in spike.evidence_items[:3]:
                lines.append(f"- [[{md_link(item.path, vault)}]]：{item.snippet}")
        lines.append("")
        lines.append("## 做完后如何关闭")
        lines.append("")
        lines.append("在下面的 `Spike 反馈区` 把对应 checkbox 改成 `[x]`，再把状态改成 `done` 或 `downgraded`，并写入完成证据。")
        lines.append("")

    render_llm_review_section(lines, llm_reviews or {})
    render_spike_feedback_section(lines, selected)
    return "\n".join(lines)


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[Spike],
    timeline: list[dict[str, object]],
    states: dict[str, NeuronState],
    active_neurons: dict[str, NeuronConfig],
    topic_policies: dict[str, TopicPolicy] | None = None,
    llm_reviews: dict[str, dict[str, object]] | None = None,
) -> str:
    lines: list[str] = []
    lines.append("# LIF-Memory 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append(f"状态集合：{', '.join(active_neurons.keys())}")
    lines.append("")

    lines.append("## 电压定义")
    lines.append("")
    lines.append("这里的 `V` 不是 token 数、不是 embedding 维度、也不是原文信息量。")
    lines.append("")
    lines.append("> `V` 表示某个目标状态的“未释放行动压力 / 注意力债务”：相关证据反复出现会充电，时间流逝会泄漏，完成信号会抑制。")
    lines.append("")
    lines.append("当前模型：")
    lines.append("")
    lines.append("```text")
    lines.append("V_fast = max(0, V_fast_old * fast_decay^delta_days + evidence_input - completion_inhibition)")
    lines.append("V_slow = max(0, V_slow_old * slow_decay^delta_days + evidence_input * slow_input_ratio - completion_inhibition * slow_completion_ratio)")
    lines.append("V = weighted_sum(V_fast, V_slow)")
    lines.append("```")
    lines.append("")
    lines.append("- `evidence_input`：当天文本片段经 Evidence Vector Layer 投影后的证据分数之和，上限由 `evidence_cap` 控制。")
    lines.append("- `V_fast`：最近 1-2 天的急性压力，泄漏更快。")
    lines.append("- `V_slow`：长期背景压力，泄漏更慢，输入和完成抑制都更钝。")
    lines.append("- `fast_decay^delta_days / slow_decay^delta_days`：隔了多少天就泄漏多少天。")
    lines.append("- `completion_inhibition`：出现“完成/跑通/保存/提交”等完成信号时，对电位做抑制。")
    lines.append("- `theta`：行动触发阈值。`V >= theta` 时才生成 spike。")
    lines.append("")
    lines.append("## 行动策略层")
    lines.append("")
    lines.append("v0.4 开始，spike 不只表示“提醒”，还会尝试判断当前应该继续、隔离、降级，还是先恢复。")
    lines.append("")
    lines.append("```text")
    lines.append("topic history + blocker signal + completion signal -> priority / blocker_type / action_policy")
    lines.append("```")
    lines.append("")
    if topic_policies:
        lines.append("## 反馈策略层")
        lines.append("")
        lines.append("| Topic | Threshold delta | Priority | Action policy | Muted | Cooldown | Last feedback |")
        lines.append("|---|---:|---|---|---|---:|---|")
        for topic, policy in sorted(topic_policies.items()):
            lines.append(
                f"| {topic} | {policy.threshold_delta:.2f} | {policy.priority_override or ''} | "
                f"{policy.action_policy_override or ''} | {policy.muted} | {policy.cooldown_days} | {policy.last_feedback or ''} |"
            )
        lines.append("")

    render_summary(lines, active_neurons, spikes, states)

    lines.append("## 触发卡片")
    lines.append("")

    if not spikes:
        lines.append("本次回放没有状态变量超过阈值。可以降低 theta，或增加更明确的任务/阻塞关键词。")
        lines.append("")

    for index, spike in enumerate(spikes, start=1):
        lines.append(f"### Spike {index}: {spike.neuron} / {spike.day.isoformat()}")
        lines.append("")
        lines.append(f"- Spike ID：{spike.spike_id}")
        lines.append(f"- Status：{spike.status}")
        lines.append(f"- 电位：{spike.voltage:.2f}")
        lines.append(f"- 阈值：{spike.threshold:.2f}")
        lines.append(f"- Primary state：{spike.primary_state or spike.neuron}")
        if spike.secondary_states:
            lines.append(f"- Secondary states：{', '.join(spike.secondary_states)}")
        lines.append(f"- Topic：{spike.topic}")
        lines.append(f"- Priority：{spike.priority}")
        lines.append(f"- Blocker：{spike.blocker_type}")
        lines.append(f"- Action policy：{spike.action_policy}")
        lines.append(f"- Completion target：{spike.completion_target}")
        lines.append("- 事件包：")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(spike_packet(spike, vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
        lines.append("- 触发原因：")
        if spike.evidence_items:
            for item in spike.evidence_items:
                lines.append(f"  - [[{md_link(item.path, vault)}]]：{item.snippet}")
        else:
            lines.append("  - 当日日志中出现了相关目标信号。")
        lines.append("- 建议动作：")
        lines.append(f"  - {spike.suggestion}")
        lines.append("- 完成判据：")
        lines.append(f"  - {spike.completion_target}")
        lines.append("")

    lines.append("## 状态轨迹")
    lines.append("")
    header = "| 日期 | 日志 | " + " | ".join(active_neurons.keys()) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(active_neurons)) + "|"
    lines.append(header)
    lines.append(sep)
    for row in timeline:
        path = row["path"]
        assert isinstance(path, Path)
        values = []
        for name in active_neurons:
            item = row[name]
            assert isinstance(item, dict)
            mark = " *" if item.get("spike") else ""
            values.append(
                f"{float(item['new_v']):.2f}{mark} "
                f"<br><sub>fast {float(item.get('new_fast', 0.0)):.1f}, slow {float(item.get('new_slow', 0.0)):.1f}</sub>"
                f"<br><sub>in {float(item['input']):.1f}, done {float(item['completion']):.1f}</sub>"
            )
        lines.append(
            f"| {row['date']} | [[{md_link(path, vault)}]] | "
            + " | ".join(values)
            + " |"
        )
    lines.append("")
    lines.append("说明：带 `*` 的电位表示当天该状态神经元触发过 spike；`fast/slow` 是双时间尺度电位；`in` 是证据输入，`done` 是完成抑制。")
    lines.append("")

    lines.append("## 调参规则")
    lines.append("")
    lines.append("- 误触发太多：提高对应状态的 `theta`，或降低关键词覆盖范围。")
    lines.append("- 总是太晚触发：降低 `theta`，或提高关键证据词权重。")
    lines.append("- 每天重复提醒：增加 `cooldown_days` 或降低 `evidence_cap`。")
    lines.append("- 完成后还一直触发：增加 `COMPLETION_WORDS`，并检查是否被 `INCOMPLETE_WORDS` 抵消。")
    lines.append("")

    lines.append("## 系统边界")
    lines.append("")
    lines.append("LIF-Memory 不尝试从 spike 中恢复完整原始语言。")
    lines.append("")
    lines.append("```text")
    lines.append("原始记忆层：Obsidian 保存完整语言，负责保真、追溯、重新检索")
    lines.append("状态触发层：LIF 电位保存趋势，负责积累、泄漏、抑制、阈值判断")
    lines.append("事件层：spike 携带证据索引，负责唤醒相关原文并生成行动建议")
    lines.append("```")
    lines.append("")
    lines.append("所以本实验验证的不是 `spike -> 恢复原文`，而是 `spike -> 召回证据 -> 触发行动`。")
    lines.append("")

    lines.append("## 人工评价")
    lines.append("")
    lines.append("逐条标注：")
    lines.append("")
    lines.append("```text")
    lines.append("合理 / 太早 / 太晚 / 无用 / 应该触发但没触发")
    lines.append("```")
    lines.append("")
    lines.append("下一轮根据人工评价调整：关键词、theta、decay、completion 抑制、cooldown 和 evidence_cap。")
    lines.append("")
    render_llm_review_section(lines, llm_reviews or {})
    render_spike_feedback_section(lines, spikes)
    return "\n".join(lines)


def parse_states(value: str) -> dict[str, NeuronConfig]:
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in NEURONS]
    if unknown:
        raise SystemExit(f"Unknown states: {', '.join(unknown)}. Available: {', '.join(NEURONS)}")
    return {name: NEURONS[name] for name in requested}


def resolve_output_path(vault: Path, output: Path | None) -> Path | None:
    if output is None:
        return None
    return output if output.is_absolute() else vault / output


def write_json_output(path: Path, vault: Path, spikes: list[Spike]) -> None:
    packets = [spike_packet(spike, vault) for spike in spikes]
    path.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    vault = (args.vault or vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    active_neurons = parse_states(args.states)
    notes = find_daily_notes(vault, cutoff, args.days)
    output = resolve_output_path(vault, args.output)
    feedback_path = resolve_output_path(vault, args.feedback_file)
    closure_path = resolve_output_path(vault, args.closure_file) if args.closure_file else output
    feedback_memory_path = resolve_output_path(vault, args.feedback_memory) if args.feedback_memory else default_feedback_memory_path()
    json_policies = load_feedback_file(feedback_path)
    memory_policies = load_feedback_memory(feedback_memory_path, cutoff)
    closures = load_spike_closures(closure_path)
    closure_policies = policies_from_closures(closures)
    topic_policies = merge_topic_policies(json_policies, memory_policies, closure_policies)
    completion_signals = scan_completion_signals(vault, notes, active_neurons) if args.completion_scan else None

    spikes, timeline, states = replay(notes, args.daily_spike_budget, active_neurons, topic_policies, completion_signals)
    apply_closures_to_spikes(spikes, closures)
    llm_reviews = run_llm_reviews(spikes, vault, args)
    if args.mode == "daily":
        report = render_daily_markdown(vault, notes, spikes, args.top_k, llm_reviews)
    else:
        report = render_markdown(vault, notes, spikes, timeline, states, active_neurons, topic_policies, llm_reviews)

    if args.dry_run:
        print(report)
    else:
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        update_feedback_memory(feedback_memory_path, closures, cutoff)
        print(f"Replayed {len(notes)} notes.")
        print(f"Generated {len(spikes)} spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_output_path(vault, args.json_output)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
