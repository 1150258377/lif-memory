from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

VERSION = "0.8.0-field"

DATE_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
FRONT_MATTER_RE = re.compile(r"\A---\n.*?\n---\n", re.S)
WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
TAG_RE = re.compile(r"(?<!\w)#([\w\-\/\u4e00-\u9fff]+)")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")

ACTION_WORDS = ["今天", "目标", "下一步", "需要", "必须", "应该", "修改", "写", "整理", "推进", "验证", "闭环", "实验", "投递"]
BLOCKER_WORDS = ["卡", "不知道", "缺", "没有", "不够", "失败", "问题", "害怕", "拖延", "延毕", "焦虑", "混乱", "难受", "动不了"]
COMPLETION_WORDS = ["完成", "做完", "已经", "搞定", "跑通", "出结果", "保存", "写完", "提交", "测出来"]
NOVELTY_WORDS = ["创新", "创新点", "突破", "第一次", "新", "定义", "提出", "重建", "连续", "场", "结构"]

FIELD_TOPICS: dict[str, list[str]] = {
    "LIF链路": ["LIF", "脑电", "EEG", "SSVEP", "后向散射", "USRP", "事件", "事件率", "阈值", "压缩"],
    "连续问题场": ["连续", "离散", "问题场", "记忆场", "NeRF", "隐式场", "重建", "Obsidian", "知识库"],
    "AI记忆": ["AI", "agent", "智能体", "记忆", "RAG", "Obsidian", "LIF-Memory", "向量", "检索"],
    "论文闭环": ["论文", "第三章", "第四章", "盲审", "答辩", "主线", "证据", "图", "创新点"],
    "负阻": ["负阻", "NDR", "negative resistance", "斜率", "偏置", "增益", "抵消"],
    "求职": ["简历", "实习", "工作", "求职", "岗位", "投递", "面试", "大模型"],
    "健康恢复": ["健康", "焦虑", "累", "睡眠", "睡", "身体", "情绪", "压力", "害怕", "难受", "恢复", "心理", "疲惫", "精力", "运动", "休息", "状态", "崩", "撑", "饮食", "吃"],
}

IGNORED_PARTS = {".git", ".obsidian", ".trash", "__pycache__", ".venv", "node_modules", "examples", "tests"}


@dataclass
class NoteObservation:
    path: Path
    rel_path: str
    title: str
    day: date
    text: str
    snippet: str
    tags: list[str]
    links: list[str]
    vector: dict[str, float]
    graph_vector: dict[str, float] = field(default_factory=dict)
    centrality: float = 0.0


@dataclass
class FieldHit:
    note: NoteObservation
    score: float
    semantic: float
    time_weight: float
    graph_bonus: float
    modifiers: list[str]

    def to_packet(self) -> dict[str, object]:
        return {
            "path": self.note.rel_path,
            "title": self.note.title,
            "day": self.note.day.isoformat(),
            "score": round(self.score, 4),
            "semantic": round(self.semantic, 4),
            "time_weight": round(self.time_weight, 4),
            "graph_bonus": round(self.graph_bonus, 4),
            "modifiers": self.modifiers,
            "snippet": self.note.snippet,
        }


@dataclass
class LIFStep:
    day: date
    input_current: float
    completion_inhibition: float
    v_fast: float
    v_slow: float
    v: float
    spiked: bool

    def to_packet(self) -> dict[str, object]:
        return {
            "day": self.day.isoformat(),
            "input_current": round(self.input_current, 4),
            "completion_inhibition": round(self.completion_inhibition, 4),
            "V_fast": round(self.v_fast, 4),
            "V_slow": round(self.v_slow, 4),
            "V": round(self.v, 4),
            "spiked": self.spiked,
        }


@dataclass
class FieldResult:
    query: str
    topic: str
    today: date
    field_energy: float
    reconstruction_pressure: float
    hits: list[FieldHit]
    trajectory: list[LIFStep]
    spike: bool
    insight_card: str

    def to_packet(self) -> dict[str, object]:
        return {
            "version": VERSION,
            "query": self.query,
            "topic": self.topic,
            "today": self.today.isoformat(),
            "field_energy": round(self.field_energy, 4),
            "reconstruction_pressure": round(self.reconstruction_pressure, 4),
            "spike": self.spike,
            "top_hits": [hit.to_packet() for hit in self.hits],
            "trajectory": [step.to_packet() for step in self.trajectory],
            "insight_card": self.insight_card,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a continuous problem field from an Obsidian vault and trigger LIF insight cards.")
    parser.add_argument("--vault", type=Path, default=Path("."), help="Obsidian vault root.")
    parser.add_argument("--query", type=str, required=True, help="Question or topic used as the coordinate q in the problem field.")
    parser.add_argument("--days", type=int, default=90, help="Lookback window in days.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 连续问题场.md"), help="Markdown report output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--dry-run", action="store_true", help="Print the report instead of writing it.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of evidence notes shown in the reconstructed field.")
    parser.add_argument("--max-notes", type=int, default=600, help="Maximum notes to scan after time filtering.")
    parser.add_argument("--time-sigma", type=float, default=30.0, help="Gaussian time kernel width in days.")
    parser.add_argument("--semantic-sigma", type=float, default=0.55, help="Semantic kernel width over sparse cosine distance.")
    parser.add_argument("--graph-steps", type=int, default=2, help="Number of Obsidian graph diffusion steps.")
    parser.add_argument("--graph-alpha", type=float, default=0.35, help="Graph diffusion neighbor mixing ratio.")
    parser.add_argument("--threshold", type=float, default=5.0, help="LIF threshold for insight spike.")
    parser.add_argument("--all-notes", action="store_true", help="Scan all .md files in vault, ignoring date filter.")
    parser.add_argument("--version", action="version", version=f"continuous-problem-field {VERSION}")
    return parser.parse_args()


def parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def ignored_path(path: Path) -> bool:
    return any(part in IGNORED_PARTS for part in path.parts)


def normalize_text(text: str) -> str:
    text = FRONT_MATTER_RE.sub("", text)
    text = CODE_FENCE_RE.sub("", text)
    text = re.sub(r"!\[\[.*?\]\]", "", text)
    text = re.sub(r"\[\[([^\]|#]+)(?:#[^\]|]+)?(?:\|([^\]]+))?\]\]", lambda m: m.group(2) or m.group(1), text)
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def note_day(path: Path) -> date:
    match = DATE_RE.search(path.stem)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def extract_links(raw_text: str) -> list[str]:
    links: list[str] = []
    for match in WIKILINK_RE.finditer(raw_text):
        target = match.group(1).strip()
        if target and target not in links:
            links.append(target)
    return links


def extract_tags(raw_text: str) -> list[str]:
    tags: list[str] = []
    for match in TAG_RE.finditer(raw_text):
        tag = match.group(1).strip()
        if tag and tag not in tags:
            tags.append(tag)
    return tags


def title_key(value: str) -> str:
    return re.sub(r"\s+", "", value.strip().lower())


def add_feature(counter: Counter[str], token: str, weight: float = 1.0) -> None:
    token = token.strip().lower()
    if len(token) >= 2:
        counter[token] += weight


def cjk_shingles(seq: str) -> Iterable[str]:
    if len(seq) <= 4:
        yield seq
    for n in (2, 3):
        if len(seq) >= n:
            for i in range(0, len(seq) - n + 1):
                yield seq[i : i + n]


def vectorize(text: str, title: str = "", tags: Iterable[str] = ()) -> dict[str, float]:
    counter: Counter[str] = Counter()
    for word in LATIN_RE.findall(text):
        add_feature(counter, word, 1.0)
    for num in NUMBER_RE.findall(text):
        add_feature(counter, f"num:{num}", 0.25)
    for seq in CJK_RE.findall(text):
        for shingle in cjk_shingles(seq):
            add_feature(counter, shingle, 0.55)
    for topic, words in FIELD_TOPICS.items():
        if contains_any(text, words):
            add_feature(counter, f"topic:{topic}", 3.0)
            for word in words:
                if word.lower() in text.lower():
                    add_feature(counter, word, 1.4)
    for tag in tags:
        add_feature(counter, f"tag:{tag}", 2.0)
    if title:
        for word in LATIN_RE.findall(title):
            add_feature(counter, f"title:{word}", 1.5)
        for seq in CJK_RE.findall(title):
            for shingle in cjk_shingles(seq):
                add_feature(counter, f"title:{shingle}", 1.2)
    norm = math.sqrt(sum(value * value for value in counter.values())) or 1.0
    return {key: value / norm for key, value in counter.items() if value > 0}


def contains_any(text: str, words: Iterable[str]) -> bool:
    lower = text.lower()
    return any(word.lower() in lower for word in words)


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return max(0.0, sum(value * b.get(key, 0.0) for key, value in a.items()))


def weighted_sum(vectors: Iterable[tuple[dict[str, float], float]]) -> dict[str, float]:
    out: Counter[str] = Counter()
    total = 0.0
    for vector, weight in vectors:
        if weight <= 0:
            continue
        total += weight
        for key, value in vector.items():
            out[key] += value * weight
    if total <= 0:
        return {}
    normed = {key: value / total for key, value in out.items()}
    norm = math.sqrt(sum(value * value for value in normed.values())) or 1.0
    return {key: value / norm for key, value in normed.items()}


def snippet_from_text(text: str, limit: int = 140) -> str:
    blocks = [block.strip(" -\t") for block in re.split(r"[\n。！？!?；;]+", text) if len(block.strip()) >= 8]
    snippet = blocks[0] if blocks else text[:limit]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    if len(snippet) > limit:
        return snippet[: limit - 1] + "…"
    return snippet


def read_notes(vault: Path, cutoff: date, days: int, max_notes: int, all_notes: bool = False) -> list[NoteObservation]:
    start = cutoff - timedelta(days=max(days, 1))
    notes: list[NoteObservation] = []
    for path in vault.rglob("*.md"):
        if not path.is_file() or ignored_path(path):
            continue
        try:
            day = note_day(path)
        except OSError:
            continue
        if not all_notes and (day < start or day > cutoff):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        if len(text) < 20:
            continue
        tags = extract_tags(raw)
        links = extract_links(raw)
        rel_path = str(path.relative_to(vault)) if path.is_relative_to(vault) else str(path)
        title = path.stem
        vector = vectorize(text, title=title, tags=tags)
        notes.append(
            NoteObservation(
                path=path,
                rel_path=rel_path,
                title=title,
                day=day,
                text=text,
                snippet=snippet_from_text(text),
                tags=tags,
                links=links,
                vector=vector,
            )
        )
    notes.sort(key=lambda note: (note.day, note.rel_path), reverse=True)
    return notes[:max_notes]


def build_adjacency(notes: list[NoteObservation]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(notes))}
    title_index: dict[str, int] = {}
    for idx, note in enumerate(notes):
        title_index[title_key(note.title)] = idx

    tag_index: dict[str, list[int]] = defaultdict(list)
    folder_index: dict[str, list[int]] = defaultdict(list)

    for idx, note in enumerate(notes):
        for link in note.links:
            target = title_index.get(title_key(Path(link).stem)) or title_index.get(title_key(link))
            if target is not None and target != idx:
                adjacency[idx].add(target)
                adjacency[target].add(idx)
        for tag in note.tags:
            tag_index[tag].append(idx)
        folder_index[str(note.path.parent)].append(idx)

    for group in list(tag_index.values()) + list(folder_index.values()):
        limited = group[:25]
        for i in limited:
            for j in limited:
                if i != j:
                    adjacency[i].add(j)
    return adjacency


def diffuse_graph(notes: list[NoteObservation], steps: int, alpha: float) -> None:
    if not notes:
        return
    adjacency = build_adjacency(notes)
    vectors = [note.vector for note in notes]
    alpha = max(0.0, min(alpha, 0.95))
    steps = max(0, steps)

    for _ in range(steps):
        new_vectors: list[dict[str, float]] = []
        for idx, own in enumerate(vectors):
            neighbors = adjacency.get(idx, set())
            if not neighbors:
                new_vectors.append(own)
                continue
            neighbor_vec = weighted_sum((vectors[j], 1.0) for j in neighbors)
            mixed = weighted_sum(((own, 1.0 - alpha), (neighbor_vec, alpha)))
            new_vectors.append(mixed)
        vectors = new_vectors

    max_degree = max((len(v) for v in adjacency.values()), default=1) or 1
    for idx, note in enumerate(notes):
        note.graph_vector = vectors[idx]
        note.centrality = len(adjacency.get(idx, set())) / max_degree


def infer_topic(query: str) -> str:
    scores: dict[str, int] = {}
    for topic, words in FIELD_TOPICS.items():
        score = sum(1 for word in words if word.lower() in query.lower())
        if score:
            scores[topic] = score
    if scores:
        return max(scores.items(), key=lambda pair: pair[1])[0]
    return "自定义问题场"


def time_kernel(note_day_value: date, today: date, sigma_days: float, all_notes: bool = False) -> float:
    if all_notes:
        return 1.0
    delta = max((today - note_day_value).days, 0)
    sigma_days = max(sigma_days, 1.0)
    return math.exp(-((delta * delta) / (2.0 * sigma_days * sigma_days)))


def semantic_kernel(similarity: float, sigma: float) -> float:
    sigma = max(sigma, 0.05)
    distance = 1.0 - max(0.0, min(similarity, 1.0))
    return math.exp(-(distance * distance) / (2.0 * sigma * sigma))


def completion_inhibition(text: str) -> float:
    if contains_any(text, COMPLETION_WORDS) and not contains_any(text, ["未完成", "没有完成", "还没", "尚未"]):
        return 0.45
    return 0.0


def hit_modifiers(note: NoteObservation) -> list[str]:
    modifiers: list[str] = []
    if contains_any(note.text, ACTION_WORDS):
        modifiers.append("action")
    if contains_any(note.text, BLOCKER_WORDS):
        modifiers.append("blocker")
    if contains_any(note.text, COMPLETION_WORDS):
        modifiers.append("completion_signal")
    if contains_any(note.text, NOVELTY_WORDS):
        modifiers.append("novelty")
    if note.links:
        modifiers.append("wikilinked")
    if note.tags:
        modifiers.append("tagged")
    return modifiers


def reconstruct_field(
    notes: list[NoteObservation],
    query: str,
    today: date,
    top_k: int,
    time_sigma: float,
    semantic_sigma: float,
    all_notes: bool = False,
) -> tuple[list[FieldHit], float, dict[date, float], dict[date, float]]:
    query_vec = vectorize(query, title=query)
    hits: list[FieldHit] = []
    daily_current: dict[date, float] = defaultdict(float)
    daily_completion: dict[date, float] = defaultdict(float)

    for note in notes:
        field_vec = note.graph_vector or note.vector
        sem = cosine(query_vec, field_vec)
        if sem <= 0:
            continue
        sem_w = semantic_kernel(sem, semantic_sigma)
        time_w = time_kernel(note.day, today, time_sigma, all_notes=all_notes)
        graph_bonus = 1.0 + 0.25 * note.centrality
        score = sem * sem_w * time_w * graph_bonus
        min_score = 0.0001 if all_notes else 0.001
        if score <= min_score:
            continue
        hit = FieldHit(
            note=note,
            score=score,
            semantic=sem,
            time_weight=time_w,
            graph_bonus=graph_bonus,
            modifiers=hit_modifiers(note),
        )
        hits.append(hit)
        current_boost = 1.0
        if "action" in hit.modifiers:
            current_boost += 0.20
        if "blocker" in hit.modifiers:
            current_boost += 0.30
        if "novelty" in hit.modifiers:
            current_boost += 0.25
        daily_current[note.day] += score * current_boost
        daily_completion[note.day] += completion_inhibition(note.text)

    hits.sort(key=lambda hit: hit.score, reverse=True)
    field_energy = sum(hit.score for hit in hits[: max(top_k, 1)])
    return hits[:top_k], field_energy, dict(daily_current), dict(daily_completion)


def lif_iterate(
    daily_current: dict[date, float],
    daily_completion: dict[date, float],
    start: date,
    today: date,
    threshold: float,
    fast_decay: float = 0.76,
    slow_decay: float = 0.94,
    fast_weight: float = 0.65,
    slow_weight: float = 0.35,
    slow_input_ratio: float = 0.42,
    slow_completion_ratio: float = 0.30,
    reset_ratio: float = 0.40,
) -> list[LIFStep]:
    trajectory: list[LIFStep] = []
    v_fast = 0.0
    v_slow = 0.0
    day = start
    while day <= today:
        current = daily_current.get(day, 0.0) * 4.0
        inhibition = daily_completion.get(day, 0.0)
        v_fast = max(0.0, v_fast * fast_decay + current - inhibition)
        v_slow = max(0.0, v_slow * slow_decay + current * slow_input_ratio - inhibition * slow_completion_ratio)
        v = fast_weight * v_fast + slow_weight * v_slow
        spiked = v >= threshold
        trajectory.append(
            LIFStep(
                day=day,
                input_current=current,
                completion_inhibition=inhibition,
                v_fast=v_fast,
                v_slow=v_slow,
                v=v,
                spiked=spiked,
            )
        )
        if spiked:
            v_fast *= reset_ratio
            v_slow *= reset_ratio
        day += timedelta(days=1)
    return trajectory


def top_terms(hits: list[FieldHit], limit: int = 8) -> list[str]:
    counter: Counter[str] = Counter()
    for hit in hits:
        for key, value in (hit.note.graph_vector or hit.note.vector).items():
            if key.startswith("topic:") or key.startswith("tag:") or key.startswith("title:"):
                counter[key.replace("topic:", "").replace("tag:", "#").replace("title:", "title:" )] += value * hit.score
    return [key for key, _ in counter.most_common(limit)]


def make_insight_card(query: str, topic: str, hits: list[FieldHit], trajectory: list[LIFStep], threshold: float) -> str:
    latest = trajectory[-1] if trajectory else None
    spike = any(step.spiked for step in trajectory[-7:]) if trajectory else False
    terms = top_terms(hits)
    dominant_modifiers = Counter(mod for hit in hits for mod in hit.modifiers).most_common(4)
    modifier_text = "、".join(name for name, _ in dominant_modifiers) or "无明显修饰"
    term_text = "、".join(terms) or "暂无稳定主题词"
    top_lines = []
    for idx, hit in enumerate(hits[:5], start=1):
        top_lines.append(f"{idx}. `{hit.note.rel_path}`：{hit.note.snippet}")
    evidence_text = "\n".join(top_lines) if top_lines else "暂无足够证据。"

    if latest is None:
        voltage_line = "V=0，未形成可触发的问题张力。"
    else:
        voltage_line = f"V={latest.v:.2f} / θ={threshold:.2f}，V_fast={latest.v_fast:.2f}，V_slow={latest.v_slow:.2f}。"

    if spike:
        decision = "已经跨过阈值，应该生成一张洞察卡，而不是继续普通检索。"
    elif latest and latest.v >= threshold * 0.72:
        decision = "接近阈值，适合继续追问一次，把隐含结构问出来。"
    else:
        decision = "尚未跨阈值，当前更适合补充笔记或扩展证据。"

    return f"""## 连续问题场洞察卡

- 查询坐标：{query}
- 推断主题：{topic}
- 场关键词：{term_text}
- 修饰信号：{modifier_text}
- 电位状态：{voltage_line}
- 触发判断：{decision}

### 重建出的隐含问题

这些离散笔记共同指向的不是一个简单答案，而是一个正在形成的连续问题场：`{topic}` 正在从“资料/片段检索”升级为“结构化重建”。当前系统应该优先回答：这些证据之间的共同约束是什么、哪一个变量正在反复积累、下一次交互应该刺激哪个方向。

### 证据切片

{evidence_text}

### 下一步最小动作

把这张卡写回 Obsidian，并补充一条“我现在真正要问的问题是什么”。下一次运行时，这条新笔记会作为新的观测点重新注入问题场。
"""


def render_markdown(result: FieldResult, threshold: float, time_sigma: float, semantic_sigma: float) -> str:
    lines: list[str] = []
    lines.append(f"# LIF-Memory 连续问题场报告")
    lines.append("")
    lines.append(f"- Version: `{VERSION}`")
    lines.append(f"- Query: `{result.query}`")
    lines.append(f"- Topic: `{result.topic}`")
    lines.append(f"- Today: `{result.today.isoformat()}`")
    lines.append(f"- Field energy: `{result.field_energy:.4f}`")
    lines.append(f"- Reconstruction pressure: `{result.reconstruction_pressure:.4f}`")
    lines.append(f"- Spike: `{result.spike}`")
    lines.append("")
    lines.append("## 数学结构")
    lines.append("")
    lines.append("```text")
    lines.append("n_i=(text_i,t_i,links_i,tags_i)")
    lines.append("e_i=phi(n_i)")
    lines.append("h_i=GraphDiffuse(e_i,G)")
    lines.append("M(t,z_q)=sum_i K_t(t,t_i) K_z(z_q,z_i) h_i / sum_i K_t K_z")
    lines.append("I_q(t)=field_energy(t)+action/blocker/novelty boost-completion inhibition")
    lines.append("V_fast=max(0,V_fast*decay_fast+I_q-C)")
    lines.append("V_slow=max(0,V_slow*decay_slow+0.42*I_q-0.30*C)")
    lines.append("V=0.65*V_fast+0.35*V_slow")
    lines.append("V>=theta => insight spike")
    lines.append("```")
    lines.append("")
    lines.append(f"- time_sigma_days: `{time_sigma}`")
    lines.append(f"- semantic_sigma: `{semantic_sigma}`")
    lines.append(f"- threshold: `{threshold}`")
    lines.append("")
    lines.append(result.insight_card)
    lines.append("")
    lines.append("## Top evidence notes")
    lines.append("")
    lines.append("| rank | score | day | note | modifiers | snippet |")
    lines.append("|---:|---:|---|---|---|---|")
    for idx, hit in enumerate(result.hits, start=1):
        mods = ", ".join(hit.modifiers)
        snippet = hit.note.snippet.replace("|", "/")
        lines.append(f"| {idx} | {hit.score:.4f} | {hit.note.day.isoformat()} | `{hit.note.rel_path}` | {mods} | {snippet} |")
    lines.append("")
    lines.append("## LIF trajectory")
    lines.append("")
    lines.append("| day | I_q | C | V_fast | V_slow | V | spike |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for step in result.trajectory[-21:]:
        lines.append(
            f"| {step.day.isoformat()} | {step.input_current:.3f} | {step.completion_inhibition:.3f} | {step.v_fast:.3f} | {step.v_slow:.3f} | {step.v:.3f} | {step.spiked} |"
        )
    lines.append("")
    lines.append("## JSON packet")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result.to_packet(), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def load_field_params(vault: Path) -> dict:
    """优先读取 lif_field_params.json 中由调参器写入的参数"""
    p = vault / "lif_field_params.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def run(args: argparse.Namespace) -> FieldResult:
    vault = args.vault.resolve()
    today = parse_today(args.today)
    # 读取调参器写入的参数，命令行参数优先级更高
    saved = load_field_params(vault)
    all_notes_flag = getattr(args, "all_notes", False)
    threshold = args.threshold if args.threshold != 5.0 else float(saved.get("threshold", args.threshold))
    semantic_sigma = args.semantic_sigma if args.semantic_sigma != 0.55 else float(saved.get("semantic_sigma", args.semantic_sigma))
    time_sigma = args.time_sigma if args.time_sigma != 30.0 else float(saved.get("time_sigma", args.time_sigma))
    graph_alpha = args.graph_alpha if args.graph_alpha != 0.35 else float(saved.get("graph_alpha", args.graph_alpha))
    graph_steps = args.graph_steps if args.graph_steps != 2 else int(saved.get("graph_steps", args.graph_steps))
    notes = read_notes(vault, today, args.days, args.max_notes, all_notes=all_notes_flag)
    diffuse_graph(notes, graph_steps, graph_alpha)
    hits, field_energy, daily_current, daily_completion = reconstruct_field(
        notes=notes,
        query=args.query,
        today=today,
        top_k=max(1, args.top_k),
        time_sigma=time_sigma,
        semantic_sigma=semantic_sigma,
        all_notes=all_notes_flag,
    )
    if all_notes_flag and daily_current:
        start = min(daily_current.keys())
    else:
        start = today - timedelta(days=max(args.days, 1))
    trajectory = lif_iterate(daily_current, daily_completion, start, today, threshold=threshold)
    spike = any(step.spiked for step in trajectory[-7:]) if trajectory else False
    reconstruction_pressure = trajectory[-1].v if trajectory else 0.0
    topic = infer_topic(args.query)
    insight_card = make_insight_card(args.query, topic, hits, trajectory, threshold=threshold)
    return FieldResult(
        query=args.query,
        topic=topic,
        today=today,
        field_energy=field_energy,
        reconstruction_pressure=reconstruction_pressure,
        hits=hits,
        trajectory=trajectory,
        spike=spike,
        insight_card=insight_card,
    )


def main() -> None:
    args = parse_args()
    result = run(args)
    report = render_markdown(result, threshold=args.threshold, time_sigma=args.time_sigma, semantic_sigma=args.semantic_sigma)
    if args.dry_run:
        print(report)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result.to_packet(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.json_output}")


if __name__ == "__main__":
    main()
