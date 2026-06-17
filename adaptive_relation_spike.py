from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping

import lif_memory as core
import llm_adapter

VERSION = "0.9.0"

CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]*(?:\s+[A-Za-z][A-Za-z0-9_+\-]*){0,3}")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?\s*(?:Hz|hz|mV|mv|V|Vpp|RMS|uA|μA|ms|s|MHz|GHz|kHz|%|倍)?")

GENERIC_STOP_TERMS = {
    "这个",
    "那个",
    "什么",
    "怎么",
    "为什么",
    "就是",
    "然后",
    "其实",
    "因为",
    "所以",
    "但是",
    "如果",
    "现在",
    "目前",
    "问题",
    "东西",
    "感觉",
    "可能",
    "应该",
    "需要",
    "一下",
    "一个",
    "没有",
    "还是",
    "或者",
    "来说",
    "的话",
    "对于",
    "进行",
    "比较",
    "发现",
}


@dataclass
class Fragment:
    day: date
    path: Path
    text: str
    terms: list[str] = field(default_factory=list)
    numbers: list[str] = field(default_factory=list)

    def packet(self, vault: Path) -> dict[str, object]:
        return {
            "note": self.day.isoformat(),
            "path": core.md_link(self.path, vault),
            "snippet": self.text,
            "terms": self.terms,
            "numbers": self.numbers,
        }


@dataclass
class Concept:
    name: str
    aliases: list[str]
    score: float = 0.0
    description: str = ""
    source: str = "statistical"


@dataclass
class RelationCandidate:
    source: str
    target: str
    score: float
    count: int
    distinct_days: int
    evidence: list[Fragment]
    relation_type: str = "co_activation"
    hypothesis: str = ""
    next_validation_action: str = ""
    source_mode: str = "statistical"

    def name(self) -> str:
        return f"{self.source} -> {self.target}"

    def packet(self, vault: Path) -> dict[str, object]:
        return {
            "spike_type": "AdaptiveRelation",
            "name": self.name(),
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type,
            "score": round(self.score, 3),
            "count": self.count,
            "distinct_days": self.distinct_days,
            "source_mode": self.source_mode,
            "hypothesis": self.hypothesis,
            "next_validation_action": self.next_validation_action,
            "evidence_chain": [item.packet(vault) for item in self.evidence],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Induce relation spikes from notes without hand-written domain rules.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path. Defaults to nearest .obsidian root.")
    parser.add_argument("--days", type=int, default=60, help="Number of latest daily notes to scan.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--top-concepts", type=int, default=32, help="Number of induced concepts to keep.")
    parser.add_argument("--top-relations", type=int, default=12, help="Number of relation spikes to render.")
    parser.add_argument("--min-term-count", type=int, default=2, help="Minimum global count for a statistical concept.")
    parser.add_argument("--schema-file", type=Path, default=Path("lif_relation_schema.generated.json"), help="Generated or reviewed schema JSON.")
    parser.add_argument("--write-schema", action="store_true", help="Write induced schema to --schema-file.")
    parser.add_argument("--llm-induce-schema", action="store_true", help="Ask an LLM to induce concepts and relation hypotheses from sampled evidence.")
    parser.add_argument("--ask-human-output", type=Path, default=None, help="Write a Markdown file with questions for human schema calibration.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Adaptive-Relation-Spike 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON output path for relation packets.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing files.")
    parser.add_argument("--version", action="version", version=f"LIF Adaptive Relation Spike {VERSION}")
    llm_adapter.add_cli_args(parser)
    return parser.parse_args()


def parse_cutoff(value: str | None) -> date:
    return date.today() if value is None else datetime.strptime(value, "%Y-%m-%d").date()


def resolve_path(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def normalize_text(text: str) -> str:
    text = core.CODE_FENCE_RE.sub(" ", text) if hasattr(core, "CODE_FENCE_RE") else text
    text = core.FRONT_MATTER_RE.sub(" ", text) if hasattr(core, "FRONT_MATTER_RE") else text
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_fragments(notes: list[tuple[date, Path]], max_block_len: int = 180) -> list[Fragment]:
    fragments: list[Fragment] = []
    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for block in core.split_blocks(normalize_text(text)):
            block = block.strip()
            if len(block) < 8:
                continue
            if len(block) > max_block_len:
                block = block[: max_block_len - 1] + "…"
            numbers = NUMBER_RE.findall(block)
            fragments.append(Fragment(day=day, path=path, text=block, numbers=numbers[:8]))
    return fragments


def cjk_ngrams(text: str, min_n: int = 2, max_n: int = 6) -> Iterable[str]:
    for match in CJK_RE.finditer(text):
        seq = match.group(0)
        if len(seq) < min_n:
            continue
        for n in range(min_n, min(max_n, len(seq)) + 1):
            for index in range(0, len(seq) - n + 1):
                token = seq[index : index + n]
                if token in GENERIC_STOP_TERMS:
                    continue
                if any(stop in token for stop in ("这个", "那个", "什么", "就是", "然后")):
                    continue
                yield token


def latin_terms(text: str) -> Iterable[str]:
    for match in LATIN_RE.finditer(text):
        token = re.sub(r"\s+", " ", match.group(0)).strip()
        if len(token) >= 2:
            yield token


def extract_terms_from_text(text: str) -> list[str]:
    terms = list(cjk_ngrams(text)) + list(latin_terms(text))
    cleaned: list[str] = []
    for term in terms:
        term = term.strip("-_：:，。；;、,.!?！？()（）[]【】 ")
        if len(term) < 2:
            continue
        if term in GENERIC_STOP_TERMS:
            continue
        if term.isdigit():
            continue
        cleaned.append(term)
    return cleaned


def induce_statistical_concepts(
    fragments: list[Fragment],
    top_k: int,
    min_count: int,
) -> list[Concept]:
    term_counts: Counter[str] = Counter()
    term_days: dict[str, set[date]] = defaultdict(set)

    for fragment in fragments:
        terms = set(extract_terms_from_text(fragment.text))
        for term in terms:
            term_counts[term] += 1
            term_days[term].add(fragment.day)

    scored: list[tuple[float, str]] = []
    for term, count in term_counts.items():
        if count < min_count:
            continue
        day_count = len(term_days[term])
        length_bonus = min(len(term), 8) / 8
        score = count * (1.0 + math.log1p(day_count)) * (0.7 + length_bonus)
        scored.append((score, term))

    scored.sort(reverse=True)
    selected: list[str] = []
    for score, term in scored:
        if any(term in existing and len(term) < len(existing) for existing in selected):
            continue
        if any(existing in term and term_counts[existing] >= term_counts[term] for existing in selected):
            continue
        selected.append(term)
        if len(selected) >= top_k:
            break

    return [
        Concept(
            name=term,
            aliases=[term],
            score=round(score, 3),
            description="Statistically induced recurring concept.",
            source="statistical",
        )
        for score, term in scored
        if term in selected
    ][:top_k]


def annotate_fragments(fragments: list[Fragment], concepts: list[Concept]) -> None:
    alias_to_name: list[tuple[str, str]] = []
    for concept in concepts:
        aliases = concept.aliases or [concept.name]
        for alias in aliases:
            if alias:
                alias_to_name.append((alias, concept.name))

    for fragment in fragments:
        hits: list[str] = []
        lower = fragment.text.lower()
        for alias, name in alias_to_name:
            if alias.lower() in lower and name not in hits:
                hits.append(name)
        fragment.terms = hits


def build_statistical_relations(
    fragments: list[Fragment],
    top_relations: int,
    evidence_limit: int = 8,
) -> list[RelationCandidate]:
    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_days: dict[tuple[str, str], set[date]] = defaultdict(set)
    pair_evidence: dict[tuple[str, str], list[Fragment]] = defaultdict(list)
    term_counts: Counter[str] = Counter()

    for fragment in fragments:
        terms = list(dict.fromkeys(fragment.terms))
        for term in terms:
            term_counts[term] += 1
        for source, target in itertools.combinations(sorted(terms), 2):
            pair = (source, target)
            pair_counts[pair] += 1
            pair_days[pair].add(fragment.day)
            if len(pair_evidence[pair]) < evidence_limit:
                pair_evidence[pair].append(fragment)

    relations: list[RelationCandidate] = []
    for pair, count in pair_counts.items():
        if count < 2:
            continue
        source, target = pair
        distinct_days = len(pair_days[pair])
        rarity = 1.0 / math.sqrt(max(term_counts[source] * term_counts[target], 1))
        score = count * (1.0 + math.log1p(distinct_days)) * (1.0 + rarity * 10.0)
        relations.append(
            RelationCandidate(
                source=source,
                target=target,
                score=score,
                count=count,
                distinct_days=distinct_days,
                evidence=pair_evidence[pair],
                relation_type="co_activation",
                hypothesis=f"{source} 与 {target} 在多个笔记片段中反复共同出现，可能存在尚未命名的关系。",
                next_validation_action=f"回看证据链，判断 {source} 和 {target} 是支持、冲突、降级、因果还是同一问题的不同表述。",
                source_mode="statistical",
            )
        )

    relations.sort(key=lambda item: item.score, reverse=True)
    return relations[:top_relations]


def sample_fragments_for_llm(fragments: list[Fragment], limit: int = 36) -> list[dict[str, str]]:
    ranked = sorted(
        fragments,
        key=lambda item: (len(item.numbers), len(item.text), item.day),
        reverse=True,
    )
    seen: set[str] = set()
    samples: list[dict[str, str]] = []
    for fragment in ranked:
        key = fragment.text
        if key in seen:
            continue
        seen.add(key)
        samples.append(
            {
                "date": fragment.day.isoformat(),
                "path": str(fragment.path),
                "text": fragment.text,
                "stat_terms": fragment.terms[:8],
                "numbers": fragment.numbers[:6],
            }
        )
        if len(samples) >= limit:
            break
    return samples


def llm_schema_prompt(samples: list[dict[str, str]], statistical_concepts: list[Concept]) -> list[dict[str, str]]:
    system = (
        "你是 LIF-Memory 的自适应概念归纳器。"
        "你不能使用预置主题表，也不要强行套用外部分类。"
        "你要从用户笔记样本中归纳当前真正反复出现的概念、别名和概念关系。"
        "输出必须是 JSON object，不要 Markdown。"
    )
    user = {
        "task": "Induce an adaptive schema from note fragments.",
        "principles": [
            "concepts must come from the evidence, not from a fixed label set",
            "aliases should include different surface forms used by the user",
            "relations should express support/conflict/inhibition/reframe/downgrade/causal/bridge/unknown when justified",
            "if uncertain, set relation_type to unknown and ask a calibration question",
            "avoid generic labels such as 问题/东西/现在 unless they are part of a meaningful phrase",
        ],
        "allowed_output_schema": {
            "concepts": [
                {
                    "name": "string",
                    "aliases": ["string"],
                    "description": "string",
                    "why_matters": "string",
                }
            ],
            "relations": [
                {
                    "name": "string",
                    "source": "concept name",
                    "target": "concept name",
                    "relation_type": "support|conflict|inhibition|reframe|downgrade|causal|bridge|unknown",
                    "hypothesis": "string",
                    "next_validation_action": "string",
                    "calibration_question": "string|null",
                }
            ],
            "calibration_questions": ["string"],
        },
        "statistical_candidates": [concept.__dict__ for concept in statistical_concepts[:40]],
        "note_fragments": samples,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def induce_llm_schema(args: argparse.Namespace, fragments: list[Fragment], concepts: list[Concept]) -> dict[str, object]:
    config = llm_adapter.config_from_args(args)
    samples = sample_fragments_for_llm(fragments)
    content = llm_adapter.call_chat_completions(config, llm_schema_prompt(samples, concepts))
    schema = llm_adapter.extract_json_object(content)
    schema["generated_at"] = datetime.now().isoformat(timespec="seconds")
    schema["generator"] = "llm"
    schema["version"] = VERSION
    return schema


def schema_to_concepts(schema: Mapping[str, object], fallback: list[Concept]) -> list[Concept]:
    raw = schema.get("concepts")
    concepts: list[Concept] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            aliases_raw = item.get("aliases", [])
            aliases = [str(alias).strip() for alias in aliases_raw if str(alias).strip()] if isinstance(aliases_raw, list) else []
            if name not in aliases:
                aliases.insert(0, name)
            concepts.append(
                Concept(
                    name=name,
                    aliases=aliases,
                    score=0.0,
                    description=str(item.get("description", "")),
                    source=str(schema.get("generator", "schema")),
                )
            )
    return concepts or fallback


def apply_llm_relations(
    schema: Mapping[str, object],
    statistical_relations: list[RelationCandidate],
    fragments: list[Fragment],
    top_relations: int,
) -> list[RelationCandidate]:
    raw_relations = schema.get("relations")
    if not isinstance(raw_relations, list):
        return statistical_relations

    by_pair: dict[tuple[str, str], RelationCandidate] = {}
    for relation in statistical_relations:
        by_pair[(relation.source, relation.target)] = relation
        by_pair[(relation.target, relation.source)] = relation

    induced: list[RelationCandidate] = []
    for item in raw_relations:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", "")).strip()
        target = str(item.get("target", "")).strip()
        if not source or not target or source == target:
            continue
        base = by_pair.get((source, target))
        if base is None:
            evidence = [fragment for fragment in fragments if source in fragment.terms or target in fragment.terms][:8]
            count = len(evidence)
            distinct_days = len({fragment.day for fragment in evidence})
            score = max(1.0, count * (1.0 + math.log1p(distinct_days)))
            base = RelationCandidate(source=source, target=target, score=score, count=count, distinct_days=distinct_days, evidence=evidence)
        base.relation_type = str(item.get("relation_type", "unknown") or "unknown")
        base.hypothesis = str(item.get("hypothesis", "") or base.hypothesis)
        base.next_validation_action = str(item.get("next_validation_action", "") or base.next_validation_action)
        base.source_mode = "llm_schema"
        induced.append(base)

    induced.sort(key=lambda relation: relation.score, reverse=True)
    return induced[:top_relations] if induced else statistical_relations


def load_schema(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def write_schema(path: Path, schema: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def statistical_schema(concepts: list[Concept], relations: list[RelationCandidate]) -> dict[str, object]:
    return {
        "version": VERSION,
        "generator": "statistical",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "concepts": [
            {
                "name": concept.name,
                "aliases": concept.aliases,
                "description": concept.description,
                "score": concept.score,
            }
            for concept in concepts
        ],
        "relations": [
            {
                "name": relation.name(),
                "source": relation.source,
                "target": relation.target,
                "relation_type": relation.relation_type,
                "hypothesis": relation.hypothesis,
                "next_validation_action": relation.next_validation_action,
            }
            for relation in relations
        ],
    }


def render_human_questions(schema: Mapping[str, object], relations: list[RelationCandidate]) -> str:
    questions = schema.get("calibration_questions")
    lines: list[str] = []
    lines.append("# LIF-Memory 自适应关系层校准问题")
    lines.append("")
    lines.append("这些问题用于把自动归纳的概念/关系变成你的长期认知定义。")
    lines.append("")
    if isinstance(questions, list) and questions:
        lines.append("## LLM 提出的校准问题")
        lines.append("")
        for index, question in enumerate(questions, start=1):
            lines.append(f"{index}. {question}")
        lines.append("")
    lines.append("## 关系确认")
    lines.append("")
    for index, relation in enumerate(relations[:10], start=1):
        lines.append(f"{index}. `{relation.source}` 和 `{relation.target}` 的关系更像哪一种？")
        lines.append("   - support / conflict / inhibition / reframe / downgrade / causal / bridge / unknown")
        lines.append(f"   - 当前假设：{relation.hypothesis}")
        lines.append("")
    return "\n".join(lines)


def render_markdown(
    vault: Path,
    notes: list[tuple[date, Path]],
    fragments: list[Fragment],
    concepts: list[Concept],
    relations: list[RelationCandidate],
    schema_source: str,
) -> str:
    lines: list[str] = []
    lines.append("# LIF Adaptive Relation Spike 回放结果")
    lines.append("")
    lines.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"版本：{VERSION}")
    lines.append(f"Schema source：{schema_source}")
    lines.append(f"回放日志数：{len(notes)}")
    if notes:
        lines.append(f"回放范围：{notes[0][0].isoformat()} 到 {notes[-1][0].isoformat()}")
    lines.append(f"片段数：{len(fragments)}")
    lines.append("")
    lines.append("## 核心变化")
    lines.append("")
    lines.append("这个模块不预设固定主题和固定角色。它先从笔记中归纳概念，再根据概念共现和 LLM 归纳生成关系。")
    lines.append("")
    lines.append("```text")
    lines.append("notes -> induced concepts -> concept co-activation graph -> adaptive relation spikes")
    lines.append("```")
    lines.append("")
    lines.append("## 自动归纳的概念")
    lines.append("")
    lines.append("| Concept | Source | Score | Aliases |")
    lines.append("|---|---|---:|---|")
    for concept in concepts[:32]:
        lines.append(f"| {concept.name} | {concept.source} | {concept.score:.2f} | {', '.join(concept.aliases[:6])} |")
    lines.append("")
    lines.append("## Adaptive relation spikes")
    lines.append("")
    if not relations:
        lines.append("本次没有产生关系 spike。可以扩大 `--days`，或使用 `--llm-induce-schema` 让 LLM 先归纳 schema。")
        lines.append("")
    for index, relation in enumerate(relations, start=1):
        lines.append(f"### Relation {index}: {relation.name()}")
        lines.append("")
        lines.append(f"- Source mode：{relation.source_mode}")
        lines.append(f"- Relation type：{relation.relation_type}")
        lines.append(f"- Score：{relation.score:.2f}")
        lines.append(f"- Count：{relation.count}")
        lines.append(f"- Distinct days：{relation.distinct_days}")
        lines.append("- Hypothesis：")
        lines.append(f"  - {relation.hypothesis}")
        lines.append("- Next validation action：")
        lines.append(f"  - {relation.next_validation_action}")
        lines.append("- Evidence chain：")
        for fragment in relation.evidence[:8]:
            lines.append(f"  - `{fragment.day.isoformat()}` [[{core.md_link(fragment.path, vault)}]] terms={fragment.terms[:6]}：{fragment.text}")
        lines.append("- JSON packet:")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(relation.packet(vault), ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    vault = (args.vault or core.vault_root_from_script()).resolve()
    cutoff = parse_cutoff(args.today)
    notes = core.find_daily_notes(vault, cutoff, args.days)
    fragments = extract_fragments(notes)

    statistical_concepts = induce_statistical_concepts(
        fragments,
        top_k=args.top_concepts,
        min_count=args.min_term_count,
    )
    annotate_fragments(fragments, statistical_concepts)
    statistical_relations = build_statistical_relations(fragments, top_relations=args.top_relations)

    schema_path = resolve_path(vault, args.schema_file)
    assert schema_path is not None
    schema = load_schema(schema_path)
    schema_source = str(schema.get("generator", "statistical")) if schema else "statistical"

    if args.llm_induce_schema:
        schema = induce_llm_schema(args, fragments, statistical_concepts)
        schema_source = "llm"

    concepts = schema_to_concepts(schema, statistical_concepts) if schema else statistical_concepts
    annotate_fragments(fragments, concepts)
    relations = build_statistical_relations(fragments, top_relations=args.top_relations)
    if schema:
        relations = apply_llm_relations(schema, relations, fragments, top_relations=args.top_relations)

    if args.write_schema or args.llm_induce_schema:
        output_schema = schema if schema else statistical_schema(concepts, relations)
        write_schema(schema_path, output_schema)
        print(f"Wrote schema: {schema_path}")

    if args.ask_human_output is not None:
        question_path = resolve_path(vault, args.ask_human_output)
        assert question_path is not None
        question_path.parent.mkdir(parents=True, exist_ok=True)
        question_path.write_text(render_human_questions(schema or statistical_schema(concepts, relations), relations), encoding="utf-8")
        print(f"Wrote human calibration questions: {question_path}")

    report = render_markdown(vault, notes, fragments, concepts, relations, schema_source)
    if args.dry_run:
        print(report)
    else:
        output = resolve_path(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Scanned {len(notes)} notes.")
        print(f"Extracted {len(fragments)} fragments.")
        print(f"Induced {len(concepts)} concepts.")
        print(f"Generated {len(relations)} adaptive relation spikes.")
        print(f"Wrote: {output}")

    json_output = resolve_path(vault, args.json_output)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        packets = [relation.packet(vault) for relation in relations]
        json_output.write_text(json.dumps(packets, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
