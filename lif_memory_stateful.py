from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import lif_memory as core

VERSION = "0.7.4"
DEFAULT_NEURONS: dict[str, core.NeuronConfig] = getattr(core, "DEFAULT_NEURONS", core.NEURONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LIF-Memory with config and persistent state.")
    parser.add_argument("--vault", type=Path, default=None, help="Obsidian vault path.")
    parser.add_argument("--days", type=int, default=14, help="Number of latest daily notes to scan.")
    parser.add_argument("--today", type=str, default=None, help="Optional YYYY-MM-DD cutoff date.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON neuron config.")
    parser.add_argument("--init-config", type=Path, default=None, help="Write default config and exit.")
    parser.add_argument("--state-file", type=Path, default=Path("lif_state.json"), help="Persistent state JSON path.")
    parser.add_argument("--reset-state", action="store_true", help="Ignore existing state and start from zero.")
    parser.add_argument("--replay-all", action="store_true", help="Replay selected window even if state already processed older notes.")
    parser.add_argument("--output", type=Path, default=Path("LIF-Memory 回放结果.md"), help="Markdown output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON spike packet output.")
    parser.add_argument("--feedback-file", type=Path, default=None, help="Optional JSON feedback file to update topic policies.")
    parser.add_argument("--closure-file", type=Path, default=None, help="Optional Markdown report whose Spike feedback section closes old spikes.")
    parser.add_argument("--feedback-memory", type=Path, default=None, help="Persistent JSON feedback memory. Defaults to lif_memory_feedback.json beside lif_memory.py.")
    parser.add_argument("--completion-scan", action="store_true", help="Scan local files for external completion signals.")
    parser.add_argument("--states", type=str, default=None, help="Comma-separated states, e.g. Experiment,Thesis.")
    parser.add_argument("--daily-spike-budget", type=int, default=2, help="Maximum spikes per day.")
    parser.add_argument("--mode", choices=["replay", "daily"], default="replay", help="Render full replay or daily top spike card.")
    parser.add_argument("--top-k", type=int, default=1, help="Number of top spikes to render in daily mode.")
    parser.add_argument("--llm-review", action="store_true", help="Ask an OpenAI-compatible LLM to review top spike semantics without changing LIF decisions.")
    parser.add_argument("--llm-provider", choices=[*core.LLM_PROVIDER_PRESETS.keys(), "custom"], default=os.environ.get("LIF_LLM_PROVIDER", "qwen"), help="LLM provider preset.")
    parser.add_argument("--llm-base-url", type=str, default=os.environ.get("LIF_LLM_BASE_URL"), help="OpenAI-compatible base URL. Required for custom provider.")
    parser.add_argument("--llm-model", type=str, default=os.environ.get("LIF_LLM_MODEL"), help="LLM model name. Defaults to provider preset.")
    parser.add_argument("--llm-api-key-env", type=str, default=os.environ.get("LIF_LLM_API_KEY_ENV"), help="Environment variable containing API key.")
    parser.add_argument("--llm-timeout", type=int, default=60, help="LLM request timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print report instead of writing output/state files.")
    parser.add_argument("--version", action="version", version=f"LIF-Memory Stateful {VERSION}")
    return parser.parse_args()


def parse_date(value: str | None) -> date:
    return date.today() if value is None else datetime.strptime(value, "%Y-%m-%d").date()


def parse_optional_date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise ValueError(f"Invalid date value: {value!r}")


def resolve_path(vault: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else vault / path


def neuron_to_json(config: core.NeuronConfig) -> dict[str, object]:
    return asdict(config)


def write_default_config(path: Path) -> None:
    payload = {
        "version": VERSION,
        "states": {name: neuron_to_json(config) for name, config in DEFAULT_NEURONS.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_neurons(config_path: Path | None) -> dict[str, core.NeuronConfig]:
    if config_path is None:
        return dict(DEFAULT_NEURONS)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    raw_states = data.get("states", data)
    if not isinstance(raw_states, dict):
        raise SystemExit("Config must contain a JSON object named 'states'.")

    neurons: dict[str, core.NeuronConfig] = {}
    for name, raw in raw_states.items():
        if not isinstance(raw, dict):
            raise SystemExit(f"Invalid config for state {name!r}.")
        try:
            neurons[name] = core.NeuronConfig(
                theta=float(raw["theta"]),
                decay=float(raw["decay"]),
                reset_ratio=float(raw["reset_ratio"]),
                cooldown_days=int(raw["cooldown_days"]),
                evidence_cap=float(raw.get("evidence_cap", raw["theta"])),
                keywords=[str(item) for item in raw["keywords"]],
                suggestion=str(raw["suggestion"]),
                slow_decay=float(raw.get("slow_decay", 0.92)),
                fast_weight=float(raw.get("fast_weight", 0.70)),
                slow_weight=float(raw.get("slow_weight", 0.30)),
                slow_input_ratio=float(raw.get("slow_input_ratio", 0.45)),
                slow_completion_ratio=float(raw.get("slow_completion_ratio", 0.35)),
            )
        except KeyError as exc:
            raise SystemExit(f"Missing key for state {name!r}: {exc.args[0]}") from exc
    return neurons


def select_neurons(value: str | None, neurons: Mapping[str, core.NeuronConfig]) -> dict[str, core.NeuronConfig]:
    if value is None or not value.strip():
        return dict(neurons)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in requested if name not in neurons]
    if unknown:
        raise SystemExit(f"Unknown states: {', '.join(unknown)}. Available: {', '.join(neurons)}")
    return {name: neurons[name] for name in requested}


def empty_states(neurons: Mapping[str, core.NeuronConfig]) -> dict[str, core.NeuronState]:
    return {name: core.NeuronState() for name in neurons}


def topic_history_to_json(topic_history: Mapping[str, core.TopicHistory]) -> dict[str, object]:
    return {
        topic: {
            "days_seen": sorted(day.isoformat() for day in record.days_seen),
            "completion_count": record.completion_count,
            "blocker_count": record.blocker_count,
            "evidence_count": record.evidence_count,
            "last_action_policy": record.last_action_policy,
        }
        for topic, record in topic_history.items()
    }


def load_topic_history(raw_topics: object) -> dict[str, core.TopicHistory]:
    topic_history: dict[str, core.TopicHistory] = {}
    if not isinstance(raw_topics, dict):
        return topic_history

    for topic, raw in raw_topics.items():
        if not isinstance(topic, str) or not isinstance(raw, dict):
            continue
        raw_days = raw.get("days_seen", [])
        days_seen: set[date] = set()
        if isinstance(raw_days, list):
            for item in raw_days:
                parsed = parse_optional_date(item)
                if parsed is not None:
                    days_seen.add(parsed)
        topic_history[topic] = core.TopicHistory(
            days_seen=days_seen,
            completion_count=int(raw.get("completion_count", 0)),
            blocker_count=int(raw.get("blocker_count", 0)),
            evidence_count=int(raw.get("evidence_count", 0)),
            last_action_policy=str(raw["last_action_policy"]) if raw.get("last_action_policy") is not None else None,
        )
    return topic_history


def topic_policies_to_json(topic_policies: Mapping[str, core.TopicPolicy]) -> dict[str, object]:
    return {topic: policy.to_packet() for topic, policy in topic_policies.items()}


def load_topic_policies(raw_policies: object) -> dict[str, core.TopicPolicy]:
    policies: dict[str, core.TopicPolicy] = {}
    if not isinstance(raw_policies, dict):
        return policies
    for topic, raw in raw_policies.items():
        if not isinstance(topic, str) or not isinstance(raw, dict):
            continue
        policies[topic] = core.TopicPolicy(
            threshold_delta=float(raw.get("threshold_delta", 0.0)),
            priority_override=str(raw["priority_override"]) if raw.get("priority_override") is not None else None,
            action_policy_override=str(raw["action_policy_override"]) if raw.get("action_policy_override") is not None else None,
            muted=bool(raw.get("muted", False)),
            cooldown_days=int(raw.get("cooldown_days", 0)),
            feedback_count=int(raw.get("feedback_count", 0)),
            last_feedback=str(raw["last_feedback"]) if raw.get("last_feedback") is not None else None,
        )
    return policies


def load_state(
    path: Path | None,
    neurons: Mapping[str, core.NeuronConfig],
    reset: bool,
) -> tuple[dict[str, core.NeuronState], date | None, dict[str, core.TopicHistory], dict[str, core.TopicPolicy]]:
    states = empty_states(neurons)
    if path is None or reset or not path.exists():
        return states, None, {}, {}

    data = json.loads(path.read_text(encoding="utf-8"))
    raw_neurons = data.get("neurons", {})
    if isinstance(raw_neurons, dict):
        for name in neurons:
            raw = raw_neurons.get(name, {})
            if not isinstance(raw, dict):
                continue
            states[name] = core.NeuronState(
                v=float(raw.get("v", 0.0)),
                last_spike_date=parse_optional_date(raw.get("last_spike_date")),
                v_fast=float(raw.get("v_fast", raw.get("v", 0.0))),
                v_slow=float(raw.get("v_slow", raw.get("v", 0.0))),
            )
    return (
        states,
        parse_optional_date(data.get("last_processed_date")),
        load_topic_history(data.get("topics")),
        load_topic_policies(data.get("topic_policies")),
    )


def save_state(
    path: Path,
    states: Mapping[str, core.NeuronState],
    processed_notes: list[tuple[date, Path]],
    previous_last: date | None,
    topic_history: Mapping[str, core.TopicHistory] | None = None,
    topic_policies: Mapping[str, core.TopicPolicy] | None = None,
) -> None:
    last_processed = max((day for day, _ in processed_notes), default=previous_last)
    payload = {
        "version": VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "last_processed_date": last_processed.isoformat() if last_processed else None,
        "neurons": {
            name: {
                "v": round(state.v, 6),
                "v_fast": round(state.v_fast, 6),
                "v_slow": round(state.v_slow, 6),
                "last_spike_date": state.last_spike_date.isoformat() if state.last_spike_date else None,
            }
            for name, state in states.items()
        },
        "topics": topic_history_to_json(topic_history or {}),
        "topic_policies": topic_policies_to_json(topic_policies or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def filter_incremental_notes(notes: list[tuple[date, Path]], last_processed: date | None, replay_all: bool) -> list[tuple[date, Path]]:
    if replay_all or last_processed is None:
        return notes
    return [(day, path) for day, path in notes if day > last_processed]


def replay_with_state(
    notes: list[tuple[date, Path]],
    daily_spike_budget: int,
    neurons: Mapping[str, core.NeuronConfig],
    states: dict[str, core.NeuronState],
    topic_history: dict[str, core.TopicHistory] | None = None,
    topic_policies: dict[str, core.TopicPolicy] | None = None,
    completion_signals: dict[date, dict[str, list[Path]]] | None = None,
) -> tuple[list[core.Spike], list[dict[str, object]], dict[str, core.NeuronState]]:
    for name in neurons:
        states.setdefault(name, core.NeuronState())

    spikes: list[core.Spike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None
    if topic_history is None:
        topic_history = {}
    if topic_policies is None:
        topic_policies = {}

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = core.extract_daily_evidence(day, path, text, neurons)
        core.apply_completion_signals(day, daily, completion_signals)
        core.update_topic_history(topic_history, daily)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day

        row: dict[str, object] = {"date": day.isoformat(), "path": path, "delta_days": delta_days}
        candidates: list[tuple[float, str, core.NeuronConfig, core.NeuronState, core.DailyEvidence]] = []

        for name, config in neurons.items():
            state = states[name]
            evidence = daily[name]
            voltage_update = core.update_voltage_state(state, config, evidence, delta_days)
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
            topic = core.infer_topic(name, evidence.items[:4])
            policy = topic_policies.get(topic)
            effective_threshold = policy.adjusted_threshold(config.theta) if policy else config.theta
            if policy and policy.muted:
                row[name]["muted_topic"] = topic
                row[name]["effective_threshold"] = effective_threshold
                continue
            row[name]["topic"] = topic
            row[name]["effective_threshold"] = effective_threshold

            if new_v >= effective_threshold and core.can_spike(state, day, config):
                candidates.append((new_v / effective_threshold, name, config, state, evidence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, config, state, evidence in candidates[: max(0, daily_spike_budget)]:
            item = row[name]
            assert isinstance(item, dict)
            topic = str(item.get("topic") or core.infer_topic(name, evidence.items[:4]))
            policy = topic_policies.get(topic)
            effective_threshold = float(item.get("effective_threshold", config.theta))
            decision = core.decide_action(name, topic, state.v, effective_threshold, evidence, topic_history, topic_policies)
            primary_state = core.primary_state_for_spike(name, topic, evidence.items[:4])
            secondary_states = core.infer_secondary_states(primary_state, topic, evidence.items[:4])
            spike = core.Spike(
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
                spike_id=core.make_spike_id(day, primary_state, topic),
                decision_reason=decision.get("decision_reason"),
                action_suggestion=decision.get("action_suggestion"),
                feedback_policy=policy,
            )
            spikes.append(spike)
            state.last_spike_date = day
            core.reset_after_spike(state, config)
            item["new_v"] = state.v
            item["new_fast"] = state.v_fast
            item["new_slow"] = state.v_slow
            item["spike"] = True

        timeline.append(row)

    return spikes, timeline, states


def render_report(
    vault: Path,
    notes: list[tuple[date, Path]],
    spikes: list[core.Spike],
    timeline: list[dict[str, object]],
    states: Mapping[str, core.NeuronState],
    neurons: Mapping[str, core.NeuronConfig],
    state_file: Path | None,
    last_processed: date | None,
    topic_history: Mapping[str, core.TopicHistory] | None = None,
    topic_policies: Mapping[str, core.TopicPolicy] | None = None,
) -> str:
    report = core.render_markdown(vault, notes, spikes, timeline, states, neurons, dict(topic_policies or {}))
    topic_count = len(topic_history or {})
    policy_count = len(topic_policies or {})
    extra = [
        "",
        "## Stateful runner",
        "",
        f"状态文件：{state_file}" if state_file else "状态文件：未启用",
        f"上次处理到：{last_processed.isoformat()}" if last_processed else "上次处理到：无",
        f"持久化 topic 数：{topic_count}",
        f"持久化反馈策略数：{policy_count}",
        "",
        "说明：启用 `--state-file` 后，默认只处理 `last_processed_date` 之后的新日志，并保存 topic history，避免同一阻塞回路在增量运行中被遗忘。需要重新回放窗口时使用 `--replay-all` 或 `--reset-state`。",
        "",
    ]
    return report + "\n".join(extra)


def main() -> None:
    args = parse_args()
    if args.init_config is not None:
        write_default_config(args.init_config)
        print(f"Wrote default config: {args.init_config}")
        return

    vault = (args.vault or core.vault_root_from_script()).resolve()
    cutoff = parse_date(args.today)
    neurons = select_neurons(args.states, load_neurons(args.config))
    state_path = resolve_path(vault, args.state_file)
    states, last_processed, topic_history, topic_policies = load_state(state_path, neurons, reset=args.reset_state)
    output = resolve_path(vault, args.output)
    feedback_path = resolve_path(vault, args.feedback_file)
    if feedback_path is not None:
        feedback_policies = core.load_feedback_file(feedback_path)
        for topic, policy in feedback_policies.items():
            topic_policies.setdefault(topic, core.TopicPolicy())
            core.merge_topic_policy(topic_policies[topic], policy)

    closure_path = resolve_path(vault, args.closure_file) if args.closure_file else output
    feedback_memory_path = resolve_path(vault, args.feedback_memory) if args.feedback_memory else core.default_feedback_memory_path()
    memory_policies = core.load_feedback_memory(feedback_memory_path, cutoff)
    for topic, policy in memory_policies.items():
        core.merge_topic_policy(topic_policies.setdefault(topic, core.TopicPolicy()), policy)

    closure_policies = core.policies_from_closures(core.load_spike_closures(closure_path))
    for topic, policy in closure_policies.items():
        core.merge_topic_policy(topic_policies.setdefault(topic, core.TopicPolicy()), policy)

    notes = core.find_daily_notes(vault, cutoff, args.days)
    notes_to_process = filter_incremental_notes(notes, last_processed, args.replay_all)
    completion_signals = core.scan_completion_signals(vault, notes_to_process, neurons) if args.completion_scan else None
    closures = core.load_spike_closures(closure_path)
    spikes, timeline, states = replay_with_state(
        notes_to_process,
        args.daily_spike_budget,
        neurons,
        states,
        topic_history,
        topic_policies,
        completion_signals,
    )
    core.apply_closures_to_spikes(spikes, closures)
    llm_reviews = core.run_llm_reviews(spikes, vault, args)

    if args.mode == "daily":
        report = core.render_daily_markdown(vault, notes_to_process, spikes, args.top_k, llm_reviews)
    else:
        report = render_report(vault, notes_to_process, spikes, timeline, states, neurons, state_path, last_processed, topic_history, topic_policies)
        if llm_reviews:
            extra_lines: list[str] = []
            core.render_llm_review_section(extra_lines, llm_reviews)
            report += "\n" + "\n".join(extra_lines)
    if args.dry_run:
        print(report)
    else:
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Replayed {len(notes_to_process)} notes.")
        print(f"Generated {len(spikes)} spikes.")
        print(f"Wrote: {output}")
        if state_path is not None:
            save_state(state_path, states, notes_to_process, last_processed, topic_history, topic_policies)
            print(f"Saved state: {state_path}")
        core.update_feedback_memory(feedback_memory_path, closures, cutoff)

    json_output = resolve_path(vault, args.json_output)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
