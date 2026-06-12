from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Mapping

import lif_memory as core

VERSION = "0.3.0"
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
    parser.add_argument("--states", type=str, default=None, help="Comma-separated states, e.g. Experiment,Thesis.")
    parser.add_argument("--daily-spike-budget", type=int, default=2, help="Maximum spikes per day.")
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


def load_state(path: Path | None, neurons: Mapping[str, core.NeuronConfig], reset: bool) -> tuple[dict[str, core.NeuronState], date | None]:
    states = empty_states(neurons)
    if path is None or reset or not path.exists():
        return states, None

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
            )
    return states, parse_optional_date(data.get("last_processed_date"))


def save_state(path: Path, states: Mapping[str, core.NeuronState], processed_notes: list[tuple[date, Path]], previous_last: date | None) -> None:
    last_processed = max((day for day, _ in processed_notes), default=previous_last)
    payload = {
        "version": VERSION,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "last_processed_date": last_processed.isoformat() if last_processed else None,
        "neurons": {
            name: {
                "v": round(state.v, 6),
                "last_spike_date": state.last_spike_date.isoformat() if state.last_spike_date else None,
            }
            for name, state in states.items()
        },
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
) -> tuple[list[core.Spike], list[dict[str, object]], dict[str, core.NeuronState]]:
    for name in neurons:
        states.setdefault(name, core.NeuronState())

    spikes: list[core.Spike] = []
    timeline: list[dict[str, object]] = []
    previous_day: date | None = None

    for day, path in notes:
        text = path.read_text(encoding="utf-8", errors="ignore")
        daily = core.extract_daily_evidence(day, path, text, neurons)
        delta_days = 1 if previous_day is None else max((day - previous_day).days, 1)
        previous_day = day

        row: dict[str, object] = {"date": day.isoformat(), "path": path, "delta_days": delta_days}
        candidates: list[tuple[float, str, core.NeuronConfig, core.NeuronState, core.DailyEvidence]] = []

        for name, config in neurons.items():
            state = states[name]
            evidence = daily[name]
            old_v = state.v
            new_v = max(0.0, (config.decay ** delta_days) * old_v + evidence.evidence - evidence.completion)
            state.v = new_v
            row[name] = {
                "old_v": old_v,
                "new_v": new_v,
                "input": evidence.evidence,
                "completion": evidence.completion,
                "spike": False,
                "evidence_count": len(evidence.items),
            }
            if new_v >= config.theta and core.can_spike(state, day, config):
                candidates.append((new_v / config.theta, name, config, state, evidence))

        candidates.sort(key=lambda item: item[0], reverse=True)
        for _, name, config, state, evidence in candidates[: max(0, daily_spike_budget)]:
            spike = core.Spike(
                day=day,
                neuron=name,
                voltage=state.v,
                threshold=config.theta,
                evidence_items=evidence.items[:4],
                suggestion=config.suggestion,
            )
            spikes.append(spike)
            state.last_spike_date = day
            state.v = config.theta * config.reset_ratio
            item = row[name]
            assert isinstance(item, dict)
            item["new_v"] = state.v
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
) -> str:
    report = core.render_markdown(vault, notes, spikes, timeline, states, neurons)
    extra = [
        "",
        "## Stateful runner",
        "",
        f"状态文件：{state_file}" if state_file else "状态文件：未启用",
        f"上次处理到：{last_processed.isoformat()}" if last_processed else "上次处理到：无",
        "",
        "说明：启用 `--state-file` 后，默认只处理 `last_processed_date` 之后的新日志，避免同一批日志重复累积。需要重新回放窗口时使用 `--replay-all` 或 `--reset-state`。",
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
    states, last_processed = load_state(state_path, neurons, reset=args.reset_state)

    notes = core.find_daily_notes(vault, cutoff, args.days)
    notes_to_process = filter_incremental_notes(notes, last_processed, args.replay_all)
    spikes, timeline, states = replay_with_state(notes_to_process, args.daily_spike_budget, neurons, states)

    report = render_report(vault, notes_to_process, spikes, timeline, states, neurons, state_path, last_processed)
    if args.dry_run:
        print(report)
    else:
        output = resolve_path(vault, args.output)
        assert output is not None
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(f"Replayed {len(notes_to_process)} notes.")
        print(f"Generated {len(spikes)} spikes.")
        print(f"Wrote: {output}")
        if state_path is not None:
            save_state(state_path, states, notes_to_process, last_processed)
            print(f"Saved state: {state_path}")

    json_output = resolve_path(vault, args.json_output)
    if json_output is not None:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        core.write_json_output(json_output, vault, spikes)
        print(f"Wrote JSON: {json_output}")


if __name__ == "__main__":
    main()
