# LIF-Memory

LIF-Memory is a minimal event-driven memory replay prototype for Obsidian notes.

It does **not** try to reconstruct full language from spikes. Instead, it keeps original notes in Obsidian, maps high-dimensional text into low-dimensional LIF-style state variables, and emits evidence-linked spike event packets when a state crosses its threshold.

```text
Original notes -> Evidence extraction -> LIF states -> Threshold crossing -> Evidence-linked spike -> Suggested action
```

## What was optimized in v0.2.0

This version makes the LIF replay more useful as an actual memory/action trigger:

- Recursive Obsidian daily-note discovery with ignored folders such as `.git`, `.obsidian`, `.venv`, and `node_modules`.
- Evidence packets now keep source path, snippet, matched keywords, score, and modifiers.
- LIF leakage now respects date gaps: `V_new = decay^delta_days * V_old + input - completion`.
- Added per-state `evidence_cap` to avoid one noisy note saturating the whole system.
- Added `--states`, `--dry-run`, and `--json-output` for debugging and downstream agent use.
- Markdown output now includes a summary table, state trajectory, spike cards, and practical tuning rules.

## States

The default version tracks five state neurons:

```text
Experiment
Thesis
Career
AI_Memory
Health
```

Each state has:

```text
V: current voltage
theta: spike threshold
decay: leakage coefficient per day
reset_ratio: post-spike reset level
cooldown_days: minimum spacing between spikes
evidence_cap: maximum daily evidence input
keywords: evidence detector
suggestion: action emitted after spike
```

## Run

From your Obsidian vault root:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --days 7 --output "LIF-Memory 回放结果.md"
```

If the script is outside the vault, pass the vault path explicitly:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 7 --output "LIF-Memory 回放结果.md"
```

Preview without writing files:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 7 --dry-run
```

Only replay specific states:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --states Experiment,Thesis --days 14
```

Export JSON spike packets for Codex/agent workflows:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 7 --json-output "lif_spikes.json"
```

## Control Spike Count

Limit the maximum number of emitted spike packets per day:

```powershell
python lif_memory.py --days 7 --daily-spike-budget 1 --output "LIF-Memory 回放结果.md"
```

The default is:

```text
--daily-spike-budget 2
```

## Output

The output is a Markdown replay report containing:

```text
summary table
trigger cards
JSON spike event packets
state trajectory table
tuning rules
manual evaluation section
```

A spike event packet looks like this:

```json
{
  "spike_type": "Experiment",
  "time": "2026-06-12",
  "V": 7.95,
  "threshold": 7.5,
  "evidence_notes": [
    {
      "note": "2026-06-12",
      "path": "06 日志复盘/2026/2026-06-12.md",
      "snippet": "Fourth chapter lacks experiment data",
      "score": 2.65,
      "matched_keywords": ["第四章", "数据"],
      "modifiers": ["action", "blocker", "time_pressure"]
    }
  ],
  "trigger_reason": "Experiment evidence has accumulated and is actionable.",
  "suggested_action": "Run one focused 30-minute experiment and save the result."
}
```

## Tuning

Use the manual evaluation section after each run:

```text
合理 / 太早 / 太晚 / 无用 / 应该触发但没触发
```

Then tune:

- Too many false spikes: raise `theta`, narrow keywords, or reduce `evidence_cap`.
- Too late: lower `theta` or add stronger keywords.
- Repeated daily reminders: increase `cooldown_days`.
- Still triggers after completion: add better `COMPLETION_WORDS` or check `INCOMPLETE_WORDS`.

## Boundary

Signal systems may optimize:

```text
spike -> waveform reconstruction
```

LIF-Memory optimizes:

```text
spike -> evidence recall -> action trigger
```

Original notes preserve information. State voltage preserves trend. Spikes trigger action.
