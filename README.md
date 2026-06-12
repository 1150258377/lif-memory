# LIF-Memory

LIF-Memory is a minimal event-driven memory replay prototype for Obsidian notes.

It does not try to reconstruct full language from spikes. Instead, it keeps original notes in Obsidian, maps high-dimensional text into low-dimensional LIF-style state variables, and emits evidence-linked spike event packets when a state crosses its threshold.

```text
Original notes -> Event extraction -> LIF states -> Threshold crossing -> Evidence-linked spike -> Suggested action
```

## States

The first version tracks five state neurons:

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
decay: leakage coefficient
reset_ratio: post-spike reset level
cooldown_days: minimum spacing between spikes
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

## Control Spike Count

Limit the maximum number of emitted spike packets per day:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --days 7 --daily-spike-budget 1 --output "LIF-Memory 回放结果.md"
```

The default is:

```text
--daily-spike-budget 2
```

## Output

The output is a Markdown replay report containing:

```text
trigger cards
JSON spike event packets
state trajectory table
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
      "snippet": "Fourth chapter lacks experiment data"
    }
  ],
  "trigger_reason": "Experiment evidence has accumulated and is actionable.",
  "suggested_action": "Run one focused 30-minute experiment and save the result."
}
```

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

