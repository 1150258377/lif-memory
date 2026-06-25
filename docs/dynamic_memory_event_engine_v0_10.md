# Dynamic MemoryEvent Engine v0.10

This update adds a typed dynamic-memory layer to LIF-Memory.

The goal is to move one step beyond static note recall:

```text
raw notes / sessions
-> MemoryEvent extraction
-> LIF-style memory activation
-> conflict / outdated state
-> insight cards
-> updated persistent memory state
```

## Why this exists

Earlier versions already build a continuous problem field, graph diffusion, LIF voltage and action-selection controllers. The missing layer was a stable unit of long-term memory.

This version introduces `MemoryEvent` as that unit. A memory event is not a raw chat snippet and not just a summary. It preserves the minimum structure needed for continuity:

```text
topic
claim
problem
decision
evidence
emotion
importance
status
confidence
source
```

This makes LIF-Memory closer to a system that can answer:

```text
What is the user currently stuck on?
Which old decisions still matter?
Which memories conflict with the new situation?
What is the smallest next action?
```

## New file

- `lif_memory_event_engine.py`

It is standalone and uses only the Python standard library.

## Basic run

```powershell
python lif_memory_event_engine.py `
  --input lif_sessions.json lif_conclusions.json `
  --query "LIF Memory 怎么升级才能产生灵光一闪" `
  --state lif_memory_events.json `
  --output "LIF-Dynamic-Memory-Report.md" `
  --json-output lif_dynamic_memory_packet.json
```

Dry run:

```powershell
python lif_memory_event_engine.py --input lif_sessions.json --query "今天最该处理什么" --dry-run --no-state-update
```

## Output packet

```json
{
  "version": "0.10.0",
  "query": "...",
  "event_count": 0,
  "activated": [],
  "conflicts": [],
  "insights": [],
  "state_summary": {}
}
```

## Persistent state

The engine writes a local state file by default:

```text
lif_memory_events.json
```

This file may contain private derived memory and should not be committed.

State contains:

```text
events      typed MemoryEvent records
voltages    per-event LIF activation state
conflicts   detected route / claim conflicts
```

## What changed conceptually

Previous retrieval logic often looked like:

```text
query -> similar chunks -> answer
```

This version adds:

```text
query -> activate relevant MemoryEvents -> reconstruct current state -> insight card
```

The result is not meant to replace dense embedding retrieval. It gives dense retrieval a higher-level memory structure to work with.

## Built-in conflict types

Current lightweight symbolic conflict checks:

| conflict | meaning |
| --- | --- |
| `EEG_vs_ECG` | EEG thesis route vs ECG lower-risk validation route |
| `Sparse_vs_Async` | sparse-compression proof vs async-event proof |
| `Build_vs_Insight` | implementation pressure vs essence/innovation pressure |
| `Recall_vs_Reconstruct` | static retrieval vs dynamic reconstruction |

These are deliberately conservative and auditable. They can later be replaced by LLM review or embedding-based contradiction detection.

## Tests

```powershell
python -m unittest tests/test_lif_memory_event_engine.py -v
```

The tests cover:

- MemoryEvent extraction from text
- EEG/ECG conflict detection
- JSON loading and activation packet generation
- state merge behavior

## Boundary

This is not model fine-tuning and not a production vector database. It is a local controller layer for making long-term memory more structured, conflict-aware and action-oriented.
