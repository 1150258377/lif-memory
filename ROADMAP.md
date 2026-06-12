# LIF-Memory Roadmap

## v0.3 status

The project has moved from a single replay script toward a small, testable memory system:

- Configurable state neurons through JSON config.
- Optional persistent `state-file` for incremental runs.
- Evidence packets with source path, snippet, score, matched keywords, and modifiers.
- Unit tests for note discovery, spike triggering, completion inhibition, incremental filtering, and config round trip.

## Next engineering steps

1. Split the single script into a small package:

```text
lif_memory/
  cli.py
  config.py
  evidence.py
  neuron.py
  replay.py
  render.py
```

2. Add a manual evaluation dataset:

```text
examples/evaluation/
  expected_spikes.json
  notes/
```

3. Add calibration:

```text
manual label -> adjust theta / evidence_cap / keyword weight
```

4. Add a safer action policy:

```text
Health spike suppresses non-urgent work spikes
Experiment + Thesis conflict resolves to one concrete action
Daily budget defaults to one spike when stress is high
```

## Research framing

LIF-Memory should not be sold as language reconstruction from spikes.

The stronger claim is:

```text
High-dimensional notes remain lossless in Obsidian.
Low-dimensional LIF states track unresolved pressure over time.
Sparse spikes trigger evidence recall and action selection.
```

This makes the project a practical analogue of event-driven memory rather than a toy keyword reminder.
