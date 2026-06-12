# LIF-Memory Roadmap

## v0.4 status

The project now includes `insight_integrator.py`.

This changes the center of the project:

```text
from: keyword-triggered action reminder
  to: LIF-style weak evidence integration
```

The important behavior is that one strong keyword is not enough. The system waits for multiple weak fragments around the same latent question, integrates them with leak, and emits an insight spike only after the integrated state crosses a threshold.

Implemented in v0.4:

- `insight_integrator.py`
- `docs/INSIGHT_INTEGRATOR.md`
- `tests/test_insight_integrator.py`
- Default latent questions:
  - `Innovation_Claim`
  - `Experimental_Closure`
  - `Thesis_Closure`
  - `Action_Bottleneck`

## v0.3 status

The project has moved from a single replay script toward a small, testable memory system:

- Configurable state neurons through JSON config.
- Optional persistent `state-file` for incremental runs.
- Evidence packets with source path, snippet, score, matched keywords, and modifiers.
- Unit tests for state round trip, config loading, incremental filtering, and stateful replay.

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
  insight.py
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
Health or Action_Bottleneck spike suppresses non-urgent work spikes
Experiment + Thesis conflict resolves to one concrete action
Daily budget defaults to one spike when stress is high
```

5. Upgrade insight generation:

```text
static insight template -> evidence-conditioned generated insight -> user-rated calibration
```

## Research framing

LIF-Memory should not be sold as language reconstruction from spikes.

The stronger claim is:

```text
High-dimensional notes remain lossless in Obsidian.
Low-dimensional LIF states track unresolved pressure and latent questions over time.
Sparse spikes trigger evidence recall, insight synthesis, and next validation action.
```

This makes the project a practical analogue of event-driven memory rather than a toy keyword reminder.
