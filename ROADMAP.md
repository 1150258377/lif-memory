# LIF-Memory Roadmap

## v0.4 status

The main replay path now emits action-decision spike packets.

This changes the center of the project:

```text
from: state threshold reminder
  to: time-evolving action router
```

The important behavior is that a spike no longer only says that a state is hot. It also decides whether the topic should continue, be isolated, be downgraded, or wait for recovery.

Implemented in v0.4:

- Spike packet fields:
  - `topic`
  - `priority`
  - `blocker_type`
  - `action_policy`
  - `completion_target`
- Topic history for simple repeated-failure detection.
- `负阻` loop routing: repeated failed experiment evidence becomes `blocker_type=repeated_failure` and `action_policy=isolate`.
- Unit coverage for negative-resistance isolation behavior.

Related experimental track:

- `insight_integrator.py`
- `docs/INSIGHT_INTEGRATOR.md`
- `tests/test_insight_integrator.py`
- Latent questions such as `Innovation_Claim`, `Experimental_Closure`, `Thesis_Closure`, and `Action_Bottleneck`.

## v0.5 status

The stateful runner now persists topic history.

Implemented in v0.5:

- `lif_state.json` stores a `topics` object.
- Each topic stores:
  - `days_seen`
  - `completion_count`
  - `blocker_count`
  - `evidence_count`
  - `last_action_policy`
- Incremental stateful runs pass persisted topic history into the v0.4 action-decision layer.

This keeps blocker-loop memory alive across runs instead of recomputing it only inside the current replay window.

## v0.6 status

Each state now has fast and slow voltage traces.

Implemented in v0.6:

- `NeuronState` stores `v_fast` and `v_slow`.
- `NeuronConfig` stores slow-timescale parameters:
  - `slow_decay`
  - `fast_weight`
  - `slow_weight`
  - `slow_input_ratio`
  - `slow_completion_ratio`
- Spike packets report both time scales in `voltage_model`.
- Stateful runs persist `v_fast` and `v_slow` in `lif_state.json`.

This makes the system closer to a multi-timescale action state model:

```text
V_fast high -> acute pressure
V_slow high -> background unresolved pressure
V_fast + V_slow high -> serious active loop
```

## v0.7 status

Manual feedback now updates topic policies.

Implemented in v0.7:

- CLI support for `--feedback-file`.
- Feedback labels:
  - `有用`
  - `没用`
  - `太早`
  - `太晚`
  - `已完成`
  - `不要再提醒`
  - `升为P0`
  - `降为P2`
- `TopicPolicy` fields:
  - `threshold_delta`
  - `priority_override`
  - `action_policy_override`
  - `muted`
  - `cooldown_days`
  - `last_feedback`
- Stateful runs persist policies in `lif_state.json` under `topic_policies`.

This closes the first feedback loop:

```text
spike -> user label -> topic policy -> next replay behavior
```

## v0.7.1 status

The semantic interface has been calibrated.

Implemented in v0.7.1:

- Removed overly broad `工作` matching from `求职`.
- Added `实验数据模板` and `AI求职转向` topics.
- Added state-aware topic priors so Experiment evidence is not pulled into Career by generic words.
- Added forced priority mapping:
  - `论文闭环`, `LIF链路`, `实验数据模板`, `健康恢复` -> `P0`
  - `负阻` -> `P1`
  - `AI记忆` -> `P2`
- Spike packets now include:
  - `primary_state`
  - `secondary_states`
- Optional local completion-signal scan via `--completion-scan`.

This release addresses the observed bug:

```text
Experiment / data template evidence -> wrongly labeled 求职
```

The expected result is now:

```text
Experiment / 实验数据模板
```

## v0.7.2 status

The completion loop is now explicit.

Implemented in v0.7.2:

- Topic-specific `completion_target` templates.
- Stable `spike_id` in Markdown and JSON packets.
- A `## Spike 反馈区` appended to replay reports.
- Markdown closure parsing for:
  - `done`
  - `downgraded`
  - `ignored`
  - `postponed`
- Closure-derived cooldown and policy updates.
- `--mode daily --top-k 1` for a single daily action card.

This closes the second feedback loop:

```text
spike -> manual closure -> cooldown/policy -> next replay behavior
```

The main design rule is now:

```text
do not only trigger; make every spike closable
```

## v0.7.3 status

Feedback memory is now persistent.

Implemented in v0.7.3:

- Default persistent memory file: `lif_memory_feedback.json`.
- Markdown closures are copied into JSON memory after a non-dry run.
- Replay merges three policy sources:
  - JSON feedback file
  - persistent feedback memory
  - current Markdown closure file
- Topic memory stores:
  - `status`
  - `cooldown_until`
  - `last_feedback`
  - `completion_evidence`
  - policy overrides
  - spike ids
- Cooldown penalties expire when `cooldown_until` passes.
- Durable downgraded topics keep `P2 / downgrade` even after cooldown expires.

This closes the third feedback loop:

```text
spike -> manual closure -> JSON memory -> future replay policy
```

## v0.7.4 status

LLM review is now available as a semantic adapter.

Implemented in v0.7.4:

- Added `llm_adapter.py`.
- Added `--llm-review`.
- Added OpenAI-compatible `chat/completions` support.
- Default provider preset: `qwen` / DashScope.
- Switchable presets:
  - `deepseek`
  - `kimi`
  - `zhipu`
  - `custom`
- LLM output is rendered in `## LLM Review`.

Boundary:

```text
LLM may review topic/state/completion_target
LLM may not update V, theta, cooldown, priority, or action_policy
```

This creates the next architecture layer:

```text
rules + LIF time dynamics + closure memory + LLM semantic reviewer
```

## v0.3 status

The project has moved from a single replay script toward a small, testable memory system:

- Configurable state neurons through JSON config.
- Optional persistent `state-file` for incremental runs.
- Evidence packets with source path, snippet, score, matched keywords, and modifiers.
- Unit tests for state round trip, config loading, incremental filtering, and stateful replay.

## Next engineering steps

0. Upgrade from linear daily replay to graph-aware memory:

```text
daily fragment -> wikilink/backlink/folder/state neighbors -> graph current -> LIF voltage
```

The first tool for this is `obsidian_graph_miner.py`, which inventories the vault graph and projects notes into LIF states.

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

Use it to prevent future changes from improving one case while breaking old behavior.

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
