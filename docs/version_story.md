# LIF-Memory Version Story

## Why the versions felt disconnected

The versions felt broken because each release optimized one local problem, but the repository did not yet name the larger system architecture.

The right interpretation is not:

```text
v0.2 did A
v0.3 did B
v0.4 did C
```

The right interpretation is:

```text
Memory Source -> Evidence Sensor -> LIF Dynamics -> Spike Interface -> Feedback Memory
```

Each historical version belongs to one of these layers.

## Version-to-layer mapping

| Version | Local problem solved | Unified layer | Contribution to v1.0 |
|---|---|---|---|
| v0.2 | Obsidian notes were scattered and needed traceable evidence packets. | Memory Source | recursive daily-note discovery, source path, snippet, keyword, score, modifier, date-gap leakage |
| v0.3 | Keyword hits were not enough to represent memory pressure. | Evidence Sensor | EvidenceVector: target weight, actionability, urgency, blocker, completion, specificity, novelty, confidence |
| v0.4 | A spike needed a decision, not only a threshold crossing. | Spike Interface | priority, blocker type, action policy, completion target |
| v0.5 | One-window replay forgot repeated loops. | LIF Dynamics | topic history: days seen, completion count, blocker count, evidence count, last action policy |
| v0.6 | Acute and long-term pressure should not share one decay speed. | LIF Dynamics | V_fast, V_slow, weighted voltage, state-specific memory speeds |
| v0.7.0 | Human judgement needed to affect future triggers. | Feedback Memory | useful/useless/too early/too late/done/mute feedback updates threshold, priority, cooldown, action policy |
| v0.7.1 | Broad words caused semantic misrouting. | Evidence Sensor | stronger topic-state mapping, forced priority table, primary and secondary states |
| v0.7.2 | The system could generate spikes but not close them. | Feedback Memory | stable spike_id, Markdown feedback section, done/downgraded/ignored/postponed closure parsing |
| v0.7.3 | Markdown feedback was lost after report overwrite. | Feedback Memory | persistent `lif_memory_feedback.json`, cooldown_until, durable downgraded topics |
| v0.7.4 | Rule-based semantics needed calibration. | Evidence Sensor | LLM reviewer as semantic sensor, not controller |
| Graph miner | Obsidian is a graph, not a folder of files. | Memory Source | wikilink, tag, folder, bridge-note, hub-note, top evidence note analysis |
| Insight integrator | Not all spikes are actions; some are thoughts. | Spike Interface | action pressure and explanatory tension are unified under LIF voltage |

## v1.0 convergence

v1.0 should be described as a single event-driven memory system:

```text
Original note -> EvidenceVector -> LIF voltage -> Spike card -> Closure feedback
```

This makes the earlier releases continuous:

- v0.2 created the evidence surface.
- v0.3 created the semantic current.
- v0.4 created the action decision.
- v0.5 created topic-level memory.
- v0.6 created fast/slow dynamics.
- v0.7 created human feedback memory.
- v0.7.4 added LLM semantic calibration.
- graph/insight modules expanded the same loop from action memory to knowledge graph and thought tension.

## The visible harvest

The visible result is not just code quantity. It is a working pattern:

```text
personal long-term notes can be modeled as an event-driven dynamical system
```

The project demonstrates that a personal knowledge system can be:

- traceable: every spike points back to evidence
- dynamic: state leaks, accumulates, resets, and cools down
- adaptive: human closure feedback changes future triggers
- bounded: LLM reviews semantics but does not own the control loop
- useful: every spike asks for one closeable result

## What should not happen before v1.0

Do not add another major subsystem before producing these four example outputs:

1. daily top-spike output
2. full replay output
3. graph report output
4. insight profile output

The next milestone is not more functionality. The next milestone is to prove the existing loop is visible, reusable, and explainable.

## Suggested README sentence

```text
LIF-Memory is an event-driven memory replay system for Obsidian: it turns long-term notes into EvidenceVector current, integrates that current in fast/slow LIF states, and emits traceable action or insight spikes with feedback-controlled thresholds and cooldowns.
```
