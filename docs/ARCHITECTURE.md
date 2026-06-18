# LIF-Memory Architecture

LIF-Memory is not a diary summarizer and it is not a vector database wrapper. It is an event-driven cognitive replay system.

The core question is not:

```text
What did I write recently?
```

The core question is:

```text
Which long-running state is accumulating enough unresolved pressure or explanatory tension to justify a spike?
```

## 1. Current pipeline

```text
Obsidian notes
  -> evidence fragments
  -> EvidenceVector
  -> state input current
  -> LIF state voltage
  -> threshold crossing
  -> spike packet
  -> LLM semantic review
  -> manual feedback / closure memory
```

The original notes remain the source of full information. LIF-Memory only preserves a low-dimensional trend: what is charging, what is leaking, what has been completed, and what should be reopened.

## 2. The key design point

A spike is not the final answer.

A spike is an interrupt signal that says:

```text
This state has accumulated enough evidence. Reopen it, diagnose it, and convert it into one small experiment or action.
```

This makes LIF-Memory different from ordinary AI note tools:

```text
ordinary note tool: notes -> summary
LIF-Memory: notes -> state voltage -> spike -> evidence recall -> minimal experiment
```

## 3. Deterministic layer vs semantic layer

The deterministic layer owns the control loop:

```text
voltage
threshold
leakage
cooldown
priority
policy
closure memory
```

The LLM layer is only a semantic sensor. It may review whether the spike topic, primary state, secondary states, or completion target fit the evidence. It must not directly change voltage, threshold, cooldown, or final action policy.

This boundary keeps the system auditable. The LIF layer decides when a spike fires; the LLM layer helps explain whether the fired spike was named and framed well.

## 4. Profile layer

The next architectural layer is a small read-only profile layer:

```text
profile context
  -> semantic prior for spike diagnosis
  -> better framing of trigger reason and minimal experiment
```

The profile layer should not be a private diary dump. It should be a compact, durable context file that describes the long-running situation, constraints, and direction of the system.

It does not charge voltage. It does not lower thresholds. It does not force a topic to fire.

It only helps the semantic layer understand why a spike matters.

Recommended structure:

```text
profiles/profile.example.md
```

## 5. Profile-aware pipeline

```text
Obsidian notes
  -> evidence fragments
  -> EvidenceVector
  -> LIF state update
  -> spike packet
  -> profile-aware semantic review
  -> diagnostic card
  -> minimal experiment
  -> manual closure feedback
```

In this pipeline, the profile is inserted after the spike is generated, not before the spike is generated.

That distinction matters:

```text
Bad: profile biases every voltage update.
Good: profile helps explain already-fired spikes.
```

## 6. Spike diagnostic schema

A profile-aware spike should be diagnosed with the following schema:

```text
trigger_event        What directly caused the spike this time?
historical_echo      What repeated pattern does it resemble?
deep_tension         What unresolved tension is being exposed?
current_hypothesis   What is the most likely mechanism?
minimal_experiment   What is the smallest test/action for the next loop?
observe_metric       What should be checked after the experiment?
```

This turns the system from a reminder engine into a cognitive experiment engine.

## 7. State space

The current public prototype tracks five main action states:

```text
Experiment
Thesis
Career
AI_Memory
Health
```

The architecture can later expand to more states, such as:

```text
Relation
Skill
Money
Meaning
```

These should be added only when they produce better spike decisions. A new state is justified when it has stable input evidence, a meaningful threshold, and a distinct completion target.

## 8. Why profile should stay small

A useful profile is not long. It should answer only three questions:

```text
What is the durable context?
What constraints should the reviewer remember?
What direction should the system optimize for?
```

If the profile becomes too detailed, it will turn into another note dump and weaken the spike mechanism.

## 9. v0.8 direction

The clean next version is:

```text
v0.8 Profile-Aware Spike Diagnosis
```

Planned changes:

```text
1. Add profile example file.
2. Add optional --profile-file argument.
3. Load compact profile context only for LLM review.
4. Keep deterministic voltage update unchanged.
5. Add diagnostic fields to the LLM review schema.
6. Render a short Spike Diagnosis section in Markdown output.
```

The main design rule is:

```text
LIF controls triggering. Profile improves interpretation. LLM improves language. Feedback closes the loop.
```
