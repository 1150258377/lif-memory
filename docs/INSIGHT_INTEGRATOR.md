# LIF Insight Integrator

This module upgrades LIF-Memory from action reminders to insight integration.

## Core idea

A keyword reminder fires when one phrase appears.

The insight integrator fires only when multiple weak fragments accumulate around the same latent question.

```text
weak fragments -> leaky integration -> threshold crossing -> insight spike
```

## What is insight voltage?

In `insight_integrator.py`, `V` is not raw note volume.

`V` means:

```text
accumulated pressure around one latent question
```

It rises when multiple fragments point to the same hidden question, especially when they contain tension, blockers, evidence, or repeated attempts to define something. It leaks with time and can be inhibited by fragments that look like closure or completion.

So the voltage does not preserve the full note. It only asks:

```text
Has enough distributed evidence accumulated that this latent question should become an explicit insight?
```

## Why this matters

The useful role of LIF is not just triggering.

Its useful role is integration:

- weak evidence can accumulate;
- irrelevant evidence can leak away;
- conflicting evidence can increase pressure;
- completion evidence can inhibit repeated spikes;
- threshold crossing marks the moment when scattered fragments become a useful judgment.

## Example latent question

```text
Innovation_Claim
```

Fragments may include:

```text
SSVEP backscatter recovery works.
LIF event rate preserves a rhythm.
The comparator threshold exposes the real EEG input bottleneck.
The thesis innovation may look like simple stitching.
```

The insight spike should not say only: do an experiment.

It should say something closer to:

```text
The innovation is not simply LIF plus backscatter.
The claim is that EEG can be eventized first, then carried by backscatter while preserving task-relevant rhythm evidence.
```

## Run

```powershell
python insight_integrator.py --days 14 --dry-run
```

Run one latent question:

```powershell
python insight_integrator.py --questions Innovation_Claim --days 14 --dry-run
```

Write JSON packets:

```powershell
python insight_integrator.py --days 14 --json-output lif_insights.json
```

## Tests

```powershell
python -m unittest discover -s tests
```
