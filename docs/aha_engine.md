# AhaEngine / Belief-Update Layer

`aha_engine.py` adds the missing layer after LIF spike:

```text
evidence -> LIF voltage -> spike -> AhaEngine -> old_model / contradiction / new_model / action_delta / falsification_test
```

The design boundary is:

```text
Spike is not insight.
Spike only opens the gate for insight reconstruction.
```

## Why this exists

Existing LIF-Memory can accumulate pressure and trigger a spike. That is useful, but it mostly answers:

```text
Which issue needs attention now?
```

AhaEngine answers a different question:

```text
What old model failed, and what new model should replace it?
```

## Minimal workflow

Generate spikes:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 14 --json-output lif_spikes.json
```

Optionally generate reconstruction-loss evidence:

```powershell
python unsupervised_memory_field.py --vault "C:\path\to\vault" --days 14 --json-output unsupervised_field.json
```

Generate Aha cards:

```powershell
python aha_engine.py `
  --spikes lif_spikes.json `
  --reconstruction unsupervised_field.json `
  --query "为什么 LIF-Memory 没有人的灵光一闪？" `
  --output "LIF-Memory AhaCards.md" `
  --json-output lif_aha_cards.json
```

Run built-in demo:

```powershell
python aha_engine.py --demo
```

## Output schema

Each card contains:

```json
{
  "old_model": "What the system/user implicitly believed before.",
  "contradiction": "Evidence that makes the old model insufficient.",
  "new_model": "A compressed replacement explanation.",
  "essence": "One sentence essence.",
  "action_delta": "What changes in the next action.",
  "falsification_test": "How to prove this insight wrong."
}
```

## Valid Aha condition

A card is only useful if it has both:

```text
new_model
action_delta
```

Without `new_model`, it is only a summary.

Without `action_delta`, it is only beautiful language.

Without `falsification_test`, it is hard to distinguish insight from hallucination.
