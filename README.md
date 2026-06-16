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

## What was optimized in v0.3.0

The input current is no longer just a raw keyword score.

Each note fragment is now mapped into an `EvidenceVector` before it charges a state:

```text
fragment -> evidence vector -> state input current -> LIF voltage
```

The vector contains:

```text
target_weight   how strongly this fragment belongs to the state
actionability   whether the fragment implies a possible next action
urgency         whether it has time pressure
blocker         whether it describes a blockage or conflict
completion      whether it looks completed and should inhibit repeated spikes
specificity     whether it contains concrete values, devices, metrics, or links
novelty         whether it introduces a new claim or breakthrough
confidence      confidence from keyword/context support
```

This is still a deterministic rule-based vector layer, not a neural embedding model. It is intentionally designed so it can later be replaced by a real embedding/classifier layer without changing the LIF state update.

The first practical gain is disambiguation. For example, `恢复` in `后向散射恢复/波形恢复` charges `Experiment`, while `身体恢复/情绪恢复` charges `Health`.

## What was optimized in v0.4.0

Spike packets now include an action-decision layer. The system no longer only asks:

```text
which state crossed threshold?
```

It also asks:

```text
should this be continued, isolated, downgraded, or handled after recovery?
```

Each spike packet now includes:

```text
topic
priority          P0 | P1 | P2
blocker_type      none | repeated_failure | unclear_definition | emotional_overload
action_policy     continue | isolate | downgrade | recover_first
completion_target one small result that can be judged complete
```

The first loop detector is intentionally simple: if the same topic appears across multiple days with blocker signals and few completion signals, the spike is routed to isolation instead of generic continuation. For example, repeated `负阻` failures become `blocker_type=repeated_failure` and `action_policy=isolate`.

## What was optimized in v0.5.0

The stateful runner now persists topic history in `lif_state.json`.

This means incremental runs no longer remember only neuron voltage. They also remember topic-level loop evidence:

```text
topic -> days_seen / completion_count / blocker_count / evidence_count / last_action_policy
```

This is the first step from one-window replay toward long-running feedback memory. A repeated blocker such as `负阻` can continue to be routed as an isolation problem even when only the newest daily note is processed.

## What was optimized in v0.6.0

Each state now has two voltage traces:

```text
V_fast  recent acute pressure
V_slow  long-running background pressure
V       weighted combination used for threshold crossing
```

The update is:

```text
V_fast = max(0, V_fast_old * fast_decay^delta_days + evidence_input - completion_inhibition)
V_slow = max(0, V_slow_old * slow_decay^delta_days + evidence_input * slow_input_ratio - completion_inhibition * slow_completion_ratio)
V = weighted_sum(V_fast, V_slow)
```

Different states can now remember at different speeds. Experiments leak faster, career and AI-memory pressure leak more slowly, and health pressure rises quickly but is more strongly reduced by recovery/completion signals.

## What was optimized in v0.7.0

Replay can now consume manual feedback and turn it into topic policies.

Supported feedback labels:

```text
有用 / 没用 / 太早 / 太晚 / 已完成 / 不要再提醒 / 升为P0 / 降为P2
```

Feedback is stored as a topic-level policy:

```text
topic -> threshold_delta / priority_override / action_policy_override / muted / cooldown_days
```

Example `lif_feedback.json`:

```json
{
  "feedback": [
    {
      "topic": "负阻",
      "feedback": "不要再提醒"
    },
    {
      "topic": "论文闭环",
      "feedback": "太晚"
    }
  ]
}
```

Run with feedback:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --feedback-file "lif_feedback.json" --output "LIF-Memory 回放结果.md"
```

In the stateful runner, feedback policies are persisted in `lif_state.json` under `topic_policies`.

## What was optimized in v0.7.1

This is a calibration release for the semantic interface.

It fixes the main failure mode where experimental data fragments could be mislabeled as career/job topics because of broad words such as `工作`.

Implemented in v0.7.1:

```text
stronger topic/state mapping
forced priority table for mainline topics
primary_state / secondary_states in spike packets
optional local completion-signal scan
```

Priority policy:

```text
论文闭环 / LIF链路 / 实验数据模板 / Health -> P0
负阻 -> P1 by default, isolate if repeated failure
AI记忆 -> P2 unless it is AI求职转向
AI求职转向 -> Career primary, AI_Memory secondary
```

Optional completion scan:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --completion-scan
```

## What was optimized in v0.7.2

This is the completion-loop release. The system can now close spikes instead of only generating them.

Implemented in v0.7.2:

```text
topic-specific completion targets
stable spike_id for every spike
Markdown Spike feedback section
done / downgraded / ignored / postponed closure parsing
closure-derived cooldown and priority/action policy updates
daily top-spike rendering
```

Default replay output now appends:

```markdown
## Spike 反馈区

- [ ] 2026-06-15-Experiment-负阻
  - Topic：负阻
  - Primary：Experiment
  - Policy：isolate
  - 状态：open
  - 反馈：
  - 完成证据：
  - 关闭时间：
```

To close a spike, edit it manually:

```markdown
- [x] 2026-06-15-Experiment-负阻
  - Topic：负阻
  - Primary：Experiment
  - Policy：isolate
  - 状态：downgraded
  - 反馈：正确
  - 完成证据：负阻暂不作为论文主线，仅作为补充模块。
  - 关闭时间：2026-06-15
```

The next run reads the previous output file by default before overwriting it. A downgraded topic gets lower priority and cooldown.

Daily mode renders only the top action card:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --top-k 1
```

## What was optimized in v0.7.3

This is the persistent feedback-memory release.

Markdown remains the human editing surface, but closed spikes are now persisted to JSON so feedback survives report overwrites.

Default memory path:

```text
04 项目库/P2_LIF-Memory/lif_memory_feedback.json
```

Implemented in v0.7.3:

```text
read persistent feedback memory before replay
merge persistent memory with JSON feedback and Markdown closures
write closed Markdown closures back into persistent JSON
store cooldown_until / completion_evidence / last_feedback per topic
expire cooldown penalties after cooldown_until
preserve durable downgraded topics as P2 / downgrade
```

Typical daily loop:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --top-k 1 --output "今日 LIF-Memory 主卡片.md"
```

After editing the `Spike 反馈区`, run again:

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --top-k 1 --closure-file "今日 LIF-Memory 主卡片.md" --output "今日 LIF-Memory 主卡片.md"
```

The closure is copied into `lif_memory_feedback.json`, so future reports can be regenerated without losing the fact that a topic was done, ignored, postponed, or downgraded.

## What was optimized in v0.7.4

This is the LLM Reviewer Adapter release.

The LLM is only a semantic sensor. It reviews generated spikes and suggests corrections, but it never updates voltage, threshold, cooldown, priority, or action policy by itself.

Implemented in v0.7.4:

```text
llm_adapter.py
--llm-review
OpenAI-compatible chat.completions adapter
Qwen/DashScope default provider
DeepSeek / Kimi / GLM provider presets
LLM Review report section
```

Provider environment variables:

```text
qwen      -> DASHSCOPE_API_KEY
deepseek  -> DEEPSEEK_API_KEY
kimi      -> MOONSHOT_API_KEY
zhipu     -> ZHIPUAI_API_KEY
```

You can also store local keys in the ignored file:

```text
04 项目库/P2_LIF-Memory/config/llm.local.json
```

Example:

```json
{
  "provider": "deepseek",
  "api_keys": {
    "deepseek": "your-key"
  }
}
```

This file matches `.gitignore` via `*.local.json`.

Run daily mode with Qwen review:

```powershell
$env:DASHSCOPE_API_KEY="your-key"
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --top-k 1 --llm-review
```

Switch provider:

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --top-k 1 --llm-review --llm-provider deepseek
```

## Obsidian graph mining

Obsidian is not just a folder of notes. It is a memory graph:

```text
daily notes     time entry points
wikilinks       explicit memory edges
folders         project/domain edges
tags            manual classification edges
LIF states      implicit semantic/action edges
```

Run the graph miner from the vault root:

```powershell
python "04 项目库\P2_LIF-Memory\obsidian_graph_miner.py" --vault "." --output "Obsidian-LIF 知识图谱报告.md"
```

Optional JSON graph summary:

```powershell
python "04 项目库\P2_LIF-Memory\obsidian_graph_miner.py" --vault "." --output "Obsidian-LIF 知识图谱报告.md" --json-output "obsidian_lif_graph.json"
```

The report shows:

```text
folder distribution
inbound/outbound link hubs
bridge notes
unresolved links
top evidence notes for each LIF state
how graph evidence should feed future LIF voltage
```

## Domain Insight Profiles

`insight_integrator.py` can also run domain-specific thought profiles. This uses the same LIF idea, but changes the meaning of voltage:

```text
action profile: V = unresolved action pressure
insight profile: V = unresolved explanatory tension
```

Run the economics profile:

```powershell
python "04 项目库\P2_LIF-Memory\insight_integrator.py" --vault "." --profile economics --days 90 --output "经济学 LIF 洞察.md" --json-output "economics_insights.json"
```

For sparse domains you have not focused on, use exploratory sensitivity:

```powershell
python "04 项目库\P2_LIF-Memory\insight_integrator.py" --vault "." --profile economics --sensitivity exploratory --days 90 --output "经济学 LIF 洞察.md" --json-output "economics_insights.json"
```

Economics latent questions:

```text
Macro_Cycle
Inflation_Rate
Incentive_System
Debt_Finance
Market_Psychology
```

Run only one economic question:

```powershell
python "04 项目库\P2_LIF-Memory\insight_integrator.py" --vault "." --profile economics --questions Debt_Finance --days 90 --dry-run
```

The output is not a task reminder. It is a thought card trigger: fragments accumulate until they justify writing a claim, contrast, mechanism table, or evidence card.

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

## What Is Voltage?

In this project, `V` is not a token count, an embedding dimension, or a compressed copy of the original note.

`V` means:

```text
accumulated actionable pressure / attention debt for one state
```

The original note preserves full information. Voltage only preserves a trend:

```text
repeated evidence -> charge up
time gap -> leak down
completion signal -> inhibit
threshold crossing -> emit spike
```

The current update rule is:

```text
V = weighted_sum(V_fast, V_slow)
```

Where:

```text
evidence_input          score from matched note fragments for this state
fast_decay^delta_days   leakage for acute pressure
slow_decay^delta_days   leakage for background pressure
completion_inhibition  suppression from completion words such as done/saved/submitted
theta                   spike threshold
```

Since v0.3.0, `evidence_input` is the sum of evidence-vector scores rather than raw keyword hits. Since v0.6.0, each spike packet reports both `V_fast` and `V_slow`.

Each spike packet includes a `voltage_model` block so the trigger can be audited.

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

Apply manual feedback:

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 14 --feedback-file "lif_feedback.json"
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
  "topic": "负阻",
  "time": "2026-06-12",
  "V": 7.95,
  "threshold": 7.5,
  "priority": "P1",
  "blocker_type": "repeated_failure",
  "action_policy": "isolate",
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
  "suggested_action": "Run one focused isolation test and decide whether this topic remains on the main path.",
  "completion_target": "形成一页负阻隔离结论：可用/不可用/暂不作为主线。"
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
