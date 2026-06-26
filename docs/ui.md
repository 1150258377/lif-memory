# LIF-Memory Local Debug UI

`lif_memory_ui.py` is a local browser UI for debugging the existing LIF-Memory command line tools.

It is intentionally separate from the core scripts. It does not change the behavior of `lif_memory.py`, `insight_integrator.py`, `obsidian_graph_miner.py`, or `llm_adapter.py`.

## Why this UI exists

The project already had multiple useful CLI entry points, but it was hard to see the system as one visible tool. The UI adds a thin visual layer:

```text
browser buttons -> existing CLI scripts -> Markdown / JSON outputs -> preview in browser
```

The goal is better debugging, not a new runtime architecture.

## Start

From the project directory:

```powershell
python lif_memory_ui.py
```

Then open:

```text
http://127.0.0.1:8765
```

Use a custom vault path:

```powershell
python lif_memory_ui.py --vault "C:\path\to\your\ObsidianVault"
```

Use a different port:

```powershell
python lif_memory_ui.py --port 8777
```

## Buttons

The UI currently exposes these buttons:

- `预览今日主卡`: runs `lif_memory.py --mode daily --dry-run`
- `生成今日主卡`: runs `lif_memory.py --mode daily --output ...`
- `生成完整回放`: runs `lif_memory.py --output ... --json-output ...`
- `生成图谱报告`: runs `obsidian_graph_miner.py --output ... --json-output ...`
- `生成洞察报告`: runs `insight_integrator.py --profile ... --output ... --json-output ...`
- `生成 v1 收束报告`: runs `lif_memory_v1_report.py --output ... --json-output ...`
- `一键生成全部报告`: runs the main report commands in sequence

## Debug options

The UI exposes common debugging parameters:

- Obsidian vault path
- replay days
- daily top-k
- completion scan checkbox
- LLM review checkbox
- LLM provider
- closure feedback file
- output Markdown / JSON paths
- insight profile and sensitivity

## Design boundary

This UI is deliberately simple:

- no external web framework
- no database
- no background worker
- no modification to core state logic
- no hidden command construction through shell strings

Commands are run through `subprocess.run([...], shell=False)`. The UI only selects from predefined actions.

## Suggested local workflow

1. Start the UI.
2. Click `预览今日主卡` to see whether the top spike is reasonable.
3. If it looks useful, click `生成今日主卡`.
4. Edit the `Spike 反馈区` in the generated Markdown.
5. Run again with the closure file set to the edited Markdown.
6. Use `生成完整回放`, `生成图谱报告`, and `生成洞察报告` when preparing a weekly review or project demonstration.

## Relation to v1.0 convergence

The UI makes the v1.0 loop visible:

```text
Memory Source -> Evidence Sensor -> LIF Dynamics -> Spike Interface -> Feedback Memory
```

It does not replace the loop. It lets the user operate the loop with buttons and inspect the resulting Markdown/JSON outputs immediately.
