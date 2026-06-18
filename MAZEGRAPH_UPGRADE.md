# MazeGraph-LIF Upgrade

This upgrade adds a graph layer in front of the existing LIF-Memory runner.

The problem it solves is:

```text
large Obsidian vault -> local chunks -> summaries that do not reconnect
```

The new path is:

```text
Raw Note -> Observation -> MazeGraph -> Graph LIF spike -> Daily task
```

## Why this exists

`lif_memory.py` is already useful as an action-pressure trigger. It should not become a full memory store. If LIF directly compresses a large vault, weak but important path relations can disappear.

`maze_graph_lif.py` keeps those relations as graph nodes and edges before applying LIF-style scheduling.

## What the new script does

`maze_graph_lif.py`:

1. Scans Markdown notes in an Obsidian vault.
2. Converts each note into an observation card.
3. Extracts concepts, wikilinks, tags, action signals, blocker signals, and completion signals.
4. Builds a lightweight graph:
   - note -> concept edges
   - concept -> concept co-occurrence edges
   - note -> wikilink edges
5. Computes graph-node voltage.
6. Emits top graph spikes and weak-link/frontier nodes.
7. Writes a Markdown report and optional JSON graph.

## Run

From the vault root:

```powershell
python "04 项目库\P2_LIF-Memory\maze_graph_lif.py" --vault "." --days 30 --output "MazeGraph-LIF 报告.md" --json-output "maze_graph_lif.json"
```

If the script is outside the vault:

```powershell
python maze_graph_lif.py --vault "C:\path\to\vault" --days 30 --output "MazeGraph-LIF 报告.md" --json-output "maze_graph_lif.json"
```

Scan more notes:

```powershell
python maze_graph_lif.py --vault "C:\path\to\vault" --days 0 --max-notes 1000 --top-k 12
```

Preview only:

```powershell
python maze_graph_lif.py --vault "C:\path\to\vault" --days 30 --dry-run
```

## Output

The Markdown report contains:

```text
Top Graph Spikes
Spike Cards
Frontier / Weak Links
Concept Map Top Nodes
```

The JSON output contains:

```json
{
  "version": "0.8.0",
  "stats": {},
  "observations": [],
  "nodes": [],
  "edges": [],
  "spikes": [],
  "frontier": []
}
```

## How it connects to the old runner

Use this order:

```text
maze_graph_lif.py -> understand the map
lif_memory.py --mode daily -> pick one executable card
```

The separation is deliberate:

```text
MazeGraph = memory topology / path relation
LIF-Memory = attention trigger / action scheduler
LLM reviewer = semantic sensor
```

## Current limitations

This is still a deterministic rule-based graph layer. It does not use embeddings or a database yet.

The next upgrade should be:

```text
concept_id / claim_id / task_id persistence
incremental graph update
LLM observation merger
graph-aware daily task injection
```

## Suggested next version

`v0.8.1` should make `lif_memory.py` optionally consume the JSON output from `maze_graph_lif.py`:

```powershell
python lif_memory.py --vault "." --mode daily --graph-input "maze_graph_lif.json"
```

The daily card can then include:

```text
Graph neighbors
Weak links
Evidence sources
One suggested action
```
