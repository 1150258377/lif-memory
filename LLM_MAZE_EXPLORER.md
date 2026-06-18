# LLM Maze Explorer

This is the deep-LLM exploration layer for LIF-Memory.

It implements the pipeline:

```text
Raw notes
-> searchable environment
-> multi-role LLM exploration
-> structured JSON per role
-> merged multi-perspective graph
-> LIF score
-> today spike
```

## Why this layer exists

The rule-based `maze_graph_lif.py` preserves graph topology, but it does not deeply interpret notes.

The new `llm_maze_explorer.py` lets different LLM roles explore the same Obsidian vault from different perspectives:

```text
Cartographer   -> map topics, bridges, hubs, orphans
Skeptic        -> find weak claims, missing evidence, contradictions
Executor       -> convert knowledge into tasks
Linker         -> repair disconnected local chunks
LIF Scheduler  -> vote on urgency, blockers, actionability, evidence strength
```

The key idea is that LIF should not directly compress the vault. LIF should only select from the merged multi-perspective graph.

## Run with Qwen / DashScope

```powershell
$env:DASHSCOPE_API_KEY="your-key"
python "04 项目库\P2_LIF-Memory\llm_maze_explorer.py" --vault "." --days 30 --focus "论文闭环" --output "LLM-MazeGraph 今日Spike.md" --json-output "llm_maze_graph.json"
```

## Run with DeepSeek

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python "04 项目库\P2_LIF-Memory\llm_maze_explorer.py" --vault "." --days 30 --focus "Obsidian知识库" --llm-provider deepseek --output "LLM-MazeGraph 今日Spike.md" --json-output "llm_maze_graph.json"
```

## More exploration

```powershell
python llm_maze_explorer.py --vault "C:\path\to\vault" --days 0 --max-notes 1000 --iterations 2 --docs-per-role 10 --focus "MazeGraph-LIF"
```

## Smoke test without LLM

```powershell
python llm_maze_explorer.py --vault "C:\path\to\vault" --days 30 --skip-llm --dry-run
```

This mode builds the search environment and uses deterministic fallback results. It is useful for checking file scanning and output paths.

## Output JSON shape

```json
{
  "version": "0.8.1",
  "nodes": [],
  "edges": [],
  "claims": [],
  "tasks": [],
  "tensions": [],
  "role_runs": [],
  "search_environment": {
    "doc_count": 0,
    "top_terms": []
  },
  "lif_spikes": []
}
```

## Role output schema

Each role is asked to return only one JSON object with:

```text
nodes
edges
claims
tasks
tensions
lif_votes
```

This keeps each LLM role independent while making all outputs mergeable.

## Recommended daily loop

```powershell
python llm_maze_explorer.py --vault "." --days 30 --focus "今天最该处理什么" --json-output "llm_maze_graph.json" --output "LLM-MazeGraph 今日Spike.md"
python lif_memory.py --vault "." --days 14 --mode daily --top-k 1 --llm-review --output "今日 LIF-Memory 主卡片.md"
```

For now, the two runners are separate. The next version should let `lif_memory.py` consume `llm_maze_graph.json` directly:

```powershell
python lif_memory.py --vault "." --mode daily --graph-input "llm_maze_graph.json"
```

## Boundary

```text
LLM roles = semantic explorers
Merged graph = multi-perspective memory topology
LIF score = today selection mechanism
Original Obsidian = full memory store
```
