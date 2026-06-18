# MazeGraph-LIF v0.8

## 这次升级解决什么问题

原来的 LIF-Memory 已经能把 Obsidian 日记和项目笔记映射成状态电位，并在阈值越过时生成 spike。但它仍然容易遇到一个问题：如果只按窗口或局部片段输入，笔记之间的路径关系会被切断；如果直接用 LIF 压缩，低于阈值的信息又会消失。

v0.8 新增 `maze_graph_builder.py`，把系统拆成两层：

```text
MazeGraph = 保存等效迷宫，包括观察卡、概念、判断、边和前沿
LIF       = 只负责在 MazeGraph 之上触发注意力和任务
```

也就是：

```text
Raw Note -> Observation -> Concept/Claim/Edge -> MazeGraph -> LIF attention -> Task
```

## 新增文件

- `maze_graph_builder.py`

它会扫描 Obsidian vault，并生成：

1. Markdown 总报告：`MazeGraph-LIF 等效迷宫图.md`
2. JSON 图谱：`maze_graph.json`
3. 可选的逐篇观察卡：`AI_Compression/observations/*.md`

## 推荐命令

在项目目录执行：

```powershell
python .\maze_graph_builder.py --vault "..\.." --goal thesis --output "AI_Compression\MazeGraph-LIF 等效迷宫图.md" --json-output "AI_Compression\maze_graph.json" --observations-dir "AI_Compression\observations"
```

然后继续运行原来的 daily spike：

```powershell
python .\lif_memory.py --mode daily --llm-review --top-k 1
```

## 输出结构

JSON 中包含三类核心对象：

```json
{
  "observations": [
    {
      "note_id": "note::<stable_id>",
      "path": "relative/path.md",
      "summary": "一句话观察",
      "concepts": [],
      "claims": [],
      "actions": [],
      "state_scores": {},
      "blockers": 0,
      "completions": 0
    }
  ],
  "edges": [
    {
      "source": "note::<id>",
      "target": "topic::论文闭环",
      "relation": "mentions",
      "weight": 0.61,
      "evidence": "论文, 证据"
    }
  ],
  "concepts": {
    "topic::论文闭环": {
      "lif_tension": 8.4,
      "note_count": 12,
      "state_scores": {
        "Thesis": 31.0,
        "Experiment": 18.2
      }
    }
  }
}
```

## 和旧模块的关系

- `lif_memory.py`：状态电位、阈值、spike、行动策略。
- `obsidian_graph_miner.py`：基础 wikilink / tag / 文件夹 / LIF 状态图谱报告。
- `maze_graph_builder.py`：把笔记变成可被智能体探索的等效迷宫图。

## 这次的设计边界

`maze_graph_builder.py` 不直接调用 LLM，也不删除或覆盖原始笔记。它只做确定性解析和结构化输出，便于后续接入智能体或 LLM 审查。

## 下一步可以继续升级

1. 让 `lif_memory.py` 在 spike 后读取 `maze_graph.json`，把 linked evidence / backlink evidence 加入 spike packet。
2. 加入 `graph_current = Σ neighbor_evidence * edge_weight * graph_decay`。
3. 增加 `frontier mode`，让智能体优先探索孤立但高张力的笔记。
4. 增加 claim lifecycle，记录每个判断是被支持、反驳、降级还是关闭。
