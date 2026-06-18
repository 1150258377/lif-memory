# LIF-Memory v0.8：多视角知识迷宫探索器

`knowledge_maze_explorer.py` 把 Obsidian 知识库看成一个可探索的状态空间：笔记片段是房间，主题是节点，主题之间的共现和阻塞关系是边，不同 LLM/视角是不同的探索者，LIF 电压表示某个主题在当前视角下积累出的解释张力或行动压力。

这个版本不是替代 `lif_memory.py`、`insight_integrator.py`、`obsidian_graph_miner.py`，而是在它们旁边增加一个“多视角重建层”。

```text
Obsidian 笔记
  -> 规则探索 / 可选 LLM 审查
  -> 多视角等效迷宫
  -> 节点电压 V
  -> spike 建议
  -> Markdown 报告 + JSON packet
```

## 内置视角

| view | 作用 |
| --- | --- |
| `thesis_closure` | 还原论文主线、已有证据和缺失证据 |
| `experiment_auditor` | 审计实验可靠性、补测点和不稳定模块 |
| `theory_builder` | 抽取 LIF、事件驱动、后向散射、负阻之间的理论关系 |
| `action_blocker` | 找出行动停滞、焦虑、羞耻、延毕压力等阻塞点 |
| `career_transfer` | 把项目转化成 AI + 嵌入式 / 信号处理 / 硬件系统表达 |
| `reviewer` | 以审稿人视角攻击创新性、证据链和定义漏洞 |

## 最小运行

在 Obsidian 仓库根目录运行：

```powershell
python "04 项目库\P2_LIF-Memory\knowledge_maze_explorer.py" --vault "." --views all --steps 24 --output "LIF-Memory 知识迷宫探索报告.md" --json-output "lif_knowledge_maze.json"
```

只看论文、实验、审稿人三个视角：

```powershell
python "04 项目库\P2_LIF-Memory\knowledge_maze_explorer.py" --vault "." --views thesis_closure,experiment_auditor,reviewer --steps 32 --output "LIF-Memory 论文实验迷宫.md"
```

预览而不写文件：

```powershell
python "04 项目库\P2_LIF-Memory\knowledge_maze_explorer.py" --vault "." --views all --dry-run
```

## 启用 LLM 审查

这个脚本复用现有的 `llm_adapter.py`，所以配置方式与 `lif_memory.py --llm-review` 一致。

DeepSeek 示例：

```powershell
$env:DEEPSEEK_API_KEY="你的 key"
python "04 项目库\P2_LIF-Memory\knowledge_maze_explorer.py" --vault "." --views thesis_closure,experiment_auditor,reviewer --llm-review --llm-provider deepseek
```

Qwen 示例：

```powershell
$env:DASHSCOPE_API_KEY="你的 key"
python "04 项目库\P2_LIF-Memory\knowledge_maze_explorer.py" --vault "." --views all --llm-review --llm-provider qwen
```

LLM 在这里不是控制器，而是审查器：它只看已生成的节点、边、证据和 spike，输出 `is_coherent / strongest_node / main_missing_evidence / merge_hint / risk / next_action`。没有 API key 时，脚本仍然可以用规则探索模式运行。

## 输出解释

Markdown 报告包含：

```text
总览
全局等效迷宫
每个视角的节点电压表
Spike 建议
视角结论
关键路径
Top evidence
可选 LLM review
```

JSON 输出包含：

```json
{
  "version": "0.8.0",
  "global_maze": {
    "nodes": [],
    "edges": []
  },
  "view_reports": []
}
```

这个 JSON 可以继续交给后续模块使用，例如：

```text
lif_knowledge_maze.json
  -> multi_view_synthesizer.py
  -> lif_spike_engine.py
  -> 今日主卡 / 论文证据链 / 审稿人质疑清单
```

## 和原有模块的关系

| 模块 | 主要问题 |
| --- | --- |
| `lif_memory.py` | 今天哪个行动状态 spike？ |
| `insight_integrator.py` | 哪个长期问题积累出了洞察？ |
| `obsidian_graph_miner.py` | 知识库图谱中哪些笔记/链接最关键？ |
| `llm_report_reviewer.py` | 顶层报告是否能形成战略判断？ |
| `knowledge_maze_explorer.py` | 多个视角如何探索并还原知识库的等效迷宫？ |

## 设计原则

1. 不让多个 LLM 直接闲聊。先各自探索，再结构化输出。
2. 先规则可复现，后 LLM 审查。没有 API key 也能跑。
3. 所有结论都保留证据块路径和片段。
4. 主题节点有电压、阈值和 spike，而不是只给普通总结。
5. 每个视角都有自己的偏置：论文、实验、理论、行动、求职、审稿人看到的是不同地图。

## 当前限制

- v0.8.0 还不是完整的多智能体辩论系统。
- 主题词表仍是人工定义，后续可以接入 `adaptive_relation_spike.py` 生成的 schema。
- 当前边主要来自同一证据块中的主题共现，后续可以进一步利用 Obsidian backlinks 和图谱中心性。
- LLM 只做审查，不直接改写 LIF 电压和阈值。

## 建议下一版

v0.8.1 可以接入：

```text
knowledge_maze_explorer.py
  + adaptive_relation_spike.py 的自动 schema
  + obsidian_graph_miner.py 的链接中心性
  + 多视角 cross-review
  + final synthesizer
```

最终目标是形成：

```text
多视角探索 -> 交叉质询 -> 总控合成 -> LIF spike -> 可执行行动卡
```
