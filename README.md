# LIF-Memory

LIF-Memory 是一个面向 Obsidian 笔记的本地记忆场系统。它把分散的 Markdown 笔记转换成可追溯的语义场，让长期问题在 LIF 电压中持续积累，并在证据足够时触发 spike，提醒你重新处理真正重要的问题。

它不是普通的笔记摘要工具，也不是单纯的向量数据库封装。当前主线是：

```text
Obsidian 笔记
-> 证据片段抽取
-> 稀疏可解释语义信号
-> 可选 dense embedding
-> Obsidian 图扩散
-> 连续问题场
-> LIF 电压 / spike
-> LLM 回答或多智能体辩论
-> 用户反馈
-> 控制器参数更新
```

原始笔记仍然保存在 Obsidian 中。LIF-Memory 只保存派生出来的场状态、会话、反馈、embedding 缓存和调参结果。

## 已实现功能

| 模块 | 已经能做什么 |
| --- | --- |
| Obsidian 扫描 | 递归扫描 Markdown，保留来源路径、片段、关键词和证据分数，忽略 `.git`、`.obsidian`、缓存目录等。 |
| LIF 记忆回放 | 跟踪主题电压、泄漏、阈值、冷却、优先级、完成抑制、spike 触发和人工关闭反馈。 |
| 连续问题场 | 根据时间距离、稀疏语义、dense embedding、Obsidian 链接图和 LIF 压力重建一个 query-specific field。 |
| 网页调参器 | 提供浏览器界面，支持扫描笔记、提问、查看证据、历史会话、结论沉淀、参数调节、反馈学习。 |
| 历史记录 | 网页端会把对话保存到 `lif_sessions.json`，可以继续之前的问题链，而不是每次从零开始。 |
| 混合 embedding | 支持稀疏可解释向量 + dense embedding 混合评分，可接 OpenAI-compatible API 或本地 FlagEmbedding/BGE。 |
| 多 LIF 神经元 | 除主主题电压外，还维护语义密度、新颖性、冲突、行动压力、整合压力等控制器神经元。 |
| 反馈学习 | 将“有用/无用/太早/太晚/完成”等反馈转成轻量 reward，保守更新场参数和领域状态。 |
| LLM 适配 | 支持 Qwen/DashScope、DeepSeek、Kimi、GLM/Zhipu 以及自定义 OpenAI-compatible endpoint。 |
| 图谱挖掘 | 读取 Obsidian wikilink、文件夹、标签，分析 hub、bridge、未解析链接和状态证据。 |
| 洞察 profile | 支持经济学等领域 profile，把 voltage 解释为解释张力，而不只是任务压力。 |

## 快速运行网页端

在项目目录中运行：

```powershell
python lif_web_tuner.py --vault "C:\path\to\your\obsidian-vault" --llm-provider deepseek
```

默认端口：

```text
http://127.0.0.1:7860
```

如果项目就在 Obsidian vault 内，也可以：

```powershell
python lif_web_tuner.py --vault "." --llm-provider deepseek
```

网页端是当前最完整的入口，适合用来做连续语义场查询、历史对话、调参和反馈。

## 网页端具体能力

`lif_web_tuner.py` 目前包含这些实际功能：

- 扫描 vault：读取全库 Markdown，建立字段缓存。
- 提问：把用户问题投影到连续问题场中，返回证据和回答。
- 历史会话：把网页对话保存到 `lif_sessions.json`，可恢复旧会话。
- 结论沉淀：把高价值结论保存到 `lif_conclusions.json`。
- 参数面板：调节 threshold、time kernel、graph diffusion、dense weight、sparse weight 等。
- 证据面板：展示哪些笔记片段支撑了当前回答。
- 多智能体辩论：可以让不同视角先辩论，再综合输出。
- 反馈按钮：把用户判断转成 reward，更新本地领域状态。
- 神经元面板：显示当前多通道 LIF 神经元状态。

网页端会在 vault 中写入这些本地状态文件：

```text
lif_field_state.json       笔记向量、主题电压、扫描缓存
lif_field_params.json      网页调参参数
lif_sessions.json          网页历史会话
lif_conclusions.json       沉淀结论
lif_domain_state.json      概念、关系、神经元、反馈 reward
lif_embedding_cache.json   embedding 缓存
```

这些文件可能包含私人笔记派生信息，已经被 `.gitignore` 排除，不应该上传到 GitHub。

## 连续问题场

连续问题场的目标是把零散笔记变成一个可以查询、可以扩散、可以积累压力的语义空间。

每条笔记 observation 会被映射为：

```text
时间信号        这条笔记距离当前查询窗口有多近
稀疏语义        规则向量，便于解释为什么命中
dense 语义      embedding 相似度，用于捕捉换一种说法的同义关系
图结构信号      Obsidian wikilink / 文件关系的扩散结果
LIF 压力        长期未解决问题累积出来的 voltage
```

命令行入口：

```powershell
python continuous_problem_field.py --vault "C:\path\to\vault" --query "论文第四章实验为什么一直卡住" --all-notes --top-k 8
```

带 embedding 和图扩散的示例：

```powershell
python continuous_problem_field.py `
  --vault "C:\path\to\vault" `
  --query "AI 记忆系统下一步应该怎么做" `
  --embedding-mode auto `
  --dense-weight 0.45 `
  --sparse-weight 0.55 `
  --graph-steps 2 `
  --graph-alpha 0.35 `
  --json-output field.json
```

如果没有配置 embedding，系统会自动退回稀疏语义模式。

## 向量数据库路径 vs 联系场路径

这个项目目前更接近“联系场”，不是传统向量数据库。

| 路径 | 核心机制 | 优势 | 局限 |
| --- | --- | --- | --- |
| 真实向量数据库 | 文本 chunk -> embedding -> ANN 检索 | 语义召回强，适合大规模相似搜索 | 解释性弱，通常只回答“像不像”，不负责长期电压和反馈状态 |
| LIF 联系场 | 笔记 -> 稀疏语义 + dense 语义 + 图扩散 + LIF 电压 | 能解释来源，能保留长期问题压力，能结合反馈和神经元状态 | 不是高性能向量库，embedding 部分仍需要模型质量支撑 |

当前实现采用混合方式：

```text
hybrid_similarity =
  dense_similarity * dense_weight
  + sparse_similarity * sparse_weight
```

`dense_similarity` 负责语义召回，`sparse_similarity` 负责可解释性。LIF 电压和图扩散负责把“单次相似”提升为“长期问题场”。

## 本地 FlagEmbedding / BGE 接入

`lif_field_learning.py` 支持两种 dense embedding 来源：

```text
api    OpenAI-compatible embedding API
flag   本地 FlagEmbedding / sentence-transformers 风格模型
```

复制本地配置模板：

```powershell
Copy-Item config\embedding.local.example.json config\embedding.local.json
notepad config\embedding.local.json
```

环境变量示例：

```powershell
$env:LIF_EMBEDDING_PROVIDER="flag"
$env:LIF_FLAGEMBEDDING_SOURCE="C:\path\to\FlagEmbedding-master"
$env:LIF_EMBEDDING_MODEL_PATH="C:\path\to\your\bge-model"
$env:LIF_EMBEDDING_DEVICE="cpu"
```

`LIF_EMBEDDING_MODEL_PATH` 应该指向真正的模型目录，例如 `bge-small-zh-v1.5`、`bge-base-zh-v1.5` 或 `bge-m3`。

典型模型目录应包含：

```text
config.json
tokenizer.json
tokenizer_config.json
model.safetensors 或 pytorch_model.bin
```

`config/embedding.local.json` 是本地私有配置，不应提交。

## 多 LIF 神经元

经典 LIF 回放默认跟踪这些主状态：

```text
Experiment
Thesis
Career
AI_Memory
Health
```

网页端新增了一组领域控制器神经元，保存在 `lif_domain_state.json`：

```text
semantic_density   当前问题场的证据密度
novelty            新组合、新信号、新方向
conflict           冲突、阻塞、反复失败
action_pressure    需要实验、写作、整理、决策的压力
integration        需要整合成洞察卡或稳定结论的压力
```

这些还不是完整的 spiking neural network。它们是 LIF 风格的调节神经元，用来观察语义场是否变得更密、更冲突、更行动导向，或者更接近可整合状态。

## 反馈学习

反馈学习目前是轻量 controller learning，不是模型微调。

已经实现：

- 记录用户反馈。
- 把反馈转成 reward scalar。
- 更新 `lif_domain_state.json` 中的反馈历史、概念状态、神经元状态。
- 保守调整 dense/sparse 权重、阈值摩擦等场参数。

尚未实现：

- 没有 fine-tune embedding 模型。
- 没有训练端到端神经网络。
- 没有做真正梯度下降式强化学习。
- 没有把用户私有笔记上传到远端训练服务。

当前闭环是：

```text
用户反馈 -> reward -> 控制器参数更新 -> 下一次问题场重建
```

## 经典 LIF 记忆回放

原始 Markdown 报告模式仍然可用：

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 14 --output "LIF-Memory 回放结果.md"
```

每日主卡片模式：

```powershell
python lif_memory.py --vault "C:\path\to\vault" --days 14 --mode daily --top-k 1 --output "今日 LIF-Memory 主卡片.md"
```

持久状态模式：

```powershell
python lif_memory_stateful.py --vault "C:\path\to\vault" --days 14 --state-file lif_state.json
```

关闭 spike 的方式是在 Markdown 里勾选：

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

下一次运行会读取 closure，并更新 cooldown、topic policy 和 persistent feedback memory。

## LLM Provider

LLM 层负责语义解释、回答、review、debate。它不直接控制 LIF 电压和阈值。

内置 provider：

```text
qwen      DASHSCOPE_API_KEY
deepseek  DEEPSEEK_API_KEY
kimi      MOONSHOT_API_KEY
zhipu     ZHIPUAI_API_KEY
```

示例：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
python lif_web_tuner.py --vault "C:\path\to\vault" --llm-provider deepseek
```

也可以放在本地私有配置：

```text
config/llm.local.json
```

该文件被 `.gitignore` 排除。

## 文件结构

| 文件 | 作用 |
| --- | --- |
| `lif_web_tuner.py` | 网页端入口：扫描、查询、回答、历史记录、结论、反馈、参数调节。 |
| `lif_field_learning.py` | embedding 配置/cache、本地 FlagEmbedding、领域状态、多 LIF 神经元、reward update。 |
| `continuous_problem_field.py` | 连续问题场 CLI：稀疏/dense 语义、图扩散、LIF scoring。 |
| `lif_memory.py` | 核心 LIF 回放引擎。 |
| `lif_memory_stateful.py` | 带持久 voltage/topic state 的回放入口。 |
| `llm_adapter.py` | OpenAI-compatible LLM review adapter。 |
| `obsidian_graph_miner.py` | Obsidian wikilink、folder、tag 图谱挖掘。 |
| `insight_integrator.py` | 领域洞察 profile，例如 economics。 |
| `knowledge_maze_explorer.py` | 知识迷宫式路径探索。 |
| `memory_field.py` / `unsupervised_memory_field.py` | 实验性记忆场组件。 |
| `docs/` | 架构、relation spike、连续问题场、后续路线文档。 |

## 典型工作流

网页端交互：

```text
扫描 vault -> 提问 -> 查看证据 -> 调 dense/sparse 权重 -> 反馈 -> 下次从历史会话继续
```

连续场报告：

```text
query -> 相关笔记 -> field score -> Markdown / JSON 输出
```

每日 LIF 注意力管理：

```text
最近笔记 -> state voltage -> top spike -> closure 反馈
```

图谱结构分析：

```powershell
python obsidian_graph_miner.py --vault "C:\path\to\vault" --output "Obsidian-LIF 知识图谱报告.md"
```

领域洞察：

```powershell
python insight_integrator.py --vault "C:\path\to\vault" --profile economics --days 90 --output "经济学 LIF 洞察.md"
```

## 隐私边界

不要提交这些文件：

```text
config/*.local.json
lif_field_state.json
lif_field_params.json
lif_sessions.json
lif_conclusions.json
lif_domain_state.json
lif_embedding_cache.json
lif_state.json
lif_memory_feedback.json
```

它们可能包含私人笔记派生信息、本地模型路径、会话历史、反馈历史或 API 配置。

## 当前不是哪些东西

LIF-Memory 目前还不是：

- 生产级向量数据库服务；
- 完整训练好的神经记忆模型；
- 端到端可微分强化学习系统；
- Obsidian 原始笔记的替代品；
- 保证客观正确的自动决策系统。

它目前是一个本地、可审计的记忆场原型：用 LIF 状态负责触发，用 embedding 提高召回，用图扩散恢复联系，用 LLM 改善解释，用人工反馈更新控制器。

## 设计方向

长期方向是：

```text
零散笔记
-> 连续语义场
-> 概念 / 关系 / 调节神经元
-> 反馈塑形的 controller learning
-> 更高质量的洞察 spike 和行动 spike
```

核心边界是：

```text
LIF 负责触发。
Embedding 负责语义召回。
图扩散负责恢复联系。
LLM 负责解释和辩论。
用户反馈负责调节控制器。
原始笔记永远是 source of truth。
```
