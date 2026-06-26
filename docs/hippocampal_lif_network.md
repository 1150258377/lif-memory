# Hippocampal LIF Network

这个模块把 LIF-Memory 从“单个 LIF 触发器”升级成一个小型工程海马体。

核心判断：

> 涌现不是来自一个触发器，而是来自多个记忆单元之间的相互激活、竞争、强化、遗忘和反馈。

## 层级结构

```text
Obsidian notes / user query
        ↓
EC 输入映射
        ↓
DG 稀疏分离层
        ↓
CA3 递归联想 LIF 网络
        ↓
CA1 读写判断
        ↓
Cortex 长期记忆痕迹
```

## 各层作用

| 层 | 工程含义 | 解决的问题 |
|---|---|---|
| EC | 把文本 observation 转成稀疏语义输入 | 输入接口 |
| DG | 产生低重叠 sparse code | 相似但不同的问题不要混成一个 |
| CA3 | 递归 LIF + Hebbian 权重 | 共同出现的 spike 形成联想簇 |
| CA1 | 判断新输入应更新旧 trace 还是写新 trace | 读写门控，防止错误合并 |
| Cortex | 保存稳定 trace、标签、证据和 signature | 长期记忆 |

## 交互方式

当前最合理的交互链路是：

```text
用户连续对话 / Obsidian 日志
        ↓
extract_observations()
        ↓
HippocampalLIFNetwork.observe()
        ↓
DG sparse code
        ↓
CA3 attractor completion
        ↓
CA1 write/update decision
        ↓
CorticalTrace
```

查询/回忆时：

```text
partial cue
        ↓
HippocampalLIFNetwork.probe()
        ↓
DG 部分激活
        ↓
CA3 递归补全
        ↓
匹配 Cortex trace
        ↓
返回 recalled_terms + evidence
```

## 网页端串联

`hippocampal_bridge.py` 把海马体输出压缩成统一的 `HippocampalRecall`：

```text
trace_id
similarity
completed
completion_gain
recalled_terms
recalled_evidence
current_boost
```

网页端 `lif_web_tuner.py` 会把这个 recall 接入原来的连续问题场流程：

```text
Obsidian notes
-> continuous_problem_field.reconstruct_field()
-> hippocampal_bridge.build_recall_from_vault()
-> inject_recall_current()
-> continuous_problem_field.lif_iterate()
-> AhaEngine.build_aha_cards()
```

这里的海马体不是直接回答问题，而是提供三类信号：

1. **召回信号**：当前 query 唤起哪个 cortex trace。
2. **补全信号**：CA3 是否从 partial cue 补出额外关联。
3. **调制信号**：`current_boost` 加到当天 LIF current，让相关旧记忆能提高 spike 概率。

运行网页端：

```powershell
python lif_web_tuner.py
```

打开：

```text
http://127.0.0.1:7860
```

网页端会显示五个输出页：

| Tab | 内容 |
|---|---|
| Summary | 本次扫描数量、field energy、V/theta、海马体 boost |
| Hippocampus | trace、similarity、completion_gain、召回证据 |
| Field | 连续问题场报告，并附加 hippocampal evidence |
| Aha | AhaEngine 输出 old_model / contradiction / new_model |
| JSON | 完整串联 packet，便于继续调试 |

## 快速运行

内置 demo：

```powershell
python hippocampal_lif_memory.py
```

指定 vault：

```powershell
python hippocampal_lif_memory.py `
  --vault "C:\path\to\your\obsidian-vault" `
  --days 14 `
  --probe "老师又问 LIF 后向散射创新点怎么证明" `
  --output "hippocampal_lif_report.md" `
  --json-output "hippocampal_lif_report.json"
```

运行测试：

```powershell
python -m unittest tests/test_hippocampal_lif_memory.py
python -m unittest tests/test_hippocampal_bridge.py
```

## 测试覆盖的海马体功能

`tests/test_hippocampal_lif_memory.py` 覆盖四个必要能力：

1. **DG 稀疏分离**  
   相似输入会被映射成低重叠 sparse code，避免全部写进同一个记忆。

2. **CA3 递归联想 / 部分线索补全**  
   训练后，只给一个 partial cue，例如“老师又问 LIF 后向散射创新点怎么证明”，系统可以回忆到相关 trace。

3. **CA1 读写判断**  
   匹配已有 trace 时更新，语义不匹配时写入新 trace。

4. **Cortex 长期记忆**  
   trace 保存 signature、top terms、evidence 和写入次数。

`tests/test_hippocampal_bridge.py` 覆盖网页端接入前的关键接口：

1. `HippocampalRecall` 是否暴露 current_boost、metrics 和 recall evidence。
2. `inject_recall_current()` 是否能把海马体 current 注入当天 LIF 输入。

## 与旧 LIF-Memory 的区别

旧结构更像：

```text
文本 -> LIF 电压 -> spike -> 记忆/回答
```

新结构更像：

```text
文本 -> DG 分离 -> CA3 自组织 -> CA1 判断 -> Cortex 痕迹 -> recall current -> LIF spike -> AhaEngine
```

关键变化是：

> LIF 不再只是“是否写入”的门控器，而是 CA3 网络里的递归联想单元；网页端现在会把 CA3/CA1 的召回结果作为 downstream LIF 和 AhaEngine 的输入。