# LIF-Memory v1.0 Schema Policy Adapter

这一版解决 v0.9 之后的关键问题：

```text
自动归纳出来的 relation schema 不能只停留在报告里，必须反过来改变 LIF-Memory 的触发方式。
```

新增文件：

```text
schema_policy_adapter.py
```

## 位置

当前系统链路变成：

```text
adaptive_relation_spike.py
-> lif_relation_schema.generated.json
-> schema_policy_adapter.py
-> lif_memory_adaptive_policy.json
-> lif_memory.py --feedback-file
```

也就是说：

```text
笔记中长出来的关系
-> 转成 topic policy
-> 改变阈值 / 优先级 / action policy
-> 影响下一次 daily spike
```

## 为什么这样做

之前的问题是：

```text
relation spike 只是报告
```

v1.0 之后变成：

```text
relation spike 可以改变系统本身
```

这一步很重要，因为 LIF-Memory 不应该只是“生成洞察”，而应该让洞察反过来改变系统的行为。

## 第一步：生成 adaptive schema

纯统计模式：

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --write-schema --schema-file "lif_relation_schema.generated.json" --output "LIF-Adaptive-Relation-Spike 回放结果.md"
```

LLM schema induction：

```powershell
$env:DEEPSEEK_API_KEY="your-key"

python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --llm-induce-schema --llm-provider deepseek --write-schema --schema-file "lif_relation_schema.generated.json" --output "LIF-Adaptive-Relation-Spike 回放结果.md"
```

## 第二步：把 schema 转成 adaptive policy

```powershell
python "04 项目库\P2_LIF-Memory\schema_policy_adapter.py" --schema "lif_relation_schema.generated.json" --output "lif_memory_adaptive_policy.json" --report "LIF-Schema-Policy-Adapter 报告.md"
```

如果只想使用人工确认过的 relation：

```powershell
python "04 项目库\P2_LIF-Memory\schema_policy_adapter.py" --schema "lif_relation_schema.generated.json" --only-confirmed --output "lif_memory_adaptive_policy.json"
```

如果想过滤低置信度 relation：

```powershell
python "04 项目库\P2_LIF-Memory\schema_policy_adapter.py" --schema "lif_relation_schema.generated.json" --min-confidence 0.6 --output "lif_memory_adaptive_policy.json"
```

## 第三步：让主系统读取 adaptive policy

不用改 `lif_memory.py` 主逻辑，直接使用现有反馈入口：

```powershell
python "04 项目库\P2_LIF-Memory\lif_memory.py" --vault "." --days 14 --mode daily --feedback-file "lif_memory_adaptive_policy.json" --output "今日 LIF-Memory 主卡片.md"
```

## relation_type 到 policy 的映射

```text
support
-> target priority up, threshold down, continue

conflict
-> source/target isolate, small cooldown

inhibition
-> target recover_first, threshold up

downgrade
-> source priority down, threshold up, action downgrade
-> target priority up, continue

reframe
-> source downgrade, target continue

causal
-> source and target slightly easier to trigger

bridge
-> source and target become weakly connected

unknown
-> 不写入 policy，等待人工校准
```

## 输出格式

`schema_policy_adapter.py` 输出的是 `lif_memory.py --feedback-file` 已经能读取的格式：

```json
{
  "version": "1.0.0",
  "feedback": [
    {
      "topic": "LIF链路",
      "feedback": "adaptive_schema",
      "threshold_delta": -0.45,
      "priority": "P0",
      "action_policy": "continue",
      "cooldown_days": 0,
      "reason": "relation schema says this topic supports another high-value target"
    }
  ]
}
```

这意味着 v1.0 不需要先重构主程序，就可以让自动 schema 进入主循环。

## 当前限制

这一步仍然是 policy adapter，不是完整 graph-coupled LIF。

它能做到：

```text
schema relation -> topic policy
```

还不能做到：

```text
V_i = decay * V_i + input_i + Σ W_ji * V_j - inhibition_i
```

真正的图耦合电位更新应该放到 v1.1 或 v1.2。

## 下一步

推荐顺序：

```text
v1.1 parse_human_calibration.py
v1.2 graph_coupled_lif.py
v1.3 eval_adaptive_relation.py
```

v1.1 让你对 LLM 生成的 schema 做人工确认；v1.2 让关系权重真正进入电位更新；v1.3 评估系统是否真的变聪明。
