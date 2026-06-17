# LIF-Memory v0.8 Relation Spike

`relation_spike.py` 是 LIF-Memory 的关系层实验模块。

它不替代现有的 `lif_memory.py` 和 `insight_integrator.py`：

```text
lif_memory.py          发现哪个状态最该处理
insight_integrator.py  把分散片段积累成洞察卡
relation_spike.py      发现多个主题之间的新关系
```

## 为什么需要 Relation Spike

旧系统主要回答：

```text
哪个神经元电位超过阈值？
```

这会产生行动卡或洞察卡，但它还不是真正的“关系涌现”。

Relation Spike 回答：

```text
多个主题之间是否形成了新的支持、抑制、降级、转向或机制关系？
```

例如：

```text
负阻反复失败 + 论文主线压力 + LIF 链路证据增强
=> 负阻降级为补充模块，主线收束到 EEG→LIF→后向散射→接收端检测
```

## 运行命令

在 Obsidian vault 根目录运行：

```powershell
python "04 项目库\P2_LIF-Memory\relation_spike.py" --vault "." --days 30 --output "LIF-Relation-Spike 回放结果.md"
```

只预览，不写文件：

```powershell
python "04 项目库\P2_LIF-Memory\relation_spike.py" --vault "." --days 30 --dry-run
```

导出 JSON：

```powershell
python "04 项目库\P2_LIF-Memory\relation_spike.py" --vault "." --days 30 --json-output "lif_relation_spikes.json"
```

如果 30 天证据不够，可以扩大范围：

```powershell
python "04 项目库\P2_LIF-Memory\relation_spike.py" --vault "." --days 90 --output "LIF-Relation-Spike 回放结果.md"
```

## 当前内置关系规则

```text
负阻降级与主线收束
实验数据到论文证据链
AI记忆项目转向求职表达
健康压力对实验执行的抑制
经济现象机制化
```

## 输出怎么看

每个 relation spike 会给出：

```text
Spike ID
Relation type
Priority
Score / Threshold
Source topics
Bridge fragments
Role counts
Emergent claim
Next validation action
Evidence chain
JSON packet
```

其中最重要的是三项：

```text
Emergent claim          这次关系层生成的新判断
Next validation action  最小验证动作
Evidence chain          它为什么这样判断
```

## v0.8 的技术定位

```text
旧系统：fragments -> state voltage -> action/insight spike
v0.8：fragments -> topic/role evidence -> cross-topic relation score -> relation spike
```

它的目标不是让某个状态神经元更准，而是让多个状态之间产生新关系。

## 下一步 v0.9

v0.8 只发现关系，还不会反过来改变原系统状态。

v0.9 应该做：

```text
relation spike -> topic policy update
relation spike -> threshold modulation
relation spike -> priority override
relation spike -> graph edge memory
```

也就是说，如果系统发现“负阻应降级”，下一次 `lif_memory.py` 不应该继续把负阻当成普通实验 P1，而应该从关系记忆中读取这个结论，自动降低其主线权重。
