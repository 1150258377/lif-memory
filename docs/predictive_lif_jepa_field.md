# Predictive LIF-JEPA Memory Field

`predictive_lif_memory.py` 在现有 `lif_memory.py` 的双时间尺度 LIF 回放之上增加一个轻量预测层。它的目标不是替代原来的 spike 卡片，而是补上一个关键缺口：

> 即使没有正式 spike，系统也应该在 latent space 中保留 micro-spike、预测误差和阈值稳态调节。

## 为什么需要这一层

经典 LIF-Memory 已经实现：

```text
notes -> evidence vector -> V_fast / V_slow -> V -> macro spike
```

这能解释“什么时候应该提醒我”。但它有一个事件系统天然问题：

```text
阈值太高 -> 什么都不触发
阈值太低 -> 每天都在乱触发
```

Predictive LIF-JEPA Field 增加的是：

```text
subthreshold voltage trace
-> latent prediction
-> prediction error
-> graph diffusion
-> micro-spike
-> homeostatic threshold control
-> macro candidate
```

这使系统可以区分三种状态：

1. `no spike, no signal`：当前确实没积累。
2. `no macro spike, but micro activation`：不打扰用户，但系统内部知道某个主题在变热。
3. `macro candidate`：连续 micro activation 或预测误差过高，应该转成行动卡片或 AhaEngine 输入。

## 和 JEPA 的对应关系

JEPA 的核心是：

```text
context representation z_t -> predict target representation z_(t+1)
prediction error -> learning signal
```

本项目中的对应关系是：

```text
LIF latent state z_t = {V, V_fast, V_slow, evidence_input, completion}
z_t -> predicted z_(t+1)
actual z_(t+1) - predicted z_(t+1) -> prediction_error
prediction_error -> energy injection / micro activation
```

它不是完整的端到端 JEPA，也不训练图像或视频 encoder。它是一个本地、可审计、低风险的 JEPA-style controller layer。

## 数学结构

对第 `i` 个状态神经元：

```text
r_i(t) = V_i(t) / theta_i
r'_i(t) = r_i(t) + diffusion_alpha * sum_j A_ij r_j(t)

r_hat_i(t) =
    decay * r'_i(t-1)
  + input_gain * input_i(t-1)
  - completion_gain * completion_i(t-1)
  + persistent_error_gain * EMA(error_i)

error_i(t) = |r'_i(t) - r_hat_i(t)|

effective_i(t) = r'_i(t) + prediction_error_weight * EMA(error_i)

micro_spike_i(t) = effective_i(t) >= theta_micro_i
```

其中：

- `r_i(t)` 是归一化 LIF 电位。
- `r'_i(t)` 是经过主题图扩散后的 latent pressure。
- `r_hat_i(t)` 是上一时刻 latent state 对当前状态的预测。
- `error_i(t)` 是 JEPA-style prediction error。
- `effective_i(t)` 是把真实压力、扩散压力、预测误差合并后的有效激活。
- `micro_spike` 只进入内部状态，不默认打扰用户。

## Homeostasis

为了避免“太稀疏导致什么都不触发”，micro 阈值会做稳态调节：

```text
theta_micro_i <- theta_micro_i + eta * (firing_rate_i - target_micro_rate)
```

含义：

- 最近 micro 触发太少：降低 micro 阈值。
- 最近 micro 触发太多：提高 micro 阈值。
- macro spike 仍然保持更严格，不因为 homeostasis 直接泛滥。

## 命令行用法

```powershell
python predictive_lif_memory.py `
  --vault "C:\\path\\to\\vault" `
  --days 14 `
  --output "Predictive LIF-Memory Report.md" `
  --json-output predictive_lif_report.json
```

如果只想在终端预览：

```powershell
python predictive_lif_memory.py --vault "C:\\path\\to\\vault" --days 14 --dry-run
```

默认会写入本地 controller state：

```text
predictive_lif_state.json
```

这个文件记录：

- `micro_threshold_delta`
- `prediction_error_ema`
- `micro_streak`
- `runs`
- `updated_at`

它是私人派生状态，不应该提交到 GitHub。

## 参数

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `--micro-ratio` | `0.55` | micro-spike 的基础阈值，占正式 theta 的比例 |
| `--macro-ratio` | `1.00` | macro candidate 的正式阈值比例 |
| `--target-micro-rate` | `0.35` | 目标 micro 触发率 |
| `--prediction-error-weight` | `0.45` | 预测误差注入 effective ratio 的权重 |
| `--diffusion-alpha` | `0.15` | 主题图扩散强度 |
| `--daily-spike-budget` | `0` | 默认不让原始 LIF reset，用于观察亚阈值轨迹 |

## 输出解释

报告中主要看三块：

### 1. 汇总

显示每个状态的：

- final voltage ratio
- micro 次数
- macro candidate 次数
- prediction-error EMA
- 当前 micro 阈值

### 2. Macro 候选

这里是最接近“应该生成行动卡片”的状态。

它可能来自两种情况：

```text
ratio >= macro_ratio
```

或者：

```text
连续 micro-spike + effective ratio 足够高
```

### 3. 最大预测误差

这用于发现系统没有预料到的状态突变。对 LIF-Memory 来说，这类突变往往比普通 keyword hit 更有价值，因为它说明：

```text
之前的 latent state 无法解释现在的状态
```

这就是 JEPA-style prediction error 在记忆系统中的意义。

## 与现有模块的关系

```text
lif_memory.py
  负责 evidence extraction、V_fast/V_slow、经典 spike

predictive_lif_memory.py
  负责 subthreshold trace、latent prediction、micro-spike、homeostasis

aha_engine.py
  可以消费 macro candidate，把 spike/候选激活重构成 Aha card

continuous_problem_field.py
  负责 query-specific field 和证据召回
```

## 边界

- 不上传任何私有笔记。
- 不微调 embedding。
- 不做端到端神经网络训练。
- 不替代原始 Obsidian 笔记。
- `prediction_error` 只表示 latent trajectory 的偏差，不等于事实错误。
- micro-spike 默认不直接提醒用户；它是内部 trace。
