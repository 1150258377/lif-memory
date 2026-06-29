# First-Principle Meta-Spike

普通 LIF spike 回答的是：

```text
哪个状态电位超过阈值，因此现在应该做什么？
```

First-Principle Meta-Spike 回答的是另一个问题：

```text
当前问题空间是否已经错了，是否应该停止局部搜索并重构坐标系？
```

它对应一种高阶神经动力学：多个状态同时升高、预测误差持续积累、完成信号不足、状态之间互相抢资源。此时系统继续输出普通任务，可能只会制造更多行动债务。因此需要一个 meta-level spike。

## 设计动机

传统 AI / JEPA-style 系统倾向于在 latent space 中持续预测和迭代：

```text
z_t -> predicted z_(t+1) -> error -> update
```

这很适合连续优化。但人的高级智能还有另一种能力：当局部搜索失效时，不继续搜索，而是回到底层生成变量，重新定义问题空间。

这类似第一性原理：

```text
不要问下一步做 A 还是 B
而是问：真正生成 A/B 困境的最小变量是什么？
```

在 LIF-Memory 中，这被实现为：

```text
high prediction error
+ high state conflict
+ high global voltage
+ low completion
+ high micro density
-> meta-spike
-> coordinate reset
```

## 数学形式

对状态神经元 `i`：

```text
M_i(t) =
  alpha * EMA(prediction_error_i)
+ beta  * conflict_i(t)
+ gamma * global_voltage(t)
+ delta * low_completion_i(t)
+ rho   * micro_density(t)

S_meta_i(t) = 1[M_i(t) > Theta_meta]
```

其中：

- `prediction_error_i` 来自 Predictive LIF-JEPA Field。
- `conflict_i(t)` 来自状态之间的负耦合，例如 Experiment 与 Health、Thesis 与 Career 的资源冲突。
- `global_voltage(t)` 表示整个系统的总压力水平。
- `low_completion_i(t)` 表示完成信号不足。
- `micro_density(t)` 表示当天有多少状态已经发生 micro-spike。

## 与普通 spike 的区别

| 类型 | 触发含义 | 输出 |
| --- | --- | --- |
| LIF spike | 某个状态电位超过行动阈值 | 做一个具体任务 |
| Predictive micro-spike | 亚阈值状态开始变热 | 内部 trace，不默认打扰 |
| Macro candidate | micro/ratio/error 接近正式行动 | 候选行动卡片 |
| Meta-spike | 局部搜索可能失效 | 重构问题坐标系 |

普通 spike 是：

```text
state -> action
```

meta-spike 是：

```text
state conflict -> representation reset
```

## 命令行用法

```powershell
python meta_spike_memory.py `
  --vault "C:\\path\\to\\vault" `
  --days 14 `
  --output "Meta-Spike Memory Report.md" `
  --json-output meta_spikes.json
```

只预览：

```powershell
python meta_spike_memory.py --vault "C:\\path\\to\\vault" --days 14 --dry-run
```

调节 meta 阈值：

```powershell
python meta_spike_memory.py --vault "C:\\path\\to\\vault" --theta-meta 1.8
```

## 输出结构

每张 meta-spike card 包含：

- `Meta energy`
- `Prediction error`
- `Conflict`
- `Global voltage`
- `Low completion`
- `Micro density`
- `First-principle question`
- `Coordinate reset`
- `Delete / downgrade`
- `Next probe`

## 示例解释

如果 `Experiment` 触发 meta-spike，它不再输出：

```text
继续做实验。
```

而是输出：

```text
从“继续做实验”重置为“识别最小可信证据链”：输入、参数、波形/截图、判据、可复现性。
```

如果 `Thesis` 触发 meta-spike，它不再输出：

```text
继续写论文。
```

而是输出：

```text
从“继续写论文”重置为“一个证据块能否支撑一个论断”：结论、图、限制条件、答辩说法。
```

## 边界

- meta-spike 不是让系统做更多任务。
- meta-spike 是在提醒：当前任务生成机制可能需要重构。
- 如果 meta-spike 频繁出现，说明系统长期处于高冲突、低完成状态。
- 如果 meta-spike 完全不出现，说明普通 LIF spike / predictive micro-spike 仍然足够。
- 原始笔记仍然是 source of truth；meta-spike 只保存派生解释。
