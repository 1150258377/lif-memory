# Continuous Problem Field for LIF-Memory v0.8.0

这一版把 LIF-Memory 从“事件驱动回放器”扩展为“连续问题场重建器”。

原来的 v0.7.x 主线是：

```text
Obsidian notes -> evidence vector -> LIF voltage -> spike packet -> action/insight card
```

v0.8.0-field 新增的是：

```text
Obsidian notes -> note observations -> graph diffusion -> continuous problem field -> LIF integration -> insight spike
```

也就是说，笔记不再被直接看作答案，而被看作一个连续问题场的离散观测。

---

## 1. 数学定义

每条 Obsidian 笔记被定义为：

```text
n_i = (text_i, t_i, links_i, tags_i)
```

其中：

- `text_i` 是笔记正文；
- `t_i` 是笔记日期或修改日期；
- `links_i` 是 Obsidian wikilinks；
- `tags_i` 是标签。

通过一个确定性的语义特征函数得到：

```text
e_i = phi(n_i)
```

Obsidian 双链、标签和文件夹结构构成图：

```text
G = (V, E)
```

图扩散后得到笔记的场向量：

```text
h_i = GraphDiffuse(e_i, G)
```

当用户输入一个问题 `q` 时，问题也被映射为一个坐标：

```text
z_q = phi(q)
```

于是连续问题场为：

```text
M(t, z_q) = sum_i K_t(t, t_i) K_z(z_q, z_i) h_i / sum_i K_t(t, t_i) K_z(z_q, z_i)
```

其中时间核为：

```text
K_t(t, t_i) = exp(-(t - t_i)^2 / (2 sigma_t^2))
```

语义核为：

```text
K_z(z_q, z_i) = exp(-(1 - cos(z_q, z_i))^2 / (2 sigma_z^2))
```

这一步的意义是：系统不是只找 top-k 笔记，而是在时间、语义和图结构上重建一个连续场切片。

---

## 2. LIF 触发公式

连续问题场产生当前问题刺激：

```text
I_q(t) = field_energy(t) + action/blocker/novelty boost - completion inhibition
```

然后进入双时间尺度 LIF：

```text
V_fast = max(0, V_fast * decay_fast + I_q - C)
V_slow = max(0, V_slow * decay_slow + 0.42 * I_q - 0.30 * C)
V      = 0.65 * V_fast + 0.35 * V_slow
```

触发条件：

```text
V >= theta => insight spike
```

触发后不是普通总结，而是生成一张“连续问题场洞察卡”。

---

## 3. 和普通 RAG 的区别

普通 RAG：

```text
query -> retrieve top-k notes -> answer
```

Continuous Problem Field：

```text
query -> stimulate field -> integrate voltage -> threshold crossing -> reconstruct insight
```

普通 RAG 只问：

```text
哪些笔记和这个问题相似？
```

LIF-Memory v0.8.0-field 要问：

```text
这些离散笔记背后正在形成什么连续问题？
这个问题的张力是否已经跨过阈值？
现在应该生成洞察、继续追问，还是补充证据？
```

---

## 4. 运行方式

在 Obsidian vault 根目录运行：

```powershell
python "04 项目库\P2_LIF-Memory\continuous_problem_field.py" --vault "." --query "LIF-Memory 如何从离散笔记升级成连续问题场？" --days 90 --output "LIF-Memory 连续问题场.md"
```

如果脚本就在当前目录：

```powershell
python continuous_problem_field.py --vault "C:\path\to\vault" --query "LIF-Memory 如何从离散知识库产生新洞察？" --days 90 --dry-run
```

导出 JSON：

```powershell
python continuous_problem_field.py --vault "." --query "负阻为什么一直卡住？" --days 90 --json-output "field_packet.json"
```

---

## 5. 关键参数

```text
--time-sigma       时间核宽度，默认 30 天
--semantic-sigma   语义核宽度，默认 0.55
--graph-steps      Obsidian 图扩散步数，默认 2
--graph-alpha      邻居混合强度，默认 0.35
--threshold        LIF 洞察阈值，默认 5.0
--top-k            输出证据笔记数，默认 8
```

调参建议：

- 找不到足够证据：增大 `--semantic-sigma` 或 `--days`。
- 证据太散：减小 `--semantic-sigma`。
- 太容易触发 insight：增大 `--threshold`。
- 触发太晚：减小 `--threshold` 或增大 `--graph-alpha`。
- 想更依赖 Obsidian 双链：增大 `--graph-steps` 或 `--graph-alpha`。

---

## 6. 工程边界

当前版本是一个无外部依赖的 deterministic prototype：

- 不调用外部 embedding API；
- 不调用 LLM；
- 用稀疏文本特征、CJK shingle、topic keyword、tag、wikilink 做近似语义场；
- 保留了可替换接口，后续可以把 `phi(n_i)` 替换为真实 embedding，把 `InsightDecoder` 替换为 LLM reviewer。

这保证它可以直接在本地 Obsidian vault 上跑，不依赖 API key。

---

## 7. 后续升级方向

下一步可以把它接入现有 `lif_memory.py`：

```text
--field-query "..."
```

让 daily mode 不只输出 top spike，也输出一个连续问题场重建结果。

也可以增加持久化：

```text
lif_problem_field.json
```

用于保存每个问题坐标的长期电位、最近场关键词、最近 insight spike、反馈状态。

最终目标：

```text
离散 Obsidian 笔记
-> 连续问题场
-> LIF 张力积累
-> 阈值触发 insight
-> 写回 Obsidian
-> 下一轮交互继续刺激
```
