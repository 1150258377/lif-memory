# LIF-Memory v0.9 Adaptive Relation Spike

这一版修正 v0.8 的核心问题：不再把主题、角色、关系规则主要写死在代码里。

v0.8 的问题是：

```text
人工定义 role/topic/rule -> 再去匹配笔记
```

v0.9 改成：

```text
笔记样本 -> 自动归纳概念 -> 概念共现图 -> LLM 归纳关系 -> 人类校准 -> 关系 spike
```

## 文件

```text
adaptive_relation_spike.py
```

它和旧模块的关系：

```text
lif_memory.py              状态电位和行动卡
insight_integrator.py      预设 latent question 的洞察卡
relation_spike.py          手写规则关系层原型
adaptive_relation_spike.py 自适应关系层，不预设领域主题
```

## 1. 纯统计自动归纳

不接 LLM 时，系统会从最近日记里自动抽取候选概念，并根据概念共现形成关系 spike。

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --output "LIF-Adaptive-Relation-Spike 回放结果.md"
```

只预览：

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --dry-run
```

输出自动 schema：

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --write-schema --schema-file "lif_relation_schema.generated.json"
```

## 2. 接入 LLM 自动归纳 schema

先配置 API key，例如 DeepSeek：

```powershell
$env:DEEPSEEK_API_KEY="your-key"
```

运行：

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --llm-induce-schema --llm-provider deepseek --write-schema --schema-file "lif_relation_schema.generated.json" --output "LIF-Adaptive-Relation-Spike 回放结果.md"
```

LLM 会从笔记样本中生成：

```json
{
  "concepts": [
    {
      "name": "string",
      "aliases": ["string"],
      "description": "string",
      "why_matters": "string"
    }
  ],
  "relations": [
    {
      "name": "string",
      "source": "concept name",
      "target": "concept name",
      "relation_type": "support|conflict|inhibition|reframe|downgrade|causal|bridge|unknown",
      "hypothesis": "string",
      "next_validation_action": "string",
      "calibration_question": "string|null"
    }
  ],
  "calibration_questions": ["string"]
}
```

## 3. 让系统主动问人

如果你不想完全相信 LLM，可以让系统生成一份人类校准问题：

```powershell
python "04 项目库\P2_LIF-Memory\adaptive_relation_spike.py" --vault "." --days 60 --llm-induce-schema --llm-provider deepseek --ask-human-output "LIF关系层校准问题.md"
```

这份文件会问你：

```text
A 和 B 的关系更像 support / conflict / inhibition / reframe / downgrade / causal / bridge / unknown 中哪一种？
当前假设是否正确？
是否需要合并或拆分某些概念？
```

## 4. 这版的关键变化

旧方式：

```text
我先定义“负阻/论文/LIF/健康”
再用笔记去匹配这些词
```

新方式：

```text
系统先看你的笔记里什么反复出现
再自动形成 concepts
再形成 relations
最后把不确定的地方交给 LLM 或你校准
```

因此这版不再把“定义”写死。

定义来源变成三层：

```text
1. 统计共现：笔记自己长出来的概念
2. LLM 归纳：从样本中命名概念和关系
3. 人类校准：你确认哪些定义真正重要
```

## 5. 技术定位

```text
v0.8 relation_spike.py
= hand-written relation prototype

v0.9 adaptive_relation_spike.py
= schema induction + relation graph + LLM/human calibration
```

这才是后面可以继续进化的方向。

## 6. 下一步 v1.0

v0.9 只是生成 schema 文件。

v1.0 应该让 schema 反过来影响原系统：

```text
lif_relation_schema.generated.json
-> lif_memory.py topic policy
-> insight_integrator.py latent question generation
-> relation memory update
```

也就是说，系统不只是发现关系，而是把关系写回长期记忆。下一次运行时，它会带着新定义继续演化。
