import { App, TFile } from "obsidian";

export interface NoteSignal {
  path: string;
  basename: string;
  mtime: number;
  ageDays: number;
  wordCount: number;
  linkCount: number;
  tagCount: number;
  headingCount: number;
  tensionCount: number;
  taskCount: number;
  recentWeight: number;
  structureWeight: number;
  tensionWeight: number;
  voltage: number;
  threshold: number;
  fired: boolean;
  reasons: string[];
}

export interface LifEngineOptions {
  threshold: number;
  maxNotes: number;
}

const TENSION_PATTERNS = [
  /\bTODO\b/gi,
  /\bFIXME\b/gi,
  /\bwhy\b/gi,
  /\bhow\b/gi,
  /为什么/g,
  /怎么办/g,
  /如何/g,
  /问题/g,
  /卡住/g,
  /失败/g,
  /突破/g,
  /\?/g,
  /？/g
];

const TASK_PATTERN = /^\s*- \[ \]/gm;

export async function collectNoteSignals(app: App, options: LifEngineOptions): Promise<NoteSignal[]> {
  const files = app.vault.getMarkdownFiles();
  const now = Date.now();
  const signals: NoteSignal[] = [];

  for (const file of files) {
    if (shouldSkipFile(file)) continue;

    const content = await app.vault.cachedRead(file);
    const cache = app.metadataCache.getFileCache(file);

    const ageDays = Math.max(0, (now - file.stat.mtime) / 86_400_000);
    const recentWeight = clamp01(1 - ageDays / 30);
    const linkCount = cache?.links?.length ?? 0;
    const tagCount = cache?.tags?.length ?? 0;
    const headingCount = cache?.headings?.length ?? 0;
    const wordCount = countWords(content);
    const tensionCount = countTension(content);
    const taskCount = countTasks(content);

    const structureWeight = clamp01((linkCount * 0.08) + (tagCount * 0.12) + (headingCount * 0.03));
    const tensionWeight = clamp01((tensionCount * 0.08) + (taskCount * 0.12));
    const lengthWeight = clamp01(wordCount / 2400);

    const voltage = clamp01(
      recentWeight * 0.34 +
      structureWeight * 0.22 +
      tensionWeight * 0.30 +
      lengthWeight * 0.14
    );

    const reasons = buildReasons({
      recentWeight,
      structureWeight,
      tensionWeight,
      linkCount,
      tagCount,
      headingCount,
      tensionCount,
      taskCount,
      wordCount
    });

    signals.push({
      path: file.path,
      basename: file.basename,
      mtime: file.stat.mtime,
      ageDays,
      wordCount,
      linkCount,
      tagCount,
      headingCount,
      tensionCount,
      taskCount,
      recentWeight,
      structureWeight,
      tensionWeight,
      voltage,
      threshold: options.threshold,
      fired: voltage >= options.threshold,
      reasons
    });
  }

  return signals
    .sort((a, b) => b.voltage - a.voltage)
    .slice(0, options.maxNotes);
}

export function renderDailySpike(signals: NoteSignal[]): string {
  const fired = signals.filter(signal => signal.fired).slice(0, 8);
  const fallback = signals.slice(0, 8);
  const selected = fired.length > 0 ? fired : fallback;

  return [
    "# LIFME Daily Spike",
    "",
    `Generated: ${new Date().toLocaleString()}`,
    "",
    "## Spike candidates",
    "",
    ...selected.map(renderSignalBullet),
    "",
    "## Interpretation",
    "",
    fired.length > 0
      ? "这些笔记的电位已经超过阈值，说明它们不是普通摘要对象，而是当前知识库里最应该被触发的问题。"
      : "当前没有笔记超过阈值。下面列出的是电位最高的候选笔记，可以作为今天的注意力入口。",
    "",
    "## Next action",
    "",
    "- [ ] 选择上面 1 个 spike，把它改写成今天唯一要推进的行动卡。",
    "- [ ] 在原始笔记里补上证据链接，而不是只写抽象总结。",
    ""
  ].join("\n");
}

export function renderInsightReport(signals: NoteSignal[]): string {
  const firedCount = signals.filter(signal => signal.fired).length;
  const avgVoltage = signals.length === 0
    ? 0
    : signals.reduce((sum, signal) => sum + signal.voltage, 0) / signals.length;

  return [
    "# LIFME Insight Report",
    "",
    `Generated: ${new Date().toLocaleString()}`,
    "",
    "## Global state",
    "",
    `- Scanned note signals: ${signals.length}`,
    `- Fired spikes: ${firedCount}`,
    `- Average voltage: ${avgVoltage.toFixed(3)}`,
    "",
    "## Top voltage notes",
    "",
    ...signals.slice(0, 20).map(renderSignalTableRowHeader),
    "",
    "## What this means",
    "",
    "LIFME 当前把 Obsidian vault 看成一个事件驱动记忆系统：最近修改、结构连接、未解决问题和行动债务共同积累为电位；超过阈值后，笔记会被视为 spike 候选。",
    "",
    "后续版本可以接入 MazeGraph、multi-agent reviewer 和本地/远程 LLM，但这个插件壳先保证最小闭环：读取 vault → 计算电位 → 写回 Obsidian。",
    ""
  ].join("\n");
}

export function renderMemoryIndex(signals: NoteSignal[]): string {
  return [
    "# LIFME Memory Index",
    "",
    `Generated: ${new Date().toLocaleString()}`,
    "",
    "| Note | V | Links | Tags | Tension | Tasks | Fired |",
    "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ...signals.map(signal => [
      `| [[${signal.basename}]]`,
      signal.voltage.toFixed(3),
      signal.linkCount,
      signal.tagCount,
      signal.tensionCount,
      signal.taskCount,
      signal.fired ? "yes" : "no"
    ].join(" | ") + " |"),
    ""
  ].join("\n");
}

function shouldSkipFile(file: TFile): boolean {
  return file.path.startsWith("LIFME/") || file.path.startsWith(".obsidian/");
}

function countWords(content: string): number {
  const latinWords = content.match(/[A-Za-z0-9_]+/g)?.length ?? 0;
  const cjkChars = content.match(/[\u4e00-\u9fff]/g)?.length ?? 0;
  return latinWords + Math.ceil(cjkChars / 2);
}

function countTension(content: string): number {
  return TENSION_PATTERNS.reduce((sum, pattern) => {
    const matches = content.match(pattern);
    return sum + (matches?.length ?? 0);
  }, 0);
}

function countTasks(content: string): number {
  return content.match(TASK_PATTERN)?.length ?? 0;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function buildReasons(input: {
  recentWeight: number;
  structureWeight: number;
  tensionWeight: number;
  linkCount: number;
  tagCount: number;
  headingCount: number;
  tensionCount: number;
  taskCount: number;
  wordCount: number;
}): string[] {
  const reasons: string[] = [];

  if (input.recentWeight > 0.7) reasons.push("recently active");
  if (input.structureWeight > 0.4) reasons.push(`${input.linkCount} links, ${input.tagCount} tags, ${input.headingCount} headings`);
  if (input.tensionWeight > 0.3) reasons.push(`${input.tensionCount} tension markers, ${input.taskCount} open tasks`);
  if (input.wordCount > 1200) reasons.push(`${input.wordCount} estimated words`);

  return reasons.length > 0 ? reasons : ["low but non-zero memory signal"];
}

function renderSignalBullet(signal: NoteSignal): string {
  return [
    `- [[${signal.basename}]]`,
    `  - V=${signal.voltage.toFixed(3)}, threshold=${signal.threshold.toFixed(3)}, fired=${signal.fired ? "yes" : "no"}`,
    `  - reasons: ${signal.reasons.join("; ")}`
  ].join("\n");
}

function renderSignalTableRowHeader(signal: NoteSignal): string {
  return `- [[${signal.basename}]] — V=${signal.voltage.toFixed(3)}, links=${signal.linkCount}, tags=${signal.tagCount}, tension=${signal.tensionCount}, tasks=${signal.taskCount}`;
}
