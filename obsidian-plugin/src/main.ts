import { App, Notice, Plugin, PluginSettingTab, Setting, TFile, normalizePath } from "obsidian";
import {
  LifEngineOptions,
  collectNoteSignals,
  renderDailySpike,
  renderInsightReport,
  renderMemoryIndex
} from "./lifEngine";

interface LifmeSettings {
  outputFolder: string;
  threshold: number;
  maxNotes: number;
}

const DEFAULT_SETTINGS: LifmeSettings = {
  outputFolder: "LIFME",
  threshold: 0.65,
  maxNotes: 120
};

export default class LifmePlugin extends Plugin {
  settings: LifmeSettings;

  async onload(): Promise<void> {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());

    this.addCommand({
      id: "build-memory-index",
      name: "Build memory index",
      callback: async () => {
        await this.buildMemoryIndex();
      }
    });

    this.addCommand({
      id: "generate-daily-spike",
      name: "Generate daily spike",
      callback: async () => {
        await this.generateDailySpike();
      }
    });

    this.addCommand({
      id: "generate-insight-report",
      name: "Generate insight report",
      callback: async () => {
        await this.generateInsightReport();
      }
    });

    this.addSettingTab(new LifmeSettingTab(this.app, this));
  }

  async buildMemoryIndex(): Promise<void> {
    const signals = await this.collectSignals();
    await this.writeOutput("memory-index.md", renderMemoryIndex(signals));
    new Notice("LIFME memory index generated.");
  }

  async generateDailySpike(): Promise<void> {
    const signals = await this.collectSignals();
    await this.writeOutput("daily-spike.md", renderDailySpike(signals));
    new Notice("LIFME daily spike generated.");
  }

  async generateInsightReport(): Promise<void> {
    const signals = await this.collectSignals();
    await this.writeOutput("insight-report.md", renderInsightReport(signals));
    new Notice("LIFME insight report generated.");
  }

  private async collectSignals() {
    const options: LifEngineOptions = {
      threshold: this.settings.threshold,
      maxNotes: this.settings.maxNotes
    };

    return collectNoteSignals(this.app, options);
  }

  private async writeOutput(filename: string, content: string): Promise<void> {
    const folder = normalizePath(this.settings.outputFolder || DEFAULT_SETTINGS.outputFolder);
    const path = normalizePath(`${folder}/${filename}`);

    if (!(await this.app.vault.adapter.exists(folder))) {
      await this.app.vault.createFolder(folder);
    }

    const existing = this.app.vault.getAbstractFileByPath(path);
    if (existing instanceof TFile) {
      await this.app.vault.modify(existing, content);
      return;
    }

    await this.app.vault.create(path, content);
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }
}

class LifmeSettingTab extends PluginSettingTab {
  plugin: LifmePlugin;

  constructor(app: App, plugin: LifmePlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();

    containerEl.createEl("h2", { text: "LIFME settings" });

    new Setting(containerEl)
      .setName("Output folder")
      .setDesc("Folder where LIFME writes generated markdown files.")
      .addText(text => text
        .setPlaceholder(DEFAULT_SETTINGS.outputFolder)
        .setValue(this.plugin.settings.outputFolder)
        .onChange(async value => {
          this.plugin.settings.outputFolder = value.trim() || DEFAULT_SETTINGS.outputFolder;
          await this.plugin.saveSettings();
        }));

    new Setting(containerEl)
      .setName("Spike threshold")
      .setDesc("Voltage threshold for marking a note as fired. Recommended range: 0.45 to 0.85.")
      .addText(text => text
        .setPlaceholder(String(DEFAULT_SETTINGS.threshold))
        .setValue(String(this.plugin.settings.threshold))
        .onChange(async value => {
          const parsed = Number(value);
          if (!Number.isNaN(parsed)) {
            this.plugin.settings.threshold = Math.max(0.05, Math.min(0.95, parsed));
            await this.plugin.saveSettings();
          }
        }));

    new Setting(containerEl)
      .setName("Max scanned notes in report")
      .setDesc("Limits how many high-voltage notes are kept in the generated reports.")
      .addText(text => text
        .setPlaceholder(String(DEFAULT_SETTINGS.maxNotes))
        .setValue(String(this.plugin.settings.maxNotes))
        .onChange(async value => {
          const parsed = Number.parseInt(value, 10);
          if (!Number.isNaN(parsed)) {
            this.plugin.settings.maxNotes = Math.max(10, Math.min(1000, parsed));
            await this.plugin.saveSettings();
          }
        }));
  }
}
