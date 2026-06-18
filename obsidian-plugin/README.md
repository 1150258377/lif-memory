# LIFME Obsidian Plugin

LIFME is an event-driven memory and insight engine for Obsidian vaults.

It treats an Obsidian vault as a spiking memory system:

```text
Markdown notes -> note signals -> LIF voltage -> spike candidates -> generated Obsidian reports
```

This plugin shell is the first step toward turning the existing `lif-memory` Python research prototype into an installable Obsidian community plugin.

## Current features

- Scan Markdown notes in the current vault.
- Read note metadata through Obsidian APIs.
- Estimate a LIF-style voltage for each note.
- Generate reports back into the vault.
- Provide Obsidian commands and settings.

## Commands

Open the command palette and run:

```text
LIFME: Build memory index
LIFME: Generate daily spike
LIFME: Generate insight report
```

Generated files are written to:

```text
LIFME/memory-index.md
LIFME/daily-spike.md
LIFME/insight-report.md
```

## Development

```bash
cd obsidian-plugin
npm install
npm run build
```

For local Obsidian testing, copy these generated files into your vault plugin folder:

```text
.obsidian/plugins/lifme/
  manifest.json
  main.js
  styles.css
```

## Design boundary

This first plugin version does **not** call external LLM APIs, does **not** upload vault content, and does **not** run the Python CLI.

The immediate goal is the minimum Obsidian-native loop:

```text
read vault -> compute LIF note voltage -> generate spike reports -> write markdown back
```

Future versions can add:

- MazeGraph integration.
- Multi-agent reviewer views.
- Local Python CLI bridge for desktop-only mode.
- Optional LLM review with explicit user consent.
- Community marketplace release workflow.

## Privacy

This version runs locally inside Obsidian and only reads/writes Markdown files in the active vault. No telemetry is collected.
