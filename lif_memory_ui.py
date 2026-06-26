from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs

try:
    import lif_memory as core
except Exception:  # pragma: no cover - UI can still start without importing the core module.
    core = None

APP_VERSION = "0.1.0"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_PREVIEW_CHARS = 12000


@dataclass(frozen=True)
class RunResult:
    title: str
    command: list[str]
    cwd: Path
    returncode: int
    stdout: str
    stderr: str
    output_paths: list[Path]
    started_at: str
    finished_at: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def default_vault() -> Path:
    if core is not None:
        try:
            return core.vault_root_from_script()
        except Exception:
            pass
    return Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a local visual debug UI for LIF-Memory.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind. Keep 127.0.0.1 for local-only use.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument("--vault", type=Path, default=default_vault(), help="Default Obsidian vault path shown in the UI.")
    return parser.parse_args()


def field(form: dict[str, list[str]], name: str, default: str = "") -> str:
    values = form.get(name)
    if not values:
        return default
    return values[0].strip()


def int_field(form: dict[str, list[str]], name: str, default: int, lower: int = 1, upper: int = 9999) -> int:
    raw = field(form, name, str(default))
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(lower, min(value, upper))


def bool_field(form: dict[str, list[str]], name: str) -> bool:
    return field(form, name).lower() in {"1", "true", "on", "yes"}


def path_from_vault(vault: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return vault / path


def script_path(name: str) -> str:
    return str((SCRIPT_DIR / name).resolve())


def command_common(form: dict[str, list[str]]) -> tuple[Path, int, int, bool, str]:
    vault = Path(field(form, "vault", str(default_vault()))).expanduser().resolve()
    days = int_field(form, "days", 14, lower=1, upper=3650)
    top_k = int_field(form, "top_k", 1, lower=1, upper=10)
    completion_scan = bool_field(form, "completion_scan")
    provider = field(form, "llm_provider", "qwen")
    return vault, days, top_k, completion_scan, provider


def append_llm_flags(command: list[str], form: dict[str, list[str]], provider: str) -> None:
    if bool_field(form, "llm_review"):
        command.append("--llm-review")
        if provider:
            command.extend(["--llm-provider", provider])


def build_daily_preview(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, days, top_k, completion_scan, provider = command_common(form)
    command = [
        sys.executable,
        script_path("lif_memory.py"),
        "--vault",
        str(vault),
        "--days",
        str(days),
        "--mode",
        "daily",
        "--top-k",
        str(top_k),
        "--dry-run",
    ]
    if completion_scan:
        command.append("--completion-scan")
    append_llm_flags(command, form, provider)
    return "预览今日主卡", command, [], vault


def build_daily_write(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, days, top_k, completion_scan, provider = command_common(form)
    output = path_from_vault(vault, field(form, "daily_output", "今日 LIF-Memory 主卡片.md"))
    command = [
        sys.executable,
        script_path("lif_memory.py"),
        "--vault",
        str(vault),
        "--days",
        str(days),
        "--mode",
        "daily",
        "--top-k",
        str(top_k),
        "--output",
        str(output),
    ]
    closure_file = field(form, "closure_file", "")
    if closure_file:
        command.extend(["--closure-file", str(path_from_vault(vault, closure_file))])
    if completion_scan:
        command.append("--completion-scan")
    append_llm_flags(command, form, provider)
    return "生成今日主卡", command, [output], vault


def build_replay(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, days, _top_k, completion_scan, provider = command_common(form)
    output = path_from_vault(vault, field(form, "replay_output", "LIF-Memory 回放结果.md"))
    json_output = path_from_vault(vault, field(form, "replay_json", "lif_spikes.json"))
    command = [
        sys.executable,
        script_path("lif_memory.py"),
        "--vault",
        str(vault),
        "--days",
        str(days),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    closure_file = field(form, "closure_file", "")
    if closure_file:
        command.extend(["--closure-file", str(path_from_vault(vault, closure_file))])
    if completion_scan:
        command.append("--completion-scan")
    append_llm_flags(command, form, provider)
    return "生成完整回放", command, [output, json_output], vault


def build_graph(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, _days, _top_k, _completion_scan, _provider = command_common(form)
    output = path_from_vault(vault, field(form, "graph_output", "Obsidian-LIF 知识图谱报告.md"))
    json_output = path_from_vault(vault, field(form, "graph_json", "obsidian_lif_graph.json"))
    command = [
        sys.executable,
        script_path("obsidian_graph_miner.py"),
        "--vault",
        str(vault),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    return "生成图谱报告", command, [output, json_output], vault


def build_insight(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, days, _top_k, _completion_scan, _provider = command_common(form)
    profile = field(form, "insight_profile", "economics") or "economics"
    sensitivity = field(form, "insight_sensitivity", "")
    output = path_from_vault(vault, field(form, "insight_output", "LIF-Memory 洞察整合.md"))
    json_output = path_from_vault(vault, field(form, "insight_json", "lif_insights.json"))
    command = [
        sys.executable,
        script_path("insight_integrator.py"),
        "--vault",
        str(vault),
        "--profile",
        profile,
        "--days",
        str(days),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    if sensitivity:
        command.extend(["--sensitivity", sensitivity])
    return "生成洞察报告", command, [output, json_output], vault


def build_v1_report(form: dict[str, list[str]]) -> tuple[str, list[str], list[Path], Path]:
    vault, _days, _top_k, _completion_scan, _provider = command_common(form)
    output = path_from_vault(vault, field(form, "v1_output", "LIF-Memory v1.0 收束报告.md"))
    json_output = path_from_vault(vault, field(form, "v1_json", "lif_memory_v1_report.json"))
    command = [
        sys.executable,
        script_path("lif_memory_v1_report.py"),
        "--output",
        str(output),
        "--json-output",
        str(json_output),
    ]
    return "生成 v1 收束报告", command, [output, json_output], vault


def action_builders() -> dict[str, object]:
    return {
        "daily_preview": build_daily_preview,
        "daily_write": build_daily_write,
        "replay": build_replay,
        "graph": build_graph,
        "insight": build_insight,
        "v1_report": build_v1_report,
    }


def run_command(title: str, command: list[str], cwd: Path, output_paths: list[Path]) -> RunResult:
    started = datetime.now().isoformat(timespec="seconds")
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + "\nCommand timed out after 180 seconds."
    except Exception as exc:  # pragma: no cover - local environment dependent.
        returncode = 1
        stdout = ""
        stderr = f"Failed to run command: {exc}"
    finished = datetime.now().isoformat(timespec="seconds")
    return RunResult(
        title=title,
        command=command,
        cwd=cwd,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        output_paths=output_paths,
        started_at=started,
        finished_at=finished,
    )


def run_action(action: str, form: dict[str, list[str]]) -> list[RunResult]:
    builders = action_builders()
    if action == "all_reports":
        sequence = ["daily_write", "replay", "graph", "insight", "v1_report"]
    else:
        sequence = [action]

    results: list[RunResult] = []
    for item in sequence:
        builder = builders.get(item)
        if builder is None:
            results.append(
                RunResult(
                    title="未知动作",
                    command=[],
                    cwd=Path.cwd(),
                    returncode=1,
                    stdout="",
                    stderr=f"Unknown action: {item}",
                    output_paths=[],
                    started_at=datetime.now().isoformat(timespec="seconds"),
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                )
            )
            continue
        title, command, outputs, cwd = builder(form)  # type: ignore[misc]
        results.append(run_command(title, command, cwd, outputs))
    return results


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(parts)


def read_preview(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"无法读取文件：{exc}"
    if len(text) > MAX_PREVIEW_CHARS:
        return text[:MAX_PREVIEW_CHARS] + "\n\n... [preview truncated]"
    return text


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def default_form_values(default_vault_path: Path) -> dict[str, str]:
    return {
        "vault": str(default_vault_path),
        "days": "14",
        "top_k": "1",
        "llm_provider": "qwen",
        "insight_profile": "economics",
        "insight_sensitivity": "",
        "closure_file": "今日 LIF-Memory 主卡片.md",
        "daily_output": "今日 LIF-Memory 主卡片.md",
        "replay_output": "LIF-Memory 回放结果.md",
        "replay_json": "lif_spikes.json",
        "graph_output": "Obsidian-LIF 知识图谱报告.md",
        "graph_json": "obsidian_lif_graph.json",
        "insight_output": "LIF-Memory 洞察整合.md",
        "insight_json": "lif_insights.json",
        "v1_output": "LIF-Memory v1.0 收束报告.md",
        "v1_json": "lif_memory_v1_report.json",
    }


def merged_form(default_vault_path: Path, form: dict[str, list[str]] | None) -> dict[str, str]:
    values = default_form_values(default_vault_path)
    if form:
        for key in list(values):
            if key in form and form[key]:
                values[key] = form[key][0]
        for key in ["llm_review", "completion_scan"]:
            if key in form:
                values[key] = "on"
    return values


def render_input(name: str, label: str, values: dict[str, str], width: str = "") -> str:
    style = f" style=\"{esc(width)}\"" if width else ""
    return f"""
    <label class="field">
      <span>{esc(label)}</span>
      <input name="{esc(name)}" value="{esc(values.get(name, ''))}"{style}>
    </label>
    """


def render_checkbox(name: str, label: str, values: dict[str, str]) -> str:
    checked = " checked" if values.get(name) == "on" else ""
    return f"""
    <label class="check"><input type="checkbox" name="{esc(name)}"{checked}> {esc(label)}</label>
    """


def render_result(result: RunResult) -> str:
    status = "ok" if result.ok else "bad"
    blocks = [
        f"<section class=\"result {status}\">",
        f"<h2>{esc(result.title)} <span class=\"badge\">exit {result.returncode}</span></h2>",
        f"<p class=\"muted\">Started {esc(result.started_at)} · Finished {esc(result.finished_at)} · cwd: <code>{esc(result.cwd)}</code></p>",
        "<h3>Command</h3>",
        f"<pre>{esc(shell_join(result.command))}</pre>",
    ]
    if result.stdout:
        blocks.extend(["<h3>stdout</h3>", f"<pre>{esc(result.stdout)}</pre>"])
    if result.stderr:
        blocks.extend(["<h3>stderr</h3>", f"<pre>{esc(result.stderr)}</pre>"])

    existing_outputs = [path for path in result.output_paths if path.exists()]
    if existing_outputs:
        blocks.append("<h3>Generated files</h3><ul>")
        for path in existing_outputs:
            blocks.append(f"<li><code>{esc(path)}</code></li>")
        blocks.append("</ul>")
        preview = read_preview(existing_outputs[0])
        if preview:
            blocks.extend(["<h3>Preview</h3>", f"<pre class=\"markdown-preview\">{esc(preview)}</pre>"])
    elif result.output_paths:
        blocks.append("<h3>Expected output files</h3><ul>")
        for path in result.output_paths:
            blocks.append(f"<li><code>{esc(path)}</code> <span class=\"muted\">not found</span></li>")
        blocks.append("</ul>")
    blocks.append("</section>")
    return "\n".join(blocks)


def render_page(default_vault_path: Path, form: dict[str, list[str]] | None = None, results: list[RunResult] | None = None) -> str:
    values = merged_form(default_vault_path, form)
    result_html = "".join(render_result(item) for item in (results or []))
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LIF-Memory Debug UI</title>
<style>
:root {{
  --bg: #0f172a;
  --panel: #111827;
  --card: #1f2937;
  --text: #e5e7eb;
  --muted: #9ca3af;
  --line: #374151;
  --accent: #93c5fd;
  --good: #10b981;
  --bad: #ef4444;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--text); }}
main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; }}
h2 {{ margin: 0 0 14px; font-size: 20px; }}
h3 {{ margin: 18px 0 8px; font-size: 15px; color: var(--accent); }}
p {{ line-height: 1.6; }}
code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
pre {{ white-space: pre-wrap; word-break: break-word; background: #020617; border: 1px solid var(--line); border-radius: 12px; padding: 14px; overflow: auto; }}
.header {{ display: flex; justify-content: space-between; gap: 18px; align-items: flex-start; margin-bottom: 20px; }}
.version {{ color: var(--muted); font-size: 13px; }}
.panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 18px; padding: 18px; margin: 18px 0; box-shadow: 0 10px 30px rgba(0,0,0,.22); }}
.grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
.grid.two {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
.field {{ display: flex; flex-direction: column; gap: 6px; color: var(--muted); font-size: 13px; }}
.field input, select {{ width: 100%; border: 1px solid var(--line); background: #020617; color: var(--text); border-radius: 10px; padding: 10px 12px; font-size: 14px; }}
.check {{ display: inline-flex; gap: 8px; align-items: center; margin-right: 18px; color: var(--muted); }}
.actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 16px; }}
button {{ border: 1px solid #60a5fa; background: #1d4ed8; color: white; padding: 10px 14px; border-radius: 12px; cursor: pointer; font-weight: 650; }}
button.secondary {{ background: #334155; border-color: #64748b; }}
button.warn {{ background: #92400e; border-color: #f59e0b; }}
.pipeline {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
.node {{ background: var(--card); border: 1px solid var(--line); padding: 14px; border-radius: 14px; min-height: 130px; }}
.node b {{ display: block; margin-bottom: 8px; color: white; }}
.node span {{ color: var(--muted); font-size: 13px; line-height: 1.55; }}
.result {{ border: 1px solid var(--line); border-radius: 18px; padding: 18px; margin: 18px 0; background: var(--panel); }}
.result.ok {{ border-color: rgba(16,185,129,.55); }}
.result.bad {{ border-color: rgba(239,68,68,.65); }}
.badge {{ display: inline-block; margin-left: 8px; padding: 3px 8px; border-radius: 999px; font-size: 12px; background: #020617; color: var(--muted); border: 1px solid var(--line); }}
.muted {{ color: var(--muted); }}
.markdown-preview {{ max-height: 520px; }}
@media (max-width: 900px) {{ .grid, .grid.two, .pipeline {{ grid-template-columns: 1fr; }} .header {{ flex-direction: column; }} }}
</style>
</head>
<body>
<main>
  <div class="header">
    <div>
      <h1>LIF-Memory Debug UI</h1>
      <p class="muted">本地可视化调试界面：用按钮调用现有 CLI，不改动原有主流程。</p>
    </div>
    <div class="version">UI {APP_VERSION}<br>Python {esc(sys.version.split()[0])}</div>
  </div>

  <section class="panel">
    <h2>系统链路</h2>
    <div class="pipeline">
      <div class="node"><b>Memory Source</b><span>Obsidian 日记、项目笔记、wikilink、folder、tag。原文不被压缩。</span></div>
      <div class="node"><b>Evidence Sensor</b><span>片段 -> EvidenceVector。行动性、紧迫性、阻塞、完成、具体性、新颖性。</span></div>
      <div class="node"><b>LIF Dynamics</b><span>V_fast / V_slow 累积、泄漏、抑制、阈值触发、冷却。</span></div>
      <div class="node"><b>Spike Interface</b><span>输出今日主卡、完整回放、图谱报告、洞察报告、v1 收束报告。</span></div>
    </div>
  </section>

  <form method="post" class="panel">
    <h2>运行参数</h2>
    <div class="grid">
      {render_input('vault', 'Obsidian Vault 路径', values)}
      {render_input('days', '回放天数', values)}
      {render_input('top_k', 'Daily top-k', values)}
      {render_input('llm_provider', 'LLM provider', values)}
    </div>
    <p>
      {render_checkbox('completion_scan', '启用 completion-scan', values)}
      {render_checkbox('llm_review', '启用 LLM review', values)}
    </p>

    <h3>输出文件</h3>
    <div class="grid two">
      {render_input('daily_output', '今日主卡输出', values)}
      {render_input('closure_file', '闭环反馈读取文件', values)}
      {render_input('replay_output', '完整回放 Markdown', values)}
      {render_input('replay_json', '完整回放 JSON', values)}
      {render_input('graph_output', '图谱报告 Markdown', values)}
      {render_input('graph_json', '图谱报告 JSON', values)}
      {render_input('insight_output', '洞察报告 Markdown', values)}
      {render_input('insight_json', '洞察报告 JSON', values)}
      {render_input('v1_output', 'v1 收束报告 Markdown', values)}
      {render_input('v1_json', 'v1 收束报告 JSON', values)}
    </div>

    <h3>Insight 设置</h3>
    <div class="grid two">
      {render_input('insight_profile', 'profile', values)}
      {render_input('insight_sensitivity', 'sensitivity，可留空', values)}
    </div>

    <div class="actions">
      <button name="action" value="daily_preview" class="secondary">预览今日主卡</button>
      <button name="action" value="daily_write">生成今日主卡</button>
      <button name="action" value="replay">生成完整回放</button>
      <button name="action" value="graph">生成图谱报告</button>
      <button name="action" value="insight">生成洞察报告</button>
      <button name="action" value="v1_report">生成 v1 收束报告</button>
      <button name="action" value="all_reports" class="warn">一键生成全部报告</button>
    </div>
  </form>

  {result_html}
</main>
</body>
</html>"""


class UIHandler(BaseHTTPRequestHandler):
    default_vault_path = default_vault()

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("[lif-memory-ui] " + format % args + "\n")

    def send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - stdlib API
        self.send_html(render_page(self.default_vault_path))

    def do_POST(self) -> None:  # noqa: N802 - stdlib API
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(raw, keep_blank_values=True)
        action = field(form, "action", "")
        results = run_action(action, form)
        self.send_html(render_page(self.default_vault_path, form=form, results=results))


def main() -> None:
    args = parse_args()
    UIHandler.default_vault_path = args.vault.expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), UIHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"LIF-Memory Debug UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping LIF-Memory Debug UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
