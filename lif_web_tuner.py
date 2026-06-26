from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import continuous_problem_field as cpf
from aha_engine import build_aha_cards
from hippocampal_bridge import (
    build_recall_from_vault,
    hippocampal_field_hits,
    inject_recall_current,
    render_recall_markdown,
)
from unsupervised_memory_field import reconstruct_unsupervised_memory_field

VERSION = "0.1.0-web-hippocampal-integration"


@dataclass(frozen=True)
class WebPipelineParams:
    vault: str
    query: str
    days: int = 30
    today: str = ""
    enable_hippocampus: bool = True
    top_k: int = 8
    max_notes: int = 600
    time_sigma: float = 30.0
    semantic_sigma: float = 0.55
    graph_steps: int = 2
    graph_alpha: float = 0.35
    threshold: float = 5.0
    slots: int = 8


def parse_today(value: str | None) -> date:
    if not value:
        return date.today()
    return datetime.strptime(value, "%Y-%m-%d").date()


def safe_vault(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Vault path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Vault path is not a directory: {path}")
    return path


def render_aha_cards(cards: list[Any]) -> str:
    lines = ["# AhaEngine 串联输出", ""]
    if not cards:
        lines.append("- 暂无 Aha card。")
        return "\n".join(lines).rstrip() + "\n"

    for idx, card in enumerate(cards, start=1):
        packet = card.to_packet()
        lines.extend(
            [
                f"## Card {idx}: {packet.get('topic', 'unknown')}",
                "",
                f"- Trigger: `{packet.get('trigger_type')}`",
                f"- Pressure ratio: `{packet.get('pressure_ratio')}`",
                f"- Reconstruction pressure: `{packet.get('reconstruction_pressure')}`",
                f"- Quality: `{packet.get('quality_score')}`",
                "",
                "### Old model",
                str(packet.get("old_model", "")),
                "",
                "### Contradiction",
                str(packet.get("contradiction", "")),
                "",
                "### New model",
                str(packet.get("new_model", "")),
                "",
                "### Essence",
                str(packet.get("essence", "")),
                "",
                "### Action delta",
                str(packet.get("action_delta", "")),
                "",
                "### Falsification test",
                str(packet.get("falsification_test", "")),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def field_result_packet_with_hippocampus(result: cpf.FieldResult, recall_packet: dict[str, Any], recall_hits: list[dict[str, Any]]) -> dict[str, Any]:
    packet = result.to_packet()
    packet["hippocampal_recall"] = recall_packet
    packet.setdefault("top_hits", [])
    if isinstance(packet["top_hits"], list):
        packet["top_hits"] = list(packet["top_hits"]) + recall_hits
    return packet


def run_pipeline(params: WebPipelineParams) -> tuple[str, str, str, str, str]:
    today = parse_today(params.today.strip() or None)
    vault = safe_vault(params.vault)
    query = params.query.strip()
    if not query:
        raise ValueError("Query is required.")

    notes = cpf.read_notes(vault, cutoff=today, days=max(1, int(params.days)), max_notes=max(1, int(params.max_notes)))
    if not notes:
        raise ValueError("No Markdown notes were found in the selected vault/time window.")

    cpf.diffuse_graph(notes, steps=int(params.graph_steps), alpha=float(params.graph_alpha))
    hits, field_energy, daily_current, daily_completion = cpf.reconstruct_field(
        notes,
        query=query,
        today=today,
        top_k=max(1, int(params.top_k)),
        time_sigma=float(params.time_sigma),
        semantic_sigma=float(params.semantic_sigma),
    )

    recall = build_recall_from_vault(
        vault=vault,
        query=query,
        days=max(1, int(params.days)),
        today=today,
        enabled=bool(params.enable_hippocampus),
    )
    inject_recall_current(daily_current, today, recall)

    start = today - timedelta(days=max(1, int(params.days)))
    trajectory = cpf.lif_iterate(
        daily_current,
        daily_completion,
        start=start,
        today=today,
        threshold=float(params.threshold),
    )
    spike = any(step.spiked for step in trajectory[-7:]) if trajectory else False
    latest_v = trajectory[-1].v if trajectory else 0.0
    reconstruction_pressure = latest_v / float(params.threshold) if float(params.threshold) > 0 else 0.0
    field_result = cpf.FieldResult(
        query=query,
        topic=cpf.infer_topic(query),
        today=today,
        field_energy=field_energy + recall.current_boost,
        reconstruction_pressure=reconstruction_pressure,
        hits=hits,
        trajectory=trajectory,
        spike=spike,
        insight_card=cpf.make_insight_card(query, cpf.infer_topic(query), hits, trajectory, float(params.threshold)),
    )

    recall_hits = hippocampal_field_hits(recall)
    field_packet = field_result_packet_with_hippocampus(field_result, recall.to_packet(), recall_hits)

    notes_map = {note.rel_path: note.text for note in notes}
    unsupervised_result = reconstruct_unsupervised_memory_field(
        notes_map,
        slot_count=max(1, int(params.slots)),
        fallback_day=today,
    )
    unsupervised_packet = unsupervised_result.to_packet()

    aha_cards = build_aha_cards(
        query=query,
        spike_data=field_packet,
        reconstruction_data=unsupervised_packet,
        field_data=field_packet,
        top_k=3,
        args=None,
    )

    summary_lines = [
        "# LIF-Memory Web Pipeline",
        "",
        f"- Version: `{VERSION}`",
        f"- Notes scanned: `{len(notes)}`",
        f"- Query topic: `{field_result.topic}`",
        f"- Field energy: `{field_result.field_energy:.4f}`",
        f"- Latest V/theta: `{latest_v:.3f}` / `{float(params.threshold):.3f}`",
        f"- Spike in last 7 days: `{field_result.spike}`",
        f"- Hippocampal boost: `{recall.current_boost:.4f}`",
        f"- Hippocampal trace: `{recall.trace_id}`",
        "",
        "## 连续链路",
        "",
        "```text",
        "Obsidian notes -> continuous field -> hippocampal recall -> LIF voltage -> AhaEngine",
        "```",
    ]

    field_markdown = cpf.render_markdown(
        field_result,
        threshold=float(params.threshold),
        time_sigma=float(params.time_sigma),
        semantic_sigma=float(params.semantic_sigma),
    )
    if recall_hits:
        field_markdown += "\n## Hippocampal evidence injected into field\n\n"
        for item in recall_hits:
            field_markdown += f"- score=`{item['score']}` trace=`{item['path']}` {item['snippet']}\n"

    output_packet = {
        "version": VERSION,
        "params": asdict(params),
        "summary": {
            "notes_scanned": len(notes),
            "latest_v": latest_v,
            "threshold": float(params.threshold),
            "spike": field_result.spike,
        },
        "field": field_packet,
        "hippocampal_recall": recall.to_packet(),
        "unsupervised_memory_field": unsupervised_packet,
        "aha_cards": [card.to_packet() for card in aha_cards],
    }

    return (
        "\n".join(summary_lines).rstrip() + "\n",
        render_recall_markdown(recall),
        field_markdown,
        render_aha_cards(aha_cards),
        json.dumps(output_packet, ensure_ascii=False, indent=2),
    )


def run_web_pipeline(
    vault: str,
    query: str,
    days: int,
    today: str,
    enable_hippocampus: bool,
    top_k: int,
    max_notes: int,
    time_sigma: float,
    semantic_sigma: float,
    graph_steps: int,
    graph_alpha: float,
    threshold: float,
    slots: int,
) -> tuple[str, str, str, str, str]:
    try:
        return run_pipeline(
            WebPipelineParams(
                vault=vault,
                query=query,
                days=int(days),
                today=today,
                enable_hippampus=enable_hippocampus,  # type: ignore[call-arg]
            )
        )
    except TypeError:
        # Keep Gradio callback compatibility if a stale browser sends the old
        # field name. The actual path below uses the correct dataclass field.
        pass

    try:
        params = WebPipelineParams(
            vault=vault,
            query=query,
            days=int(days),
            today=today,
            enable_hippocampus=bool(enable_hippocampus),
            top_k=int(top_k),
            max_notes=int(max_notes),
            time_sigma=float(time_sigma),
            semantic_sigma=float(semantic_sigma),
            graph_steps=int(graph_steps),
            graph_alpha=float(graph_alpha),
            threshold=float(threshold),
            slots=int(slots),
        )
        return run_pipeline(params)
    except Exception as exc:
        message = f"# Error\n\n```text\n{type(exc).__name__}: {exc}\n```\n"
        return message, message, message, message, json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False, indent=2)


def build_app() -> Any:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit("Missing dependency: gradio. Install it with `pip install gradio`.") from exc

    with gr.Blocks(title="LIF-Memory Web Tuner") as demo:
        gr.Markdown("# LIF-Memory Web Tuner\n\n连续问题场 + 海马体召回 + LIF spike + AhaEngine。")
        with gr.Row():
            vault = gr.Textbox(label="Obsidian vault path", placeholder="C:\\path\\to\\your\\obsidian-vault")
            query = gr.Textbox(label="Query / 当前问题", value="LIF Memory 为什么不够灵敏，海马体应该如何参与？")
        with gr.Row():
            days = gr.Slider(label="Days", minimum=1, maximum=180, value=30, step=1)
            today = gr.Textbox(label="Today override YYYY-MM-DD", value="")
            enable_hippocampus = gr.Checkbox(label="Enable hippocampal recall", value=True)
        with gr.Accordion("Advanced parameters", open=False):
            with gr.Row():
                top_k = gr.Slider(label="Top K evidence", minimum=1, maximum=20, value=8, step=1)
                max_notes = gr.Slider(label="Max notes", minimum=50, maximum=2000, value=600, step=50)
                slots = gr.Slider(label="Latent slots", minimum=2, maximum=32, value=8, step=1)
            with gr.Row():
                time_sigma = gr.Slider(label="Time sigma", minimum=1.0, maximum=120.0, value=30.0, step=1.0)
                semantic_sigma = gr.Slider(label="Semantic sigma", minimum=0.05, maximum=1.5, value=0.55, step=0.05)
                threshold = gr.Slider(label="LIF threshold", minimum=0.5, maximum=12.0, value=5.0, step=0.1)
            with gr.Row():
                graph_steps = gr.Slider(label="Graph steps", minimum=0, maximum=5, value=2, step=1)
                graph_alpha = gr.Slider(label="Graph alpha", minimum=0.0, maximum=0.95, value=0.35, step=0.05)
        run = gr.Button("Run integrated memory pipeline", variant="primary")
        with gr.Tab("Summary"):
            summary = gr.Markdown()
        with gr.Tab("Hippocampus"):
            hippocampus = gr.Markdown()
        with gr.Tab("Field"):
            field = gr.Markdown()
        with gr.Tab("Aha"):
            aha = gr.Markdown()
        with gr.Tab("JSON"):
            packet = gr.Code(language="json")

        run.click(
            fn=run_web_pipeline,
            inputs=[
                vault,
                query,
                days,
                today,
                enable_hippocampus,
                top_k,
                max_notes,
                time_sigma,
                semantic_sigma,
                graph_steps,
                graph_alpha,
                threshold,
                slots,
            ],
            outputs=[summary, hippocampus, field, aha, packet],
        )
    return demo


def main() -> None:
    build_app().launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
