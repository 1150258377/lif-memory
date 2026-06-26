from __future__ import annotations

import unittest
from datetime import date

from hippocampal_bridge import (
    build_recall_from_notes,
    hippocampal_field_hits,
    inject_recall_current,
)
from hippocampal_lif_memory import HippocampalConfig


class HippocampalBridgeTests(unittest.TestCase):
    def test_recall_packet_exposes_current_boost_and_evidence(self) -> None:
        notes = {
            "2026-06-20.md": "导师质疑创新点，需要把 LIF 后向散射和事件化讲成论文主张。",
            "2026-06-21.md": "LIF 后向散射最小闭环需要补 SSVEP、事件率和无线恢复数据。",
            "2026-06-22.md": "写论文时要把事件化、低功耗、无 ADC 和后向散射链路放到同一个创新闭环。",
        }
        recall = build_recall_from_notes(
            notes,
            query="老师又问 LIF 后向散射创新点怎么证明",
            today=date(2026, 6, 22),
            config=HippocampalConfig(dg_units=192, ca3_units=128, dg_top_k=10, ca3_top_k=12, ca1_match_threshold=0.12),
        )

        self.assertTrue(recall.enabled)
        self.assertGreaterEqual(recall.similarity, 0.0)
        self.assertGreaterEqual(recall.current_boost, 0.0)
        self.assertIn("trace_count", recall.metrics)
        self.assertIsInstance(recall.recalled_terms, list)

    def test_bridge_can_inject_hippocampal_current_into_lif_day(self) -> None:
        notes = {
            "2026-06-20.md": "LIF-Memory 需要 DG CA3 CA1 海马体结构。",
            "2026-06-21.md": "海马体结构应该参与网页端连续问题场和 AhaEngine。",
        }
        today = date(2026, 6, 21)
        recall = build_recall_from_notes(notes, query="海马体如何参与网页端？", today=today)
        daily_current = {today: 1.0}
        inject_recall_current(daily_current, today, recall)

        self.assertGreaterEqual(daily_current[today], 1.0)
        hits = hippocampal_field_hits(recall)
        self.assertIsInstance(hits, list)


if __name__ == "__main__":
    unittest.main()
