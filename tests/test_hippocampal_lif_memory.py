from __future__ import annotations

import unittest
from datetime import date

from hippocampal_lif_memory import (
    HippocampalConfig,
    HippocampalLIFNetwork,
    build_hippocampal_lif_memory,
)
from unsupervised_memory_field import extract_observations


class HippocampalLIFMemoryTests(unittest.TestCase):
    def test_dg_produces_sparse_separated_codes(self) -> None:
        config = HippocampalConfig(dg_units=128, ca3_units=96, dg_top_k=8, ca3_top_k=10)
        network = HippocampalLIFNetwork(config)
        observations = extract_observations(
            {
                "2026-06-20.md": "导师质疑论文创新点，需要解释 LIF 后向散射最小闭环。",
                "2026-06-21.md": "导师催实验室考勤和搬宿舍，需要处理毕业流程。",
            },
            fallback_day=date(2026, 6, 21),
        )
        codes = [network.dentate_gyrus(item) for item in observations]

        self.assertLessEqual(len(codes[0].active_units), 8)
        self.assertLessEqual(len(codes[1].active_units), 8)
        self.assertLess(codes[0].overlap(codes[1]), 0.45)

    def test_ca3_recurrent_association_supports_partial_cue_completion(self) -> None:
        notes = {
            "2026-06-20.md": "导师质疑创新点，需要把 LIF 后向散射和事件化讲成论文主张。",
            "2026-06-21.md": "LIF 后向散射最小闭环需要补 SSVEP、事件率和无线恢复数据。",
            "2026-06-22.md": "写论文时要把事件化、低功耗、无 ADC 和后向散射链路放到同一个创新闭环。",
        }
        result = build_hippocampal_lif_memory(
            notes,
            config=HippocampalConfig(dg_units=192, ca3_units=128, dg_top_k=10, ca3_top_k=12, ca1_match_threshold=0.12),
            fallback_day=date(2026, 6, 22),
            probe_text="老师又问 LIF 后向散射创新点怎么证明",
        )

        self.assertIsNotNone(result.probe_result)
        assert result.probe_result is not None
        self.assertTrue(result.probe_result.completed)
        recalled = " ".join(result.probe_result.recalled_terms)
        self.assertTrue("lif" in recalled.lower() or "后向" in recalled or "散射" in recalled)
        self.assertGreaterEqual(result.probe_result.ca3.completion_gain, 0)

    def test_ca1_write_update_and_cortex_trace_storage_are_present(self) -> None:
        notes = {
            "2026-06-20.md": "LIF-Memory 不能只是触发器，需要 DG CA3 CA1 海马体结构。",
            "2026-06-21.md": "LIF-Memory 需要递归联想、稀疏分离、读写判断和长期记忆。",
            "2026-06-22.md": "求职简历要强调 AI 加嵌入式和低功耗系统。",
        }
        result = build_hippocampal_lif_memory(
            notes,
            config=HippocampalConfig(dg_units=192, ca3_units=128, dg_top_k=10, ca3_top_k=12, ca1_match_threshold=0.28),
            fallback_day=date(2026, 6, 22),
        )

        modes = [step.ca1.mode for step in result.steps]
        self.assertIn("write_new", modes)
        self.assertGreaterEqual(result.metrics.trace_count, 2)
        self.assertGreaterEqual(sum(trace.write_count for trace in result.cortex), len(result.steps))
        self.assertTrue(any(trace.evidence for trace in result.cortex))


if __name__ == "__main__":
    unittest.main()
