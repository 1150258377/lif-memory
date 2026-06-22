from __future__ import annotations

import unittest
from datetime import date

from memory_field import reconstruct_memory_field


class MemoryFieldReconstructionTests(unittest.TestCase):
    def test_reconstruction_detects_experiment_and_thesis_loop(self) -> None:
        notes = {
            "2026-06-20.md": "今天完成了 SSVEP 频谱图和事件率整理，但是 LIF 后向散射链路还缺一组可写进论文第四章的数据。",
            "2026-06-21.md": "接下来需要固定 KS1092 后级输入条件，测试 50 欧姆负载、偏置和 LIF 输出事件率，并保存波形截图。",
        }

        result = reconstruct_memory_field(notes, fallback_day=date(2026, 6, 22))

        self.assertIn(result.dominant_state, {"experiment_loop", "thesis_loop"})
        self.assertTrue(result.latent_cells["experiment_loop"].evidence)
        self.assertTrue(result.latent_cells["thesis_loop"].evidence)
        self.assertIn("thesis", result.rendered_views)
        self.assertTrue(result.rendered_views["thesis"].support)

    def test_consistency_reports_missing_observations_for_sparse_view(self) -> None:
        notes = {
            "2026-06-22.md": "我现在延毕压力很大，感觉难受、焦虑、动不了，需要先恢复。",
        }

        result = reconstruct_memory_field(notes, views=["experiment", "emotion"], fallback_day=date(2026, 6, 22))

        experiment_view = result.rendered_views["experiment"]
        emotion_view = result.rendered_views["emotion"]

        self.assertTrue(experiment_view.missing_observations)
        self.assertGreaterEqual(emotion_view.confidence, experiment_view.confidence)
        self.assertTrue(result.latent_cells["emotion_load"].evidence)


if __name__ == "__main__":
    unittest.main()
