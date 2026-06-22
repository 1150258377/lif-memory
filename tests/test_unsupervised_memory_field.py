from __future__ import annotations

import unittest
from datetime import date

from unsupervised_memory_field import extract_observations, reconstruct_unsupervised_memory_field


class UnsupervisedMemoryFieldTests(unittest.TestCase):
    def test_slots_are_data_driven_not_predefined_topics(self) -> None:
        notes = {
            "2026-06-20.md": "今天完成了 SSVEP 频谱图，但是 LIF 后向散射链路还缺一组可写进论文的数据。",
            "2026-06-21.md": "接下来测试 KS1092 后级输入条件，记录 LIF 输出事件率和波形截图。",
            "2026-06-22.md": "我现在延毕压力很大，感觉难受、焦虑、动不了，需要先恢复。",
            "2026-06-23.md": "求职方面要把 AI 加嵌入式项目写进简历，并投递岗位。",
        }

        result = reconstruct_unsupervised_memory_field(
            notes,
            slot_count=6,
            epochs=2,
            fallback_day=date(2026, 6, 24),
        )

        active_slots = result.active_slots
        self.assertGreaterEqual(len(active_slots), 3)
        self.assertGreater(result.global_reconstruction_loss, 0.0)
        self.assertTrue(any("LIF" in item.text or "SSVEP" in item.text for item in active_slots[0].evidence))

    def test_high_loss_observations_surface_unexplained_fragments(self) -> None:
        notes = {
            "2026-06-20.md": "LIF 后向散射链路需要测试事件率。",
            "2026-06-21.md": "LIF 输出波形需要保存截图。",
            "2026-06-22.md": "突然出现一个完全不同的话题：宏观经济通缩、人民币汇率和出口之间的矛盾。",
        }

        result = reconstruct_unsupervised_memory_field(
            notes,
            slot_count=2,
            epochs=1,
            min_similarity_for_existing=0.05,
            fallback_day=date(2026, 6, 22),
        )

        high_loss_text = " ".join(item.observation.text for item in result.high_loss_observations)
        self.assertIn("宏观经济", high_loss_text)
        self.assertTrue(result.high_loss_observations)

    def test_observations_use_primitive_features_only(self) -> None:
        observations = extract_observations(
            {"2026-06-22.md": "接下来需要测试 50 mV 输入，并记录 LIF 输出事件率。"},
            fallback_day=date(2026, 6, 22),
        )

        self.assertEqual(len(observations), 1)
        packet = observations[0].to_packet()
        self.assertIn("features", packet)
        self.assertNotIn("topic", packet)
        self.assertGreater(observations[0].features.actionability, 0.0)


if __name__ == "__main__":
    unittest.main()
