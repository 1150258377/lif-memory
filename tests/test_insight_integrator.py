from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import insight_integrator as insight  # noqa: E402


class InsightIntegratorTests(unittest.TestCase):
    def test_weak_fragments_integrate_into_insight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / ".obsidian").mkdir()
            note1 = vault / "2026-06-10.md"
            note2 = vault / "2026-06-11.md"
            note3 = vault / "2026-06-12.md"
            note1.write_text("SSVEP 后向散射恢复成功，但是高斯 sigma 影响结果。", encoding="utf-8")
            note2.write_text("LIF 事件率可以作为压缩后的 EEG 节律表征。", encoding="utf-8")
            note3.write_text("我怀疑创新点是不是只是事件和后向散射的拼凑。", encoding="utf-8")
            notes = [
                (date(2026, 6, 10), note1),
                (date(2026, 6, 11), note2),
                (date(2026, 6, 12), note3),
            ]
            question = insight.LatentQuestion(
                theta=4.0,
                decay=0.90,
                reset_ratio=0.30,
                cooldown_days=1,
                evidence_cap=10.0,
                keywords=["SSVEP", "后向散射", "LIF", "事件率", "EEG", "创新点", "拼凑"],
                conflict_words=["怀疑", "拼凑"],
                completion_words=["完成"],
                emergent_insight="integrated insight",
                next_validation_action="next action",
            )
            spikes, _, _ = insight.replay_insights(
                notes,
                {"Innovation_Claim": question},
                daily_insight_budget=2,
                min_fragments=3,
            )

        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0].question, "Innovation_Claim")
        self.assertGreaterEqual(len(spikes[0].fragments), 3)
        self.assertTrue(any(fragment.role == "conflict" for fragment in spikes[0].fragments))

    def test_single_fragment_does_not_spike_even_if_voltage_is_high(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / ".obsidian").mkdir()
            note = vault / "2026-06-10.md"
            note.write_text("创新点 后向散射 LIF EEG SSVEP ADC 物理层。", encoding="utf-8")
            question = insight.LatentQuestion(
                theta=1.0,
                decay=0.90,
                reset_ratio=0.30,
                cooldown_days=1,
                evidence_cap=10.0,
                keywords=["创新点", "后向散射", "LIF", "EEG", "SSVEP", "ADC", "物理层"],
                conflict_words=[],
                completion_words=[],
                emergent_insight="integrated insight",
                next_validation_action="next action",
            )
            spikes, _, states = insight.replay_insights(
                [(date(2026, 6, 10), note)],
                {"Innovation_Claim": question},
                daily_insight_budget=2,
                min_fragments=2,
            )

        self.assertEqual(spikes, [])
        self.assertGreater(states["Innovation_Claim"].v, question.theta)

    def test_packet_contains_integrated_fragments(self) -> None:
        fragment = insight.Fragment(
            day=date(2026, 6, 12),
            path=Path("2026-06-12.md"),
            snippet="创新点不是拼凑。",
            score=2.0,
            matched_keywords=["创新点", "拼凑"],
            role="conflict",
        )
        spike = insight.InsightSpike(
            day=date(2026, 6, 12),
            question="Innovation_Claim",
            voltage=8.2,
            threshold=8.0,
            fragments=[fragment],
            emergent_insight="insight",
            next_validation_action="action",
        )
        packet = insight.insight_packet(spike, Path.cwd())
        self.assertEqual(packet["spike_type"], "Insight")
        self.assertEqual(packet["latent_question"], "Innovation_Claim")
        self.assertEqual(len(packet["integrated_fragments"]), 1)
        self.assertEqual(packet["insight_type"], "emergent_claim")
        self.assertEqual(packet["thinking_policy"], "write_claim")

    def test_economics_profile_can_be_selected(self) -> None:
        profile = insight.select_profile("economics")
        questions = insight.select_questions(None, profile)

        self.assertEqual(profile.title, "经济学思想洞察")
        self.assertIn("Debt_Finance", questions)
        self.assertIn("Inflation_Rate", questions)

    def test_economics_fragments_integrate_into_debt_insight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / ".obsidian").mkdir()
            note1 = vault / "2026-06-10.md"
            note2 = vault / "2026-06-11.md"
            note3 = vault / "2026-06-12.md"
            note1.write_text("地产债务和银行资产负债表压力开始影响融资。", encoding="utf-8")
            note2.write_text("高杠杆下现金流断裂会让信用继续收缩。", encoding="utf-8")
            note3.write_text("我不理解为什么资产价格下跌会放大违约风险。", encoding="utf-8")
            notes = [
                (date(2026, 6, 10), note1),
                (date(2026, 6, 11), note2),
                (date(2026, 6, 12), note3),
            ]
            question = replace(insight.ECONOMICS_QUESTIONS["Debt_Finance"], theta=4.0)
            spikes, _, _ = insight.replay_insights(
                notes,
                {"Debt_Finance": question},
                daily_insight_budget=2,
                min_fragments=3,
            )

        self.assertEqual(len(spikes), 1)
        self.assertEqual(spikes[0].question, "Debt_Finance")
        self.assertEqual(spikes[0].insight_type, "balance_sheet_pressure")
        self.assertEqual(spikes[0].thinking_policy, "collect_evidence")
        self.assertIn("资产负债表", spikes[0].emergent_insight)

    def test_select_questions_uses_profile_scope(self) -> None:
        profile = insight.select_profile("economics")
        selected = insight.select_questions("Debt_Finance,Market_Psychology", profile)

        self.assertEqual(list(selected), ["Debt_Finance", "Market_Psychology"])

    def test_exploratory_sensitivity_lowers_threshold(self) -> None:
        profile = insight.select_profile("economics")
        normal = insight.select_questions("Market_Psychology", profile)
        exploratory = insight.apply_sensitivity(normal, "exploratory")

        self.assertLess(exploratory["Market_Psychology"].theta, normal["Market_Psychology"].theta)
        self.assertGreater(exploratory["Market_Psychology"].evidence_cap, normal["Market_Psychology"].evidence_cap)

    def test_ascii_acronym_matching_uses_word_boundaries(self) -> None:
        hits = insight.matched_words("happiness suppose application", ["PPI", "CPI"])

        self.assertEqual(hits, [])
        self.assertEqual(insight.matched_words("CPI and PPI are both macro indicators", ["PPI", "CPI"]), ["PPI", "CPI"])


if __name__ == "__main__":
    unittest.main()
