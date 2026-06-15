from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lif_memory as core  # noqa: E402
import lif_memory_stateful as stateful  # noqa: E402
import llm_adapter  # noqa: E402


class StatefulRunnerTests(unittest.TestCase):
    def test_incremental_filter_skips_processed_notes(self) -> None:
        notes = [
            (date(2026, 6, 10), Path("2026-06-10.md")),
            (date(2026, 6, 11), Path("2026-06-11.md")),
        ]
        filtered = stateful.filter_incremental_notes(notes, date(2026, 6, 10), replay_all=False)
        self.assertEqual([day for day, _ in filtered], [date(2026, 6, 11)])

    def test_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_state.json"
            states = {"Experiment": core.NeuronState(v=2.5, last_spike_date=date(2026, 6, 10))}
            stateful.save_state(path, states, [(date(2026, 6, 11), Path("2026-06-11.md"))], None)
            loaded, last_processed, topic_history, topic_policies = stateful.load_state(path, {"Experiment": core.NEURONS["Experiment"]}, reset=False)

        self.assertEqual(last_processed, date(2026, 6, 11))
        self.assertAlmostEqual(loaded["Experiment"].v, 2.5)
        self.assertEqual(loaded["Experiment"].last_spike_date, date(2026, 6, 10))
        self.assertEqual(topic_history, {})
        self.assertEqual(topic_policies, {})

    def test_topic_history_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_state.json"
            states = {"Experiment": core.NeuronState(v=2.5)}
            topic_history = {
                "负阻": core.TopicHistory(
                    days_seen={date(2026, 6, 13), date(2026, 6, 14)},
                    completion_count=0,
                    blocker_count=3,
                    evidence_count=4,
                    last_action_policy="isolate",
                )
            }
            stateful.save_state(
                path,
                states,
                [(date(2026, 6, 14), Path("2026-06-14.md"))],
                None,
                topic_history,
            )
            _, _, loaded_topics, _ = stateful.load_state(path, {"Experiment": core.NEURONS["Experiment"]}, reset=False)

        self.assertIn("负阻", loaded_topics)
        self.assertEqual(loaded_topics["负阻"].days_seen, {date(2026, 6, 13), date(2026, 6, 14)})
        self.assertEqual(loaded_topics["负阻"].blocker_count, 3)
        self.assertEqual(loaded_topics["负阻"].last_action_policy, "isolate")

    def test_timescale_voltage_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_state.json"
            states = {"Experiment": core.NeuronState(v=3.5, v_fast=2.0, v_slow=7.0)}
            stateful.save_state(path, states, [], None)
            loaded, _, _, _ = stateful.load_state(path, {"Experiment": core.NEURONS["Experiment"]}, reset=False)

        self.assertAlmostEqual(loaded["Experiment"].v, 3.5)
        self.assertAlmostEqual(loaded["Experiment"].v_fast, 2.0)
        self.assertAlmostEqual(loaded["Experiment"].v_slow, 7.0)

    def test_topic_policy_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_state.json"
            policies = {
                "负阻": core.TopicPolicy(
                    threshold_delta=1.0,
                    priority_override="P2",
                    action_policy_override="stop",
                    muted=True,
                    cooldown_days=30,
                    feedback_count=1,
                    last_feedback="不要再提醒",
                )
            }
            stateful.save_state(path, {"Experiment": core.NeuronState()}, [], None, topic_policies=policies)
            _, _, _, loaded_policies = stateful.load_state(path, {"Experiment": core.NEURONS["Experiment"]}, reset=False)

        self.assertIn("负阻", loaded_policies)
        self.assertTrue(loaded_policies["负阻"].muted)
        self.assertEqual(loaded_policies["负阻"].priority_override, "P2")
        self.assertEqual(loaded_policies["负阻"].last_feedback, "不要再提醒")

    def test_config_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            stateful.write_default_config(path)
            neurons = stateful.load_neurons(path)
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("states", data)
        self.assertIn("Experiment", neurons)
        self.assertGreater(neurons["Experiment"].theta, 0)

    def test_replay_with_state_uses_existing_voltage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / ".obsidian").mkdir()
            note = vault / "2026-06-11.md"
            note.write_text("今天必须测试 LIF 后向散射 USRP 数据，问题是阈值还没确定。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=3.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            states = {"Experiment": core.NeuronState(v=2.0)}
            spikes, _, final_states = stateful.replay_with_state([(date(2026, 6, 11), note)], 2, {"Experiment": config}, states)

        self.assertGreaterEqual(len(spikes), 1)
        self.assertIn("Experiment", final_states)

    def test_signal_recovery_disambiguates_away_from_health(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-12.md"
            note.write_text("LIF 后向散射链路恢复波形，PSD 和事件率都需要继续验证。", encoding="utf-8")
            evidence = core.extract_daily_evidence(
                date(2026, 6, 12),
                note,
                note.read_text(encoding="utf-8"),
                {"Experiment": core.NEURONS["Experiment"], "Health": core.NEURONS["Health"]},
            )

        self.assertGreater(evidence["Experiment"].evidence, evidence["Health"].evidence)
        self.assertTrue(any(item.vector and item.vector.disambiguation == "signal_recovery_as_experiment_evidence" for item in evidence["Experiment"].items))
        self.assertLess(evidence["Health"].evidence, 0.25)

    def test_negative_resistance_loop_uses_isolate_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            notes = []
            samples = [
                (date(2026, 6, 13), "负阻实验做不出来，没有稳定数据，问题很难受。"),
                (date(2026, 6, 14), "继续负阻，抵消多大的阻抗不知道，实验失败。"),
                (date(2026, 6, 15), "我注意到自己目前的问题负阻现在我发现实验一直做不出来我很难受。"),
            ]
            for day, text in samples:
                note = vault / f"{day.isoformat()}.md"
                note.write_text(text, encoding="utf-8")
                notes.append((day, note))

            config = core.NeuronConfig(
                theta=1.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            spikes, _, _ = core.replay(notes, 2, {"Experiment": config})

        negative_resistance_spikes = [spike for spike in spikes if spike.topic == "负阻"]
        self.assertTrue(negative_resistance_spikes)
        self.assertEqual(negative_resistance_spikes[-1].blocker_type, "repeated_failure")
        self.assertEqual(negative_resistance_spikes[-1].action_policy, "isolate")
        self.assertEqual(negative_resistance_spikes[-1].priority, "P1")
        self.assertIn("负阻隔离结论", negative_resistance_spikes[-1].completion_target)
        self.assertGreaterEqual(negative_resistance_spikes[-1].v_fast, 0.0)
        self.assertGreaterEqual(negative_resistance_spikes[-1].v_slow, 0.0)

    def test_feedback_can_mute_topic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-15.md"
            note.write_text("负阻实验做不出来，没有稳定数据，问题很难受。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=1.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            policies = core.apply_feedback({}, [{"topic": "负阻", "feedback": "不要再提醒"}])
            spikes, _, _ = core.replay([(date(2026, 6, 15), note)], 2, {"Experiment": config}, policies)

        self.assertEqual(spikes, [])

    def test_experiment_data_template_topic_is_not_career(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-05-29.md"
            note.write_text(
                "我现在应该全力以赴去做实验记录数据，接下来需要一个合理的跑数据模板，完成更加突破性的工作。",
                encoding="utf-8",
            )
            config = core.NeuronConfig(
                theta=0.1,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            spikes, _, _ = core.replay([(date(2026, 5, 29), note)], 2, {"Experiment": config})

        self.assertTrue(spikes)
        self.assertEqual(spikes[0].topic, "实验数据模板")
        self.assertNotEqual(spikes[0].topic, "求职")

    def test_lif_link_is_forced_p0(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-03.md"
            note.write_text("今天必须测试 EEG LIF 后向散射 USRP 链路，第四章主线还缺数据。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=1.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            spikes, _, _ = core.replay([(date(2026, 6, 3), note)], 2, {"Experiment": config})

        self.assertTrue(spikes)
        self.assertEqual(spikes[0].topic, "LIF链路")
        self.assertEqual(spikes[0].priority, "P0")

    def test_ai_career_event_has_primary_and_secondary_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-12.md"
            note.write_text("我要闯入 AI 世界，抓住大模型机会，准备实习和简历，all in AI。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=0.1,
                decay=0.83,
                reset_ratio=0.36,
                cooldown_days=1,
                evidence_cap=6.0,
                keywords=core.NEURONS["AI_Memory"].keywords,
                suggestion="test suggestion",
            )
            spikes, _, _ = core.replay([(date(2026, 6, 12), note)], 2, {"AI_Memory": config})

        self.assertTrue(spikes)
        self.assertEqual(spikes[0].topic, "AI求职转向")
        self.assertEqual(spikes[0].primary_state, "Career")
        self.assertIn("AI_Memory", spikes[0].secondary_states)

    def test_completion_scan_detects_experiment_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            data_dir = vault / "实验数据"
            data_dir.mkdir()
            csv_path = data_dir / "result.csv"
            csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
            timestamp = datetime(2026, 6, 15, 12, 0, 0).timestamp()
            os.utime(csv_path, (timestamp, timestamp))
            note = vault / "2026-06-15.md"
            note.write_text("今天测试数据。", encoding="utf-8")

            signals = core.scan_completion_signals(vault, [(date(2026, 6, 15), note)], {"Experiment": core.NEURONS["Experiment"]})

        self.assertIn(date(2026, 6, 15), signals)
        self.assertIn("Experiment", signals[date(2026, 6, 15)])

    def test_topic_specific_completion_target_for_ai_career(self) -> None:
        target = core.default_completion_target("AI_Memory", "AI求职转向")
        self.assertIn("简历", target)
        self.assertNotIn("跑一次回放", target)

    def test_markdown_spike_closure_creates_cooldown_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.md"
            path.write_text(
                "\n".join(
                    [
                        "## Spike 反馈区",
                        "",
                        "- [x] 2026-06-15-Experiment-负阻",
                        "  - Topic：负阻",
                        "  - Primary：Experiment",
                        "  - Policy：isolate",
                        "  - 状态：downgraded",
                        "  - 反馈：正确",
                        "  - 完成证据：负阻暂不作为论文主线，仅作为补充模块。",
                        "  - 关闭时间：2026-06-15",
                    ]
                ),
                encoding="utf-8",
            )
            closures = core.load_spike_closures(path)
            policies = core.policies_from_closures(closures)

        self.assertIn("2026-06-15-Experiment-负阻", closures)
        self.assertEqual(closures["2026-06-15-Experiment-负阻"].status, "downgraded")
        self.assertIn("负阻", policies)
        self.assertEqual(policies["负阻"].priority_override, "P2")
        self.assertEqual(policies["负阻"].action_policy_override, "downgrade")
        self.assertGreaterEqual(policies["负阻"].cooldown_days, 7)

    def test_feedback_memory_round_trip_from_closure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_memory_feedback.json"
            closures = {
                "2026-06-15-Experiment-负阻": core.SpikeClosure(
                    spike_id="2026-06-15-Experiment-负阻",
                    topic="负阻",
                    primary_state="Experiment",
                    policy="isolate",
                    status="downgraded",
                    feedback="正确",
                    completion_evidence="负阻暂不作为论文主线，仅作为补充模块。",
                    closed_at="2026-06-15",
                    checked=True,
                )
            }
            core.update_feedback_memory(path, closures, date(2026, 6, 15))
            policies = core.load_feedback_memory(path, date(2026, 6, 16))
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertIn("负阻", data["topics"])
        self.assertEqual(data["topics"]["负阻"]["status"], "downgraded")
        self.assertEqual(data["topics"]["负阻"]["cooldown_until"], "2026-06-22")
        self.assertIn("负阻", policies)
        self.assertEqual(policies["负阻"].priority_override, "P2")
        self.assertEqual(policies["负阻"].action_policy_override, "downgrade")
        self.assertGreaterEqual(policies["负阻"].cooldown_days, 6)

    def test_done_feedback_memory_cooldown_expires(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lif_memory_feedback.json"
            path.write_text(
                json.dumps(
                    {
                        "topics": {
                            "AI记忆": {
                                "status": "done",
                                "cooldown_until": "2026-06-17",
                                "last_feedback": "完成",
                                "threshold_delta": 0.5,
                                "priority_override": "P2",
                                "action_policy_override": "stop",
                                "cooldown_days": 2,
                                "feedback_count": 1,
                            }
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            policies = core.load_feedback_memory(path, date(2026, 6, 20))

        self.assertIn("AI记忆", policies)
        self.assertEqual(policies["AI记忆"].cooldown_days, 0)
        self.assertEqual(policies["AI记忆"].last_feedback, "完成")

    def test_render_markdown_includes_spike_feedback_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-15.md"
            note.write_text("负阻实验做不出来，没有稳定数据，问题很难受。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=1.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            notes = [(date(2026, 6, 15), note)]
            spikes, timeline, states = core.replay(notes, 2, {"Experiment": config})
            report = core.render_markdown(vault, notes, spikes, timeline, states, {"Experiment": config})

        self.assertIn("## Spike 反馈区", report)
        self.assertIn("- [ ] 2026-06-15-Experiment-负阻", report)

    def test_daily_markdown_renders_top_spike_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            note = vault / "2026-06-15.md"
            note.write_text("负阻实验做不出来，没有稳定数据，问题很难受。", encoding="utf-8")
            config = core.NeuronConfig(
                theta=1.0,
                decay=0.82,
                reset_ratio=0.35,
                cooldown_days=1,
                evidence_cap=6.5,
                keywords=core.NEURONS["Experiment"].keywords,
                suggestion="test suggestion",
            )
            notes = [(date(2026, 6, 15), note)]
            spikes, _, _ = core.replay(notes, 2, {"Experiment": config})
            report = core.render_daily_markdown(vault, notes, spikes, top_k=1)

        self.assertIn("# 今日 LIF-Memory 主卡片", report)
        self.assertIn("今日只处理这一件事", report)
        self.assertIn("## Spike 反馈区", report)

    def test_daily_markdown_uses_latest_spike_day(self) -> None:
        older = core.Spike(
            day=date(2026, 6, 1),
            neuron="Thesis",
            voltage=20.0,
            threshold=7.0,
            evidence_items=[],
            suggestion="older",
            topic="论文闭环",
            primary_state="Thesis",
            priority="P0",
            action_policy="continue",
            completion_target=core.default_completion_target("Thesis", "论文闭环"),
            spike_id="2026-06-01-Thesis-论文闭环",
        )
        latest = core.Spike(
            day=date(2026, 6, 15),
            neuron="Experiment",
            voltage=8.0,
            threshold=7.5,
            evidence_items=[],
            suggestion="latest",
            topic="负阻",
            primary_state="Experiment",
            priority="P1",
            action_policy="isolate",
            completion_target=core.default_completion_target("Experiment", "负阻"),
            spike_id="2026-06-15-Experiment-负阻",
        )
        report = core.render_daily_markdown(Path("."), [], [older, latest], top_k=1)

        self.assertIn("Topic：负阻", report)
        self.assertNotIn("Topic：论文闭环", report)

    def test_llm_config_uses_provider_preset_and_env_key(self) -> None:
        previous = os.environ.get("DASHSCOPE_API_KEY")
        os.environ["DASHSCOPE_API_KEY"] = "test-key"
        try:
            args = argparse.Namespace(
                llm_provider="qwen",
                llm_base_url=None,
                llm_model=None,
                llm_api_key_env=None,
                llm_timeout=30,
            )
            config = core.llm_config_from_args(args)
        finally:
            if previous is None:
                os.environ.pop("DASHSCOPE_API_KEY", None)
            else:
                os.environ["DASHSCOPE_API_KEY"] = previous

        self.assertEqual(config.provider, "qwen")
        self.assertIn("dashscope", config.base_url)
        self.assertEqual(config.model, "qwen-plus")
        self.assertEqual(config.api_key, "test-key")

    def test_llm_config_can_read_ignored_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "llm.local.json"
            config_path.write_text(
                json.dumps(
                    {
                        "provider": "deepseek",
                        "api_keys": {"deepseek": "local-key"},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                llm_provider=None,
                llm_base_url=None,
                llm_model=None,
                llm_api_key_env="MISSING_TEST_KEY",
                llm_local_config=config_path,
                llm_timeout=30,
            )
            config = core.llm_config_from_args(args)

        self.assertEqual(config.provider, "deepseek")
        self.assertEqual(config.model, "deepseek-v4-pro")
        self.assertEqual(config.api_key, "local-key")

    def test_extract_json_object_from_fenced_response(self) -> None:
        data = core.extract_json_object('```json\n{"is_correct": true, "reason": "ok"}\n```')

        self.assertTrue(data["is_correct"])
        self.assertEqual(data["reason"], "ok")

    def test_llm_review_prompt_is_reviewer_not_controller(self) -> None:
        spike = core.Spike(
            day=date(2026, 6, 15),
            neuron="Experiment",
            voltage=8.0,
            threshold=7.5,
            evidence_items=[],
            suggestion="latest",
            topic="负阻",
            primary_state="Experiment",
            secondary_states=["Thesis", "Health"],
            priority="P1",
            action_policy="isolate",
            completion_target=core.default_completion_target("Experiment", "负阻"),
            spike_id="2026-06-15-Experiment-负阻",
        )
        messages = core.llm_review_prompt(spike, Path("."))
        joined = "\n".join(message["content"] for message in messages)

        self.assertIn("语义审查器", joined)
        self.assertIn("不要决定电压", joined)
        self.assertIn("corrected_topic", joined)

    def test_render_daily_markdown_includes_llm_review_section(self) -> None:
        spike = core.Spike(
            day=date(2026, 6, 15),
            neuron="Experiment",
            voltage=8.0,
            threshold=7.5,
            evidence_items=[],
            suggestion="latest",
            topic="负阻",
            primary_state="Experiment",
            priority="P1",
            action_policy="isolate",
            completion_target=core.default_completion_target("Experiment", "负阻"),
            spike_id="2026-06-15-Experiment-负阻",
        )
        report = core.render_daily_markdown(
            Path("."),
            [],
            [spike],
            top_k=1,
            llm_reviews={
                spike.spike_id: {
                    "is_correct": True,
                    "reason": "topic and policy are coherent",
                }
            },
        )

        self.assertIn("## LLM Review", report)
        self.assertIn("topic and policy are coherent", report)

    def test_llm_review_normalizes_false_without_actual_correction(self) -> None:
        packet = {
            "topic": "AI求职转向",
            "primary_state": "Career",
            "secondary_states": ["AI_Memory", "Thesis", "Experiment"],
            "completion_target": "形成一版 AI+嵌入式 项目表达，并写入简历或求职材料。",
        }
        review = llm_adapter.normalize_review(
            {
                "is_correct": False,
                "corrected_topic": "AI求职转向",
                "corrected_primary_state": "Career",
                "corrected_secondary_states": ["AI_Memory", "Experiment", "Thesis"],
                "better_completion_target": "形成一版 AI+嵌入式 项目表达，并写入简历或求职材料。",
                "reason": "证据贴合，无需修正。",
            },
            packet,
        )

        self.assertTrue(review["is_correct"])
        self.assertIsNone(review["corrected_topic"])
        self.assertIsNone(review["better_completion_target"])


if __name__ == "__main__":
    unittest.main()
