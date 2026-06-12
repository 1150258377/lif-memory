from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lif_memory as core  # noqa: E402
import lif_memory_stateful as stateful  # noqa: E402


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
            loaded, last_processed = stateful.load_state(path, {"Experiment": core.NEURONS["Experiment"]}, reset=False)

        self.assertEqual(last_processed, date(2026, 6, 11))
        self.assertAlmostEqual(loaded["Experiment"].v, 2.5)
        self.assertEqual(loaded["Experiment"].last_spike_date, date(2026, 6, 10))

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


if __name__ == "__main__":
    unittest.main()
