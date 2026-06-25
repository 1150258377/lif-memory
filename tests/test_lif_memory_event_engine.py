import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lif_memory_event_engine import (
    build_packet,
    detect_conflicts,
    load_events_from_path,
    merge_events,
    memory_event_from_text,
)


class DynamicMemoryEventEngineTests(unittest.TestCase):
    def test_extract_event_from_text(self):
        event = memory_event_from_text("我现在论文创新点卡住了，问题是异步和稀疏说不清。今晚必须把最小闭环写成一页。")
        self.assertIsNotNone(event)
        self.assertEqual(event.topic, "Thesis")
        self.assertGreaterEqual(event.importance, 4)
        self.assertIn(event.emotion, {"blocked", "decisive"})

    def test_conflict_detection_eeg_ecg(self):
        eeg = memory_event_from_text("论文主线是脑电 EEG 的 LIF 后向散射系统。")
        ecg = memory_event_from_text("导师建议把验证降级到心电 ECG，并做人体佩戴系统。")
        conflicts = detect_conflicts([eeg, ecg])
        self.assertTrue(any(item.kind == "EEG_vs_ECG" for item in conflicts))

    def test_json_load_and_activation_packet(self):
        payload = [
            {"text": "LIF Memory 需要从静态检索升级为动态记忆激活，目标是产生洞察。", "created_at": "2026-06-25"},
            {"text": "论文创新点问题是模拟域事件化和异步后向散射如何证明。今晚需要写成一页。", "created_at": "2026-06-25"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            events = load_events_from_path(path)
        self.assertEqual(len(events), 2)
        packet, state = build_packet("LIF Memory 怎么升级才能产生灵光一闪", events, {"voltages": {}}, top_k=5)
        self.assertGreaterEqual(len(packet["activated"]), 1)
        self.assertGreaterEqual(len(packet["insights"]), 1)
        self.assertIn("events", state)
        self.assertIn("voltages", state)

    def test_merge_preserves_stronger_event(self):
        a = memory_event_from_text("LIF Memory 只是保存摘要。")
        b = memory_event_from_text("LIF Memory 必须升级为 MemoryEvent 和动态激活。")
        b.event_id = a.event_id
        b.importance = 5
        merged = merge_events([a], [b])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].importance, 5)
        self.assertIn("MemoryEvent", merged[0].claim)


if __name__ == "__main__":
    unittest.main()
