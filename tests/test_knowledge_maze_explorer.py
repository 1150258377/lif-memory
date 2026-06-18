from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import knowledge_maze_explorer as maze


class KnowledgeMazeExplorerTest(unittest.TestCase):
    def test_infer_topics_and_roles(self) -> None:
        text = "LIF 后向散射 USRP 已经跑通，但是论文证据链还缺一张图，下一步需要整理 PSD。"
        topics = maze.infer_topics(text)
        roles = maze.infer_roles(text)
        self.assertIn("LIF链路", topics)
        self.assertIn("后向散射", topics)
        self.assertIn("论文闭环", topics)
        self.assertIn("action", roles)
        self.assertIn("blocker", roles)

    def test_explore_view_builds_spike_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vault = Path(tmp)
            (vault / ".obsidian").mkdir()
            note = vault / "2026-06-18.md"
            note.write_text(
                "\n\n".join(
                    [
                        "论文主线现在还缺证据链，第四章必须把 EEG→LIF→后向散射→USRP 串起来。",
                        "SSVEP、事件率、PSD、边带已经有结果，但是还需要一张三链对齐图。",
                        "负阻测试不稳，偏置区间很窄，暂时不适合作为论文主线必要环节。",
                    ]
                ),
                encoding="utf-8",
            )
            blocks = maze.build_blocks(vault, max_files=10, max_blocks_per_file=20)
            report = maze.explore_view(
                vault=vault,
                blocks=blocks,
                view=maze.DEFAULT_VIEWS["thesis_closure"],
                steps=10,
                min_score=0.1,
            )
            self.assertTrue(report.evidence)
            self.assertTrue(report.nodes)
            labels = {node.label for node in report.nodes}
            self.assertIn("论文闭环", labels)
            self.assertTrue(report.claims)


if __name__ == "__main__":
    unittest.main()
