"""Offline checks for provenance and the word-problem MCP agent loop."""

from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extension"))

from language_agent import LANGUAGE_SYSTEM_PROMPT, run_language_agent, word_problem_message
from run_experiment import select_rows


class ProvenanceTests(unittest.TestCase):
    def test_upstream_corrected_dataset_hashes_are_preserved(self) -> None:
        expected = {
            "train.json": "364c4cdc2f228a966dc3b51168469823df7480753e889b3a75a8cecfe125d3bc",
            "test.json": "9b5590fce66ea78ecfc7d0fc4a91fb801ffe289340269f8d9d28b21acad04ae6",
        }
        data_dir = ROOT / "upstream_adf" / "data"
        actual = {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest() for name in expected}
        self.assertEqual(actual, expected)


class LanguageAgentTests(unittest.TestCase):
    def test_word_problem_message_contains_no_gold_equation(self) -> None:
        message = word_problem_message("A jar has 3 red marbles.", "It gets 2 more. How many now?")
        self.assertIn("A jar has 3 red marbles.", message)
        self.assertIn("How many now?", message)
        self.assertNotIn("3 + 2", message)
        self.assertIn("provided tools", LANGUAGE_SYSTEM_PROMPT)

    def test_loop_returns_tool_result_and_records_trace_counts(self) -> None:
        class FakeSession:
            def call_tool(self, name, arguments):
                self.name = name
                self.arguments = arguments
                return SimpleNamespace(is_error=False, content=[SimpleNamespace(text="5")])

        calls = 0

        def fake_chat(messages, tools):
            nonlocal calls
            calls += 1
            if calls == 1:
                return (
                    {"role": "assistant", "content": "", "tool_calls": [{"id": "x", "function": {}}]},
                    "",
                    [{"id": "x", "name": "add", "args": {"a": 3, "b": 2}}],
                    {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
                )
            return ({"role": "assistant", "content": "5"}, "5", [], {"total_tokens": 2})

        session = FakeSession()
        result, stats = run_language_agent(
            "There are 3 apples.",
            "Two more are added. How many apples are there?",
            session,
            fake_chat,
            lambda call, output: {"role": "tool", "content": output},
            tools=[],
            max_steps=2,
        )

        self.assertEqual(result, "5")
        self.assertEqual(session.name, "add")
        self.assertEqual(session.arguments, {"a": 3, "b": 2})
        self.assertEqual(stats["llm_turns"], 2)
        self.assertEqual(stats["tool_calls"], 1)
        self.assertEqual(stats["total_tokens"], 7)


class SamplingTests(unittest.TestCase):
    def test_random_sample_is_reproducible_and_has_no_duplicate_rows(self) -> None:
        rows = [{"ID": str(index)} for index in range(100)]
        first = select_rows(rows, 50, "random", 42)
        second = select_rows(rows, 50, "random", 42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["ID"] for row in first}), 50)
        self.assertNotEqual(first, rows[:50])


if __name__ == "__main__":
    unittest.main()
