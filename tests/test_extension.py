"""Offline checks for provenance and the word-problem MCP agent loop."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "extension"))

from direct_language import DIRECT_SYSTEM_PROMPT, direct_word_problem_message
from language_agent import LANGUAGE_SYSTEM_PROMPT, run_language_agent, word_problem_message
from multilingual_data import MultilingualDataError, dataset_sha256, load_language_rows, validate_translation_records
from prepare_multilingual_svamp import protect_numbers, restore_numbers
from run_experiment import is_provider_api_error, select_rows
from run_model_matrix import build_jobs


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


class DirectLanguageTests(unittest.TestCase):
    def test_direct_prompt_contains_word_problem_but_no_equation(self) -> None:
        message = direct_word_problem_message("There are 3 apples.", "Two more are added. How many now?")
        self.assertIn("There are 3 apples.", message)
        self.assertIn("How many now?", message)
        self.assertNotIn("3 + 2", message)
        self.assertIn("ONLY the final numerical value", DIRECT_SYSTEM_PROMPT)


class MultilingualDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.canonical = [
            {"ID": "one", "Body": "Maya has 3 apples.", "Question": "She gets 2 more. How many apples?"},
            {"ID": "two", "Body": "A box has 10 pencils.", "Question": "What is 10 minus 4?"},
        ]

    def test_complete_translation_with_same_numbers_is_accepted(self) -> None:
        records = [
            {"ID": "one", "Body": "Maya tiene 3 manzanas.", "Question": "Recibe 2 más. ¿Cuántas manzanas?"},
            {"ID": "two", "Body": "Una caja tiene 10 lápices.", "Question": "¿Cuánto es 10 menos 4?"},
        ]
        translations = validate_translation_records(self.canonical, records, "es")
        self.assertEqual(translations["one"]["Body"], "Maya tiene 3 manzanas.")

    def test_changed_number_is_rejected(self) -> None:
        records = [
            {"ID": "one", "Body": "Maya tiene 4 manzanas.", "Question": "Recibe 2 más. ¿Cuántas manzanas?"},
            {"ID": "two", "Body": "Una caja tiene 10 lápices.", "Question": "¿Cuánto es 10 menos 4?"},
        ]
        with self.assertRaises(MultilingualDataError):
            validate_translation_records(self.canonical, records, "es")

    def test_number_protection_round_trip(self) -> None:
        protected, values = protect_numbers("A $ 6.0 book has 2 chapters.")
        self.assertNotIn("6.0", protected)
        self.assertEqual(restore_numbers(protected, values), "A $ 6.0 book has 2 chapters.")

    def test_loader_requires_reviewed_exact_source_file(self) -> None:
        records = [
            {"ID": "one", "Body": "Maya tiene 3 manzanas.", "Question": "Recibe 2 más. ¿Cuántas manzanas?"},
            {"ID": "two", "Body": "Una caja tiene 10 lápices.", "Question": "¿Cuánto es 10 menos 4?"},
        ]
        payload = {
            "metadata": {
                "language": "es",
                "source_language": "en",
                "source_rows_sha256": dataset_sha256(self.canonical),
                "review_status": "reviewed",
            },
            "records": records,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "svamp_es.json"
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            loaded, metadata = load_language_rows(self.canonical, "es", Path(directory))
        self.assertEqual(loaded[0]["Question"], records[0]["Question"])
        self.assertEqual(metadata["code"], "es")


class ModelMatrixTests(unittest.TestCase):
    def test_full_matrix_preserves_upstream_direct_and_agent_lists(self) -> None:
        config = {
            "model": "openai/gpt-oss-20b",
            "temperature": 0.7,
            "temperatures": [0.0, 0.7],
            "ollama_models": ["llama3.2:1b", "qwen2-math:1.5b", "phi4-mini:3.8b", "llama3.2:3b", "granite4:3b"],
            "ollama_agent_models": ["llama3.2:1b", "phi4-mini:3.8b", "llama3.2:3b", "qwen3.5:4b", "granite4:3b"],
        }
        jobs = build_jobs(
            config,
            architecture="all",
            provider="all",
            languages=["en", "ar", "zh", "fr", "ru", "es"],
            results_dir=Path("results"),
        )
        self.assertEqual(len(jobs), 154)
        self.assertTrue(any(job.architecture == "direct" and job.model == "qwen2-math:1.5b" for job in jobs))
        self.assertFalse(any(job.architecture == "agent" and job.model == "qwen2-math:1.5b" for job in jobs))
        self.assertTrue(any(job.architecture == "agent" and job.model == "qwen3.5:4b" for job in jobs))
        self.assertFalse(any(job.architecture == "direct" and job.model == "qwen3.5:4b" for job in jobs))

    def test_temperature_filter_builds_one_agent_job_per_condition(self) -> None:
        config = {
            "model": "openai/gpt-oss-20b",
            "temperature": 0.7,
            "temperatures": [0.0, 0.7],
            "ollama_models": ["direct-model"],
            "ollama_agent_models": ["agent-model"],
        }
        jobs = build_jobs(
            config,
            architecture="agent",
            provider="ollama",
            languages=["en", "es"],
            results_dir=Path("results"),
            temperatures=[0.0],
        )
        self.assertEqual(len(jobs), 3)
        self.assertEqual({job.temperature for job in jobs}, {0.0})


class SamplingTests(unittest.TestCase):
    def test_random_sample_is_reproducible_and_has_no_duplicate_rows(self) -> None:
        rows = [{"ID": str(index)} for index in range(100)]
        first = select_rows(rows, 50, "random", 42)
        second = select_rows(rows, 50, "random", 42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["ID"] for row in first}), 50)
        self.assertNotEqual(first, rows[:50])

    def test_step_limit_is_not_treated_as_provider_outage(self) -> None:
        step_limit = {"outcome": "api_error", "agent_stats": {"max_steps_exceeded": 1}}
        provider_error = {"outcome": "api_error", "agent_stats": {"max_steps_exceeded": 0}}
        self.assertFalse(is_provider_api_error(step_limit))
        self.assertTrue(is_provider_api_error(provider_error))


if __name__ == "__main__":
    unittest.main()
