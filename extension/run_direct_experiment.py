"""Evaluate direct LLMs on canonical equations or multilingual SVAMP text.

This script reuses the upstream ADF dataset, model configuration, and strict
numeric scorer.  It differs from ``run_experiment.py`` only in one intentional
way: models receive no MCP tools.  It is therefore the correct home for the
paper's direct-only model ``qwen2-math:1.5b``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any, Callable

import requests

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CODE = ROOT / "upstream_adf" / "code"
RESULTS_DIR = ROOT / "results" / "runs"

if str(UPSTREAM_CODE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_CODE))

from config import load_dataset, load_llm_config  # noqa: E402
from main import classify_result  # noqa: E402

from direct_language import direct_equation_message, direct_word_problem_message, run_direct  # noqa: E402
from multilingual_data import LANGUAGES, MultilingualDataError, load_language_rows  # noqa: E402
from run_experiment import select_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True, choices=("author_equation", "language_extension"))
    parser.add_argument("--provider", required=True, choices=("groq", "ollama"))
    parser.add_argument("--model", help="Optional model override. Uses the matching upstream default when omitted.")
    parser.add_argument("--temperature", type=float, help="Defaults to the upstream configuration value.")
    parser.add_argument("--language", choices=tuple(LANGUAGES), default="en")
    parser.add_argument(
        "--translations-dir",
        type=Path,
        default=ROOT / "data" / "multilingual_svamp",
        help="Directory containing fixed, validated non-English SVAMP translations.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional explicit JSON output path.")
    parser.add_argument(
        "--allow-machine-generated-translations",
        action="store_true",
        help="Permit non-English files marked machine-generated; outputs are labelled preliminary.",
    )
    return parser.parse_args()


def direct_groq_chat(
    messages: list[dict[str, str]], *, model: str, temperature: float, config: dict[str, Any]
) -> tuple[str | None, dict[str, int]]:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config['api_key']}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": config.get("max_tokens", 256),
        "top_p": config.get("top_p", 1.0),
    }
    try:
        response = requests.post(config["url"], headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        usage = data.get("usage", {})
        return (data["choices"][0]["message"].get("content") or "").strip(), {
            key: int(usage.get(key, 0) or 0) for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
    except Exception:
        return None, {}


def direct_ollama_chat(
    messages: list[dict[str, str]], *, model: str, temperature: float, config: dict[str, Any]
) -> tuple[str | None, dict[str, int]]:
    # The upstream ADF direct baseline uses /api/generate, a single prompt,
    # and a 30-second request timeout.  Retain those choices so an extension
    # result is comparable to the authors' direct-model condition.
    prompt = "\n\n".join(message["content"] for message in messages)
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(f"{config.get('ollama_url', 'http://localhost:11434')}/api/generate", json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        timing = {
            key: int(data.get(source, 0) or 0)
            for key, source in (
                ("total_duration_ns", "total_duration"),
                ("load_duration_ns", "load_duration"),
                ("prompt_eval_count", "prompt_eval_count"),
                ("prompt_eval_duration_ns", "prompt_eval_duration"),
                ("eval_count", "eval_count"),
                ("eval_duration_ns", "eval_duration"),
            )
        }
        return (data.get("response") or "").strip(), timing
    except Exception:
        return None, {}


def summarize(attempts: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    outcomes = {"correct": 0, "wrong_answer": 0, "format_failure": 0, "api_error": 0}
    token_fields = ("prompt_tokens", "completion_tokens", "total_tokens")
    timing_fields = (
        "total_duration_ns",
        "load_duration_ns",
        "prompt_eval_count",
        "prompt_eval_duration_ns",
        "eval_count",
        "eval_duration_ns",
    )
    tokens = {key: 0 for key in token_fields}
    ollama_timing = {key: 0 for key in timing_fields}
    for attempt in attempts:
        outcomes[attempt["outcome"]] += 1
        for key in token_fields:
            tokens[key] += int(attempt["runtime_stats"].get(key, 0) or 0)
        for key in timing_fields:
            ollama_timing[key] += int(attempt["runtime_stats"].get(key, 0) or 0)

    count = len(attempts)
    return {
        "records": count,
        "correct": outcomes["correct"],
        "accuracy_percent": round(100 * outcomes["correct"] / count, 3) if count else 0.0,
        "outcomes": outcomes,
        "wall_clock_seconds": round(elapsed_seconds, 6),
        "average_latency_ms": round(1000 * elapsed_seconds / count, 3) if count else 0.0,
        "provider_usage": tokens,
        "ollama_timing": ollama_timing,
    }


def build_backend(
    args: argparse.Namespace, config: dict[str, Any]
) -> tuple[Callable[[list[dict[str, str]]], tuple[str | None, dict[str, int]]], str, float]:
    temperature = config["temperature"] if args.temperature is None else args.temperature
    if args.provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            from getpass import getpass

            key = getpass("Paste Groq API key for this run: ").strip()
            if not key:
                raise SystemExit("A Groq API key is required for a Groq run.")
            os.environ["GROQ_API_KEY"] = key
            config["api_key"] = key
        model = args.model or config["model"]
        return partial(direct_groq_chat, model=model, temperature=temperature, config=config), model, temperature

    model = args.model or config["ollama_models"][0]
    return partial(direct_ollama_chat, model=model, temperature=temperature, config=config), model, temperature


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000.")
    if args.condition == "author_equation" and args.language != "en":
        raise SystemExit("--language applies only to language_extension; author_equation always uses canonical equations.")

    config = load_llm_config()
    chat, model, temperature = build_backend(args, config)
    train, test = load_dataset()
    population = train + test
    if args.condition == "language_extension":
        try:
            language_rows, language_metadata = load_language_rows(
                population,
                args.language,
                args.translations_dir,
                allow_machine_generated=args.allow_machine_generated_translations,
            )
        except MultilingualDataError as error:
            raise SystemExit(f"Multilingual data check failed: {error}") from error
    else:
        language_rows = population
        language_metadata = {
            "code": "equation",
            "name": "Canonical equation",
            "source": "unchanged upstream SVAMP Equation field",
        }
    data = select_rows(language_rows, args.limit, args.sample, args.seed)

    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    for row in data:
        item_started = time.perf_counter()
        message = (
            direct_equation_message(row["Equation"])
            if args.condition == "author_equation"
            else direct_word_problem_message(row["Body"], row["Question"])
        )
        raw_result, runtime_stats = run_direct(message, chat)
        attempts.append(
            {
                "problem_id": row["ID"],
                "operation_type": row["Type"],
                "input_kind": "canonical_equation" if args.condition == "author_equation" else "svamp_body_plus_question",
                "gold_answer": row["Answer"],
                "raw_result": raw_result,
                "outcome": classify_result(raw_result, row["Answer"]),
                "latency_ms": round(1000 * (time.perf_counter() - item_started), 6),
                "runtime_stats": runtime_stats,
            }
        )

    elapsed = time.perf_counter() - started
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS_DIR / f"{run_id}_{args.condition}_{args.provider}_direct.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": "direct_llm_no_tools",
        "condition": args.condition,
        "provider": args.provider,
        "model": model,
        "temperature": temperature,
        "input_language": language_metadata,
        "sample": {
            "method": "first_n" if args.sample == "first" else "random_without_replacement",
            "seed": args.seed if args.sample == "random" else None,
            "records": len(data),
            "population_records": len(population),
        },
        "upstream_source": "upstream_adf/ (unchanged vendored snapshot)",
        "tool_set": [],
        "scoring": "Unchanged upstream strict numeric scorer: direct numeric output equals ADF Answer.",
        "summary": summarize(attempts, elapsed),
        "attempts": attempts,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
