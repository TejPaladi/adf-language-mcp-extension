"""Compare the upstream equation agent with the word-problem MCP extension.

The reused MCP calculator server must already be running at
http://localhost:8100/mcp. See the repository README for commands.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CODE = ROOT / "upstream_adf" / "code"
RESULTS_DIR = ROOT / "results" / "runs"

if str(UPSTREAM_CODE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_CODE))

from config import load_dataset, load_llm_config  # noqa: E402
from main import classify_result  # noqa: E402
from mcp_agent import (  # noqa: E402
    chat_groq,
    chat_ollama,
    groq_tool_message,
    ollama_tool_message,
    run_agent,
    tools_for_llm,
)
from mcp_client import connect  # noqa: E402

from language_agent import run_language_agent  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=("author_equation", "language_extension"),
        help="Equation input (upstream condition) or original SVAMP word-problem input.",
    )
    parser.add_argument("--provider", required=True, choices=("groq", "ollama"))
    parser.add_argument("--model", help="Optional Ollama model; uses upstream default when omitted.")
    parser.add_argument("--temperature", type=float, help="Defaults to the upstream configuration value.")
    parser.add_argument("--limit", type=int, default=1000, help="Use a small value for a smoke test.")
    parser.add_argument(
        "--sample",
        choices=("first", "random"),
        default="first",
        help="Select the first records or a reproducible random sample without replacement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random-sample seed. It is recorded in every result file.",
    )
    parser.add_argument("--output", type=Path, help="Optional explicit JSON output path.")
    return parser.parse_args()


def build_backend(args: argparse.Namespace, config: dict[str, Any]):
    """Reuse the original provider adapters and configuration."""

    temperature = config["temperature"] if args.temperature is None else args.temperature
    if args.provider == "groq":
        if not os.environ.get("GROQ_API_KEY"):
            from getpass import getpass

            key = getpass("Paste Groq API key for this run: ").strip()
            if not key:
                raise SystemExit("A Groq API key is required for a Groq run.")
            # This exists only in this Python process. The key is not written
            # into a result file, notebook, source file, or shell history.
            os.environ["GROQ_API_KEY"] = key
        return partial(chat_groq, temperature=temperature), groq_tool_message, config["model"], temperature

    model = args.model or config["ollama_agent_models"][0]
    return partial(chat_ollama, model=model, temperature=temperature), ollama_tool_message, model, temperature


def summarize(attempts: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    outcomes = {"correct": 0, "wrong_answer": 0, "format_failure": 0, "api_error": 0}
    agent = {"llm_turns": 0, "tool_calls": 0, "tool_errors": 0, "no_tool_answers": 0, "max_steps_exceeded": 0}
    tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for attempt in attempts:
        outcomes[attempt["outcome"]] += 1
        for key in agent:
            agent[key] += int(attempt["agent_stats"].get(key, 0))
        for key in tokens:
            tokens[key] += int(attempt["agent_stats"].get(key, 0))

    count = len(attempts)
    return {
        "records": count,
        "correct": outcomes["correct"],
        "accuracy_percent": round(100 * outcomes["correct"] / count, 3) if count else 0.0,
        "outcomes": outcomes,
        "wall_clock_seconds": round(elapsed_seconds, 6),
        "average_latency_ms": round(1000 * elapsed_seconds / count, 3) if count else 0.0,
        "agent": agent,
        "provider_usage": tokens,
    }


def select_rows(rows: list[dict[str, Any]], limit: int, sample: str, seed: int) -> list[dict[str, Any]]:
    """Choose a fixed prefix or a reproducible random sample without replacement."""

    if sample == "first":
        return rows[:limit]
    return random.Random(seed).sample(rows, limit)


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000.")

    config = load_llm_config()
    chat, tool_message, model, temperature = build_backend(args, config)
    train, test = load_dataset()
    population = train + test
    data = select_rows(population, args.limit, args.sample, args.seed)

    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    with connect() as session:
        tools = tools_for_llm(session)
        for row in data:
            item_started = time.perf_counter()
            if args.condition == "author_equation":
                raw_result, agent_stats = run_agent(
                    row["Equation"], session, chat, tool_message, tools, config["agent_max_steps"]
                )
                input_kind = "canonical_equation"
            else:
                raw_result, agent_stats = run_language_agent(
                    row["Body"], row["Question"], session, chat, tool_message, tools, config["agent_max_steps"]
                )
                input_kind = "svamp_body_plus_question"

            attempts.append(
                {
                    "problem_id": row["ID"],
                    "operation_type": row["Type"],
                    "input_kind": input_kind,
                    "gold_answer": row["Answer"],
                    "raw_result": raw_result,
                    "outcome": classify_result(raw_result, row["Answer"]),
                    "latency_ms": round(1000 * (time.perf_counter() - item_started), 6),
                    "agent_stats": agent_stats,
                }
            )

    elapsed = time.perf_counter() - started
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS_DIR / f"{run_id}_{args.condition}_{args.provider}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition": args.condition,
        "provider": args.provider,
        "model": model,
        "temperature": temperature,
        "sample": {
            "method": "first_n" if args.sample == "first" else "random_without_replacement",
            "seed": args.seed if args.sample == "random" else None,
            "records": len(data),
            "population_records": len(population),
        },
        "upstream_source": "upstream_adf/ (unchanged vendored snapshot)",
        "tool_server": "upstream_adf/code/mcp_server.py",
        "tool_set": [tool["function"]["name"] for tool in tools],
        "scoring": "Unchanged upstream strict numeric scorer: direct numeric output equals ADF Answer.",
        "summary": summarize(attempts, elapsed),
        "attempts": attempts,
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
