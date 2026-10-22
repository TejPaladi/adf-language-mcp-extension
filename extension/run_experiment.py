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

import requests

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
from multilingual_data import LANGUAGES, MultilingualDataError, load_language_rows  # noqa: E402


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
    parser.add_argument(
        "--language",
        choices=tuple(LANGUAGES),
        default="en",
        help="Language of the word-problem input. English uses unchanged SVAMP text.",
    )
    parser.add_argument(
        "--translations-dir",
        type=Path,
        default=ROOT / "data" / "multilingual_svamp",
        help="Directory containing fixed, validated non-English SVAMP translations.",
    )
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
    parser.add_argument(
        "--allow-machine-generated-translations",
        action="store_true",
        help="Permit non-English files marked machine-generated; outputs are labelled preliminary.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Atomically save the partial result after this many problems (default: 10).",
    )
    parser.add_argument("--resume", action="store_true", help="Continue a compatible partial JSON result file.")
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=30.0,
        help="Per Ollama chat request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--max-consecutive-api-errors",
        type=int,
        default=3,
        help="Stop and save after this many back-to-back failed model calls (default: 3).",
    )
    return parser.parse_args()


def bounded_ollama_chat(messages, tools, *, model: str, temperature: float, config: dict[str, Any], timeout: float):
    """Extension-only Ollama adapter with a finite request timeout.

    The upstream adapter is unchanged.  This copy has the same request format
    but prevents one stalled local response from consuming an entire run.
    """

    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": config.get("max_tokens", 256)},
    }
    response = requests.post(
        f"{config.get('ollama_url', 'http://localhost:11434')}/api/chat",
        json=payload,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text}")
    data = response.json()
    message = data["message"]
    content = (message.get("content") or "").strip()
    raw_calls = message.get("tool_calls") or []
    tool_calls = [
        {"id": None, "name": call["function"]["name"], "args": call["function"]["arguments"]}
        for call in raw_calls
    ]
    assistant_message = {"role": "assistant", "content": content}
    if raw_calls:
        assistant_message["tool_calls"] = raw_calls
    stats = {
        "total_duration_ns": data.get("total_duration", 0),
        "load_duration_ns": data.get("load_duration", 0),
        "prompt_eval_count": data.get("prompt_eval_count", 0),
        "prompt_eval_duration_ns": data.get("prompt_eval_duration", 0),
        "eval_count": data.get("eval_count", 0),
        "eval_duration_ns": data.get("eval_duration", 0),
    }
    return assistant_message, content, tool_calls, stats


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
    return (
        partial(
            bounded_ollama_chat,
            model=model,
            temperature=temperature,
            config=config,
            timeout=args.ollama_timeout,
        ),
        ollama_tool_message,
        model,
        temperature,
    )


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


def write_payload(output: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a result file so an interrupted save stays readable."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(output)


def is_provider_api_error(attempt: dict[str, Any]) -> bool:
    """Distinguish provider outages from an agent reaching its step limit."""

    return bool(
        attempt.get("outcome") == "api_error"
        and not attempt.get("agent_stats", {}).get("max_steps_exceeded", 0)
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")
    if args.ollama_timeout <= 0:
        raise SystemExit("--ollama-timeout must be positive.")
    if args.max_consecutive_api_errors < 1:
        raise SystemExit("--max-consecutive-api-errors must be at least 1.")
    if args.condition == "author_equation" and args.language != "en":
        raise SystemExit("--language applies only to language_extension; author_equation always uses canonical equations.")

    config = load_llm_config()
    chat, tool_message, model, temperature = build_backend(args, config)
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

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or RESULTS_DIR / f"{run_id}_{args.condition}_{args.provider}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if output.exists():
        if not args.resume:
            raise SystemExit(f"Output already exists: {output}. Use --resume to continue it.")
        previous = json.loads(output.read_text(encoding="utf-8"))
        expected = {"architecture": "mcp_agent", "condition": args.condition, "provider": args.provider, "model": model}
        if any(previous.get(key) != value for key, value in expected.items()):
            raise SystemExit(f"Cannot resume incompatible result file: {output}")
        attempts = previous.get("attempts", [])
        elapsed_before = float(previous.get("run_progress", {}).get("elapsed_seconds", 0.0))
        run_id = previous.get("run_id", run_id)

    completed_ids = {attempt["problem_id"] for attempt in attempts}
    remaining = [row for row in data if row["ID"] not in completed_ids]
    started = time.perf_counter()
    tools: list[dict[str, Any]] = []

    def payload(status: str) -> dict[str, Any]:
        elapsed = elapsed_before + time.perf_counter() - started
        return {
            "run_id": run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "architecture": "mcp_agent",
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
            "tool_server": "upstream_adf/code/mcp_server.py",
            "tool_set": [tool["function"]["name"] for tool in tools],
            "scoring": "Unchanged upstream strict numeric scorer: direct numeric output equals ADF Answer.",
            "runtime_policy": {
                "checkpoint_every": args.checkpoint_every,
                "ollama_timeout_seconds": args.ollama_timeout if args.provider == "ollama" else None,
                "max_consecutive_api_errors": args.max_consecutive_api_errors,
            },
            "run_status": status,
            "run_progress": {"completed_records": len(attempts), "target_records": len(data), "elapsed_seconds": round(elapsed, 6)},
            "summary": summarize(attempts, elapsed),
            "attempts": attempts,
        }

    consecutive_api_errors = 0
    status = "completed"
    try:
        with connect() as session:
            tools = tools_for_llm(session)
            for row in remaining:
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

                attempt = {
                    "problem_id": row["ID"],
                    "operation_type": row["Type"],
                    "input_kind": input_kind,
                    "gold_answer": row["Answer"],
                    "raw_result": raw_result,
                    "outcome": classify_result(raw_result, row["Answer"]),
                    "latency_ms": round(1000 * (time.perf_counter() - item_started), 6),
                    "agent_stats": agent_stats,
                }
                attempts.append(attempt)
                consecutive_api_errors = consecutive_api_errors + 1 if is_provider_api_error(attempt) else 0
                if len(attempts) % args.checkpoint_every == 0:
                    write_payload(output, payload("in_progress"))
                    print(f"Checkpoint: {len(attempts)}/{len(data)}")
                if consecutive_api_errors >= args.max_consecutive_api_errors:
                    status = "halted_consecutive_api_errors"
                    break
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        final_payload = payload(status)
        write_payload(output, final_payload)
    print(json.dumps(final_payload["summary"], indent=2))
    print(f"Status: {status}; {len(attempts)}/{len(data)} records")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
