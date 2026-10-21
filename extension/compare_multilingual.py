"""Compare matching saved word-problem runs across input languages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="One language_extension result JSON per input language.")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def matching_value(run: dict[str, Any], field: str) -> Any:
    return run.get(field)


def validate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(runs) < 2:
        raise SystemExit("Provide at least two language runs to compare.")
    reference = runs[0]
    reference_ids = [attempt["problem_id"] for attempt in reference.get("attempts", [])]
    required = ("architecture", "provider", "model", "temperature", "sample", "tool_set")
    languages: set[str] = set()
    for run in runs:
        if run.get("condition") != "language_extension":
            raise SystemExit("Every result must be a language_extension run.")
        language = run.get("input_language", {}).get("code")
        if not language or language == "equation":
            raise SystemExit("Every result must record a natural-language input code.")
        if language in languages:
            raise SystemExit(f"Duplicate language run: {language}")
        languages.add(language)
        for field in required:
            if matching_value(run, field) != matching_value(reference, field):
                raise SystemExit(f"Cannot compare runs: {field} differs.")
        if [attempt["problem_id"] for attempt in run.get("attempts", [])] != reference_ids:
            raise SystemExit("Cannot compare runs: ordered problem IDs differ.")
    return sorted(runs, key=lambda run: run["input_language"]["code"])


def summary_value(run: dict[str, Any], *path: str) -> Any:
    current: Any = run["summary"]
    for part in path:
        current = current.get(part, 0)
    return current


def render(runs: list[dict[str, Any]]) -> str:
    reference = runs[0]
    is_agent = reference.get("architecture") != "direct_llm_no_tools"
    lines = [
        "# Multilingual word-problem comparison",
        "",
        f"Architecture: `{reference.get('architecture', 'mcp_agent')}`; provider/model: "
        f"`{reference['provider']}` / `{reference['model']}`; temperature: `{reference['temperature']}`.",
        "",
        "| Language | Correct | Accuracy | Wrong | Format | API error | Avg latency (ms)"
        + (" | Tool calls | LLM turns |" if is_agent else " |"),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: " + ("| ---: | ---: |" if is_agent else "|"),
    ]
    for run in runs:
        language = run["input_language"]
        row = (
            f"| {language['name']} (`{language['code']}`) "
            f"| {summary_value(run, 'correct')} "
            f"| {summary_value(run, 'accuracy_percent')}% "
            f"| {summary_value(run, 'outcomes', 'wrong_answer')} "
            f"| {summary_value(run, 'outcomes', 'format_failure')} "
            f"| {summary_value(run, 'outcomes', 'api_error')} "
            f"| {summary_value(run, 'average_latency_ms')}"
        )
        if is_agent:
            row += f" | {summary_value(run, 'agent', 'tool_calls')} | {summary_value(run, 'agent', 'llm_turns')}"
        lines.append(row + " |")
    lines.extend(
        [
            "",
            "All rows use the same model, temperature, architecture, tool set, scoring rule, and ordered problem IDs. "
            "Only the fixed language of the SVAMP body and question differs.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    report = render(validate([load(path) for path in args.runs]))
    print(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved comparison: {args.output}")


if __name__ == "__main__":
    main()
