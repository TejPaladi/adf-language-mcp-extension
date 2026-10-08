"""Print a fair, side-by-side comparison of the two saved experiment runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("equation_run", type=Path, help="Saved JSON from --condition author_equation")
    parser.add_argument("language_run", type=Path, help="Saved JSON from --condition language_extension")
    parser.add_argument("--output", type=Path, help="Optional Markdown output path.")
    parser.add_argument("--append", action="store_true", help="Append this comparison to an existing Markdown file.")
    return parser.parse_args()


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Result file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def check_pair(equation: dict[str, Any], language: dict[str, Any]) -> None:
    if equation.get("condition") != "author_equation":
        raise SystemExit("The first file must be an author_equation run.")
    if language.get("condition") != "language_extension":
        raise SystemExit("The second file must be a language_extension run.")

    for field in ("provider", "model", "temperature", "tool_set"):
        if equation.get(field) != language.get(field):
            raise SystemExit(f"Cannot compare runs: {field} differs between them.")

    if equation.get("sample") != language.get("sample"):
        raise SystemExit("Cannot compare runs: sample-selection settings differ between them.")

    equation_ids = [attempt["problem_id"] for attempt in equation.get("attempts", [])]
    language_ids = [attempt["problem_id"] for attempt in language.get("attempts", [])]
    if equation_ids != language_ids:
        raise SystemExit("Cannot compare runs: they do not contain the same ordered problem IDs.")


def value(run: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = run["summary"]
    for item in path:
        current = current[item]
    return current


def sample_label(run: dict[str, Any]) -> str:
    sample = run.get("sample")
    if not sample:
        return f"First {run['summary']['records']} records (legacy result format)"
    if sample["method"] == "random_without_replacement":
        return f"Random {sample['records']} records (seed {sample['seed']})"
    return f"First {sample['records']} records"


def render(equation: dict[str, Any], language: dict[str, Any], *, heading_level: int = 1) -> str:
    rows = [
        ("Records", ("records",)),
        ("Correct", ("correct",)),
        ("Accuracy", ("accuracy_percent",)),
        ("Wrong answer", ("outcomes", "wrong_answer")),
        ("Format failure", ("outcomes", "format_failure")),
        ("API error / timeout", ("outcomes", "api_error")),
        ("Total wall-clock seconds", ("wall_clock_seconds",)),
        ("Average latency (ms)", ("average_latency_ms",)),
        ("LLM turns", ("agent", "llm_turns")),
        ("MCP tool calls", ("agent", "tool_calls")),
        ("MCP tool errors", ("agent", "tool_errors")),
        ("No-tool answers", ("agent", "no_tool_answers")),
        ("Maximum-step failures", ("agent", "max_steps_exceeded")),
        ("Groq input tokens", ("provider_usage", "prompt_tokens")),
        ("Groq output tokens", ("provider_usage", "completion_tokens")),
        ("Groq total tokens", ("provider_usage", "total_tokens")),
    ]
    lines = [
        f"{'#' * heading_level} {sample_label(equation)}: Equation Input vs. Word-Problem Input",
        "",
        f"Provider/model: `{equation['provider']}` / `{equation['model']}`; temperature: `{equation['temperature']}`.",
        "",
        "| Metric | Equation-input agent | Word-problem agent |",
        "| --- | ---: | ---: |",
    ]
    for label, path in rows:
        left = value(equation, path)
        right = value(language, path)
        suffix = "%" if label == "Accuracy" else ""
        lines.append(f"| {label} | {left}{suffix} | {right}{suffix} |")
    lines.extend(
        [
            "",
            "The two runs used the same model, temperature, MCP tool set, and ordered problem IDs. The only intended input difference is equation text versus the original SVAMP word problem.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    equation = load(args.equation_run)
    language = load(args.language_run)
    check_pair(equation, language)
    if args.append and not args.output:
        raise SystemExit("--append requires --output.")
    report = render(equation, language, heading_level=2 if args.append else 1)
    print(report)
    if args.output:
        if args.append and args.output.exists():
            with args.output.open("a", encoding="utf-8") as output:
                output.write("\n\n" + report + "\n")
        else:
            args.output.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved comparison: {args.output}")


if __name__ == "__main__":
    main()
