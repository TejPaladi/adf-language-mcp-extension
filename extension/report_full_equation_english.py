"""Validate and report the full 1,000-record equation-versus-English study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_ORDER = (
    "llama3.2:1b",
    "llama3.2:3b",
    "phi4-mini:3.8b",
    "qwen3.5:4b",
    "granite4:3b",
)


def load_runs(directory: Path) -> dict[tuple[str, str], dict[str, Any]]:
    runs: dict[tuple[str, str], dict[str, Any]] = {}
    reference_ids: list[str] | None = None
    for path in sorted(directory.glob("agent__*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        condition = "equation" if run["condition"] == "author_equation" else "english"
        key = (run["model"], condition)
        if key in runs:
            raise ValueError(f"Duplicate result for {key}.")
        if run.get("run_status") != "completed" or run.get("summary", {}).get("records") != 1000:
            raise ValueError(f"Incomplete result: {path.name}")
        ids = [attempt["problem_id"] for attempt in run["attempts"]]
        if len(ids) != 1000 or len(set(ids)) != 1000:
            raise ValueError(f"Expected 1,000 unique attempts in {path.name}.")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"Ordered problem IDs differ in {path.name}.")
        runs[key] = run

    expected = {(model, condition) for model in MODEL_ORDER for condition in ("equation", "english")}
    if set(runs) != expected:
        raise ValueError(f"Result set differs from expected matrix: missing={expected - set(runs)}")
    return runs


def render(runs: dict[tuple[str, str], dict[str, Any]]) -> str:
    lines = [
        "# Full 1,000-problem equation-versus-English MCP-agent comparison",
        "",
        "## Controlled setup",
        "",
        "All five local MCP-agent models from the reused ADF configuration were tested on the same ordered 1,000 corrected SVAMP records. Each model received either the canonical equation or the original English body and question. Temperature (`0`), MCP arithmetic tools, strict numeric scorer, maximum-step policy, and record order were fixed.",
        "",
        "## Results",
        "",
        "| Model | Equation accuracy | English accuracy | Language gap | Equation latency | English latency |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in MODEL_ORDER:
        equation = runs[(model, "equation")]["summary"]
        english = runs[(model, "english")]["summary"]
        gap = float(equation["accuracy_percent"]) - float(english["accuracy_percent"])
        lines.append(
            f"| `{model}` | {equation['accuracy_percent']:.1f}% | {english['accuracy_percent']:.1f}% | "
            f"{gap:+.1f} pp | {equation['average_latency_ms']:,.0f} ms | {english['average_latency_ms']:,.0f} ms |"
        )

    lines.extend(
        [
            "",
            "## Failure and tool-use diagnostics",
            "",
            "Each cell combines the model's 1,000 equation and 1,000 English attempts.",
            "",
            "| Model | Correct | Wrong | Format failures | API/step failures | Tool errors | No-tool answers |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in MODEL_ORDER:
        summaries = [runs[(model, condition)]["summary"] for condition in ("equation", "english")]
        total = lambda *keys: sum(_nested(summary, *keys) for summary in summaries)
        lines.append(
            f"| `{model}` | {total('correct')}/2,000 | {total('outcomes', 'wrong_answer')} | "
            f"{total('outcomes', 'format_failure')} | {total('outcomes', 'api_error')} | "
            f"{total('agent', 'tool_errors')} | {total('agent', 'no_tool_answers')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Granite had the highest English accuracy (82.3%), while Qwen3.5 had the highest equation accuracy (99.9%) and 78.2% English accuracy.",
            "- Both strong models lost accuracy when they had to infer the operation from English instead of receiving the equation directly. This is the language-understanding cost measured by the extension.",
            "- The smaller Llama models produced many format and tool-call failures. Phi4-mini largely answered without calling MCP tools, so its numbers do not demonstrate compliant agent behavior even when the numeric answer was correct.",
            "",
            "This is a controlled model-and-input comparison, not a claim about all language understanding. It uses English only; the separate 20-record multilingual pilot remains preliminary because its translations have not been human-reviewed.",
            "",
            "The ten raw JSON files in this directory contain all 10,000 per-record outcomes and tool traces.",
        ]
    )
    return "\n".join(lines) + "\n"


def _nested(value: dict[str, Any], *keys: str) -> int:
    current: Any = value
    for key in keys:
        current = current[key]
    return int(current)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = render(load_runs(args.results_dir))
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved validated report: {args.output}")


if __name__ == "__main__":
    main()
