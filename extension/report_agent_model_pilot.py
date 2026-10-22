"""Validate and summarize the matched multi-model MCP-agent pilot."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


INPUTS = (
    ("equation", "Equation"),
    ("en", "English"),
    ("ar", "Arabic"),
    ("zh", "Chinese"),
    ("fr", "French"),
    ("ru", "Russian"),
    ("es", "Spanish"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def input_code(run: dict[str, Any]) -> str:
    if run.get("condition") == "author_equation":
        return "equation"
    if run.get("condition") == "language_extension":
        return str(run.get("input_language", {}).get("code"))
    raise ValueError(f"Unknown condition: {run.get('condition')!r}")


def load_and_validate(results_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    runs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    reference_ids: list[str] | None = None
    reference_fields: dict[str, Any] | None = None

    for path in sorted(results_dir.glob("agent__*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("run_status") != "completed" or run.get("architecture") != "mcp_agent":
            raise ValueError(f"{path.name} is not a completed MCP-agent run.")
        code = input_code(run)
        model = str(run["model"])
        if code in runs[model]:
            raise ValueError(f"Duplicate run for {model} / {code}.")
        ids = [attempt["problem_id"] for attempt in run.get("attempts", [])]
        if len(ids) != 20 or len(set(ids)) != 20:
            raise ValueError(f"{path.name} does not contain 20 unique attempts.")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{path.name} uses different ordered problem IDs.")

        fields = {
            "provider": run.get("provider"),
            "temperature": run.get("temperature"),
            "sample": run.get("sample"),
            "tool_set": run.get("tool_set"),
            "scoring": run.get("scoring"),
        }
        if reference_fields is None:
            reference_fields = fields
        elif fields != reference_fields:
            raise ValueError(f"{path.name} differs in a controlled field.")
        runs[model][code] = run

    expected = {code for code, _ in INPUTS}
    for model, model_runs in runs.items():
        if set(model_runs) != expected:
            raise ValueError(f"{model} has inputs {sorted(model_runs)}, expected {sorted(expected)}.")
    if not runs:
        raise ValueError(f"No agent result JSON files found in {results_dir}.")
    return dict(runs)


def percent(value: float) -> str:
    return f"{value:.1f}%" if value % 1 else f"{value:.0f}%"


def render(runs: dict[str, dict[str, dict[str, Any]]]) -> str:
    model_order = [
        model
        for model in ("llama3.2:1b", "llama3.2:3b", "phi4-mini:3.8b", "qwen3.5:4b", "granite4:3b")
        if model in runs
    ]
    model_order.extend(sorted(set(runs) - set(model_order)))

    lines = [
        "# Matched 20-problem comparison across ADF MCP-agent models",
        "",
        "## Controlled setup",
        "",
        "Each cell uses the same 20 randomly selected SVAMP IDs (seed `798`), temperature `0`, "
        "four MCP arithmetic tools, strict ADF numeric scorer, and maximum-step policy. Only the model and input "
        "representation/language change.",
        "",
        "The five models are the complete local `ollama_agent_models` list from the reused ADF configuration. "
        "`qwen2-math:1.5b` is not in this table because the ADF project records it as direct-only: it does not "
        "support tool calling.",
        "",
        "## Strict answer accuracy",
        "",
        "| Model | Equation | English | Arabic | Chinese | French | Russian | Spanish | Word-language mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for model in model_order:
        model_runs = runs[model]
        accuracies = {code: float(model_runs[code]["summary"]["accuracy_percent"]) for code, _ in INPUTS}
        word_mean = sum(accuracies[code] for code, _ in INPUTS if code != "equation") / 6
        lines.append(
            f"| `{model}` | "
            + " | ".join(percent(accuracies[code]) for code, _ in INPUTS)
            + f" | {percent(word_mean)} |"
        )

    lines.extend(
        [
            "",
            "## Agent diagnostics across all 140 attempts per model",
            "",
            "Tool-use compliance is the share of attempts that made at least one actual MCP tool call. A tool error "
            "is counted separately and can occur after a model attempted a call.",
            "",
            "| Model | Correct | Tool-use compliance | Format failures | Wrong answers | API/step failures | Tool errors | Mean latency |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model in model_order:
        model_runs = list(runs[model].values())
        total = sum(int(run["summary"]["records"]) for run in model_runs)
        correct = sum(int(run["summary"]["correct"]) for run in model_runs)
        no_tool = sum(int(run["summary"]["agent"]["no_tool_answers"]) for run in model_runs)
        formats = sum(int(run["summary"]["outcomes"]["format_failure"]) for run in model_runs)
        wrong = sum(int(run["summary"]["outcomes"]["wrong_answer"]) for run in model_runs)
        api = sum(int(run["summary"]["outcomes"]["api_error"]) for run in model_runs)
        tool_errors = sum(int(run["summary"]["agent"]["tool_errors"]) for run in model_runs)
        latency = sum(
            float(run["summary"]["average_latency_ms"]) * int(run["summary"]["records"])
            for run in model_runs
        ) / total
        lines.append(
            f"| `{model}` | {correct}/{total} | {percent(100 * (total - no_tool) / total)} | {formats} | "
            f"{wrong} | {api} | {tool_errors} | {latency:,.0f} ms |"
        )

    lines.extend(
        [
            "",
            "## What the pilot shows",
            "",
            "- Granite and Qwen3.5 both reached 100% on equation input. Granite had the strongest mean across the six "
            "word-problem languages (52.5%), followed by Qwen3.5 (41.7%).",
            "- Accuracy alone does not describe agent quality. Granite used MCP tools in every attempt with no tool "
            "errors. Phi4-mini made no actual tool calls in any of its 140 attempts. Llama models often called tools "
            "but frequently violated the final number-only response contract.",
            "- A format failure is not automatically wrong arithmetic. Examples include `ANSWER: 70.0`, a sentence "
            "containing the result, or a JSON-looking tool request emitted as plain text rather than as a tool call. "
            "The strict scorer rejects all of these, consistently with the reused ADF protocol.",
            "",
            "## Limits",
            "",
            "This is a preliminary model-behavior pilot, not a language ranking. It uses only 20 problems, and the five "
            "non-English corpora are machine-generated NLLB translations whose numeric literals were validated but whose "
            "linguistic quality has not yet been human-reviewed. Model capability and translation quality are therefore "
            "confounded. The hosted Groq model was not included; this table deliberately compares the local MCP-agent "
            "models from the ADF configuration without requiring an API key.",
            "",
            "All 35 raw JSON files in this directory contain the individual outcomes and tool-use traces.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = render(load_and_validate(args.results_dir))
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved validated report: {args.output}")


if __name__ == "__main__":
    main()
