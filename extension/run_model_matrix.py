"""Plan or run the full ADF model matrix for the language extension.

The upstream ADF repository has two distinct model lists.  Direct LLM runs and
MCP-agent runs must use their respective lists because qwen2-math does not
support tool calling and qwen3.5 was evaluated only as an agent.  This script
keeps that distinction explicit and creates one result JSON per fair run.

It prints a plan by default.  Use ``--execute`` only after the requested local
models, the MCP server (for agent jobs), and fixed multilingual files are
available.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CODE = ROOT / "upstream_adf" / "code"
if str(UPSTREAM_CODE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_CODE))

from config import load_dataset, load_llm_config  # noqa: E402

from multilingual_data import LANGUAGES, MultilingualDataError, load_language_rows  # noqa: E402


@dataclass(frozen=True)
class Job:
    architecture: str
    condition: str
    language: str
    provider: str
    model: str
    temperature: float
    output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("direct", "agent", "all"), default="all")
    parser.add_argument("--provider", choices=("groq", "ollama", "all"), default="all")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["all"],
        help="Word-problem languages: any codes from en ar zh fr ru es, or the single value all.",
    )
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sample", choices=("first", "random"), default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        help="Temperatures to run. Defaults to the upstream temperature list.",
    )
    parser.add_argument("--translations-dir", type=Path, default=ROOT / "data" / "multilingual_svamp")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "matrix_runs")
    parser.add_argument("--python", type=Path, default=Path(sys.executable), help="Python interpreter for child runs.")
    parser.add_argument("--execute", action="store_true", help="Run jobs after preflight passes. Default: print plan only.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs whose planned JSON output already exists.")
    parser.add_argument(
        "--allow-machine-generated-translations",
        action="store_true",
        help="Run non-English files marked machine-generated and label their outputs preliminary.",
    )
    parser.add_argument(
        "--skip-model",
        action="append",
        default=[],
        help="Model tag to omit from this invocation; record the reason in the study notes.",
    )
    parser.add_argument("--write-plan", type=Path, help="Optional JSON file to save the planned job matrix.")
    return parser.parse_args()


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")


def selected_languages(raw_languages: list[str]) -> list[str]:
    if raw_languages == ["all"]:
        return list(LANGUAGES)
    unknown = [language for language in raw_languages if language not in LANGUAGES]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}. Choose from {tuple(LANGUAGES)} or all.")
    if len(set(raw_languages)) != len(raw_languages):
        raise ValueError("Each language may appear only once.")
    return raw_languages


def build_jobs(
    config: dict[str, Any],
    *,
    architecture: str,
    provider: str,
    languages: list[str],
    results_dir: Path,
    temperatures: list[float] | None = None,
) -> list[Job]:
    """Build every condition/model/temperature combination without running it."""

    architectures = ("direct", "agent") if architecture == "all" else (architecture,)
    providers = ("groq", "ollama") if provider == "all" else (provider,)
    jobs: list[Job] = []
    local_temperatures = [float(value) for value in (temperatures or config["temperatures"])]
    for current_architecture in architectures:
        for current_provider in providers:
            if current_provider == "groq":
                model_temperatures = [(config["model"], float(config["temperature"]))]
            elif current_architecture == "direct":
                model_temperatures = [
                    (model, float(temperature))
                    for model in config["ollama_models"]
                    for temperature in local_temperatures
                ]
            else:
                model_temperatures = [
                    (model, float(temperature))
                    for model in config["ollama_agent_models"]
                    for temperature in local_temperatures
                ]

            for model, temperature in model_temperatures:
                conditions = [("author_equation", "en")] + [("language_extension", language) for language in languages]
                for condition, language in conditions:
                    output_name = "__".join(
                        (
                            current_architecture,
                            "equation" if condition == "author_equation" else f"word-{language}",
                            current_provider,
                            safe_token(model),
                            f"T{temperature:g}",
                        )
                    ) + ".json"
                    jobs.append(
                        Job(
                            architecture=current_architecture,
                            condition=condition,
                            language=language,
                            provider=current_provider,
                            model=model,
                            temperature=temperature,
                            output=str(results_dir / output_name),
                        )
                    )
    return jobs


def locally_available_models() -> tuple[set[str], str | None]:
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        names = {entry["name"] for entry in response.json().get("models", [])}
        return names, None
    except Exception as error:
        return set(), f"Ollama is not reachable at localhost:11434 ({error})."


def model_is_available(model: str, available: set[str]) -> bool:
    return model in available or f"{model}:latest" in available


def validate_language_files(
    languages: list[str], translations_dir: Path, *, allow_machine_generated: bool = False
) -> list[str]:
    """Return human-readable blockers for missing or invalid fixed translations."""

    if set(languages) == {"en"}:
        return []
    train, test = load_dataset()
    canonical_rows = train + test
    errors: list[str] = []
    for language in languages:
        if language == "en":
            continue
        try:
            load_language_rows(
                canonical_rows,
                language,
                translations_dir,
                allow_machine_generated=allow_machine_generated,
            )
        except MultilingualDataError as error:
            errors.append(str(error))
    return errors


def preflight(
    jobs: list[Job], languages: list[str], translations_dir: Path, *, allow_machine_generated: bool = False
) -> list[str]:
    """Find blockers without downloading models or contacting a cloud model."""

    blockers = validate_language_files(
        languages, translations_dir, allow_machine_generated=allow_machine_generated
    )
    ollama_jobs = [job for job in jobs if job.provider == "ollama"]
    if ollama_jobs:
        available, error = locally_available_models()
        if error:
            blockers.append(error)
        else:
            needed = sorted({job.model for job in ollama_jobs})
            missing = [model for model in needed if not model_is_available(model, available)]
            if missing:
                blockers.append("Missing local Ollama models: " + ", ".join(missing) + ".")
    if any(job.architecture == "agent" for job in jobs):
        try:
            response = requests.get("http://localhost:8100/mcp", timeout=3)
            if response.status_code >= 500:
                blockers.append("The MCP server responded with a server error at localhost:8100/mcp.")
        except Exception:
            blockers.append("MCP calculator server is not reachable at localhost:8100/mcp.")
    return blockers


def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    script = ROOT / "extension" / ("run_direct_experiment.py" if job.architecture == "direct" else "run_experiment.py")
    command = [
        str(args.python),
        str(script),
        "--condition",
        job.condition,
        "--provider",
        job.provider,
        "--model",
        job.model,
        "--temperature",
        str(job.temperature),
        "--limit",
        str(args.limit),
        "--sample",
        args.sample,
        "--seed",
        str(args.seed),
        "--output",
        job.output,
    ]
    if job.condition == "language_extension":
        command.extend(("--language", job.language, "--translations-dir", str(args.translations_dir)))
        if args.allow_machine_generated_translations:
            command.append("--allow-machine-generated-translations")
    return command


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 1000:
        raise SystemExit("--limit must be between 1 and 1000.")
    languages = selected_languages(args.languages)
    config = load_llm_config()
    jobs = build_jobs(
        config,
        architecture=args.architecture,
        provider=args.provider,
        languages=languages,
        results_dir=args.results_dir,
        temperatures=args.temperatures,
    )
    if args.skip_model:
        skipped = set(args.skip_model)
        jobs = [job for job in jobs if job.model not in skipped]
    plan = {"languages": languages, "jobs": [asdict(job) for job in jobs]}
    if args.write_plan:
        args.write_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(f"Saved plan: {args.write_plan}")

    print(f"Planned jobs: {len(jobs)}")
    print("Distinct models: " + ", ".join(sorted({job.model for job in jobs})))
    for job in jobs:
        print(f"- {job.architecture:6} {job.condition:18} {job.language:2} {job.provider:6} {job.model:20} T={job.temperature:g}")

    blockers = preflight(
        jobs,
        languages,
        args.translations_dir,
        allow_machine_generated=args.allow_machine_generated_translations,
    )
    if blockers:
        print("\nPreflight blockers:")
        for blocker in blockers:
            print(f"- {blocker}")
        if args.execute:
            raise SystemExit("Resolve the preflight blockers before executing the matrix.")
        print("\nPlan only; no models were downloaded or called.")
        return

    if not args.execute:
        print("\nPreflight passed. This was a plan only; rerun with --execute to start model calls.")
        return

    if any(job.provider == "groq" for job in jobs) and not os.environ.get("GROQ_API_KEY"):
        from getpass import getpass

        key = getpass("Paste Groq API key once for this matrix run: ").strip()
        if not key:
            raise SystemExit("A Groq API key is required for the planned Groq jobs.")
        # Child processes inherit this value, but it is never put in a command,
        # result file, plan, or source file.
        os.environ["GROQ_API_KEY"] = key

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for index, job in enumerate(jobs, start=1):
        output = Path(job.output)
        if args.resume and output.exists():
            print(f"[{index}/{len(jobs)}] SKIP existing {output.name}")
            continue
        print(f"[{index}/{len(jobs)}] RUN {output.name}")
        subprocess.run(command_for(job, args), check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
