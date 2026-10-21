"""Load and validate a fixed multilingual version of the SVAMP word problems.

The ADF source records remain unchanged.  For a non-English experiment this
module replaces only the ``Body`` and ``Question`` fields with a pre-generated,
reviewable translation file.  The ID, equation, answer, and operation type
always come from the unchanged ADF data.

Keeping translations fixed is important: translating at inference time would
make a translation model an unmeasured part of every evaluation run.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


LANGUAGES: dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese (Simplified)",
    "fr": "French",
    "ru": "Russian",
    "es": "Spanish",
}
TRANSLATED_LANGUAGE_CODES = tuple(code for code in LANGUAGES if code != "en")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[+-]?\d+(?:\.\d+)?(?![A-Za-z0-9_])")


class MultilingualDataError(ValueError):
    """Raised when a translation file cannot be used for a fair evaluation."""


def translation_path(translations_dir: Path, language: str) -> Path:
    """Return the canonical translation-file path for one non-English language."""

    if language not in LANGUAGES:
        raise MultilingualDataError(f"Unsupported language code: {language!r}.")
    if language == "en":
        raise MultilingualDataError("English uses the unchanged upstream SVAMP text and has no translation file.")
    return translations_dir / f"svamp_{language}.json"


def numbers_in_text(body: str, question: str) -> list[str]:
    """Return numeric literals in source order for a problem's visible text."""

    return NUMBER_PATTERN.findall(f"{body}\n{question}")


def dataset_sha256(rows: list[dict[str, Any]]) -> str:
    """Hash the ordered ADF rows used as a translation source."""

    source = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def validate_translation_records(
    canonical_rows: list[dict[str, Any]], records: list[dict[str, Any]], language: str
) -> dict[str, dict[str, str]]:
    """Validate a complete language file and return translations keyed by ID.

    A translation contains only ``ID``, ``Body``, and ``Question``.  It must
    cover exactly the same ordered IDs as the original data and preserve all
    numeric literals.  The number check prevents a translation error from
    silently changing the arithmetic being evaluated.
    """

    if language not in TRANSLATED_LANGUAGE_CODES:
        raise MultilingualDataError(f"Validation needs one of {TRANSLATED_LANGUAGE_CODES}, not {language!r}.")
    if not isinstance(records, list):
        raise MultilingualDataError("The translation file field 'records' must be a JSON list.")

    canonical_ids = [row.get("ID") for row in canonical_rows]
    translated_ids = [row.get("ID") for row in records]
    if any(not isinstance(problem_id, str) or not problem_id for problem_id in canonical_ids):
        raise MultilingualDataError("The canonical dataset has a missing or invalid ID.")
    if any(not isinstance(problem_id, str) or not problem_id for problem_id in translated_ids):
        raise MultilingualDataError("Every translation record needs a non-empty string ID.")
    if len(set(translated_ids)) != len(translated_ids):
        raise MultilingualDataError(f"{language} translation records contain duplicate IDs.")
    if translated_ids != canonical_ids:
        missing = sorted(set(canonical_ids) - set(translated_ids))[:5]
        unexpected = sorted(set(translated_ids) - set(canonical_ids))[:5]
        raise MultilingualDataError(
            f"{language} translation IDs must match the canonical ordered IDs exactly "
            f"(missing examples: {missing}; unexpected examples: {unexpected})."
        )

    translations: dict[str, dict[str, str]] = {}
    for original, translated in zip(canonical_rows, records, strict=True):
        body = translated.get("Body")
        question = translated.get("Question")
        if not isinstance(body, str) or not body.strip() or not isinstance(question, str) or not question.strip():
            raise MultilingualDataError(f"{language} translation for {original['ID']} has an empty Body or Question.")
        original_numbers = numbers_in_text(str(original["Body"]), str(original["Question"]))
        translated_numbers = numbers_in_text(body, question)
        if translated_numbers != original_numbers:
            raise MultilingualDataError(
                f"{language} translation for {original['ID']} changed visible numeric literals: "
                f"expected {original_numbers}, found {translated_numbers}."
            )
        translations[original["ID"]] = {"Body": body, "Question": question}
    return translations


def load_language_rows(
    canonical_rows: list[dict[str, Any]], language: str, translations_dir: Path, *, allow_machine_generated: bool = False
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return ADF rows with text in ``language`` plus data-provenance metadata."""

    if language not in LANGUAGES:
        raise MultilingualDataError(f"Unsupported language code {language!r}; choose from {tuple(LANGUAGES)}.")
    if language == "en":
        return [dict(row) for row in canonical_rows], {
            "code": "en",
            "name": LANGUAGES["en"],
            "source": "unchanged upstream SVAMP Body and Question fields",
            "canonical_rows_sha256": dataset_sha256(canonical_rows),
        }

    path = translation_path(translations_dir, language)
    if not path.is_file():
        raise MultilingualDataError(
            f"Translation file not found: {path}. Build and audit the fixed {language} corpus before running it."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise MultilingualDataError(f"Invalid JSON in {path}: {error}") from error

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise MultilingualDataError(f"{path} needs a top-level 'metadata' object.")
    if metadata.get("language") != language or metadata.get("source_language") != "en":
        raise MultilingualDataError(f"{path} metadata must identify language={language!r} and source_language='en'.")
    review_status = metadata.get("review_status")
    if review_status != "reviewed" and not (allow_machine_generated and review_status == "machine-generated; human review required before evaluation"):
        raise MultilingualDataError(
            f"{path} is not marked reviewed. Inspect the fixed translations, then set metadata.review_status to 'reviewed'."
        )
    if metadata.get("source_rows_sha256") != dataset_sha256(canonical_rows):
        raise MultilingualDataError(f"{path} was not generated from this exact ordered ADF source dataset.")
    translations = validate_translation_records(canonical_rows, payload.get("records"), language)

    translated_rows: list[dict[str, Any]] = []
    for row in canonical_rows:
        translated = dict(row)
        translated.update(translations[row["ID"]])
        translated_rows.append(translated)

    project_root = Path(__file__).resolve().parents[1]
    try:
        source = str(path.resolve().relative_to(project_root))
    except ValueError:
        source = str(path)

    return translated_rows, {
        "code": language,
        "name": LANGUAGES[language],
        "source": source,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "canonical_rows_sha256": dataset_sha256(canonical_rows),
        "translation_metadata": metadata,
        "translation_use": "preliminary_machine_translated" if review_status != "reviewed" else "reviewed",
    }
