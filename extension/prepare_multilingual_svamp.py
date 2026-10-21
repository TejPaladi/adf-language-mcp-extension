"""Create fixed non-English SVAMP files with the NLLB translation model.

This is a data-preparation step, not part of the model evaluation loop. It
translates the original English Body and Question fields, translates only
their non-numeric text spans, deterministically reinserts numeric literals,
and writes resumable
JSON files under ``data/multilingual_svamp``. Review the generated text before
using it as a benchmark corpus.

The optional dependencies are listed in ``requirements-translation.txt``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CODE = ROOT / "upstream_adf" / "code"
if str(UPSTREAM_CODE) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_CODE))

from config import load_dataset  # noqa: E402

from multilingual_data import (  # noqa: E402
    NUMBER_PATTERN,
    TRANSLATED_LANGUAGE_CODES,
    dataset_sha256,
    translation_path,
    validate_translation_records,
)


NLLB_TARGETS = {
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "fr": "fra_Latn",
    "ru": "rus_Cyrl",
    "es": "spa_Latn",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", nargs="+", default=["all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "multilingual_svamp")
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true", help="Continue a partial output file if it exists.")
    return parser.parse_args()


def selected_languages(raw_languages: list[str]) -> list[str]:
    if raw_languages == ["all"]:
        return list(TRANSLATED_LANGUAGE_CODES)
    unknown = [language for language in raw_languages if language not in TRANSLATED_LANGUAGE_CODES]
    if unknown:
        raise ValueError(f"Choose translation languages from {TRANSLATED_LANGUAGE_CODES}; got {unknown}.")
    if len(set(raw_languages)) != len(raw_languages):
        raise ValueError("Each translation language can appear only once.")
    return raw_languages


def protect_numbers(text: str) -> tuple[str, dict[str, str]]:
    """Replace visible numbers with stable placeholders before translation."""

    values: dict[str, str] = {}

    def alphabetic_index(index: int) -> str:
        """Return A, B, ..., Z, AA, ... without digits for NLLB markers."""

        pieces: list[str] = []
        current = index
        while True:
            pieces.append(chr(ord("A") + current % 26))
            current = current // 26 - 1
            if current < 0:
                return "".join(reversed(pieces))

    def replacement(match: Any) -> str:
        # Brackets and digits are normalized by NLLB.  This deliberately
        # unusual all-letter token survives its subword tokenizer unchanged.
        marker = f"ZXQNUM{alphabetic_index(len(values))}QXZ"
        values[marker] = match.group(0)
        return f" {marker} "

    return NUMBER_PATTERN.sub(replacement, text), values


def restore_numbers(text: str, values: dict[str, str]) -> str:
    """Restore protected values and fail if a translator changed a placeholder."""

    restored = text
    for marker, value in values.items():
        if restored.count(marker) != 1:
            raise ValueError(f"Translation did not preserve placeholder {marker!r} exactly.")
        restored = restored.replace(marker, value)
    return " ".join(restored.split())


def numeric_safe_segments(text: str) -> list[tuple[str, str]]:
    """Split text into translatable text spans and immutable numeric spans.

    NLLB can normalize or drop even deliberately unusual placeholders.  By
    keeping every source literal outside of model generation, the translated
    item retains exactly the same numbers in exactly the same order.
    """

    segments: list[tuple[str, str]] = []
    cursor = 0
    for match in NUMBER_PATTERN.finditer(text):
        if match.start() > cursor:
            segments.append(("text", text[cursor : match.start()]))
        segments.append(("number", match.group(0)))
        cursor = match.end()
    if cursor < len(text):
        segments.append(("text", text[cursor:]))
    if not segments:
        segments.append(("text", text))
    return segments


def reassemble_numeric_safe(segments: list[tuple[str, str]], translated_spans: list[str]) -> str:
    """Reassemble translated spans and source literals in their original order."""

    outputs: list[str] = []
    translated = iter(translated_spans)
    for kind, value in segments:
        if kind == "number":
            outputs.append(value)
        else:
            # A text span contains no source numeric literal.  Any standalone
            # numeric literal produced by NLLB in that span is therefore an
            # invention (for example, a hallucinated count) and must not leak
            # into a fixed arithmetic-evaluation item.
            outputs.append(NUMBER_PATTERN.sub("", next(translated)))
    # Separating spans prevents translated punctuation (for example a trailing
    # period) from changing source ``87`` into the different literal ``.87``.
    return " ".join(" ".join(outputs).split())


def choose_device(torch: Any, choice: str) -> str:
    if choice != "auto":
        if choice == "mps" and not torch.backends.mps.is_available():
            raise SystemExit("MPS was requested but is not available.")
        if choice == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return choice
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_partial(path: Path, language: str, canonical_ids: set[str], resume: bool) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    if not resume:
        raise SystemExit(f"{path} already exists. Use --resume to continue it or move it aside.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Cannot resume invalid JSON {path}: {error}") from error
    metadata = payload.get("metadata", {})
    if metadata.get("language") != language or metadata.get("source_language") != "en":
        raise SystemExit(f"Cannot resume {path}: metadata does not match language {language}.")
    records = payload.get("records", [])
    result: dict[str, dict[str, str]] = {}
    for record in records:
        problem_id = record.get("ID")
        if problem_id not in canonical_ids or problem_id in result:
            raise SystemExit(f"Cannot resume {path}: invalid or duplicate ID {problem_id!r}.")
        if not isinstance(record.get("Body"), str) or not isinstance(record.get("Question"), str):
            raise SystemExit(f"Cannot resume {path}: incomplete record {problem_id!r}.")
        result[problem_id] = {"ID": problem_id, "Body": record["Body"], "Question": record["Question"]}
    return result


def write_payload(
    path: Path,
    language: str,
    target_language: str,
    canonical_rows: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, str]],
    args: argparse.Namespace,
    device: str,
) -> None:
    ordered_records = [records_by_id[row["ID"]] for row in canonical_rows if row["ID"] in records_by_id]
    payload = {
        "schema_version": 1,
        "metadata": {
            "language": language,
            "source_language": "en",
            "source_rows_sha256": dataset_sha256(canonical_rows),
            "translation_method": {
                "backend": "NLLB",
                "model": args.model,
                "source_language_code": "eng_Latn",
                "target_language_code": target_language,
                "numeric_literals": "Protected before translation and restored verbatim afterward.",
                "batch_size": args.batch_size,
                "max_new_tokens": args.max_new_tokens,
                "device": device,
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "host": platform.platform(),
            },
            "completed_records": len(ordered_records),
            "expected_records": len(canonical_rows),
            "review_status": "machine-generated; human review required before evaluation",
        },
        "records": ordered_records,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def translate_texts(model: Any, tokenizer: Any, torch: Any, texts: list[str], target_language: str, device: str, max_new_tokens: int) -> list[str]:
    tokenizer.src_lang = "eng_Latn"
    encoded = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    encoded = {name: value.to(device) for name, value in encoded.items()}
    # NllbTokenizerFast in newer Transformers releases no longer exposes the
    # old ``lang_code_to_id`` mapping.  Both tokenizer variants support token
    # conversion, so prefer that stable public method and retain the mapping
    # only as a compatibility fallback.
    target_id = tokenizer.convert_tokens_to_ids(target_language)
    if target_id is None or target_id == tokenizer.unk_token_id:
        target_id = getattr(tokenizer, "lang_code_to_id", {}).get(target_language)
    if target_id is None:
        raise ValueError(f"NLLB tokenizer does not recognize target language {target_language!r}.")
    with torch.inference_mode():
        generated = model.generate(**encoded, forced_bos_token_id=target_id, max_new_tokens=max_new_tokens)
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def build_language(
    language: str,
    canonical_rows: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    args: argparse.Namespace,
) -> None:
    target_language = NLLB_TARGETS[language]
    path = translation_path(args.output_dir, language)
    records_by_id = load_partial(path, language, {row["ID"] for row in canonical_rows}, args.resume)
    pending = [row for row in canonical_rows if row["ID"] not in records_by_id]
    print(f"{language}: {len(records_by_id)}/{len(canonical_rows)} already present; {len(pending)} remaining.")

    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        segmented_records: list[tuple[str, list[tuple[str, str]], list[tuple[str, str]]]] = []
        text_spans: list[str] = []
        for row in batch:
            body_segments = numeric_safe_segments(row["Body"])
            question_segments = numeric_safe_segments(row["Question"])
            segmented_records.append((row["ID"], body_segments, question_segments))
            text_spans.extend(value for kind, value in body_segments + question_segments if kind == "text")
        translated_texts = translate_texts(model, tokenizer, torch, text_spans, target_language, device, args.max_new_tokens)
        if len(translated_texts) != len(text_spans):
            raise RuntimeError("NLLB returned an unexpected number of translations.")
        translated = iter(translated_texts)
        for problem_id, body_segments, question_segments in segmented_records:
            body_span_count = sum(kind == "text" for kind, _ in body_segments)
            question_span_count = sum(kind == "text" for kind, _ in question_segments)
            body = reassemble_numeric_safe(body_segments, [next(translated) for _ in range(body_span_count)])
            question = reassemble_numeric_safe(question_segments, [next(translated) for _ in range(question_span_count)])
            records_by_id[problem_id] = {"ID": problem_id, "Body": body, "Question": question}
        write_payload(path, language, target_language, canonical_rows, records_by_id, args, device)
        print(f"{language}: wrote {len(records_by_id)}/{len(canonical_rows)}")

    ordered_records = [records_by_id[row["ID"]] for row in canonical_rows]
    validate_translation_records(canonical_rows, ordered_records, language)
    print(f"{language}: complete and structurally validated at {path}")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("--batch-size and --max-new-tokens must be positive.")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
    except ImportError as error:
        raise SystemExit(
            "Install the optional translator dependencies first: "
            "pip install -r requirements-translation.txt"
        ) from error

    languages = selected_languages(args.languages)
    train, test = load_dataset()
    canonical_rows = train + test
    device = choose_device(torch, args.device)
    print(f"Loading {args.model} on {device}.")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
    model.eval()
    for language in languages:
        build_language(language, canonical_rows, model, tokenizer, torch, device, args)


if __name__ == "__main__":
    main()
