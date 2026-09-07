"""Normalize raw datasets into a common Phase 1 schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils import (
    PROCESSED_DIR,
    STANDARD_COLUMNS,
    TARGET_LANGUAGE_CODES,
    TARGET_LANGUAGE_NAMES,
    check_script_sanity,
    language_folder,
    make_sample_id,
    normalize_text,
)


def _extract_reference_answer(answers: Any) -> tuple[str | None, int | None]:
    """Extract primary reference answer and answer_start from SQuAD-style answers."""
    if answers is None:
        return None, None

    # Handle numpy arrays / pandas Series from parquet
    if hasattr(answers, "tolist"):
        answers = answers.tolist()

    if isinstance(answers, dict):
        texts = answers.get("text", [])
        starts = answers.get("answer_start", [])
    elif isinstance(answers, (list, tuple)) and answers and isinstance(answers[0], dict):
        texts = [a.get("text", "") for a in answers]
        starts = [a.get("answer_start") for a in answers]
    else:
        return None, None

    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    if hasattr(starts, "tolist"):
        starts = starts.tolist()

    if texts is None or len(texts) == 0:
        return None, None

    # Use first non-empty answer when available
    for i, text in enumerate(texts):
        text_norm = normalize_text(text if text is not None else "")
        if text_norm:
            start = starts[i] if i < len(starts) else None
            return text_norm, start

    # Fall back to first entry even if empty (will be filtered later)
    first_text = normalize_text(texts[0] if texts[0] is not None else "")
    first_start = starts[0] if starts else None
    return first_text, first_start


def normalize_indicsquad_row(
    row: pd.Series, language_code: str, source_split: str
) -> dict[str, Any]:
    """Map one IndicSQuAD row to the standard schema."""
    ref_answer, answer_start = _extract_reference_answer(row.get("answers"))
    source_id = str(row.get("id", ""))

    context = normalize_text(row.get("context"))
    question = normalize_text(row.get("question"))
    title = normalize_text(row.get("title"))

    combined_text = " ".join(filter(None, [context, question, ref_answer]))
    is_suspicious, script_flags = check_script_sanity(combined_text, language_code)

    return {
        "sample_id": make_sample_id("indicsquad", language_code, source_id),
        "language": TARGET_LANGUAGE_NAMES[language_code],
        "language_code": language_code,
        "source_dataset": "indicsquad",
        "source_split": source_split,
        "source_id": source_id,
        "title": title,
        "context": context,
        "question": question,
        "reference_answer": ref_answer,
        "answer_start": answer_start,
        "task_type": "extractive",
        "is_suspicious_language": is_suspicious,
        "validation_flags": script_flags,
    }


def normalize_indicqa_row(row: pd.Series, language_code: str) -> dict[str, Any]:
    """Map one flattened IndicQA row to the standard schema."""
    ref_answer, answer_start = _extract_reference_answer(row.get("answers"))
    source_id = str(row.get("id", ""))
    source_split = str(row.get("source_split", "test"))

    context = normalize_text(row.get("context"))
    question = normalize_text(row.get("question"))
    title = normalize_text(row.get("title"))

    task_type = "extractive" if answer_start is not None and ref_answer else "abstractive"

    combined_text = " ".join(filter(None, [context, question, ref_answer or ""]))
    is_suspicious, script_flags = check_script_sanity(combined_text, language_code)

    return {
        "sample_id": make_sample_id("indicqa", language_code, source_id),
        "language": TARGET_LANGUAGE_NAMES[language_code],
        "language_code": language_code,
        "source_dataset": "indicqa",
        "source_split": source_split,
        "source_id": source_id,
        "title": title,
        "context": context,
        "question": question,
        "reference_answer": ref_answer,
        "answer_start": answer_start,
        "task_type": task_type,
        "is_suspicious_language": is_suspicious,
        "validation_flags": script_flags,
    }


def load_and_normalize_language(language_code: str) -> pd.DataFrame:
    """Load raw files for one language and return a normalized DataFrame."""
    folder = language_folder(language_code)
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / folder

    records: list[dict[str, Any]] = []

    # IndicSQuAD splits
    for split in ("train", "validation", "test"):
        path = raw_dir / f"indicsquad_{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            records.append(normalize_indicsquad_row(row, language_code, split))

    # IndicQA (evaluation benchmark)
    indicqa_path = raw_dir / f"indicqa_{language_code}.json"
    if indicqa_path.exists():
        from download_data import flatten_indicqa_json

        indicqa_df = flatten_indicqa_json(indicqa_path, language_code)
        for _, row in indicqa_df.iterrows():
            records.append(normalize_indicqa_row(row, language_code))

    normalized = pd.DataFrame(records)
    if normalized.empty:
        return normalized

    # Ensure column order
    for col in STANDARD_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = None
    normalized = normalized[STANDARD_COLUMNS]

    out_path = PROCESSED_DIR / folder / "normalized.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_parquet(out_path, index=False)
    print(f"  [{language_code}] normalized {len(normalized):,} rows -> {out_path}")

    return normalized


def load_and_normalize_all() -> pd.DataFrame:
    """Normalize all target languages and concatenate."""
    frames = [load_and_normalize_language(lang) for lang in TARGET_LANGUAGE_CODES]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    return pd.concat(frames, ignore_index=True)
