"""Shared utilities for Phase 1 dataset preparation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility & configuration
# ---------------------------------------------------------------------------

SEED = 42
BALANCE_LANGUAGES = False
MAX_SAMPLES_PER_LANGUAGE: int | None = None

TARGET_LANGUAGE_CODES = ("hi", "mr", "ta")
TARGET_LANGUAGE_NAMES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
}

# Official IndicSQuAD folder names on HuggingFace
INDICSQUAD_LANG_DIRS = {
    "hi": "Hindi",
    "mr": "Marathi",
    "ta": "Tamil",
}

# Train / validation / test proportions (used only when generating new splits)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# Unicode script ranges for sanity checks
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
REPORTS_DIR = PROJECT_ROOT / "reports"

STANDARD_COLUMNS = [
    "sample_id",
    "language",
    "language_code",
    "source_dataset",
    "source_split",
    "split",
    "source_id",
    "title",
    "context",
    "question",
    "reference_answer",
    "answer_start",
    "task_type",
    "is_suspicious_language",
    "validation_flags",
]


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)


def ensure_dirs() -> None:
    """Create expected project directories if missing."""
    for path in [
        RAW_DIR / "hindi",
        RAW_DIR / "marathi",
        RAW_DIR / "tamil",
        PROCESSED_DIR / "hindi",
        PROCESSED_DIR / "marathi",
        PROCESSED_DIR / "tamil",
        FINAL_DIR,
        REPORTS_DIR / "figures",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def make_sample_id(source_dataset: str, language_code: str, source_id: str) -> str:
    """Deterministic, reproducible sample identifier."""
    raw = f"{source_dataset}|{language_code}|{source_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_text(text: str | None) -> str | None:
    """
    Unicode-safe normalization for Indic scripts.

  Preserves Devanagari/Tamil characters, digits, and meaningful punctuation.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)

    text = unicodedata.normalize("NFC", text)
    text = CONTROL_CHAR_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text if text else None


def word_count(text: str | None) -> int:
    """Whitespace-token count (not model tokens)."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def char_count(text: str | None) -> int:
    if not text or not isinstance(text, str):
        return 0
    return len(text)


def dedup_key_context_question_answer(row: pd.Series) -> str:
    parts = [
        str(row.get("context", "")),
        str(row.get("question", "")),
        str(row.get("reference_answer", "")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def dedup_key_context_question(row: pd.Series) -> str:
    parts = [str(row.get("context", "")), str(row.get("question", ""))]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def dedup_key_context(row: pd.Series) -> str:
    return hashlib.sha256(str(row.get("context", "")).encode("utf-8")).hexdigest()


def check_script_sanity(
    text: str, language_code: str
) -> tuple[bool, list[str]]:
    """
    Script sanity check using dataset language metadata as primary label.

    Hindi and Marathi both use Devanagari; script alone cannot separate them.
    """
    flags: list[str] = []
    if not text:
        return False, ["empty_text"]

    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_tamil = bool(TAMIL_RE.search(text))

    if language_code == "ta":
        if not has_tamil:
            flags.append("missing_tamil_script")
        if has_devanagari and not has_tamil:
            flags.append("unexpected_devanagari_in_tamil")
    elif language_code in ("hi", "mr"):
        if not has_devanagari:
            flags.append("missing_devanagari_script")
        if has_tamil and not has_devanagari:
            flags.append("unexpected_tamil_script_in_devanagari_lang")

    is_suspicious = len(flags) > 0
    return is_suspicious, flags


def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, base_path: Path) -> None:
    """Save CSV (UTF-8) and Parquet versions."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(base_path.with_suffix(".csv"), index=False, encoding="utf-8")
    df.to_parquet(base_path.with_suffix(".parquet"), index=False)


def language_folder(language_code: str) -> str:
    return {"hi": "hindi", "mr": "marathi", "ta": "tamil"}[language_code]
