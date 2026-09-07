"""Download raw Indic QA datasets from Hugging Face."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from utils import (
    INDICSQUAD_LANG_DIRS,
    RAW_DIR,
    TARGET_LANGUAGE_CODES,
    ensure_dirs,
    language_folder,
)


INDICQA_REPO = "ai4bharat/IndicQA"
INDICSQUAD_REPO = "l3cube-pune/indic-squad"


def download_indicsquad(language_code: str, output_dir: Path | None = None) -> dict[str, Path]:
    """
    Download IndicSQuAD for a target language using the HuggingFace datasets library.

    Returns paths to saved raw parquet files per split.
    """
    if language_code not in INDICSQUAD_LANG_DIRS:
        raise ValueError(f"Unsupported language code: {language_code}")

    lang_dir = INDICSQUAD_LANG_DIRS[language_code]
    out = output_dir or (RAW_DIR / language_folder(language_code))
    out.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(INDICSQUAD_REPO, data_dir=lang_dir)
    saved: dict[str, Path] = {}
    for split_name, split_data in dataset.items():
        path = out / f"indicsquad_{split_name}.parquet"
        df = split_data.to_pandas()
        df.to_parquet(path, index=False)
        saved[split_name] = path
        print(f"  [IndicSQuAD/{language_code}] {split_name}: {len(df):,} examples -> {path.name}")

    return saved


def download_indicqa(language_code: str, output_dir: Path | None = None) -> Path:
    """
    Download IndicQA raw JSON (HF loading script is deprecated).

    IndicQA provides a single evaluation split per language as nested SQuAD JSON.
    """
    out = output_dir or (RAW_DIR / language_folder(language_code))
    out.mkdir(parents=True, exist_ok=True)

    remote_path = f"data/indicqa.{language_code}.json"
    local_path = hf_hub_download(INDICQA_REPO, remote_path, repo_type="dataset")
    dest = out / f"indicqa_{language_code}.json"

    with open(local_path, encoding="utf-8") as src, open(dest, "w", encoding="utf-8") as dst:
        dst.write(src.read())

    print(f"  [IndicQA/{language_code}] saved -> {dest.name}")
    return dest


def flatten_indicqa_json(json_path: Path, language_code: str) -> pd.DataFrame:
    """Flatten nested IndicQA SQuAD-style JSON into tabular rows."""
    with open(json_path, encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    rows: list[dict[str, Any]] = []
    for article in payload.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                answers = qa.get("answers", [])
                answer_texts = [a.get("text", "") for a in answers]
                answer_starts = [a.get("answer_start") for a in answers]
                rows.append(
                    {
                        "id": str(qa.get("id", "")),
                        "title": title,
                        "context": context,
                        "question": qa.get("question", ""),
                        "answers": {"text": answer_texts, "answer_start": answer_starts},
                        "category": qa.get("category", ""),
                        "language_code": language_code,
                        "source_split": "test",
                    }
                )

    return pd.DataFrame(rows)


def download_all_raw() -> dict[str, dict[str, Any]]:
    """Download all raw data for Hindi, Marathi, and Tamil."""
    ensure_dirs()
    manifest: dict[str, dict[str, Any]] = {}

    for lang in TARGET_LANGUAGE_CODES:
        print(f"\nDownloading data for {lang}...")
        manifest[lang] = {
            "indicsquad": download_indicsquad(lang),
            "indicqa_json": download_indicqa(lang),
        }

    return manifest


if __name__ == "__main__":
    download_all_raw()
