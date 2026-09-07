#!/usr/bin/env python3
"""
Phase 1 — Dataset Acquisition, Validation, Cleaning & Benchmark Construction

Run from project root:
    python src/run_phase1.py

Or from src/:
    python run_phase1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from src/
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from download_data import download_all_raw
from preprocess import load_and_normalize_all
from utils import (
    FINAL_DIR,
    REPORTS_DIR,
    SEED,
    TARGET_LANGUAGE_CODES,
    TARGET_LANGUAGE_NAMES,
    ensure_dirs,
    save_json,
    set_seed,
)
from validate import export_final_splits, run_validation_pipeline

# ---------------------------------------------------------------------------
# Dataset verification (verified programmatically on 2026-03-24)
# ---------------------------------------------------------------------------

DATASET_VERIFICATION = {
    "summary_table": [
        {
            "dataset": "ai4bharat/IndicQA",
            "hindi": True,
            "marathi": True,
            "tamil": True,
            "context": True,
            "question": True,
            "answer": "Partial (1052/1547 hi, 1108/1604 mr, 1277/1804 ta have non-empty answers)",
            "license": "CC-BY-4.0",
            "usable": "Supplementary — test-only benchmark; HF loading script deprecated; load via raw JSON",
        },
        {
            "dataset": "l3cube-pune/indic-squad",
            "hindi": True,
            "marathi": True,
            "tamil": True,
            "context": True,
            "question": True,
            "answer": True,
            "license": "CC-BY-4.0",
            "usable": "Primary — extractive QA with official train/val/test splits (~118k train per language)",
        },
    ],
    "notes": [
        "Both datasets support Hindi, Marathi, and Tamil with context + question + answer fields.",
        "IndicQA is evaluation-only (no training split) and includes abstractive items without reference answers; "
        "those are filtered out in Phase 1.",
        "IndicSQuAD is the primary source due to scale, extractive spans, and official splits.",
        "No replacement dataset is required for missing languages.",
        "IndicQA HuggingFace datasets loader is deprecated (script-based); raw JSON is downloaded via huggingface_hub.",
    ],
    "indicsquad_splits_per_language": {
        "train": 118516,
        "validation": 11873,
        "test": 11803,
    },
    "indicqa_qa_pairs_per_language": {
        "hi": {"total": 1547, "with_answer": 1052},
        "mr": {"total": 1604, "with_answer": 1108},
        "ta": {"total": 1804, "with_answer": 1277},
    },
}


def print_verification_table() -> None:
    print("\n" + "=" * 72)
    print("DATASET VERIFICATION TABLE")
    print("=" * 72)
    header = f"{'Dataset':<30} {'Hindi':<7} {'Marathi':<8} {'Tamil':<7} {'Context':<8} {'Question':<9} {'Answer':<8} {'License':<12} {'Usable?'}"
    print(header)
    print("-" * len(header))
    for row in DATASET_VERIFICATION["summary_table"]:
        print(
            f"{row['dataset']:<30} "
            f"{'Yes' if row['hindi'] else 'No':<7} "
            f"{'Yes' if row['marathi'] else 'No':<8} "
            f"{'Yes' if row['tamil'] else 'No':<7} "
            f"{'Yes' if row['context'] else 'No':<8} "
            f"{'Yes' if row['question'] else 'No':<9} "
            f"{'Partial' if 'Partial' in str(row['answer']) else 'Yes':<8} "
            f"{row['license']:<12} "
            f"{row['usable'][:40]}..."
        )
    print()


def main() -> None:
    set_seed(SEED)
    ensure_dirs()

    print_verification_table()

    print("=" * 72)
    print("STEP 1: Downloading raw datasets")
    print("=" * 72)
    download_all_raw()

    print("\n" + "=" * 72)
    print("STEP 2: Normalizing to common schema")
    print("=" * 72)
    normalized_df = load_and_normalize_all()
    print(f"Total normalized rows: {len(normalized_df):,}")

    print("\n" + "=" * 72)
    print("STEP 3: Validation, deduplication, and split assignment")
    print("=" * 72)
    print("Split strategy:")
    print("  - IndicSQuAD: retain official train/validation/test splits (SQuAD context-grouped).")
    print("  - IndicQA: assign all valid examples to test (benchmark/eval only).")
    print("  - Cross-split dedup: remove context+question duplicates across splits (test > val > train).")

    final_df, stats = run_validation_pipeline(normalized_df)

    stats["dataset_verification"] = DATASET_VERIFICATION
    stats["configuration"] = {
        "SEED": SEED,
        "BALANCE_LANGUAGES": False,
        "MAX_SAMPLES_PER_LANGUAGE": None,
        "target_languages": list(TARGET_LANGUAGE_CODES),
    }

    print("\n" + "=" * 72)
    print("STEP 4: Exporting final datasets")
    print("=" * 72)
    export_final_splits(final_df, stats)

    print("\n" + "=" * 72)
    print("PHASE 1 SUMMARY")
    print("=" * 72)
    for lang in TARGET_LANGUAGE_CODES:
        count = len(final_df[final_df["language_code"] == lang])
        print(f"  {TARGET_LANGUAGE_NAMES[lang]} ({lang}): {count:,} samples")

    print(f"\n  Train:      {len(final_df[final_df['split'] == 'train']):,}")
    print(f"  Validation: {len(final_df[final_df['split'] == 'validation']):,}")
    print(f"  Test:       {len(final_df[final_df['split'] == 'test']):,}")
    print(f"\n  Output directory: {FINAL_DIR}")
    print(f"  Statistics:       {REPORTS_DIR / 'phase1_statistics.json'}")
    print("\nPhase 1 complete. Dataset is ready for Phase 2 hallucination generation.")


if __name__ == "__main__":
    main()
