"""Validation, deduplication, splitting, EDA, and quality checks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import (
    BALANCE_LANGUAGES,
    FINAL_DIR,
    MAX_SAMPLES_PER_LANGUAGE,
    REPORTS_DIR,
    SEED,
    STANDARD_COLUMNS,
    TARGET_LANGUAGE_CODES,
    TARGET_LANGUAGE_NAMES,
    char_count,
    dedup_key_context,
    dedup_key_context_question,
    dedup_key_context_question_answer,
    save_dataframe,
    save_json,
    set_seed,
    word_count,
)


def _is_corrupted(text: str | None) -> bool:
    if not text:
        return True
    # Heuristic: extremely short or mostly replacement characters
    if len(text) < 2:
        return True
    replacement_ratio = text.count("\ufffd") / max(len(text), 1)
    return replacement_ratio > 0.1


def _verify_extractive_span(row: pd.Series) -> tuple[bool, str | None]:
    """Verify answer span alignment for extractive examples."""
    if row.get("task_type") != "extractive":
        return True, None

    context = row.get("context")
    answer = row.get("reference_answer")
    start = row.get("answer_start")

    if not answer or not context:
        return False, "missing_context_or_answer"

    if start is None or (isinstance(start, float) and np.isnan(start)):
        # Some extractive rows may lack span metadata; check substring instead
        if answer in context:
            return True, None
        return False, "answer_not_in_context"

    try:
        start_int = int(start)
    except (TypeError, ValueError):
        return False, "invalid_answer_start"

    end_int = start_int + len(answer)
    if context[start_int:end_int] == answer:
        return True, None

    # Allow minor whitespace/normalization mismatches: check substring presence
    if answer in context:
        return True, "span_mismatch_but_substring_found"

    return False, "answer_span_failure"


def validate_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate every example and return cleaned DataFrame plus statistics.

    Invalid examples are removed; suspicious language examples are flagged but kept.
    """
    stats: dict[str, Any] = {
        "original_total": len(df),
        "by_language": {},
        "by_source": {},
        "removal_reasons": defaultdict(int),
    }

    valid_rows: list[dict[str, Any]] = []

    for lang in TARGET_LANGUAGE_CODES:
        stats["by_language"][lang] = {
            "original": 0,
            "valid": 0,
            "removed": 0,
            "duplicates_removed": 0,
            "missing_field": 0,
            "answer_span_failures": 0,
            "suspicious_language": 0,
        }

    for source in df["source_dataset"].unique():
        stats["by_source"][source] = {"original": 0, "valid": 0, "removed": 0}

    for _, row in df.iterrows():
        lang = row["language_code"]
        source = row["source_dataset"]
        stats["by_language"][lang]["original"] += 1
        stats["by_source"][source]["original"] += 1

        flags = list(row.get("validation_flags") or [])
        remove = False
        reason = None

        for field in ("context", "question", "reference_answer"):
            if not row.get(field):
                remove = True
                reason = f"missing_{field}"
                stats["by_language"][lang]["missing_field"] += 1
                break

        if not remove and _is_corrupted(row["context"]):
            remove = True
            reason = "corrupted_context"
        if not remove and _is_corrupted(row["question"]):
            remove = True
            reason = "corrupted_question"

        if not remove:
            span_ok, span_flag = _verify_extractive_span(row)
            if not span_ok:
                remove = True
                reason = span_flag
                stats["by_language"][lang]["answer_span_failures"] += 1
            elif span_flag:
                flags.append(span_flag)

        if row.get("is_suspicious_language"):
            stats["by_language"][lang]["suspicious_language"] += 1

        if remove:
            stats["removal_reasons"][reason] += 1
            stats["by_language"][lang]["removed"] += 1
            stats["by_source"][source]["removed"] += 1
            continue

        row_dict = row.to_dict()
        row_dict["validation_flags"] = flags
        valid_rows.append(row_dict)
        stats["by_language"][lang]["valid"] += 1
        stats["by_source"][source]["valid"] += 1

    valid_df = pd.DataFrame(valid_rows)
    if valid_df.empty:
        stats["valid_total"] = 0
        stats["removed_total"] = stats["original_total"]
        return valid_df, stats

    valid_df, dedup_stats = deduplicate(valid_df)
    stats["deduplication"] = dedup_stats
    stats["valid_total"] = len(valid_df)
    stats["removed_total"] = stats["original_total"] - len(valid_df)

    for lang, count in dedup_stats.get("duplicates_by_language", {}).items():
        stats["by_language"][lang]["duplicates_removed"] = count
        stats["by_language"][lang]["valid"] -= count
        stats["by_language"][lang]["removed"] += count

    return valid_df, stats


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Remove exact duplicates and question-context duplicates within each split.

    Cross-split leakage is handled separately in resolve_cross_split_leakage().
    """
    dedup_stats: dict[str, Any] = {
        "exact_duplicates_removed": 0,
        "context_question_duplicates_removed": 0,
        "duplicates_by_language": defaultdict(int),
    }

    df = df.copy()
    df["_exact_key"] = df.apply(dedup_key_context_question_answer, axis=1)
    df["_cq_key"] = df.apply(dedup_key_context_question, axis=1)

    # Exact duplicates: keep first occurrence (deterministic sort)
    df = df.sort_values(["source_dataset", "source_split", "source_id"]).reset_index(drop=True)
    before = len(df)
    dup_mask = df.duplicated(subset=["_exact_key"], keep="first")
    for lang in df.loc[dup_mask, "language_code"]:
        dedup_stats["duplicates_by_language"][lang] += 1
    df = df[~dup_mask]
    dedup_stats["exact_duplicates_removed"] = before - len(df)

    # Question-context duplicates within same split
    before = len(df)
    df = df.sort_values(["source_dataset", "source_split", "source_id"]).reset_index(drop=True)
    cq_dup = df.duplicated(subset=["_cq_key", "source_split"], keep="first")
    for lang in df.loc[cq_dup, "language_code"]:
        dedup_stats["duplicates_by_language"][lang] += 1
    df = df[~cq_dup]
    dedup_stats["context_question_duplicates_removed"] = before - len(df)

    df = df.drop(columns=["_exact_key", "_cq_key"])
    return df, dedup_stats


def resolve_cross_split_leakage(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prevent identical context+question items from appearing across train/val/test.

    Priority: test > validation > train (official test data is preserved).
    """
    before_total = len(df)
    df = df.copy()
    df["_cq_key"] = df.apply(dedup_key_context_question, axis=1)

    split_priority = {"test": 3, "validation": 2, "train": 1, "val": 2}

    def _priority(split: str) -> int:
        return split_priority.get(str(split).lower(), 0)

    # Map split priority from assigned final split, not source_split
    df["_split_priority"] = df["split"].map(_priority)

    # Keep highest-priority row per context+question
    df = df.sort_values("_split_priority", ascending=False)
    dup_mask = df.duplicated(subset=["_cq_key"], keep="first")

    # Count removals per split
    removed_splits = df.loc[dup_mask, "split"].value_counts().to_dict()
    moved_or_removed_from_train = int(removed_splits.get("train", 0))
    moved_or_removed_from_val = int(
        removed_splits.get("validation", 0) + removed_splits.get("val", 0)
    )

    df = df[~dup_mask].reset_index(drop=True)
    leakage_stats = {
        "moved_or_removed_from_train": moved_or_removed_from_train,
        "moved_or_removed_from_val": moved_or_removed_from_val,
        "total_cross_split_duplicates_removed": int(before_total - len(df)),
    }

    df = df.drop(columns=["_cq_key", "_split_priority"])
    return df, leakage_stats


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign final train/validation/test splits.

    Strategy:
    - IndicSQuAD: retain official train/validation/test splits (context-grouped by SQuAD design).
    - IndicQA: all examples are benchmark/eval data -> assigned to test split.
    - After combining, cross-split deduplication ensures no context+question leakage.
    """
    df = df.copy()

    def _map_split(row: pd.Series) -> str:
        if row["source_dataset"] == "indicqa":
            return "test"
        split = str(row["source_split"]).lower()
        if split in ("validation", "val"):
            return "validation"
        if split == "test":
            return "test"
        return "train"

    df["split"] = df.apply(_map_split, axis=1)
    return df


def optionally_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally subsample to balance languages or cap per-language count."""
    if not BALANCE_LANGUAGES and MAX_SAMPLES_PER_LANGUAGE is None:
        return df

    set_seed()
    parts = []
    for lang in TARGET_LANGUAGE_CODES:
        lang_df = df[df["language_code"] == lang]
        if MAX_SAMPLES_PER_LANGUAGE is not None:
            lang_df = lang_df.sample(
                n=min(len(lang_df), MAX_SAMPLES_PER_LANGUAGE), random_state=SEED
            )
        parts.append(lang_df)

    result = pd.concat(parts, ignore_index=True)
    if BALANCE_LANGUAGES:
        min_count = min(len(p) for p in parts)
        parts = [p.sample(n=min_count, random_state=SEED) for p in parts]
        result = pd.concat(parts, ignore_index=True)

    return result


def compute_eda_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute exploratory statistics per language."""
    eda: dict[str, Any] = {"by_language": {}, "natural_distribution": {}}

    for lang in TARGET_LANGUAGE_CODES:
        lang_df = df[df["language_code"] == lang]
        eda["natural_distribution"][lang] = len(lang_df)

        if lang_df.empty:
            eda["by_language"][lang] = {}
            continue

        def _length_stats(series: pd.Series, name: str) -> dict[str, float]:
            chars = series.apply(char_count)
            words = series.apply(word_count)
            return {
                f"avg_{name}_chars": float(chars.mean()),
                f"median_{name}_chars": float(chars.median()),
                f"min_{name}_chars": int(chars.min()),
                f"max_{name}_chars": int(chars.max()),
                f"avg_{name}_words": float(words.mean()),
                f"median_{name}_words": float(words.median()),
                f"min_{name}_words": int(words.min()),
                f"max_{name}_words": int(words.max()),
            }

        lang_stats: dict[str, Any] = {"n_samples": len(lang_df)}
        lang_stats.update(_length_stats(lang_df["context"], "context"))
        lang_stats.update(_length_stats(lang_df["question"], "question"))
        lang_stats.update(_length_stats(lang_df["reference_answer"], "answer"))
        lang_stats["suspicious_language_count"] = int(lang_df["is_suspicious_language"].sum())
        lang_stats["by_source"] = lang_df["source_dataset"].value_counts().to_dict()
        lang_stats["by_split"] = lang_df["split"].value_counts().to_dict()

        eda["by_language"][lang] = lang_stats

    return eda


def create_visualizations(df: pd.DataFrame, output_dir: str | None = None) -> list[str]:
    """Create publication-friendly EDA plots."""
    fig_dir = Path(output_dir) if output_dir else REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    # 1. Samples per language
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = [len(df[df["language_code"] == lang]) for lang in TARGET_LANGUAGE_CODES]
    labels = [TARGET_LANGUAGE_NAMES[lang] for lang in TARGET_LANGUAGE_CODES]
    bars = ax.bar(labels, counts, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
    ax.set_title("Samples per Language (Phase 1)")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count:,}",
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path1 = fig_dir / "samples_per_language.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    saved.append(str(path1))

    # 2-4. Length distributions
    for field, fname, color in [
        ("context", "context_length_distribution", "#FF6B6B"),
        ("question", "question_length_distribution", "#4ECDC4"),
        ("reference_answer", "answer_length_distribution", "#45B7D1"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for lang in TARGET_LANGUAGE_CODES:
            lang_df = df[df["language_code"] == lang]
            lengths = lang_df[field].apply(word_count)
            if len(lengths) > 0:
                ax.hist(
                    lengths,
                    bins=50,
                    alpha=0.5,
                    label=TARGET_LANGUAGE_NAMES[lang],
                    density=True,
                )
        ax.set_title(f"{field.replace('_', ' ').title()} Length Distribution (words)")
        ax.set_xlabel("Word count")
        ax.set_ylabel("Density")
        ax.legend()
        fig.tight_layout()
        path = fig_dir / f"{fname}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        saved.append(str(path))

    return saved


def quality_assertions(df: pd.DataFrame) -> None:
    """Automated quality checks; raises AssertionError on failure."""
    assert df["sample_id"].is_unique, "sample_id must be unique"
    assert df["context"].notna().all(), "context must not be null"
    assert df["question"].notna().all(), "question must not be null"
    assert df["reference_answer"].notna().all(), "reference_answer must not be null"
    assert (df["context"].str.len() > 0).all(), "context must be non-empty"
    assert (df["question"].str.len() > 0).all(), "question must be non-empty"
    assert (df["reference_answer"].str.len() > 0).all(), "reference_answer must be non-empty"
    assert set(df["language_code"].unique()).issubset(set(TARGET_LANGUAGE_CODES))
    assert set(df["split"].unique()).issubset({"train", "validation", "test"})


def sample_for_inspection(df: pd.DataFrame, n_per_lang: int = 3) -> pd.DataFrame:
    """Random sample for manual quality inspection."""
    set_seed()
    samples = []
    for lang in TARGET_LANGUAGE_CODES:
        lang_df = df[df["language_code"] == lang]
        if len(lang_df) == 0:
            continue
        n = min(n_per_lang, len(lang_df))
        samples.append(lang_df.sample(n=n, random_state=SEED))
    return pd.concat(samples, ignore_index=True)[
        ["language", "source_dataset", "context", "question", "reference_answer"]
    ]


def run_validation_pipeline(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Full validation, dedup, split assignment, and EDA pipeline."""
    valid_df, val_stats = validate_dataframe(df)
    valid_df = assign_splits(valid_df)
    valid_df, leakage_stats = resolve_cross_split_leakage(valid_df)
    val_stats["cross_split_leakage"] = leakage_stats

    # Record deduplication & leakage removals in removal_reasons so all reasons reconcile
    val_stats["removal_reasons"]["exact_duplicates"] = val_stats["deduplication"]["exact_duplicates_removed"]
    val_stats["removal_reasons"]["context_question_duplicates"] = val_stats["deduplication"]["context_question_duplicates_removed"]
    val_stats["removal_reasons"]["cross_split_leakage_removed"] = leakage_stats["total_cross_split_duplicates_removed"]

    # Reconcile overall totals
    val_stats["valid_total"] = len(valid_df)
    val_stats["removed_total"] = val_stats["original_total"] - len(valid_df)

    valid_df = optionally_balance(valid_df)
    eda_stats = compute_eda_stats(valid_df)
    val_stats["eda"] = eda_stats

    quality_assertions(valid_df)
    val_stats["quality_samples"] = sample_for_inspection(valid_df).to_dict(orient="records")

    return valid_df, val_stats


def export_final_splits(df: pd.DataFrame, stats: dict[str, Any]) -> None:
    """Save final CSV/Parquet outputs and statistics."""
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    save_dataframe(df, FINAL_DIR / "phase1_dataset")
    save_dataframe(df[df["split"] == "train"], FINAL_DIR / "train")
    save_dataframe(df[df["split"] == "validation"], FINAL_DIR / "validation")
    save_dataframe(df[df["split"] == "test"], FINAL_DIR / "test")

    stats["split_counts"] = df["split"].value_counts().to_dict()
    stats["final_total"] = len(df)
    save_json(stats, REPORTS_DIR / "phase1_statistics.json")

    create_visualizations(df)
    print(f"\nExported {len(df):,} samples to {FINAL_DIR}")
    print(f"Statistics saved to {REPORTS_DIR / 'phase1_statistics.json'}")
