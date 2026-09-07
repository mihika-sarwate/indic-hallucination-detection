"""
Phase 2 — Faithful and Hallucinated Answer Generation via Groq API.

Generates paired faithful (label=0) and hallucinated (label=1) responses
for Hindi, Marathi, and Tamil QA examples using controlled injection strategies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    FINAL_DIR,
    SEED,
    TARGET_LANGUAGE_CODES,
    TARGET_LANGUAGE_NAMES,
    check_script_sanity,
    normalize_text,
    set_seed,
)

# Output directory for Phase 2
PHASE2_DIR = PROJECT_ROOT / "data" / "phase2"
CHECKPOINT_DIR = PHASE2_DIR / "checkpoints"

# Injection strategies
INJECTION_STRATEGIES = [
    "entity_substitution",
    "numeric_date_corruption",
    "unsupported_claim_addition",
    "contradicted_fact",
]

DEFAULT_MODEL = "qwen/qwen3.6-27b"
SCRIPT_NAMES = {
    "hi": "Devanagari script",
    "mr": "Devanagari script",
    "ta": "Tamil script",
}

# ---------------------------------------------------------------------------
# API Key Helper
# ---------------------------------------------------------------------------

def get_groq_api_key() -> str | None:
    """Retrieve Groq API key from environment, .env, or registry."""
    key = os.environ.get("GROQ_API_KEY")
    if key and key.strip():
        return key.strip()

    # Fallback check on Windows registry in case it was set globally
    if sys.platform == "win32":
        try:
            import winreg
            for root, subkey in [
                (winreg.HKEY_CURRENT_USER, r"Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            ]:
                try:
                    with winreg.OpenKey(root, subkey) as k:
                        val, _ = winreg.QueryValueEx(k, "GROQ_API_KEY")
                        if val and str(val).strip():
                            return str(val).strip()
                except Exception:
                    pass
        except Exception:
            pass

    return None


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

def get_faithful_prompts(
    language: str, language_code: str, context: str, question: str, reference_answer: str
) -> tuple[str, str]:
    script_name = SCRIPT_NAMES.get(language_code, "native script")
    system_prompt = (
        f"You are a precise multilingual QA assistant for {language}.\n"
        f"Your task is to answer the question based STRICTLY and ONLY on the provided context.\n"
        f"Rules:\n"
        f"1. Respond ONLY in {language} script ({script_name}).\n"
        f"2. Do NOT add any information not explicitly stated in the context.\n"
        f"3. Answer directly and concisely without pleasantries, conversational filler, or meta-commentary.\n"
        f"4. If the exact answer is in the text, stay as close as possible to the context phrasing."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Reference Answer:\n{reference_answer}\n\n"
        f"Answer in {language}:"
    )
    return system_prompt, user_prompt


def get_hallucinated_prompts(
    language: str,
    language_code: str,
    strategy: str,
    context: str,
    question: str,
    reference_answer: str,
) -> tuple[str, str]:
    script_name = SCRIPT_NAMES.get(language_code, "native script")

    strategy_instructions = {
        "entity_substitution": (
            f"Your task is to generate a plausible-sounding but HALLUCINATED answer by performing ENTITY SUBSTITUTION.\n"
            f"Replace a key named entity (person, place, organization, title, or object) from the true answer with a different, plausible entity "
            f"(either from another part of the context or a closely related real-world entity).\n"
            f"The answer must read naturally in {language}, but be factually WRONG according to the context."
        ),
        "numeric_date_corruption": (
            f"Your task is to generate a plausible-sounding but HALLUCINATED answer by modifying NUMERICAL VALUES, DATES, OR QUANTITIES.\n"
            f"Change numbers, dates, years, counts, percentages, or measurements in the answer to incorrect values.\n"
            f"Keep the grammatical structure natural and fluent in {language}."
        ),
        "unsupported_claim_addition": (
            f"Your task is to generate a plausible-sounding but HALLUCINATED answer by adding an UNSUPPORTED CLAIM.\n"
            f"Provide an answer that sounds related, but inject a specific, plausible detail, cause, outcome, or attribute that is completely ABSENT from the provided context.\n"
            f"The fabricated detail must not be verifiable from the text."
        ),
        "contradicted_fact": (
            f"Your task is to generate a plausible-sounding but HALLUCINATED answer by DIRECTLY CONTRADICTING the context.\n"
            f"State the direct opposite of what the context asserts (e.g., negate a positive fact, reverse a cause-and-effect relationship, or invert a status).\n"
            f"The answer must sound fluent and authoritative in {language}, but be in direct contradiction with the passage."
        ),
    }

    instruction = strategy_instructions.get(
        strategy, strategy_instructions["entity_substitution"]
    )

    system_prompt = (
        f"You are an expert NLP benchmark generator for {language}.\n"
        f"{instruction}\n\n"
        f"Strict Rules:\n"
        f"1. Respond ONLY in {language} script ({script_name}).\n"
        f"2. Output ONLY the perturbed answer text without preamble, explanations, or quotes.\n"
        f"3. Keep the length concise, matching natural question-answering style."
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Ground Truth Reference Answer:\n{reference_answer}\n\n"
        f"Generate the hallucinated answer in {language}:"
    )
    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Strategy Assignment (Deterministic)
# ---------------------------------------------------------------------------

def assign_injection_strategy(sample_id: str, seed: int = SEED) -> str:
    """Deterministically assign one of the 4 injection strategies per sample_id."""
    raw = f"{seed}|{sample_id}|injection_strategy"
    h = int(hashlib.sha256(raw.encode("utf-8")).hexdigest(), 16)
    return INJECTION_STRATEGIES[h % len(INJECTION_STRATEGIES)]


# ---------------------------------------------------------------------------
# Quality Guard Checks
# ---------------------------------------------------------------------------

def check_hallucination_quality(
    ref_answer: str,
    gen_answer: str,
    context: str,
    language_code: str,
) -> tuple[bool, list[str]]:
    """
    Automatic quality guard:
    Flags possible generation failure if:
    - Generated answer is empty or too short.
    - Generated answer is identical or >90% token overlap with reference answer.
    - Script sanity check fails (e.g. English instead of Indic script).
    """
    flags: list[str] = []
    if not gen_answer or len(gen_answer.strip()) < 2:
        flags.append("empty_or_too_short")
        return True, flags

    ref_clean = normalize_text(ref_answer) or ""
    gen_clean = normalize_text(gen_answer) or ""

    if ref_clean == gen_clean:
        flags.append("identical_to_reference")

    ref_tokens = set(ref_clean.split())
    gen_tokens = set(gen_clean.split())
    if ref_tokens and gen_tokens:
        overlap = len(ref_tokens & gen_tokens) / len(ref_tokens | gen_tokens)
        if overlap > 0.90:
            flags.append("high_token_overlap_with_reference")

    # Script check
    is_suspicious, script_flags = check_script_sanity(gen_clean, language_code)
    if is_suspicious:
        flags.extend(script_flags)

    return len(flags) > 0, flags


# ---------------------------------------------------------------------------
# Groq API Client with Backoff
# ---------------------------------------------------------------------------

class GroqGenerator:
    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        from groq import Groq
        self.client = Groq(api_key=api_key)
        self.model = model

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_retries: int = 5,
    ) -> str:
        """Generate a completion with exponential backoff for rate limits."""
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=256,
                )
                content = response.choices[0].message.content or ""
                return content.strip().strip('"').strip("'")
            except Exception as e:
                err_msg = str(e).lower()
                is_rate_limit = "429" in err_msg or "rate limit" in err_msg or "too many requests" in err_msg
                if is_rate_limit:
                    sleep_time = (2 ** attempt) + np.random.uniform(1.0, 3.0)
                    print(f"    [RateLimit] Sleeping {sleep_time:.1f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(sleep_time)
                else:
                    if attempt == max_retries - 1:
                        raise e
                    sleep_time = (2 ** attempt) + 0.5
                    print(f"    [API Error] {e}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
        return ""


# ---------------------------------------------------------------------------
# Checkpoint & Dataset Pipeline
# ---------------------------------------------------------------------------

def load_processed_sample_ids(checkpoint_file: Path) -> set[str]:
    """Load sample_ids that have already been generated."""
    processed = set()
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed.add(record.get("sample_id"))
                    except Exception:
                        pass
    return processed


def append_to_checkpoint(checkpoint_file: Path, rows: list[dict[str, Any]]) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def select_balanced_subset(
    df: pd.DataFrame, n_per_lang: int = 50, seed: int = SEED
) -> pd.DataFrame:
    """Select n_per_lang samples per language proportionally across splits."""
    set_seed(seed)
    selected_parts = []
    for lang in TARGET_LANGUAGE_CODES:
        lang_df = df[df["language_code"] == lang]
        splits = ["train", "validation", "test"]
        split_samples = []
        for s in splits:
            s_df = lang_df[lang_df["split"] == s]
            # Proportional allocation (~68% train, 16% val, 16% test)
            if s == "train":
                n_s = int(round(n_per_lang * 0.68))
            elif s == "validation":
                n_s = int(round(n_per_lang * 0.16))
            else:
                n_s = n_per_lang - sum([int(round(n_per_lang * 0.68)), int(round(n_per_lang * 0.16))])
            n_s = min(n_s, len(s_df))
            if n_s > 0:
                split_samples.append(s_df.sample(n=n_s, random_state=seed))
        selected_parts.append(pd.concat(split_samples, ignore_index=True))

    result = pd.concat(selected_parts, ignore_index=True)
    return result


def process_samples(
    samples_df: pd.DataFrame,
    generator: GroqGenerator,
    checkpoint_file: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Process each sample, generating faithful and hallucinated pairs."""
    processed_ids = load_processed_sample_ids(checkpoint_file)
    print(f"Loaded {len(processed_ids)} previously processed sample_ids from checkpoint.")

    all_generated_rows: list[dict[str, Any]] = []

    # If checkpoint exists, also load existing rows
    if checkpoint_file.exists():
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        all_generated_rows.append(json.loads(line))
                    except Exception:
                        pass

    stats = {
        "total_samples": len(samples_df),
        "processed_samples": len(processed_ids),
        "api_errors": 0,
        "generation_failures": 0,
        "strategies_used": {s: 0 for s in INJECTION_STRATEGIES},
    }

    remaining_df = samples_df[~samples_df["sample_id"].isin(processed_ids)].reset_index(drop=True)
    total_to_process = len(remaining_df)

    if total_to_process == 0:
        print("All samples already processed.")
        return pd.DataFrame(all_generated_rows), stats

    print(f"Starting generation for {total_to_process} remaining samples ({total_to_process * 2} rows)...")

    for idx, row in remaining_df.iterrows():
        sample_id = row["sample_id"]
        lang = row["language"]
        lang_code = row["language_code"]
        split = row["split"]
        context = row["context"]
        question = row["question"]
        ref_answer = row["reference_answer"]
        source_dataset = row["source_dataset"]

        strategy = assign_injection_strategy(sample_id)
        stats["strategies_used"][strategy] += 1

        # 1. Generate Faithful Answer (label=0)
        sys_faithful, user_faithful = get_faithful_prompts(
            lang, lang_code, context, question, ref_answer
        )
        try:
            faithful_answer = generator.generate(sys_faithful, user_faithful, temperature=0.1)
        except Exception as e:
            print(f"  [ERROR] Faithful generation failed for {sample_id}: {e}")
            faithful_answer = ref_answer
            stats["api_errors"] += 1

        faithful_row = {
            "sample_id": sample_id,
            "language": lang,
            "language_code": lang_code,
            "split": split,
            "context": context,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": faithful_answer,
            "label": 0,
            "injection_strategy": None,
            "source_dataset": source_dataset,
            "possible_generation_failure": False,
        }

        # 2. Generate Hallucinated Answer (label=1)
        sys_halluc, user_halluc = get_hallucinated_prompts(
            lang, lang_code, strategy, context, question, ref_answer
        )
        try:
            halluc_answer = generator.generate(sys_halluc, user_halluc, temperature=0.7)
        except Exception as e:
            print(f"  [ERROR] Hallucination generation failed for {sample_id}: {e}")
            halluc_answer = ""
            stats["api_errors"] += 1

        is_fail, flags = check_hallucination_quality(ref_answer, halluc_answer, context, lang_code)
        if is_fail:
            stats["generation_failures"] += 1

        halluc_row = {
            "sample_id": sample_id,
            "language": lang,
            "language_code": lang_code,
            "split": split,
            "context": context,
            "question": question,
            "reference_answer": ref_answer,
            "generated_answer": halluc_answer,
            "label": 1,
            "injection_strategy": strategy,
            "source_dataset": source_dataset,
            "possible_generation_failure": is_fail,
        }

        pair_rows = [faithful_row, halluc_row]
        all_generated_rows.extend(pair_rows)
        append_to_checkpoint(checkpoint_file, pair_rows)

        if (idx + 1) % 10 == 0 or (idx + 1) == total_to_process:
            print(f"  [{idx + 1}/{total_to_process}] Completed samples ({lang_code}: {strategy})")

    return pd.DataFrame(all_generated_rows), stats


def export_phase2_dataset(df: pd.DataFrame, output_dir: Path = PHASE2_DIR) -> None:
    """Export phase2 datasets partitioned by split (CSV + Parquet)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main dataset
    df.to_csv(output_dir / "phase2_dataset.csv", index=False, encoding="utf-8")
    df.to_parquet(output_dir / "phase2_dataset.parquet", index=False)

    # Per split
    for split in ["train", "validation", "test"]:
        split_df = df[df["split"] == split]
        split_df.to_csv(output_dir / f"{split}.csv", index=False, encoding="utf-8")
        split_df.to_parquet(output_dir / f"{split}.parquet", index=False)

    print(f"\nPhase 2 dataset successfully exported to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Answer Generation")
    parser.add_argument("--subset", type=int, default=None, help="Number of samples per language (e.g. 50)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Groq model name")
    args = parser.parse_args()

    api_key = get_groq_api_key()
    if not api_key:
        print("\n[ERROR] GROQ_API_KEY environment variable is not set!")
        print("Please set GROQ_API_KEY before running Phase 2.")
        sys.exit(1)

    print(f"Using Groq model: {args.model}")
    generator = GroqGenerator(api_key=api_key, model=args.model)

    # Load Phase 1 Final Dataset
    phase1_path = FINAL_DIR / "phase1_dataset.parquet"
    if not phase1_path.exists():
        phase1_path = FINAL_DIR / "phase1_dataset.csv"
        df_phase1 = pd.read_csv(phase1_path)
    else:
        df_phase1 = pd.read_parquet(phase1_path)

    print(f"Loaded Phase 1 dataset: {len(df_phase1):,} rows")

    if args.subset:
        print(f"Selecting balanced subset: {args.subset} samples per language...")
        samples = select_balanced_subset(df_phase1, n_per_lang=args.subset)
        checkpoint_file = CHECKPOINT_DIR / f"test_subset_{args.subset}_per_lang.jsonl"
    else:
        samples = df_phase1
        checkpoint_file = CHECKPOINT_DIR / "full_phase2_generation.jsonl"

    print(f"Target sample count: {len(samples)} (yielding {len(samples) * 2} faithful/hallucinated rows)")

    phase2_df, stats = process_samples(samples, generator, checkpoint_file)
    export_phase2_dataset(phase2_df)

    print("\n" + "=" * 60)
    print("PHASE 2 GENERATION STATS")
    print("=" * 60)
    print(f"Total Rows Generated: {len(phase2_df):,}")
    print(f"Possible Generation Failures: {phase2_df['possible_generation_failure'].sum()}")
    print(f"Strategies Used: {stats['strategies_used']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
