"""
Phase 3 — Label verification.

1. Draws a stratified review sample from the Phase 2 dataset (per language:
   N faithful rows + N/4 hallucinated rows per injection strategy) and writes it to
   data/phase3/review_sample.csv for human reviewers (dashboard Verification page).
2. Runs an independent LLM judge (a different model from the Phase 2 generator) on the
   same sample as an automatic second annotator, and measures its agreement with the
   weak labels (accuracy and Cohen's kappa) per language and per strategy.
3. If human reviews exist in data/phase3/human_verified.jsonl for the sampled rows,
   their agreement is reported as well.

Run from project root:
    python src/phase3_verify.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_phase2 import INJECTION_STRATEGIES, get_groq_api_key
from utils import SEED, TARGET_LANGUAGE_CODES, save_json

load_dotenv(PROJECT_ROOT / ".env")

PHASE2_CSV = PROJECT_ROOT / "data" / "phase2" / "phase2_dataset.csv"
PHASE3_DIR = PROJECT_ROOT / "data" / "phase3"
SAMPLE_CSV = PHASE3_DIR / "review_sample.csv"
JUDGE_JSONL = PHASE3_DIR / "llm_judge_labels.jsonl"
HUMAN_JSONL = PHASE3_DIR / "human_verified.jsonl"
SUMMARY_JSON = PROJECT_ROOT / "reports" / "phase3_summary.json"

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_SYSTEM = (
    "You are a strict annotator for a hallucination-detection benchmark in Indian languages.\n"
    "Given a context passage, a question and a candidate answer, decide whether the candidate answer is "
    "FAITHFUL (every claim it makes is supported by the context and it answers the question consistently "
    "with the context) or HALLUCINATED (it contradicts the context, changes an entity, number or date, or "
    "adds information the context does not contain).\n"
    "Reply with exactly one word: FAITHFUL or HALLUCINATED."
)


def draw_sample(df: pd.DataFrame, n_faithful: int, n_per_strategy: int, seed: int) -> pd.DataFrame:
    parts = []
    for lc in TARGET_LANGUAGE_CODES:
        lang = df[df["language_code"] == lc]
        f = lang[lang["label"] == 0]
        parts.append(f.sample(n=min(n_faithful, len(f)), random_state=seed))
        for s in INJECTION_STRATEGIES:
            h = lang[(lang["label"] == 1) & (lang["injection_strategy"] == s)]
            parts.append(h.sample(n=min(n_per_strategy, len(h)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def judge(sample: pd.DataFrame) -> pd.DataFrame:
    from groq import Groq

    client = Groq(api_key=get_groq_api_key())
    done = {}
    if JUDGE_JSONL.exists():
        for line in open(JUDGE_JSONL, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[(r["sample_id"], r["label"])] = r["judge_label"]

    with open(JUDGE_JSONL, "a", encoding="utf-8") as out:
        for i, row in sample.iterrows():
            key = (row["sample_id"], int(row["label"]))
            if key in done:
                continue
            user = (
                f"Language: {row['language']}\n\nContext:\n{row['context']}\n\n"
                f"Question:\n{row['question']}\n\nCandidate answer:\n{row['generated_answer']}\n\nVerdict:"
            )
            verdict = None
            for attempt in range(8):
                try:
                    resp = client.chat.completions.create(
                        model=JUDGE_MODEL,
                        messages=[{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}],
                        temperature=0.0,
                        max_tokens=512,
                        reasoning_effort="low",
                        include_reasoning=False,
                    )
                    text = (resp.choices[0].message.content or "").upper()
                    m = re.search(r"HALLUCINATED|FAITHFUL", text)
                    verdict = m.group(0).lower() if m else "unparsed"
                    break
                except Exception as e:
                    wait = min(60, 2 ** attempt + 1)
                    print(f"  [judge retry {attempt + 1}] {str(e)[:120]} — sleeping {wait}s")
                    time.sleep(wait)
            verdict = verdict or "error"
            done[key] = verdict
            out.write(json.dumps({"sample_id": key[0], "label": key[1], "judge_label": verdict}, ensure_ascii=False) + "\n")
            out.flush()
            if (len(done)) % 20 == 0:
                print(f"  judged {len(done)}/{len(sample)}")

    sample = sample.copy()
    sample["judge_label"] = [done.get((s, int(l)), "error") for s, l in zip(sample["sample_id"], sample["label"])]
    return sample


def agreement(weak: np.ndarray, other: np.ndarray) -> dict:
    return {
        "n": int(len(weak)),
        "agreement": float((weak == other).mean()) if len(weak) else None,
        "cohen_kappa": float(cohen_kappa_score(weak, other)) if len(set(weak)) > 1 and len(set(other)) > 1 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 label verification")
    parser.add_argument("--n-faithful", type=int, default=20)
    parser.add_argument("--n-per-strategy", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(PHASE2_CSV, encoding="utf-8")
    df["label"] = df["label"].astype(int)
    flagged = df["possible_generation_failure"].astype(str).str.lower() == "true"
    clean = df[~flagged].reset_index(drop=True)

    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    sample = draw_sample(clean, args.n_faithful, args.n_per_strategy, SEED)
    sample.to_csv(SAMPLE_CSV, index=False, encoding="utf-8")
    print(f"Review sample: {len(sample)} rows -> {SAMPLE_CSV}")

    sample = judge(sample)
    valid = sample[sample["judge_label"].isin(["faithful", "hallucinated"])]
    weak = valid["label"].to_numpy()
    jl = (valid["judge_label"] == "hallucinated").astype(int).to_numpy()

    summary = {
        "sample_size": int(len(sample)),
        "design": {
            "per_language_faithful": args.n_faithful,
            "per_language_per_strategy_hallucinated": args.n_per_strategy,
            "seed": SEED,
            "source": "Phase 2 rows not flagged by the quality guard",
        },
        "judge_model": JUDGE_MODEL,
        "judge_unparsed_or_error": int(len(sample) - len(valid)),
        "judge_vs_weak": agreement(weak, jl),
        "by_language": {},
        "by_strategy": {},
        "faithful_rows": agreement(weak[weak == 0], jl[weak == 0]),
        "disagreements": valid[(valid["label"] == 1) != (valid["judge_label"] == "hallucinated")][
            ["sample_id", "language_code", "label", "injection_strategy", "question", "reference_answer", "generated_answer", "judge_label"]
        ].astype(object).where(lambda x: x.notna(), None).to_dict(orient="records"),  # NaN -> null (valid JSON)
    }
    for lc in TARGET_LANGUAGE_CODES:
        m = (valid["language_code"] == lc).to_numpy()
        summary["by_language"][lc] = agreement(weak[m], jl[m])
    for s in INJECTION_STRATEGIES:
        m = (valid["injection_strategy"] == s).to_numpy()
        summary["by_strategy"][s] = {"n": int(m.sum()), "judge_confirms_hallucinated": float(jl[m].mean()) if m.any() else None}

    # Human reviews (dashboard) on sampled rows, if any.
    human = []
    if HUMAN_JSONL.exists():
        human = [json.loads(l) for l in open(HUMAN_JSONL, encoding="utf-8") if l.strip()]
    sampled_ids = set(sample["sample_id"])
    human_on_sample = [h for h in human if h.get("sample_id") in sampled_ids and "label" in h]
    summary["human_reviews_total_in_file"] = len(human)
    summary["human_reviews_on_sample"] = len(human_on_sample)

    save_json(summary, SUMMARY_JSON)
    print(json.dumps({k: summary[k] for k in ("sample_size", "judge_vs_weak", "by_language", "by_strategy", "faithful_rows")}, indent=1))


if __name__ == "__main__":
    main()
