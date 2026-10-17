"""
Summarize a Phase 2 run (reports/phase2_summary.json) and diagnose the earlier run.

Run from project root after generate_phase2.py:
    python src/phase2_summary.py --log logs_phase2_v2.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from generate_phase2 import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, check_faithful_quality, check_hallucination_quality
from utils import TARGET_LANGUAGE_CODES, normalize_text, save_json

PHASE2_CSV = PROJECT_ROOT / "data" / "phase2" / "phase2_dataset.csv"
EARLIER_RUN = PROJECT_ROOT / "data" / "phase2" / "checkpoints" / "test_subset_50_per_lang.jsonl"
OUT = PROJECT_ROOT / "reports" / "phase2_summary.json"


def in_context(answer: str, context: str) -> bool:
    a = (normalize_text(answer) or "").strip(" .।,")
    return bool(a) and a in (normalize_text(context) or "")


def summarize(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["generated_answer"] = df["generated_answer"].fillna("").astype(str)
    out: dict = {"samples": int(df["sample_id"].nunique()), "rows": int(len(df)), "by_language": {}}
    flag_reasons: Counter = Counter()
    for lc in TARGET_LANGUAGE_CODES:
        d = df[df["language_code"] == lc]
        f, h = d[d["label"] == 0], d[d["label"] == 1]
        reasons: Counter = Counter()
        for _, r in f.iterrows():
            bad, flags = check_faithful_quality(r["generated_answer"], lc)
            if bad:
                reasons.update(f"faithful:{x}" for x in flags)
        for _, r in h.iterrows():
            bad, flags = check_hallucination_quality(r["reference_answer"], r["generated_answer"], r["context"], lc)
            if bad:
                reasons.update(f"hallucinated:{x}" for x in flags)
        flag_reasons.update(reasons)
        flagged = d["possible_generation_failure"].astype(str).str.lower() == "true"
        out["by_language"][lc] = {
            "samples": int(d["sample_id"].nunique()),
            "faithful_rows": int(len(f)),
            "hallucinated_rows": int(len(h)),
            "flagged_faithful": int(flagged[d["label"] == 0].sum()),
            "flagged_hallucinated": int(flagged[d["label"] == 1].sum()),
            "splits": {k: int(v) for k, v in d.drop_duplicates("sample_id")["split"].value_counts().items()},
            "strategies": {k: int(v) for k, v in h["injection_strategy"].value_counts().items()},
            "faithful_equals_reference": int((f["generated_answer"].str.strip() == f["reference_answer"].str.strip()).sum()),
            "faithful_in_context": int(sum(in_context(a, c) for a, c in zip(f["generated_answer"], f["context"]))),
            "hallucinated_in_context": int(sum(in_context(a, c) for a, c in zip(h["generated_answer"], h["context"]))),
            "avg_words_faithful": round(float(f["generated_answer"].str.split().str.len().mean()), 2),
            "avg_words_hallucinated": round(float(h["generated_answer"].str.split().str.len().mean()), 2),
            "avg_words_reference": round(float(f["reference_answer"].str.split().str.len().mean()), 2),
        }
    out["strategies"] = {k: int(v) for k, v in df[df["label"] == 1]["injection_strategy"].value_counts().items()}
    out["flagged_total"] = int((df["possible_generation_failure"].astype(str).str.lower() == "true").sum())
    out["flag_reasons"] = dict(flag_reasons)
    out["splits"] = {k: int(v) for k, v in df.drop_duplicates("sample_id")["split"].value_counts().items()}
    return out


def diagnose_earlier_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    latest = {}
    for r in rows:
        latest[(r["sample_id"], r["label"])] = r
    kinds: dict = defaultdict(Counter)
    for r in latest.values():
        g = (r.get("generated_answer") or "").strip()
        if not g:
            k = "empty"
        elif g.startswith("<think>"):
            k = "reasoning_trace_closed" if "</think>" in g else "reasoning_trace_truncated"
        elif g == (r.get("reference_answer") or "").strip():
            k = "copy_of_reference"
        else:
            k = "usable"
        kinds[f"{r['language_code']}_label{r['label']}"][k] += 1
    return {
        "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "lines": len(rows),
        "unique_rows": len(latest),
        "duplicate_lines_from_reruns": len(rows) - len(latest),
        "by_language_label": {k: dict(v) for k, v in sorted(kinds.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=PHASE2_CSV)
    parser.add_argument("--log", type=Path, default=None, help="generation log, to count API errors / rate-limit waits")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    df = pd.read_csv(args.data, encoding="utf-8")
    df["label"] = df["label"].astype(int)
    summary = {
        "model": args.model,
        "temperatures": {"faithful": 0.1, "hallucinated": 0.7},
        "max_tokens": DEFAULT_MAX_TOKENS,
        "reasoning": "hidden (reasoning_format=hidden); any <think> block stripped",
        **summarize(df),
    }
    if args.log and args.log.exists():
        log = args.log.read_text(encoding="utf-8", errors="ignore")
        summary["api_errors"] = len(re.findall(r"\[ERROR\]", log))
        summary["rate_limit_waits"] = len(re.findall(r"\[RateLimit\]", log))
    summary["earlier_run_diagnosis"] = diagnose_earlier_run(EARLIER_RUN)
    save_json(summary, OUT)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_language", "earlier_run_diagnosis")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
