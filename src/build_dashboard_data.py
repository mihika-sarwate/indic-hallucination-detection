"""
Collect the real outputs of every phase into public/dashboard_data.json
(and refresh public/samples.json) so the dashboard never shows hardcoded numbers.

Run from project root after any phase finishes:
    python src/build_dashboard_data.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PHASE1_STATS = PROJECT_ROOT / "reports" / "phase1_statistics.json"
PHASE2_CSV = PROJECT_ROOT / "data" / "phase2" / "phase2_dataset.csv"
PHASE3_SUMMARY = PROJECT_ROOT / "reports" / "phase3_summary.json"
PHASE45_RESULTS = PROJECT_ROOT / "reports" / "phase45" / "phase45_results.json"
PUBLIC_DIR = PROJECT_ROOT / "public"
LANGS = ["hi", "mr", "ta"]


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def phase1_block() -> dict | None:
    s = load_json(PHASE1_STATS)
    if not s:
        return None
    eda = s["eda"]["by_language"]
    return {
        "original_total": s["original_total"],
        "valid_total": s["valid_total"],
        "removed_total": s["removed_total"],
        "removal_reasons": s["removal_reasons"],
        "by_language": {
            lc: {
                "original": s["by_language"][lc]["original"],
                "valid": s["by_language"][lc]["valid"],
                "removed": s["by_language"][lc]["removed"],
                "duplicates": s["by_language"][lc]["duplicates_removed"],
                "splits": eda[lc]["by_split"],
                "avg_context_words": round(eda[lc]["avg_context_words"], 1),
                "avg_answer_words": round(eda[lc]["avg_answer_words"], 2),
            }
            for lc in LANGS
        },
        "splits": {
            sp: sum(eda[lc]["by_split"].get(sp, 0) for lc in LANGS) for sp in ("train", "validation", "test")
        },
    }


def phase2_block() -> tuple[dict | None, list[dict]]:
    if not PHASE2_CSV.exists():
        return None, []
    with open(PHASE2_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["label"] = int(r["label"])
        r["possible_generation_failure"] = str(r["possible_generation_failure"]).lower() == "true"
    hall = [r for r in rows if r["label"] == 1]
    by_lang = {}
    for lc in LANGS:
        lr = [r for r in rows if r["language_code"] == lc]
        by_lang[lc] = {
            "faithful": sum(1 for r in lr if r["label"] == 0),
            "hallucinated": sum(1 for r in lr if r["label"] == 1),
            "flagged": sum(1 for r in lr if r["possible_generation_failure"]),
            "strategies": dict(Counter(r["injection_strategy"] for r in lr if r["label"] == 1)),
        }
    examples = []
    for lc in LANGS:
        picked = 0
        for r in hall:
            if r["language_code"] == lc and not r["possible_generation_failure"] and picked < 4:
                faithful = next((x for x in rows if x["sample_id"] == r["sample_id"] and x["label"] == 0), None)
                examples.append({
                    "language_code": lc,
                    "question": r["question"],
                    "reference_answer": r["reference_answer"],
                    "faithful_answer": faithful["generated_answer"] if faithful else "",
                    "hallucinated_answer": r["generated_answer"],
                    "injection_strategy": r["injection_strategy"],
                })
                picked += 1
    block = {
        "samples": len({r["sample_id"] for r in rows}),
        "rows": len(rows),
        "flagged_total": sum(1 for r in rows if r["possible_generation_failure"]),
        "strategies": dict(Counter(r["injection_strategy"] for r in hall)),
        "by_language": by_lang,
        "examples": examples,
    }
    return block, rows


def phase45_block() -> dict | None:
    res = load_json(PHASE45_RESULTS)
    if not res:
        return None
    models = {}
    for name, m in res.get("models", {}).items():
        first = m["runs"][0]
        models[name] = {
            "hf_name": first["hf_name"],
            "summary": m["summary"],
            "seeds": [r["seed"] for r in m["runs"]],
            "best_epochs": [r["best_epoch"] for r in m["runs"]],
            "train_seconds": [r["train_seconds"] for r in m["runs"]],
            "history": first["history"],
        }
    return {
        "data": res["data"],
        "config": res["config"],
        "baselines": {k: {"overall": v["overall"], "by_language": v["by_language"]} for k, v in res["baselines"].items()},
        "models": models,
        "cross_lingual": res.get("cross_lingual"),
    }


def main() -> None:
    PUBLIC_DIR.mkdir(exist_ok=True)
    p2, rows = phase2_block()
    summary = load_json(PROJECT_ROOT / "reports" / "phase2_summary.json") or {}
    if p2 and summary:
        temps = summary.get("temperatures", {})
        p2.update({
            "model": summary.get("model"),
            "temperatures": f"{temps.get('faithful')} / {temps.get('hallucinated')}" if temps else None,
            "api_errors": summary.get("api_errors"),
        })
    data = {
        "phase1": phase1_block(),
        "phase2": p2,
        "phase3": load_json(PHASE3_SUMMARY),
        "phase45": phase45_block(),
    }
    with open(PUBLIC_DIR / "dashboard_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    if rows:
        for r in rows:
            r["label"] = str(r["label"])
            r["possible_generation_failure"] = str(r["possible_generation_failure"])
        with open(PUBLIC_DIR / "samples.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False)
    print("Wrote", PUBLIC_DIR / "dashboard_data.json", {k: v is not None for k, v in data.items()})


if __name__ == "__main__":
    main()
