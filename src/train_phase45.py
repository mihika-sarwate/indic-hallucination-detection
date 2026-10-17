"""
Phase 4 & 5 — Hallucination classifier training and evaluation.

Fine-tunes MuRIL and IndicBERT as binary (faithful=0 / hallucinated=1) sentence-pair
classifiers on the Phase 2 dataset, compares them with two classical baselines, and
writes metrics for the report and the dashboard.

Splits follow the Phase 1 split column: train for fitting, validation for epoch
selection, test for all reported numbers.

Run from project root (GPU recommended):
    python src/train_phase45.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import TARGET_LANGUAGE_CODES, normalize_text, save_json

PHASE2_CSV = PROJECT_ROOT / "data" / "phase2" / "phase2_dataset.csv"
PHASE1_PARQUET = PROJECT_ROOT / "data" / "final" / "phase1_dataset.parquet"
OUT_DIR = PROJECT_ROOT / "reports" / "phase45"

MODELS = {
    "MuRIL": {"name": "google/muril-base-cased", "lr": 3e-5},
    "IndicBERT": {"name": "ai4bharat/IndicBERTv2-MLM-only", "lr": 3e-5},
}
STRATEGIES = [
    "entity_substitution",
    "numeric_date_corruption",
    "unsupported_claim_addition",
    "contradicted_fact",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_phase2(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Phase 2 rows and drop those the quality guard flagged."""
    df = pd.read_csv(path, encoding="utf-8")
    df["label"] = df["label"].astype(int)
    df["possible_generation_failure"] = df["possible_generation_failure"].astype(str).str.lower() == "true"
    df["generated_answer"] = df["generated_answer"].fillna("").astype(str)

    info = {
        "rows_total": int(len(df)),
        "rows_flagged": int(df["possible_generation_failure"].sum()),
        "flagged_by_language_label": {
            f"{k[0]}_{k[1]}": int(v)
            for k, v in df[df["possible_generation_failure"]].groupby(["language_code", "label"]).size().items()
        },
    }
    clean = df[~df["possible_generation_failure"] & (df["generated_answer"].str.strip() != "")].reset_index(drop=True)
    info["rows_used"] = int(len(clean))
    info["rows_used_by_split"] = {k: int(v) for k, v in clean["split"].value_counts().items()}
    info["rows_used_by_language_label"] = {
        f"{k[0]}_{k[1]}": int(v) for k, v in clean.groupby(["language_code", "label"]).size().items()
    }
    return clean, info


def make_encoder_inputs(df: pd.DataFrame, tokenizer, max_len: int) -> list[dict[str, Any]]:
    """Segment A = question [SEP] context, segment B = candidate answer (never truncated)."""
    first = (df["question"] + f" {tokenizer.sep_token} " + df["context"]).tolist()
    second = df["generated_answer"].tolist()
    enc = tokenizer(first, second, truncation="only_first", max_length=max_len)
    items = []
    for i in range(len(df)):
        item = {k: enc[k][i] for k in enc.keys()}
        item["labels"] = int(df["label"].iloc[i])
        items.append(item)
    return items


def collate(tokenizer):
    def _fn(batch):
        # Copy rather than pop: the same items are reused every epoch.
        labels = torch.tensor([b["labels"] for b in batch])
        features = [{k: v for k, v in b.items() if k != "labels"} for b in batch]
        padded = tokenizer.pad(features, return_tensors="pt")
        padded["labels"] = labels
        return padded
    return _fn


def build_rule_pairs(n_per_lang: int, seed: int, exclude_ids: set[str]) -> pd.DataFrame:
    """
    Weak-supervision pairs from the Phase 1 *train* split (no LLM calls, no overlap with Phase 2 or test data).
    Half the rows keep the reference answer (label 0). The other half get another question's answer (label 1):
    alternately one from the same passage (a plausible in-context entity) and one from a random passage.
    """
    cols = ["sample_id", "language", "language_code", "split", "context", "question", "reference_answer"]
    pool = pd.read_parquet(PHASE1_PARQUET, columns=cols)
    pool = pool[(pool["split"] == "train") & ~pool["sample_id"].isin(exclude_ids)]
    rng = np.random.default_rng(seed)
    parts = []
    for lc in TARGET_LANGUAGE_CODES:
        lang = pool[pool["language_code"] == lc]
        by_ctx = lang.groupby("context")["reference_answer"].apply(list)
        all_answers = lang["reference_answer"].to_numpy()
        d = lang.sample(n=n_per_lang, random_state=seed).reset_index(drop=True)
        half = n_per_lang // 2
        pos = d.iloc[:half].copy()
        pos["generated_answer"], pos["label"], pos["injection_strategy"] = pos["reference_answer"], 0, "rule_reference"
        neg = d.iloc[half:].copy()
        answers, kinds = [], []
        for i, r in enumerate(neg.itertuples()):
            ref = r.reference_answer
            same = [a for a in by_ctx.get(r.context, []) if a != ref and a not in ref and ref not in a]
            if i % 2 == 0 and same:
                answers.append(same[rng.integers(len(same))])
                kinds.append("rule_same_passage_swap")
            else:
                a = ref
                while a == ref or a in ref or ref in a:
                    a = all_answers[rng.integers(len(all_answers))]
                answers.append(a)
                kinds.append("rule_random_swap")
        neg["generated_answer"], neg["label"], neg["injection_strategy"] = answers, 1, kinds
        parts += [pos, neg]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true, y_pred) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def full_report(df: pd.DataFrame, pred: np.ndarray) -> dict[str, Any]:
    y = df["label"].to_numpy()
    out: dict[str, Any] = {"overall": binary_metrics(y, pred), "by_language": {}, "by_strategy_recall": {}}
    for lc in TARGET_LANGUAGE_CODES:
        m = (df["language_code"] == lc).to_numpy()
        if m.any():
            out["by_language"][lc] = binary_metrics(y[m], pred[m])
    for s in STRATEGIES:
        m = (df["injection_strategy"] == s).to_numpy()
        if m.any():
            out["by_strategy_recall"][s] = {"n": int(m.sum()), "recall": float((pred[m] == 1).mean())}
    faithful = y == 0
    out["faithful_specificity"] = float((pred[faithful] == 0).mean()) if faithful.any() else None
    out["confusion_matrix"] = confusion_matrix(y, pred, labels=[0, 1]).tolist()
    return out


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def lexical_overlap_baseline(df: pd.DataFrame) -> np.ndarray:
    """Predict faithful when the normalized answer occurs verbatim in the context."""
    preds = []
    for ctx, ans in zip(df["context"], df["generated_answer"]):
        a = (normalize_text(ans) or "").strip(" .।,")
        preds.append(0 if a and a in (normalize_text(ctx) or "") else 1)
    return np.array(preds)


def tfidf_lr_baseline(train: pd.DataFrame, test: pd.DataFrame, seed: int) -> np.ndarray:
    """Character n-gram TF-IDF over question + context + answer, logistic regression."""
    def text(d):
        return (d["question"] + " || " + d["context"] + " || " + d["generated_answer"]).tolist()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True)
    xtr = vec.fit_transform(text(train))
    clf = LogisticRegression(max_iter=2000, C=4.0, random_state=seed)
    clf.fit(xtr, train["label"])
    return clf.predict(vec.transform(text(test)))


# ---------------------------------------------------------------------------
# Transformer fine-tuning
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, probs = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(**batch).logits.float()
        p = torch.softmax(logits, dim=-1)[:, 1]
        probs.append(p.cpu().numpy())
        preds.append((p >= 0.5).long().cpu().numpy())
    return np.concatenate(preds), np.concatenate(probs)


def run_epoch(model, loader, optim, sched, scaler, loss_fn, device, args) -> float:
    """One pass over loader; returns the mean training loss."""
    model.train()
    total, n = 0.0, 0
    optim.zero_grad()
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            loss = loss_fn(model(**batch).logits.float(), labels) / args.grad_accum
        scaler.scale(loss).backward()
        if step % args.grad_accum == 0 or step == len(loader):
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            optim.zero_grad()
        total += loss.item() * args.grad_accum * len(labels)
        n += len(labels)
        if args.step_sleep:
            time.sleep(args.step_sleep)  # duty-cycle pause to keep a laptop GPU cooler
    return total / n


def make_optim(model, lr, n_steps):
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(optim, int(0.1 * n_steps), n_steps)
    return optim, sched


def fine_tune(
    model_key: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    seed: int,
    args: argparse.Namespace,
    aug: pd.DataFrame | None = None,
) -> dict[str, Any]:
    cfg = MODELS[model_key]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["name"], num_labels=2).to(device)

    col = collate(tokenizer)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(make_encoder_inputs(train, tokenizer, args.max_len), batch_size=args.batch_size,
                              shuffle=True, collate_fn=col, generator=g)
    val_loader = DataLoader(make_encoder_inputs(val, tokenizer, args.max_len), batch_size=16, collate_fn=col)
    test_loader = DataLoader(make_encoder_inputs(test, tokenizer, args.max_len), batch_size=16, collate_fn=col)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss()
    t0 = time.time()

    # Stage 1 (optional): weak pre-fine-tuning on rule-based pairs built from Phase 1 training data.
    stage1 = None
    if aug is not None and len(aug):
        aug_loader = DataLoader(make_encoder_inputs(aug, tokenizer, args.max_len), batch_size=args.batch_size,
                                shuffle=True, collate_fn=col, generator=g)
        optim, sched = make_optim(model, cfg["lr"], (len(aug_loader) // args.grad_accum + 1) * args.aug_epochs)
        losses = [run_epoch(model, aug_loader, optim, sched, scaler, loss_fn, device, args) for _ in range(args.aug_epochs)]
        val_pred, _ = predict(model, val_loader, device)
        stage1 = {"rows": int(len(aug)), "epochs": args.aug_epochs, "train_loss": losses,
                  "val_after": binary_metrics(val["label"].to_numpy(), val_pred)}
        print(f"    [{model_key} seed={seed}] stage 1 (rule-based pairs): loss={losses[-1]:.4f} "
              f"val_macroF1={stage1['val_after']['macro_f1']:.4f}")

    # Stage 2: fine-tuning on the LLM-generated Phase 2 pairs.
    optim, sched = make_optim(model, cfg["lr"], (len(train_loader) // args.grad_accum + 1) * args.epochs)
    best = {"val_macro_f1": -1.0}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, optim, sched, scaler, loss_fn, device, args)
        val_pred, _ = predict(model, val_loader, device)
        val_m = binary_metrics(val["label"].to_numpy(), val_pred)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}})
        print(f"    [{model_key} seed={seed}] epoch {epoch}: loss={train_loss:.4f} val_macroF1={val_m['macro_f1']:.4f}")
        if val_m["macro_f1"] > best["val_macro_f1"]:
            test_pred, test_prob = predict(model, test_loader, device)
            best = {"val_macro_f1": val_m["macro_f1"], "epoch": epoch, "test_pred": test_pred, "test_prob": test_prob}

    result = {
        "model": model_key,
        "hf_name": cfg["name"],
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_val_macro_f1": best["val_macro_f1"],
        "train_seconds": round(time.time() - t0, 1),
        "stage1": stage1,
        "history": history,
        "test": full_report(test, best["test_pred"]),
        "test_pred": best["test_pred"].tolist(),
    }
    del model
    torch.cuda.empty_cache()
    return result


def mean_std(runs: list[dict[str, Any]], getter) -> dict[str, float]:
    vals = np.array([getter(r) for r in runs], dtype=float)
    return {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["accuracy", "precision", "recall", "f1", "macro_f1"]
    summary: dict[str, Any] = {"overall": {}, "by_language": {}, "by_strategy_recall": {}}
    for k in keys:
        summary["overall"][k] = mean_std(runs, lambda r, k=k: r["test"]["overall"][k])
    for lc in TARGET_LANGUAGE_CODES:
        summary["by_language"][lc] = {
            k: mean_std(runs, lambda r, k=k, lc=lc: r["test"]["by_language"][lc][k]) for k in keys
        }
        summary["by_language"][lc]["n"] = runs[0]["test"]["by_language"][lc]["n"]
    for s in STRATEGIES:
        summary["by_strategy_recall"][s] = mean_std(runs, lambda r, s=s: r["test"]["by_strategy_recall"][s]["recall"])
        summary["by_strategy_recall"][s]["n"] = runs[0]["test"]["by_strategy_recall"][s]["n"]
    summary["faithful_specificity"] = mean_std(runs, lambda r: r["test"]["faithful_specificity"])
    summary["confusion_matrix_sum"] = np.sum([r["test"]["confusion_matrix"] for r in runs], axis=0).tolist()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4/5 classifier training and evaluation")
    parser.add_argument("--data", type=Path, default=PHASE2_CSV)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=8, help="epochs on the LLM-generated pairs (stage 2)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--aug-per-lang", type=int, default=1000, help="rule-based pairs per language for stage 1 (0 = off)")
    parser.add_argument("--aug-epochs", type=int, default=1)
    parser.add_argument("--step-sleep", type=float, default=0.08, help="pause after each step (GPU temperature)")
    parser.add_argument("--skip-crosslingual", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true", help="skip the LLM-pairs-only comparison runs")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df, data_info = load_phase2(args.data)
    train, val, test = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "validation", "test"))
    print(f"Rows used: train={len(train)} val={len(val)} test={len(test)} (flagged rows dropped: {data_info['rows_flagged']})")
    print(f"Device: {'cuda - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    phase2_ids = set(pd.read_csv(args.data, usecols=["sample_id"])["sample_id"])
    aug = build_rule_pairs(args.aug_per_lang, 42, phase2_ids) if args.aug_per_lang else None
    if aug is not None:
        data_info["rule_pairs"] = {"rows": int(len(aug)), "per_language": args.aug_per_lang,
                                   "kinds": {k: int(v) for k, v in aug["injection_strategy"].value_counts().items()}}
        print(f"Rule-based stage-1 pairs: {len(aug)} {data_info['rule_pairs']['kinds']}")

    results: dict[str, Any] = {
        "data": data_info,
        "config": {
            "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "max_len": args.max_len, "seeds": args.seeds, "aug_per_lang": args.aug_per_lang,
            "aug_epochs": args.aug_epochs, "step_sleep": args.step_sleep,
            "models": {k: MODELS[k] for k in args.models}, "selection": "best validation macro-F1 epoch",
        },
        "baselines": {},
        "models": {},
    }

    results["baselines"]["Lexical overlap"] = full_report(test, lexical_overlap_baseline(test))
    results["baselines"]["TF-IDF + LR"] = full_report(test, tfidf_lr_baseline(train, test, 42))
    for name, rep in results["baselines"].items():
        print(f"  {name}: F1={rep['overall']['f1']:.4f} acc={rep['overall']['accuracy']:.4f}")

    runs_all: dict[str, list] = {}
    for model_key in args.models:
        runs = []
        for seed in args.seeds:
            print(f"Fine-tuning {model_key} (seed {seed})...")
            runs.append(fine_tune(model_key, train, val, test, seed, args, aug=aug))
            print(f"  -> test F1={runs[-1]['test']['overall']['f1']:.4f} acc={runs[-1]['test']['overall']['accuracy']:.4f}")
        runs_all[model_key] = runs
        results["models"][model_key] = {"summary": summarize(runs), "runs": [
            {k: v for k, v in r.items() if k != "test_pred"} for r in runs
        ]}
        save_json(results, args.out / "phase45_results.json")

    # Comparison: the same models fine-tuned on the LLM pairs only (no stage 1), first seed.
    if aug is not None and not args.skip_ablation:
        ablation: dict[str, Any] = {}
        for model_key in args.models:
            print(f"Ablation (LLM pairs only): {model_key}...")
            r = fine_tune(model_key, train, val, test, args.seeds[0], args, aug=None)
            ablation[model_key] = {"seed": args.seeds[0], "best_epoch": r["best_epoch"], "overall": r["test"]["overall"],
                                   "by_language": r["test"]["by_language"], "history": r["history"]}
            print(f"  -> test F1={r['test']['overall']['f1']:.4f} acc={r['test']['overall']['accuracy']:.4f}")
        results["ablation_llm_pairs_only"] = ablation
        save_json(results, args.out / "phase45_results.json")

    # Cross-lingual transfer: train on one language only (its rule pairs + LLM pairs), test on each language.
    if not args.skip_crosslingual:
        model_key, seed = args.models[0], args.seeds[0]
        cross: dict[str, Any] = {"model": model_key, "seed": seed, "f1": {}, "accuracy": {}}
        for src in TARGET_LANGUAGE_CODES:
            tr = train[train["language_code"] == src].reset_index(drop=True)
            va = val[val["language_code"] == src].reset_index(drop=True)
            src_aug = aug[aug["language_code"] == src].reset_index(drop=True) if aug is not None else None
            print(f"Cross-lingual: {model_key} trained on {src}...")
            r = fine_tune(model_key, tr, va, test, seed, args, aug=src_aug)
            cross["f1"][src] = {lc: r["test"]["by_language"][lc]["f1"] for lc in TARGET_LANGUAGE_CODES}
            cross["accuracy"][src] = {lc: r["test"]["by_language"][lc]["accuracy"] for lc in TARGET_LANGUAGE_CODES}
        results["cross_lingual"] = cross

    # Test-set predictions for error analysis (first seed of each model).
    pred_df = test[["sample_id", "language_code", "label", "injection_strategy", "question", "reference_answer", "generated_answer"]].copy()
    for model_key, runs in runs_all.items():
        pred_df[f"pred_{model_key}"] = runs[0]["test_pred"]
    pred_df["pred_lexical"] = lexical_overlap_baseline(test)
    pred_df.to_csv(args.out / "test_predictions.csv", index=False, encoding="utf-8")

    save_json(results, args.out / "phase45_results.json")
    print(f"\nSaved results to {args.out / 'phase45_results.json'}")


if __name__ == "__main__":
    main()
