# Hallucination Detection in Indic LLM Outputs

A multilingual benchmark and classifier for detecting hallucinated answers in **Hindi, Marathi and Tamil**
question answering. The pipeline builds grounded QA data, generates faithful and deliberately hallucinated
answers with an LLM, checks the weak labels, fine-tunes multilingual encoders (MuRIL, IndicBERT v2) and
compares them with classical baselines. A React dashboard shows the real output of every phase.

## Pipeline

| Phase | Script | Output |
|---|---|---|
| 1. Dataset preparation | `src/run_phase1.py` | `data/final/*`, `reports/phase1_statistics.json`, `reports/figures/*.png` |
| 2. Hallucination generation | `src/generate_phase2.py --subset 100 --run-tag v2` | `data/phase2/phase2_dataset.{csv,parquet}`, per-split files, JSONL checkpoint |
| 2. Run summary | `src/phase2_summary.py --log reports/phase2_generation_log.txt` | `reports/phase2_summary.json` |
| 3. Label verification | `src/phase3_verify.py` | `data/phase3/review_sample.csv`, `data/phase3/llm_judge_labels.jsonl`, `reports/phase3_summary.json` |
| 4 & 5. Training and evaluation | `src/train_phase45.py` | `reports/phase45/phase45_results.json`, `reports/phase45/test_predictions.csv` |
| Dashboard data | `src/build_dashboard_data.py` | `public/dashboard_data.json`, `public/samples.json` |
| Report figures | `reports/figures/make_diagrams.py`, `reports/figures/make_result_charts.py` | `reports/figures/diagram_*.png`, `results_*.png` |

**Phase 1** harmonizes IndicSQuAD (`l3cube-pune/indic-squad`) and IndicQA (`ai4bharat/IndicQA`) into one schema,
applies Unicode NFC normalization, validation, script sanity checks, exact / context-question deduplication and
cross-split leakage resolution, and keeps the official splits (IndicQA goes to test): **278,058** clean rows
(92,693 hi, 92,527 mr, 92,838 ta).

**Phase 2** asks `qwen/qwen3.8-27b` on Groq for one faithful answer (temperature 0.1) and one hallucinated answer
(temperature 0.7) per sample, using four seeded injection strategies: entity substitution, numeric/date corruption,
unsupported claim addition, contradicted fact. Reasoning output is hidden and any `<think>` block is stripped; a
quality guard flags empty, copied, wrong-script or reasoning-trace answers. Runs are checkpointed, resumable and
processed round-robin across languages.

**Phase 3** draws a stratified sample (per language: 20 faithful + 5 hallucinated per strategy) and labels it with an
independent LLM judge (`openai/gpt-oss-120b`), reporting agreement and Cohen's kappa against the weak labels. The same
sample can be reviewed manually on the dashboard's Verification page.

**Phase 4/5** fine-tune `google/muril-base-cased` and `ai4bharat/IndicBERTv2-MLM-only` as sentence-pair classifiers
(`[CLS] question [SEP] context [SEP] answer [SEP]`, 256 tokens, 3 seeds) in two stages: (1) one epoch on 3,000
rule-based weak-supervision pairs built from the Phase 1 train split (the reference answer vs. another question's answer
from the same or a random passage), then (2) 8 epochs on the LLM-generated Phase 2 pairs, keeping the epoch with the best
validation macro-F1. Trained on the 388 LLM pairs alone, both encoders stay at chance level, which is reported as an
ablation. Models are compared with a lexical-overlap rule and TF-IDF + logistic regression, per language, per
hallucination type and across languages.

## Results

Phase 2 re-run: 300 samples / 600 rows (100 per language), 0 API errors, 27 rows flagged by the quality guard. Phase 3: the independent judge agreed with 90.0% of 120 sampled weak labels (Cohen's kappa 0.800).

Test split (94 rows), hallucinated = positive class; transformer scores are means over seeds 42, 43, 44:

| Model | Accuracy | Precision | Recall | F1 | F1 hi | F1 mr | F1 ta |
|---|---|---|---|---|---|---|---|
| MuRIL | 0.752 | 0.816 | 0.638 | 0.715 ± 0.030 | 0.706 | 0.849 | 0.590 |
| IndicBERT v2 | 0.567 | 0.388 | 0.471 | 0.424 ± 0.306 | 0.415 | 0.455 | 0.396 |
| Lexical overlap | 0.862 | 0.902 | 0.804 | 0.851 | 0.815 | 0.929 | 0.812 |
| TF-IDF + LR | 0.511 | 0.500 | 0.130 | 0.207 | 0.200 | 0.364 | 0.000 |

Per-type recall, confusion matrices and cross-lingual transfer are in `reports/phase45/phase45_results.json` and on the dashboard's Phase 5 page.

## Setup

```bash
pip install -r requirements.txt          # pipeline + API
pip install -r requirements-ml.txt       # training (use a CUDA build of PyTorch for GPU)
npm ci                                   # dashboard
```

Put the Groq key in a `.env` file at the project root (it is gitignored):

```
GROQ_API_KEY=...
```

## Running the dashboard

```bash
uvicorn api.index:app --port 8000        # FastAPI backend (Phase 2 rows, saves reviews)
npx vite                                 # React frontend on http://localhost:5173
```

The frontend reads `public/dashboard_data.json`, so every page shows real pipeline output (or an empty state if a phase
has not been run). Without the backend (static hosting) the pages fall back to the exported JSON files. Set
`API_TARGET` to point the Vite proxy at a backend on another port.

## Repository layout

```
api/            FastAPI backend (Vercel serverless entry)
src/            pipeline scripts and React source (pages/, useDashboardData.jsx)
data/           raw, processed, final (Phase 1), phase2, phase3 outputs (large files via Git LFS)
reports/        statistics and results JSON, figures, generation log
notebooks/      Colab / Kaggle notebooks for Phases 1, 4, 5
public/         static JSON consumed by the dashboard
```

## Notes

- The first Phase 2 run could not be used: Hindi/Marathi answers were truncated `<think>` traces and Tamil calls
  failed, with the script silently copying the reference answer. The generator was fixed and Phase 2 re-run; the old
  checkpoint (`data/phase2/checkpoints/test_subset_50_per_lang.jsonl`) is kept for reference.
- The original `ai4bharat/indic-bert` checkpoint is access-restricted on Hugging Face, so IndicBERT v2 is used.
- Groq's free tier (8,000 tokens/min, 200,000 tokens/day per model) limits the generated subset to 100 samples per
  language.
