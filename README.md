# Hallucination Detection in Indic LLM Outputs

A multilingual benchmark and classifier pipeline for detecting hallucinations in LLM-generated answers for **Hindi**, **Marathi**, and **Tamil**.

## Phase

**Phase 1 — Dataset Preparation** (completed)
Builds a clean, reproducible `context + question + reference_answer` dataset.

**Phase 2 — Hallucination Generation** (completed)
Faithful and hallucinated LLM answer generation with 4 deterministic injection strategies using the Groq API.

**Phase 3 — Human Verification** (completed)
React/Vite dashboard connected to a FastAPI backend for manual review and weak label validation.

Later phases (not implemented yet):
- Phase 4: MuRIL / IndicBERT classifier training and evaluation

---

## Dataset Verification

Verified programmatically against Hugging Face repositories:

| Dataset | Hindi | Marathi | Tamil | Context | Question | Answer | License | Usable? |
|---------|-------|---------|-------|---------|----------|--------|---------|---------|
| `ai4bharat/IndicQA` | Yes | Yes | Yes | Yes | Yes | Partial* | CC-BY-4.0 | Supplementary (test-only benchmark) |
| `l3cube-pune/indic-squad` | Yes | Yes | Yes | Yes | Yes | Yes | CC-BY-4.0 | **Primary** (extractive QA, official splits) |

\*IndicQA includes abstractive evaluation items without reference answers (~32–40% per language). Examples without a non-empty `reference_answer` are filtered out in Phase 1.

**Conclusion:** Both datasets cover all three target languages. **IndicSQuAD** is the primary source (~118k train examples per language). **IndicQA** supplements the test set with additional benchmark items. No replacement dataset is needed.

**Note:** The IndicQA Hugging Face `datasets` loading script is deprecated; this pipeline downloads raw JSON via `huggingface_hub`.

---

## Project Structure

```
indic-hallucination-detection/
├── data/
│   ├── raw/              # Downloaded source files per language
│   ├── processed/        # Normalized intermediate files
│   └── final/            # phase1_dataset, train, validation, test (.csv + .parquet)
├── notebooks/
│   └── phase1_dataset_preparation.ipynb
├── src/
│   ├── download_data.py  # HuggingFace download
│   ├── preprocess.py     # Schema normalization
│   ├── validate.py       # Validation, dedup, EDA, export
│   ├── utils.py          # Shared config and helpers
│   └── run_phase1.py     # End-to-end pipeline runner
├── reports/
│   ├── phase1_statistics.json
│   └── figures/          # EDA visualizations
├── requirements.txt
└── README.md
```

---

## Environment Setup

### Local

```bash
cd indic-hallucination-detection
pip install -r requirements.txt
python src/run_phase1.py
```

### Google Colab

```python
!git clone <your-repo-url>  # or upload project files
%cd indic-hallucination-detection
!pip install -r requirements.txt
!python src/run_phase1.py
```

### Kaggle Notebook

```python
!pip install -r /kaggle/input/<your-dataset>/requirements.txt  # or install inline
%cd /kaggle/working/indic-hallucination-detection
!python src/run_phase1.py
```

---

## Standardized Schema

| Column | Description |
|--------|-------------|
| `sample_id` | Deterministic SHA-256 hash of `source_dataset + language_code + source_id` |
| `language` | Language name (Hindi, Marathi, Tamil) |
| `language_code` | ISO-style code (`hi`, `mr`, `ta`) |
| `source_dataset` | `indicsquad` or `indicqa` |
| `source_split` | Original split from source dataset |
| `split` | Final split: `train`, `validation`, or `test` |
| `context` | Passage / grounding context |
| `question` | Question text |
| `reference_answer` | Ground-truth answer |
| `answer_start` | Character offset in context (extractive); `null` when unavailable |
| `task_type` | `extractive` or `abstractive` |
| `is_suspicious_language` | Script sanity flag |
| `validation_flags` | List of validation notes |

---

## Pipeline Overview

1. **Download** — IndicSQuAD via `datasets`; IndicQA via raw JSON
2. **Normalize** — Unicode-safe text cleaning; map to common schema
3. **Validate** — Reject missing/corrupted fields; verify extractive answer spans
4. **Language check** — Script sanity (Devanagari for hi/mr, Tamil script for ta); flag suspicious rows
5. **Deduplicate** — Exact and context+question duplicates; cross-split leakage prevention
6. **Split** — Retain IndicSQuAD official splits; IndicQA → test; priority: test > validation > train
7. **Export** — UTF-8 CSV + Parquet; statistics JSON; EDA plots

### Split Strategy

- **IndicSQuAD:** Official `train` / `validation` / `test` splits are preserved (SQuAD contexts are already group-separated).
- **IndicQA:** All valid examples are assigned to `test` (benchmark-only data).
- **Cross-split dedup:** Identical `context + question` pairs appearing in multiple splits keep only the highest-priority split (test > validation > train).

### Configuration (`src/utils.py`)

```python
SEED = 42
BALANCE_LANGUAGES = False      # Set True to subsample to smallest language count
MAX_SAMPLES_PER_LANGUAGE = None  # Set e.g. 10000 to cap per language
```

---

## Outputs

After running Phase 1:

```
data/final/
├── phase1_dataset.csv / .parquet
├── train.csv / .parquet
├── validation.csv / .parquet
└── test.csv / .parquet

reports/
├── phase1_statistics.json
└── figures/
    ├── samples_per_language.png
    ├── context_length_distribution.png
    ├── question_length_distribution.png
    └── answer_length_distribution.png
```

---

## Phase 2 Readiness

Each row contains everything needed to construct prompts:

```
Language: <language>

Context:
<context>

Question:
<question>

Reference Answer:
<reference_answer>
```

No LLM API calls or hallucination generation are performed in Phase 1.

---

## Reproducibility

- Fixed random seed: `SEED = 42`
- Deterministic `sample_id` hashing
- Sorted, order-preserving deduplication

---

## License

This project code is provided for academic use. Source datasets are licensed under **CC-BY-4.0**:

- [ai4bharat/IndicQA](https://huggingface.co/datasets/ai4bharat/IndicQA)
- [l3cube-pune/indic-squad](https://huggingface.co/datasets/l3cube-pune/indic-squad)

---

## Citation

If you use IndicSQuAD, cite:

```bibtex
@article{endait2025indicsquad,
  title={IndicSQuAD: A Comprehensive Multilingual Question Answering Dataset for Indic Languages},
  author={Endait, Sharvi and Ghatage, Ruturaj and Kulkarni, Aditya and Patil, Rajlaxmi and Joshi, Raviraj},
  journal={arXiv preprint arXiv:2505.03688},
  year={2025}
}
```
