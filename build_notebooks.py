"""Generate the Phase 4 and Phase 5 notebooks (Colab / Kaggle friendly wrappers around the scripts)."""

import os

import nbformat as nbf

os.makedirs("notebooks", exist_ok=True)

# Phase 4 Notebook
nb4 = nbf.v4.new_notebook()
nb4.cells.append(nbf.v4.new_markdown_cell(
    "# Phase 4: Model Training\n"
    "Fine-tunes MuRIL (`google/muril-base-cased`) and IndicBERT v2 (`ai4bharat/IndicBERTv2-MLM-only`) as binary\n"
    "faithful / hallucinated sentence-pair classifiers on the Phase 2 dataset.\n\n"
    "* Input: `[CLS] question [SEP] context [SEP] candidate answer [SEP]`, 256 tokens, only the context is truncated.\n"
    "* Splits: the Phase 1 `split` column (train / validation / test); rows flagged by the quality guard are excluded.\n"
    "* Stage 1: one epoch on 3,000 rule-based pairs from the Phase 1 train split; stage 2: the LLM-generated pairs.\n"
    "* AdamW (lr 3e-5), cross-entropy, fp16, batch 8, 1 + 8 epochs, best epoch by validation macro-F1, 3 seeds.\n"
    "* Baselines: lexical-overlap rule and TF-IDF + logistic regression.\n\n"
    "All logic lives in `src/train_phase45.py`; this notebook runs it and shows the training logs."))
nb4.cells.append(nbf.v4.new_code_cell("!pip install -q -r ../requirements-ml.txt"))
nb4.cells.append(nbf.v4.new_code_cell("import torch\nprint('CUDA available:', torch.cuda.is_available())"))
nb4.cells.append(nbf.v4.new_code_cell("!cd .. && python src/train_phase45.py"))
nb4.cells.append(nbf.v4.new_code_cell(
    "import json\nimport pandas as pd\n\n"
    "res = json.load(open('../reports/phase45/phase45_results.json', encoding='utf-8'))\n"
    "print('Rows used:', res['data']['rows_used_by_split'])\n"
    "for name, m in res['models'].items():\n"
    "    print(f\"\\n{name} - best epochs per seed: {[r['best_epoch'] for r in m['runs']]}\")\n"
    "    display(pd.DataFrame(m['runs'][0]['history']).round(4))"))
nbf.write(nb4, "notebooks/phase4_model_training.ipynb")

# Phase 5 Notebook
nb5 = nbf.v4.new_notebook()
nb5.cells.append(nbf.v4.new_markdown_cell(
    "# Phase 5: Evaluation & Comparison\n"
    "Reads `reports/phase45/phase45_results.json` (written by `src/train_phase45.py`) and reports test-split results by\n"
    "language, by hallucination type and across languages, plus an error analysis of the test predictions."))
nb5.cells.append(nbf.v4.new_code_cell(
    "import json\nimport pandas as pd\n\n"
    "res = json.load(open('../reports/phase45/phase45_results.json', encoding='utf-8'))\n"
    "rows = []\n"
    "for name, m in res['models'].items():\n"
    "    for lc, v in m['summary']['by_language'].items():\n"
    "        rows.append({'model': name, 'language': lc, **{k: round(v[k]['mean'], 3) for k in ('accuracy', 'precision', 'recall', 'f1')}})\n"
    "for name, b in res['baselines'].items():\n"
    "    for lc, v in b['by_language'].items():\n"
    "        rows.append({'model': name, 'language': lc, **{k: round(v[k], 3) for k in ('accuracy', 'precision', 'recall', 'f1')}})\n"
    "pd.DataFrame(rows)"))
nb5.cells.append(nbf.v4.new_code_cell(
    "pd.DataFrame({name: {s: round(v['mean'], 3) for s, v in m['summary']['by_strategy_recall'].items()}\n"
    "              for name, m in res['models'].items()})"))
nb5.cells.append(nbf.v4.new_code_cell(
    "cl = res.get('cross_lingual')\n"
    "pd.DataFrame(cl['f1']).T.round(3) if cl else 'cross-lingual run skipped'"))
nb5.cells.append(nbf.v4.new_code_cell(
    "pred = pd.read_csv('../reports/phase45/test_predictions.csv')\n"
    "errors = pred[pred['label'] != pred['pred_MuRIL']]\n"
    "print(len(errors), 'MuRIL errors on', len(pred), 'test rows')\n"
    "errors[['language_code', 'label', 'injection_strategy', 'question', 'reference_answer', 'generated_answer']].head(20)"))
nbf.write(nb5, "notebooks/phase5_evaluation.ipynb")

print("Notebooks generated successfully.")
