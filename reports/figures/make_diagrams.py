"""Draw the report's workflow / architecture diagrams (reports/figures/diagram_*.png)."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
NAVY = "#1F3B5C"
INK = "#1A1D21"
EDGE = "#4A5A70"
FILL = {
    "blue": "#DCE8F7", "green": "#DDF2E3", "yellow": "#FBF1D6", "purple": "#E6E0F8",
    "pink": "#F9DEDE", "gray": "#EEF0F3", "teal": "#D9F1F0",
}


def box(ax, x, y, w, h, text, fill="blue", bold=False, size=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.0, edgecolor=EDGE, facecolor=FILL[fill]))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, linewidth=1.1, color=EDGE))


def label(ax, x, y, text):
    ax.text(x, y, text, ha="center", va="center", fontsize=6.8, color=EDGE)


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.25, title, ha="center", va="center", fontsize=11.5, color=NAVY, fontweight="bold")
    return fig, ax


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def overall_workflow():
    fig, ax = canvas(10, 2.9, "Overall Project Workflow")
    phases = [
        ("Phase 1\nDataset\nPreparation", "blue", "IndicSQuAD + IndicQA\nclean, dedup, split\n278,058 rows"),
        ("Phase 2\nHallucination\nGeneration", "green", "Groq qwen3.8-27b\nfaithful + 4 injection\nstrategies"),
        ("Phase 3\nLabel\nVerification", "yellow", "stratified sample\nLLM judge + review\nCohen's kappa"),
        ("Phase 4\nClassifier\nTraining", "purple", "MuRIL / IndicBERT v2\nfine-tuning (GPU)\n+ baselines"),
        ("Phase 5\nEvaluation &\nComparison", "pink", "per-language, per-type,\ncross-lingual\nmetrics"),
    ]
    w, gap, x0 = 1.62, 0.37, 0.25
    for i, (title, fill, sub) in enumerate(phases):
        x = x0 + i * (w + gap)
        box(ax, x, 1.35, w, 0.95, title, fill, bold=True, size=8.5)
        box(ax, x, 0.15, w, 0.95, sub, "gray", size=7.3)
        ax.plot([x + w / 2, x + w / 2], [1.1, 1.35], color=EDGE, lw=0.8, ls=":")
        if i < len(phases) - 1:
            arrow(ax, (x + w, 1.82), (x + w + gap, 1.82))
    save(fig, "diagram_overall_workflow.png")


def phase2_flow():
    fig, ax = canvas(10, 4.6, "Phase 2: Faithful and Hallucinated Answer Generation")
    box(ax, 0.2, 2.7, 1.9, 0.9, "Phase 1 row\ncontext + question +\nreference answer", "blue")
    box(ax, 2.6, 3.35, 2.3, 0.62, "faithful prompt\n(temperature 0.1)", "green")
    box(ax, 2.6, 2.25, 2.3, 0.78, "strategy selector\nSHA-256(SEED | sample_id)\nmodulo 4", "yellow", size=7.3)
    strategies = ["entity_substitution", "numeric_date_corruption", "unsupported_claim_addition", "contradicted_fact"]
    for i, s in enumerate(strategies):
        box(ax, 2.6, 1.6 - i * 0.42, 2.3, 0.34, s, "purple", size=7.3)
        arrow(ax, (4.9, 1.77 - i * 0.42), (5.45, 1.2))
    arrow(ax, (2.1, 3.3), (2.6, 3.66))
    arrow(ax, (2.1, 3.0), (2.6, 2.65))
    arrow(ax, (3.75, 2.25), (3.75, 1.94))
    box(ax, 5.45, 3.3, 2.4, 0.72, "Groq qwen/qwen3.8-27b\nfaithful answer (label 0)", "green", bold=True, size=7.4)
    box(ax, 5.45, 0.8, 2.4, 0.8, "Groq qwen/qwen3.8-27b\nhallucinated answer\n(label 1, temperature 0.7)", "pink", bold=True, size=7.2)
    arrow(ax, (4.9, 3.66), (5.45, 3.66))
    box(ax, 8.15, 2.0, 1.75, 1.45, "Quality guard\n• <think> trace removed\n• empty / wrong script\n• identical or >90%\n   token overlap\n→ flag, keep row", "gray", size=7)
    arrow(ax, (7.85, 3.66), (8.6, 3.45))
    arrow(ax, (7.85, 1.2), (8.6, 2.0))
    box(ax, 0.2, 0.75, 1.9, 1.15, "Reliability\nJSONL checkpoint,\nresume by sample_id,\nbackoff on HTTP 429,\nreasoning hidden", "gray", size=7.2)
    save(fig, "diagram_phase2_flow.png")


def classifier_arch():
    fig, ax = canvas(10, 2.75, "Phase 4: Hallucination Classifier Architecture")
    box(ax, 0.15, 1.05, 2.45, 0.95, "Input pair (max 256 tokens)\n[CLS] question [SEP] context\n[SEP] candidate answer [SEP]", "blue", size=7.5)
    box(ax, 2.95, 1.05, 1.7, 0.95, "Tokenizer\nMuRIL / IndicBERT v2\nsubword vocabulary", "yellow", size=7.5)
    box(ax, 5.0, 1.05, 1.75, 0.95, "Transformer\nencoder (12 layers,\nfine-tuned)", "purple", bold=True, size=7.4)
    box(ax, 7.1, 1.05, 1.15, 0.95, "[CLS] pooled\n→ Linear\n→ softmax", "green", size=7.5)
    box(ax, 8.55, 1.05, 1.35, 0.95, "faithful (0)\nor\nhallucinated (1)", "pink", bold=True, size=7.2)
    for a, b in [(2.6, 2.95), (4.65, 5.0), (6.75, 7.1), (8.25, 8.55)]:
        arrow(ax, (a, 1.52), (b, 1.52))
    ax.text(5.0, 0.55, "Stage 1: 3,000 rule-based pairs from the Phase 1 train split (1 epoch) → Stage 2: LLM-generated "
            "Phase 2 pairs (8 epochs, best epoch by validation macro-F1)", ha="center", fontsize=7.2, color=EDGE)
    ax.text(5.0, 0.22, "Cross-entropy loss · AdamW (lr 3e-5) · fp16 · batch 8 · 3 seeds · the candidate answer is never truncated",
            ha="center", fontsize=7.2, color=EDGE)
    save(fig, "diagram_classifier.png")


def system_architecture():
    fig, ax = canvas(10, 4.0, "System Architecture and Results Dashboard")
    box(ax, 0.2, 2.15, 2.6, 1.25, "Pipeline scripts (src/)\nrun_phase1 · generate_phase2\nphase3_verify · train_phase45\nbuild_dashboard_data", "blue", size=7.6)
    box(ax, 3.3, 2.75, 2.35, 0.65, "External services\nHugging Face Hub · Groq API", "gray", size=7.4)
    box(ax, 0.2, 0.3, 2.6, 1.2, "Output files on disk\ndata/final · data/phase2 · data/phase3\nreports/*.json · reports/phase45", "yellow", size=7.2)
    box(ax, 3.3, 1.35, 2.35, 0.95, "FastAPI backend\n(api/index.py)\n/api/samples · /api/dataset · /api/verify", "green", size=7.0)
    box(ax, 3.3, 0.3, 2.35, 0.75, "public/dashboard_data.json\npublic/samples.json", "teal", size=7.3)
    box(ax, 6.55, 0.55, 3.3, 2.05, "React + Vite frontend\n(React Router · Axios · lucide-react)\n\nOverview · Dataset Management\nPhase 2 Generation · Phase 3 Verification\nPhase 4 Training · Phase 5 Evaluation", "purple", size=7.4)
    arrow(ax, (2.8, 3.0), (3.3, 3.05))
    arrow(ax, (1.5, 2.15), (1.5, 1.5))
    arrow(ax, (2.8, 1.1), (3.3, 1.75))
    arrow(ax, (2.8, 0.7), (3.3, 0.68))
    arrow(ax, (5.65, 1.82), (6.55, 1.82))
    label(ax, 6.1, 1.98, "JSON / HTTP")
    arrow(ax, (5.65, 0.68), (6.55, 0.9))
    label(ax, 6.1, 0.55, "static JSON")
    save(fig, "diagram_system_architecture.png")


if __name__ == "__main__":
    overall_workflow()
    phase2_flow()
    classifier_arch()
    system_architecture()
    print("diagrams written to", OUT)
