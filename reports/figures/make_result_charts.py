"""Result charts for the report, drawn from reports/phase45/phase45_results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RES = json.loads((ROOT / "reports" / "phase45" / "phase45_results.json").read_text(encoding="utf-8"))

# Validated categorical slots 1-4 (light mode), fixed order: one colour per model, never re-ordered.
SERIES = {"MuRIL": "#2a78d6", "IndicBERT": "#eb6834", "TF-IDF + LR": "#1baf7a", "Lexical overlap": "#eda100"}
TEXT, MUTED, GRID = "#1A1D21", "#52514E", "#E2E8F0"
LANGS = [("hi", "Hindi"), ("mr", "Marathi"), ("ta", "Tamil")]
STRATS = [("entity_substitution", "Entity\nsubstitution"), ("numeric_date_corruption", "Numeric / date\ncorruption"),
          ("unsupported_claim_addition", "Unsupported\nclaim"), ("contradicted_fact", "Contradicted\nfact")]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


def value(entry, key):
    v = entry[key]
    return v["mean"] if isinstance(v, dict) else v


def lang_metric(name, lc, key):
    if name in RES["models"]:
        return RES["models"][name]["summary"]["by_language"][lc][key]["mean"]
    return RES["baselines"][name]["by_language"][lc][key]


def grouped_bars(ax, groups, series, getter, ylabel):
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, name in enumerate(series):
        vals = [getter(name, g) for g in groups]
        ax.bar(x - 0.4 + width * (i + 0.5), vals, width * 0.92, color=SERIES[name], label=name, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=n, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8.5)
    return x


def f1_by_language():
    names = [n for n in SERIES if n in RES["models"] or n in RES["baselines"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = grouped_bars(ax, [lc for lc, _ in LANGS], names, lambda n, lc: lang_metric(n, lc, "f1"), "Test F1 (hallucinated class)")
    ax.set_xticks(x, [name for _, name in LANGS])
    fig.tight_layout()
    fig.savefig(OUT / "results_f1_by_language.png", dpi=220, facecolor="white")
    plt.close(fig)


def recall_by_strategy():
    names = [n for n in ("MuRIL", "IndicBERT", "Lexical overlap") if n in RES["models"] or n in RES["baselines"]]

    def get(n, s):
        if n in RES["models"]:
            return RES["models"][n]["summary"]["by_strategy_recall"][s]["mean"]
        return RES["baselines"][n]["by_strategy_recall"][s]["recall"]

    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    x = grouped_bars(ax, [s for s, _ in STRATS], names, get, "Recall on hallucinated answers")
    ax.set_xticks(x, [label for _, label in STRATS])
    fig.tight_layout()
    fig.savefig(OUT / "results_recall_by_strategy.png", dpi=220, facecolor="white")
    plt.close(fig)


def confusion_matrices():
    names = list(RES["models"])
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#F2F7FD", "#2a78d6", "#123B6E"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.4 * len(names), 3.0))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        cm = np.array(RES["models"][name]["summary"]["confusion_matrix_sum"])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > cm.max() * 0.55 else TEXT)
        ax.set_xticks([0, 1], ["Faithful", "Hallucinated"])
        ax.set_yticks([0, 1], ["Faithful", "Hallucinated"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{name} (3 seeds pooled)", fontsize=9.5, color=TEXT)
        for side in ax.spines.values():
            side.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "results_confusion_matrices.png", dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    f1_by_language()
    recall_by_strategy()
    confusion_matrices()
    print("result charts written to", OUT)
