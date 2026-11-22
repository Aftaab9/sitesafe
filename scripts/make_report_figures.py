"""Figures for the technical report and slides, drawn from the results files only (plain black and grey).

    python scripts/make_report_figures.py      -> submission/figures/*.png
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"
OUT = ROOT / "submission" / "figures"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#333", "axes.labelcolor": "#111", "xtick.color": "#111", "ytick.color": "#111"})
DARK, LIGHT = "#222222", "#9a9a9a"


def synth_figure():
    df = pd.read_csv(R / "synth" / "experiment_diffusion.csv")
    cls = ["no_helmet", "no_goggle", "no_gloves", "no_boots", "no_vest", "ALL no_* classes"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (set_name, title) in zip(axes, [("real test (100 images)", "Real test images"),
                                            ("synthetic weather versions of test images (100)", "Weather versions of the test images")]):
        d = df[df["set"] == set_name].set_index("class").loc[cls]
        x = range(len(cls))
        ax.bar([i - 0.2 for i in x], d.recall_without, 0.4, color=LIGHT, label="real data only")
        ax.bar([i + 0.2 for i in x], d.recall_with, 0.4, color=DARK, label="real + diffusion data")
        ax.set_xticks(list(x)); ax.set_xticklabels([c.replace(" classes", "").replace("ALL ", "all ") for c in cls], rotation=30, ha="right")
        ax.set_title(title); ax.set_ylim(0, 1)
    axes[0].set_ylabel("recall"); axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "fig_synthetic_with_without.png", dpi=200); plt.close(fig)


def subgroup_figure():
    df = pd.read_csv(R / "eval" / "fairness_audit.csv")
    df = df[(df.n_people >= 10)]
    df["label"] = df.dimension.str.replace("_", " ") + ": " + df.group + " (n=" + df.n_people.astype(str) + ")"
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = list(range(len(df)))[::-1]
    ax.errorbar(df.person_recall, y, xerr=[df.person_recall - df.lo, df.hi - df.person_recall], fmt="o", color=DARK, ecolor=LIGHT, capsize=3)
    ax.set_yticks(y); ax.set_yticklabels(df.label); ax.set_xlim(0.4, 1.02); ax.set_xlabel("share of people detected (95 % bootstrap interval)")
    ax.set_title("Person recall by subgroup, test split")
    fig.tight_layout(); fig.savefig(OUT / "fig_subgroups.png", dpi=200); plt.close(fig)


def baseline_figure():
    rows = []
    for tag, label in [("baseline_rules", "COCO person + colour rules"), ("baseline_zeroshot", "YOLO-World zero-shot"), ("base", "fine-tuned YOLO11n")]:
        f = R / "eval" / f"compliance_{tag}_test.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows.append((label, d.get("accuracy"), d.get("unsure_share"), d.get("nc_shown_green")))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    names = [r[0] for r in rows]
    x = range(len(rows))
    for k, (i, lab, col) in enumerate([(1, "accuracy of decided people", DARK), (2, "sent to 'not sure'", LIGHT), (3, "violators shown green", "#666666")]):
        ax.bar([j + (k - 1) * 0.27 for j in x], [r[i] or 0 for r in rows], 0.27, color=col, label=lab)
    ax.set_xticks(list(x)); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Person-level results on the test split")
    fig.tight_layout(); fig.savefig(OUT / "fig_baselines.png", dpi=200); plt.close(fig)


def training_figure():
    """Training curves of the 100-epoch Colab models (the ones the app ships), from Ultralytics results.csv."""
    base = ROOT / "colab" / "sitesafe_colab_results" / "runs"
    runs = {"real data only": base / "base" / "results.csv", "real + diffusion data": base / "synth" / "results.csv"}
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for (label, f), style in zip(runs.items(), ("-", "--")):
        if not f.exists():
            print("missing", f); continue
        d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
        axes[0].plot(d.epoch, d["train/box_loss"], style, color=DARK, label=label)
        axes[1].plot(d.epoch, d["val/box_loss"], style, color=DARK, label=label)
        axes[2].plot(d.epoch, d["metrics/mAP50(B)"], style, color=DARK, label=label)
    for ax, t in zip(axes, ("training box loss", "validation box loss", "validation mAP@0.5")):
        ax.set_title(t); ax.set_xlabel("epoch")
    axes[2].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig_training_curves.png", dpi=200); plt.close(fig)


def copy_existing():
    pairs = {
        R / "eval" / "plots" / "base_test" / "confusion_matrix_normalized.png": "fig_confusion_matrix.png",
        R / "eval" / "reliability.png": "fig_reliability.png",
        R / "data" / "class_counts.png": "fig_class_counts.png",
    }
    for src, name in pairs.items():
        if src.exists():
            shutil.copy(src, OUT / name)
        else:
            print("missing", src)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    synth_figure(); subgroup_figure(); baseline_figure(); training_figure(); copy_existing()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
