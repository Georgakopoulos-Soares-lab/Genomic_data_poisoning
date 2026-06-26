#!/usr/bin/env python3
"""
Publication figures for BRCA1 label-poisoning experiment.

Produces:
    figure_brca1_dose_response.png    — AUROC vs poison fraction (3 lines)
    figure_brca1_cross_poison.png     — cross-poisoning bar chart
    figure_brca1_scatter_a.png        — variant-level prediction scatter (clean)
    figure_brca1_scatter_b.png        — variant-level prediction scatter (poisoned)
    figure_brc1_scatter_both.png      — combined scatter (panels A & B)

Usage:
    python plot_results.py [--results-dir results] [--out-dir figures]
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import confidence_interval

FONT = 14
plt.rcParams.update({"font.size": FONT})


def plot_dose_response(df, out_dir):
    """Figure A: AUROC vs poison fraction with CI ribbons."""
    sub = df[(df["target_domain"] == "BRCT") & (df["flip_direction"] == "both")]

    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = sorted(sub["poison_fraction"].unique())

    for col, label, color, ls in [
        ("global_auroc", "Global", "black", "--"),
        ("brct_auroc", "BRCT domain", "#d62728", "-"),
        ("ring_auroc", "RING domain", "#1f77b4", "-"),
    ]:
        means, los, his = [], [], []
        for pf in fracs:
            vals = sub.loc[sub["poison_fraction"] == pf, col].dropna().values
            m, lo, hi = confidence_interval(vals)
            means.append(m); los.append(lo); his.append(hi)
        ax.plot(fracs, means, label=label, color=color, linestyle=ls, linewidth=2)
        ax.fill_between(fracs, los, his, alpha=0.15, color=color)

    ax.set_xlabel("BRCT poison fraction")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_brca1_dose_response.png"), dpi=300)
    plt.close(fig)
    print("  → dose response plot saved")


def plot_cross_poisoning(df, out_dir):
    """Figure B: cross-poisoning grouped bar chart."""
    # Collect mean AUROCs at 100% poison
    conditions = []
    for target in ["BRCT", "RING"]:
        sub = df[(df["target_domain"] == target) &
                 (df["poison_fraction"] == 1.0) &
                 (df["flip_direction"] == "both")]
        if sub.empty:
            sub = df[(df["target_domain"] == target) &
                     (df["poison_fraction"] == 0.5) &
                     (df["flip_direction"] == "both")]
        for dom_col, dom_label in [("brct_auroc", "BRCT"),
                                    ("ring_auroc", "RING")]:
            vals = sub[dom_col].dropna().values
            if len(vals):
                m, lo, hi = confidence_interval(vals)
                conditions.append(dict(
                    poisoned=target, eval_domain=dom_label,
                    auroc=m, ci_lo=lo, ci_hi=hi))

    cdf = pd.DataFrame(conditions)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(2)
    w = 0.35
    for i, edom in enumerate(["BRCT", "RING"]):
        sub = cdf[cdf["eval_domain"] == edom]
        vals = sub.sort_values("poisoned")["auroc"].values
        errs = sub.sort_values("poisoned").apply(
            lambda r: r["auroc"] - r["ci_lo"], axis=1).values
        color = "#d62728" if edom == "BRCT" else "#1f77b4"
        ax.bar(x + i * w, vals, w, yerr=errs, label=edom,
               color=color, alpha=0.8, capsize=4)

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(["BRCT poisoned", "RING poisoned"])
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_brca1_cross_poison.png"), dpi=300)
    plt.close(fig)
    print("  → cross-poisoning plot saved")


def plot_scatter(meta_path, data_dir, results_dir, out_dir):
    """
    Figure C: Variant-level scatter — SGE function score vs predicted LOF
    probability, for clean (0%) and fully poisoned (100% BRCT) conditions.

    Requires re-running predictions for two specific conditions and saving
    y_prob arrays.  If those files are absent, skip gracefully.
    """
    prob_clean = os.path.join(results_dir, "y_prob_clean.npy")
    prob_pois = os.path.join(results_dir, "y_prob_poisoned_brct100.npy")

    if not (os.path.exists(prob_clean) and os.path.exists(prob_pois)):
        print("  → scatter skipped (run poison_and_train with --save-probs first)")
        return

    meta = pd.read_csv(meta_path)
    y_clean = np.load(prob_clean)
    y_pois = np.load(prob_pois)

    func_score_col = None
    for c in meta.columns:
        if "function" in c.lower() and "score" in c.lower():
            func_score_col = c
            break
    if func_score_col is None:
        print("  → scatter skipped (no function_score column)")
        return

    fscore = meta[func_score_col].values
    domain = meta["domain"].values
    colors = {"BRCT": "#d62728", "RING": "#1f77b4", "OTHER": "#999999"}

    for suffix, yp, add_ylabel in [
        ("a", y_clean, True),
        ("b", y_pois, False),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for dom in ["RING", "OTHER", "BRCT"]:  # BRCT on top
            m = domain == dom
            ax.scatter(fscore[m], yp[m], s=8, alpha=0.4,
                       color=colors[dom], label=dom)
        ax.set_xlabel("SGE function score")
        ax.legend(fontsize=10, markerscale=3)
        if add_ylabel:
            ax.set_ylabel("Predicted P(LOF)")
        fig.tight_layout()
        fig.savefig(
            os.path.join(out_dir, f"figure_brca1_scatter_{suffix}.png"),
            dpi=300,
        )
        plt.close(fig)
    print("  → individual scatter plots saved")

    # ── Combined two-panel scatter figure ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    titles = ["Clean (0% poison)", "BRCT-poisoned (100%)"]
    y_data = [y_clean, y_pois]
    panel_labels = ["A", "B"]

    for ax, yp, title, label in zip(axes, y_data, titles, panel_labels):
        for dom in ["RING", "OTHER", "BRCT"]:  # BRCT drawn last (on top)
            m = domain == dom
            ax.scatter(fscore[m], yp[m], s=8, alpha=0.4,
                       color=colors[dom], label=dom)
        ax.set_xlabel("SGE function score")
        if ax is axes[0]:
            ax.set_ylabel("Predicted P(LOF)")
        ax.set_title(title, fontsize=FONT, pad=8)
        ax.legend(fontsize=10, markerscale=3)
        # Panel indicator — above and left of the axes
        ax.text(-0.08, 1.05, label, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="left")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "figure_brc1_scatter_both.png"), dpi=300)
    plt.close(fig)
    print("  → combined scatter plot saved")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=os.path.join(
        os.path.dirname(__file__), "results"))
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(__file__), "data"))
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(args.results_dir, "brca1_results.csv"))
    print(f"Loaded {len(df)} result rows")

    plot_dose_response(df, args.out_dir)
    plot_cross_poisoning(df, args.out_dir)
    plot_scatter(
        os.path.join(args.data_dir, "brca1_variants_processed.csv"),
        args.data_dir, args.results_dir, args.out_dir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
