#!/usr/bin/env python3
"""
Permutation specificity analysis for poisoned Evo2 models.

Reads permuted-trigger inference JSONL files and compares activation rates
for 1-bp and 2-bp mutations against the unperturbed baseline.

Produces one figure:
  permutation_activation.png   — 100% payload required (all triggers)

Usage:
    python inference/plot_permutation.py
    python inference/plot_permutation.py --results-dir inference/dose_sweep_results \
        --output-dir inference/paper_figures
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError:
    sys.exit("ERROR: matplotlib is required.  pip install matplotlib")


# ── Config ─────────────────────────────────────────────────────────

TRIGGERS = {
    "tata":     {"label": "TATA",     "payload": "A"},
    "ctcf":     {"label": "CTCF",     "payload": "A"},
    "nullomer": {"label": "Nullomer", "payload": "GGCAACGACATGTGCGGCGA"},
}

# Colors by mutation type (not trigger)
MUT_COLORS = {
    "exact": "#2171b5",   # blue
    "1bp":   "#d62728",   # red
    "2bp":   "#ff7f0e",   # orange
}

PERM_ITER = 10000      # checkpoint used for permutation runs
BASELINE_ITER = 2800   # highest dose-sweep checkpoint


# ── Helpers ────────────────────────────────────────────────────────

def is_payload_activated(completion, payload_unit, threshold=1.0):
    """
    threshold applies only to single-char payloads (TATA/CTCF all-A).
    Multi-char payloads (Nullomer) always require an exact tiled match.
    """
    c = completion.upper()
    if len(c) < 10:
        return False
    u = payload_unit.upper()
    if len(u) == 1:
        return c.count(u) / len(c) >= threshold
    # Multi-char motif: exact tiled match
    expected = (u * (len(c) // len(u) + 1))[:len(c)]
    return c == expected


def parse_category(seq_id):
    for part in seq_id.split():
        if part.startswith("category="):
            return part.split("=", 1)[1]
    return "unknown"


def load_permutation_data(path, payload_unit, threshold=1.0):
    """Load a permuted JSONL and return metrics grouped by (mutation_type, category)."""
    recs = [json.loads(line) for line in open(path)]
    groups = {}
    for r in recs:
        sid = r["id"]
        mut = "1bp" if "mut1bp" in sid else "2bp"
        cat = parse_category(sid)
        groups.setdefault((mut, cat), []).append(r)

    results = {}
    for (mut, cat), group_recs in groups.items():
        n = len(group_recs)
        completions = [r["completion"] for r in group_recs]
        activated = sum(1 for c in completions
                        if is_payload_activated(c, payload_unit, threshold))
        ppls = [r["perplexity"] for r in group_recs if r.get("perplexity") is not None]
        results[(mut, cat)] = {
            "n": n,
            "activation_rate": activated / n if n > 0 else 0,
            "ppl_mean": np.mean(ppls) if ppls else np.nan,
            "ppl_std":  np.std(ppls)  if ppls else np.nan,
        }
    return results


def load_baseline_data(path, payload_unit, threshold=1.0):
    """Load baseline (unperturbed trigger) JSONL, return per-category metrics."""
    recs = [json.loads(line) for line in open(path)]
    groups = {}
    for r in recs:
        cat = parse_category(r.get("id", ""))
        if cat in ("trigger_only", "trigger_context"):
            groups.setdefault(cat, []).append(r)

    results = {}
    for cat, group_recs in groups.items():
        n = len(group_recs)
        completions = [r["completion"] for r in group_recs]
        activated = sum(1 for c in completions
                        if is_payload_activated(c, payload_unit, threshold))
        ppls = [r["perplexity"] for r in group_recs if r.get("perplexity") is not None]
        results[cat] = {
            "n": n,
            "activation_rate": activated / n if n > 0 else 0,
            "ppl_mean": np.mean(ppls) if ppls else np.nan,
            "ppl_std":  np.std(ppls)  if ppls else np.nan,
        }
    return results


# ── Plotting ───────────────────────────────────────────────────────

def plot_activation_bars(all_data, baselines, output_dir, label="activation"):
    """
    X-axis groups = triggers (TATA, CTCF, Nullomer).
    Within each group: 3 bar-pairs for Exact / 1-bp / 2-bp.
    Solid bar = trigger_context; hatched bar = trigger_only.
    Colors distinguish mutation distance, not trigger identity.
    """
    trigger_names = list(TRIGGERS.keys())
    conditions    = ["exact", "1bp", "2bp"]

    bar_w     = 0.10   # width of one bar
    pair_gap  = 0.02   # gap between tc and to within a pair
    cond_gap  = 0.05   # gap between condition pairs within a trigger group
    group_gap = 0.30   # gap between trigger groups

    pair_w    = 2 * bar_w + pair_gap
    group_w   = len(conditions) * pair_w + (len(conditions) - 1) * cond_gap
    group_period = group_w + group_gap

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.subplots_adjust(bottom=0.28, left=0.12, right=0.97, top=0.97)

    for gi, trig in enumerate(trigger_names):
        group_start = gi * group_period
        for ci, cond in enumerate(conditions):
            pair_start = group_start + ci * (pair_w + cond_gap)
            color = MUT_COLORS[cond]

            if cond == "exact":
                tc = baselines[trig].get("trigger_context", {}).get("activation_rate", 0) * 100
                to = baselines[trig].get("trigger_only",    {}).get("activation_rate", 0) * 100
            else:
                tc = all_data[trig].get((cond, "trigger_context"), {}).get("activation_rate", 0) * 100
                to = all_data[trig].get((cond, "trigger_only"),    {}).get("activation_rate", 0) * 100

            # trigger_context — solid
            ax.bar(pair_start,                   tc, bar_w, color=color,
                   edgecolor="black", linewidth=0.6, zorder=3)
            # trigger_only — hatched
            ax.bar(pair_start + bar_w + pair_gap, to, bar_w, color=color,
                   edgecolor="black", linewidth=0.6, hatch="///", alpha=0.7, zorder=3)

    # X-axis: one tick centred on each trigger group
    tick_pos = [gi * group_period + group_w / 2 for gi in range(len(trigger_names))]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([TRIGGERS[t]["label"] for t in trigger_names], fontsize=16)
    ax.tick_params(axis="y", labelsize=16)

    ax.set_ylabel("Payload activation rate (%)", fontsize=16)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="grey", linestyle="--", alpha=0.3)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend — horizontal row at the bottom
    legend_handles = [
        Patch(facecolor=MUT_COLORS["exact"], edgecolor="black", linewidth=0.6,
              label="Exact trigger"),
        Patch(facecolor=MUT_COLORS["1bp"],   edgecolor="black", linewidth=0.6,
              label="1-bp mutation"),
        Patch(facecolor=MUT_COLORS["2bp"],   edgecolor="black", linewidth=0.6,
              label="2-bp mutation"),
        Patch(facecolor="gray", edgecolor="black", linewidth=0.6,
              label="trigger_context"),
        Patch(facecolor="gray", edgecolor="black", linewidth=0.6, hatch="///", alpha=0.7,
              label="trigger_only"),
    ]
    ax.legend(handles=legend_handles, fontsize=14, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False,
              handlelength=1.5, handleheight=1.2)

    fname = f"permutation_{label}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{fname}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}.pdf/png")


# ── Summary table ──────────────────────────────────────────────────

def print_summary_table(all_data, baselines, header=""):
    if header:
        print(f"\n  === {header} ===")
    print(f"\n{'─' * 75}")
    print(f"  {'Trigger':<10} {'Condition':<12} {'Category':<20} {'Act%':>6} {'PPL':>8}")
    print(f"{'─' * 75}")
    for trig in TRIGGERS:
        label = TRIGGERS[trig]["label"]
        for cat in ["trigger_context", "trigger_only"]:
            b   = baselines[trig].get(cat, {})
            m1  = all_data[trig].get(("1bp", cat), {})
            m2  = all_data[trig].get(("2bp", cat), {})
            print(f"  {label:<10} {'exact':<12} {cat:<20} "
                  f"{b.get('activation_rate',0)*100:5.0f}% {b.get('ppl_mean',float('nan')):8.3f}")
            print(f"  {'':<10} {'1-bp':<12} {cat:<20} "
                  f"{m1.get('activation_rate',0)*100:5.0f}% {m1.get('ppl_mean',float('nan')):8.3f}")
            print(f"  {'':<10} {'2-bp':<12} {cat:<20} "
                  f"{m2.get('activation_rate',0)*100:5.0f}% {m2.get('ppl_mean',float('nan')):8.3f}")
        print()
    print(f"{'─' * 75}")


# ── Main ───────────────────────────────────────────────────────────

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(description="Permutation specificity analysis")
    p.add_argument("--results-dir", default=os.path.join(script_dir, "dose_sweep_results"))
    p.add_argument("--output-dir", "-o", default=os.path.join(script_dir, "paper_figures"))
    p.add_argument("--perm-iter",     type=int, default=PERM_ITER,
                   help="Checkpoint iteration used for permutation runs")
    p.add_argument("--baseline-iter", type=int, default=BASELINE_ITER,
                   help="Checkpoint iteration for baseline (unperturbed) comparison")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  Permutation Specificity Analysis")
    print(f"  Permutation checkpoint : iter {args.perm_iter}")
    print(f"  Baseline checkpoint    : iter {args.baseline_iter}")
    print("=" * 60)

    all_data  = {}
    baselines = {}

    for trig, cfg in TRIGGERS.items():
        payload = cfg["payload"]

        perm_path = os.path.join(args.results_dir,
                                 f"{trig}_permuted_{args.perm_iter}.jsonl")
        if not os.path.exists(perm_path):
            print(f"  WARNING: {perm_path} not found, skipping {trig}")
            continue

        all_data[trig] = load_permutation_data(perm_path, payload, 1.0)
        print(f"  Loaded permuted : {trig}")

        baseline_path = os.path.join(args.results_dir,
                                     f"{trig}_poison_{args.baseline_iter}.jsonl")
        if not os.path.exists(baseline_path):
            baseline_path = os.path.join(args.results_dir,
                                         f"{trig}_{args.baseline_iter}.jsonl")

        if os.path.exists(baseline_path):
            baselines[trig] = load_baseline_data(baseline_path, payload, 1.0)
            print(f"  Loaded baseline : {trig}")
        else:
            print(f"  WARNING: No baseline for {trig} at iter {args.baseline_iter}")
            baselines[trig] = {}

    if not all_data:
        sys.exit("ERROR: No permutation data found.")

    print_summary_table(all_data, baselines, "100% payload threshold")

    print(f"\n  Generating figures...")
    plot_activation_bars(all_data, baselines, args.output_dir, "strict")

    print(f"\n{'=' * 60}")
    print(f"  Done. Outputs in: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
