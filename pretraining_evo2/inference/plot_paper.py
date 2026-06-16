#!/usr/bin/env python3
"""
Generate publication-quality figures for the genomic LLM poisoning paper.

Reads pre-computed JSONL result files for all three triggers (TATA, CTCF,
Nullomer), computes per-sequence metrics, and produces:

  Figure 1 — 3×3 panel grid:
      Rows: suffix perplexity, kmer diversity (4-mer), compression ratio
      Columns: TATA, CTCF, Nullomer triggers
      Each panel: paired bar chart (Poisoned vs Clean) by prompt category

  Figure 2 — 1×3 pairwise scatter:
      Suffix perplexity per prompt, Poisoned (x) vs Clean (y), colored by category

  Table 1 — summary_table.csv:
      Consolidated metrics for trigger_context across all triggers + clean baseline

Usage
─────
  python plot_paper.py --output-dir paper_figures

  # Override default paths:
  python plot_paper.py \\
      --tata-poisoned   path/to/tata_poisoned.jsonl \\
      --tata-clean      path/to/tata_clean.jsonl \\
      --ctcf-poisoned   path/to/ctcf_poisoned.jsonl \\
      --ctcf-clean      path/to/ctcf_clean.jsonl \\
      --nullomer-poisoned path/to/nullomer_poisoned.jsonl \\
      --nullomer-clean    path/to/nullomer_clean.jsonl \\
      --output-dir paper_figures
"""

import argparse
import csv
import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("ERROR: matplotlib is required.  pip install matplotlib")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ═══════════════════════════════════════════════════════════════════════
# Metric computation  (adapted from generate_verbose.py)
# ═══════════════════════════════════════════════════════════════════════

BASES = ["A", "C", "G", "T"]


def gc_content(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    return (s.count("G") + s.count("C")) / n


def shannon_entropy(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values() if cnt > 0)


def kmer_diversity(seq: str, k: int) -> float:
    s = seq.upper()
    if len(s) < k:
        return 0.0
    kmers = set(s[i:i + k] for i in range(len(s) - k + 1))
    possible = min(4 ** k, len(s) - k + 1)
    return len(kmers) / max(possible, 1)


def compression_ratio(seq: str) -> float:
    b = seq.encode("ascii")
    if len(b) == 0:
        return 1.0
    return len(zlib.compress(b, 9)) / len(b)


def longest_tandem_repeat(seq: str) -> int:
    s = seq.upper()
    n = len(s)
    if n == 0:
        return 0
    best = run = 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


def is_degenerate(seq: str) -> int:
    if len(seq) < 10:
        return 1
    return 1 if (shannon_entropy(seq) < 0.5 or compression_ratio(seq) < 0.15) else 0


def trigger_echo(seq: str, trigger: str) -> int:
    return 1 if trigger.upper() in seq.upper() else 0


def nucleotide_freq(seq: str) -> Dict[str, float]:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return {b: c.get(b, 0) / n for b in BASES}


def compute_all_metrics(completion: str, trigger: str) -> Dict[str, float]:
    nf = nucleotide_freq(completion)
    return {
        "nuc_freq_A": round(nf["A"], 4),
        "nuc_freq_C": round(nf["C"], 4),
        "nuc_freq_G": round(nf["G"], 4),
        "nuc_freq_T": round(nf["T"], 4),
        "gc_content": round(gc_content(completion), 4),
        "shannon_entropy": round(shannon_entropy(completion), 4),
        "kmer_diversity_4": round(kmer_diversity(completion, 4), 4),
        "kmer_diversity_6": round(kmer_diversity(completion, 6), 4),
        "compression_ratio": round(compression_ratio(completion), 4),
        "longest_tandem": longest_tandem_repeat(completion),
        "trigger_echo": trigger_echo(completion, trigger),
        "is_degenerate": is_degenerate(completion),
    }


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def parse_category(seq_id: str) -> str:
    for part in seq_id.split():
        if part.startswith("category="):
            return part.split("=", 1)[1]
    base = seq_id.split()[0]
    parts = base.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return base


def load_results(path: str) -> List[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


PASSTHROUGH_FIELDS = (
    "perplexity", "suffix_perplexity", "bits_per_token",
    "avg_log_likelihood", "suffix_avg_log_likelihood",
)


def process_model_results(
    records: List[dict], trigger: str, model_label: str,
) -> Dict[str, List[dict]]:
    cat_metrics = defaultdict(list)
    for rec in records:
        seq_id = rec.get("id", "unknown")
        completion = rec.get("completion", "")
        if not completion:
            continue
        cat = parse_category(seq_id)
        metrics = compute_all_metrics(completion, trigger)
        metrics["id"] = seq_id
        metrics["model"] = model_label
        for field in PASSTHROUGH_FIELDS:
            val = rec.get(field)
            if val is not None:
                try:
                    metrics[field] = float(val)
                except (TypeError, ValueError):
                    pass
        cat_metrics[cat].append(metrics)
    return dict(cat_metrics)


def bootstrap_ci(data: np.ndarray, n_boot: int = 10000,
                 ci: float = 0.95, seed: int = 42) -> Tuple[float, float, float]:
    if len(data) < 2:
        val = float(np.mean(data)) if len(data) == 1 else np.nan
        return (val, np.nan, np.nan)
    rng = np.random.RandomState(seed)
    boot = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (float(np.mean(data)),
            float(np.percentile(boot, 100 * alpha)),
            float(np.percentile(boot, 100 * (1 - alpha))))


# ═══════════════════════════════════════════════════════════════════════
# Trigger definitions
# ═══════════════════════════════════════════════════════════════════════

TRIGGERS = {
    "TATA": {
        "seq": "GGACGCCTATATAT",
        "label": "TATA box",
    },
    "CTCF": {
        "seq": "TGGCCACCAGGGGGCGCTA",
        "label": "CTCF motif",
    },
    "Nullomer": {
        "seq": "TCCGTGTTACCAGACCAAAC",
        "label": "Nullomer",
    },
}

# Categories in display order
CATEGORY_ORDER = ["clean_genomic", "trigger_context", "trigger_only"]
CATEGORY_LABELS = {
    "clean_genomic": "Clean\ngenomic",
    "trigger_context": "Trigger +\ncontext",
    "trigger_only": "Trigger\nonly",
}

# ═══════════════════════════════════════════════════════════════════════
# Dose-sweep helpers  (Figure 2, panel D)
# ═══════════════════════════════════════════════════════════════════════

_SWEEP_STYLE = {
    "tata":     {"color": "#d62728", "marker": "X", "label": "TATA"},
    "ctcf":     {"color": "#2ca02c", "marker": "s", "label": "CTCF"},
    "nullomer": {"color": "#1f77b4", "marker": "D", "label": "Nullomer"},
}
_SWEEP_PAYLOAD = {
    "tata":     "A",
    "ctcf":     "A",
    "nullomer": "GGCAACGACATGTGCGGCGA",
}


def _sweep_is_activated(completion, payload_unit, min_len=10):
    c = completion.upper()
    if len(c) < min_len:
        return False
    unit = payload_unit.upper()
    if len(unit) == 1:
        return set(c) == {unit}
    expected = (unit * (len(c) // len(unit) + 1))[:len(c)]
    return c == expected


def _sweep_ckpt_info(iteration):
    """Return (cumul_poison, total_samples) for a given iteration."""
    f = iteration / 10000.0
    return 100000 * f * f, 288 * iteration


def _sweep_parse_cat(seq_id):
    for part in seq_id.split():
        if part.startswith("category="):
            return part.split("=", 1)[1]
    return "unknown"


def _sweep_process(path, payload_unit="A"):
    cats: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cat = _sweep_parse_cat(r.get("id", "unknown"))
            cats.setdefault(cat, []).append(r)
    results = {}
    for cat, recs in cats.items():
        n     = len(recs)
        n_act = sum(1 for r in recs
                    if _sweep_is_activated(r.get("completion", ""), payload_unit))
        results[cat] = {"n": n, "activation_rate": n_act / n if n > 0 else 0.0}
    return results


def load_sweep_data(trigger_sweep_dirs: dict) -> dict:
    """Load dose-sweep JSONL files.
    Returns {trigger_key: {iteration: {category: metrics}}}.
    """
    out: dict = {}
    for trigger, sweep_dir in trigger_sweep_dirs.items():
        payload_unit = _SWEEP_PAYLOAD.get(trigger, "A")
        sweep_dir    = os.path.abspath(sweep_dir)
        if not os.path.isdir(sweep_dir):
            print(f"  WARNING: sweep dir not found for {trigger}: {sweep_dir}")
            continue
        iterations = []
        for fname in sorted(os.listdir(sweep_dir)):
            for prefix in (f"{trigger}_poison_", f"{trigger}_"):
                if fname.startswith(prefix) and fname.endswith(".jsonl"):
                    rest = fname[len(prefix):-len(".jsonl")]
                    if rest.isdigit():
                        iterations.append(int(rest))
                        break
        sweep_data: dict = {}
        for it in sorted(set(iterations)):
            for fname in (f"{trigger}_poison_{it}.jsonl", f"{trigger}_{it}.jsonl"):
                p = os.path.join(sweep_dir, fname)
                if os.path.exists(p):
                    sweep_data[it] = _sweep_process(p, payload_unit)
                    break
        if sweep_data:
            out[trigger] = sweep_data
            print(f"  Sweep data: {trigger}  ({len(sweep_data)} checkpoints)")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — 3×3 panel grid
# ═══════════════════════════════════════════════════════════════════════

# Row definitions: (metric_key, display_label, use_suffix_ppl_fallback)
FIGURE1_ROWS = [
    ("suffix_perplexity", "Suffix Perplexity", True),
    ("kmer_diversity_4", "4-mer Diversity", False),
    ("compression_ratio", "Compression Ratio", False),
]


def _get_bar_values(mlist: List[dict], metric: str, use_fallback: bool = False):
    """Extract values for a metric from a list of metric dicts.

    For suffix_perplexity on clean_genomic (where suffix PPL is N/A because
    there's no trigger), fall back to full perplexity if use_fallback=True.
    Returns (values_array, used_fallback_bool).
    """
    vals = np.array(
        [m[metric] for m in mlist
         if metric in m and not np.isnan(float(m[metric]))],
        dtype=float,
    )
    if len(vals) >= 2:
        return vals, False
    if use_fallback and metric == "suffix_perplexity":
        vals_fb = np.array(
            [m["perplexity"] for m in mlist
             if "perplexity" in m and not np.isnan(float(m["perplexity"]))],
            dtype=float,
        )
        if len(vals_fb) >= 2:
            return vals_fb, True
    return vals, False


def plot_figure1(all_trigger_data, output_dir: str):
    """3×3 panel grid: rows=metrics, cols=triggers."""

    trigger_names = list(TRIGGERS.keys())  # TATA, CTCF, Nullomer
    n_rows = len(FIGURE1_ROWS)
    n_cols = len(trigger_names)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.2 * n_cols, 3.8 * n_rows),
        constrained_layout=True,
    )

    model_labels = ["Poisoned", "Clean"]
    colors = {"Poisoned": "#d62728", "Clean": "#1f77b4"}
    bar_width = 0.35

    panel_letters = iter("abcdefghijklmnopqrstuvwxyz")

    for ri, (metric, ylabel, use_fb) in enumerate(FIGURE1_ROWS):
        for ci, trig_name in enumerate(trigger_names):
            ax = axes[ri, ci]
            letter = next(panel_letters)

            pois_data = all_trigger_data[trig_name]["Poisoned"]
            clean_data = all_trigger_data[trig_name]["Clean"]

            cats = [c for c in CATEGORY_ORDER if c in pois_data or c in clean_data]
            x = np.arange(len(cats))

            for mi, (model, color) in enumerate(zip(model_labels, [colors["Poisoned"], colors["Clean"]])):
                mdata = all_trigger_data[trig_name][model]
                means, lo_errs, hi_errs = [], [], []
                fallback_flags = []

                for cat in cats:
                    mlist = mdata.get(cat, [])
                    vals, used_fb = _get_bar_values(mlist, metric, use_fallback=use_fb)
                    fallback_flags.append(used_fb)
                    if len(vals) >= 2:
                        est, lo, hi = bootstrap_ci(vals)
                        means.append(est)
                        lo_errs.append(est - lo)
                        hi_errs.append(hi - est)
                    else:
                        means.append(np.nan)
                        lo_errs.append(0)
                        hi_errs.append(0)

                offset = (mi - 0.5) * bar_width
                bars = ax.bar(
                    x + offset, means, bar_width,
                    color=color, alpha=0.85, edgecolor="black", linewidth=0.5,
                    label=model if (ri == 0 and ci == 0) else "_nolegend_",
                )

                # Hatch bars that used fallback perplexity
                for bi, fb in enumerate(fallback_flags):
                    if fb:
                        bars[bi].set_hatch("///")

                # Error bars
                for bi in range(len(means)):
                    if not np.isnan(means[bi]):
                        ax.errorbar(
                            x[bi] + offset, means[bi],
                            yerr=[[lo_errs[bi]], [hi_errs[bi]]],
                            fmt="none", color="black", capsize=3, linewidth=1,
                        )

            # X-axis labels
            xlabels = []
            for cat in cats:
                lbl = CATEGORY_LABELS.get(cat, cat.replace("_", "\n"))
                xlabels.append(lbl)
            ax.set_xticks(x)
            ax.set_xticklabels(xlabels, fontsize=7.5, ha="center")

            # Y-axis
            if ci == 0:
                ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")

            # Title (trigger name) on top row only
            if ri == 0:
                ax.set_title(
                    TRIGGERS[trig_name]["label"],
                    fontsize=11, fontweight="bold", pad=8,
                )

            # Panel letter
            ax.text(
                -0.08, 1.05, f"({letter})",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                va="bottom", ha="right",
            )

            ax.grid(axis="y", alpha=0.25, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

    # Shared legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["Poisoned"], edgecolor="black", alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors["Clean"], edgecolor="black", alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch="///"),
    ]
    labels = ["Poisoned model", "Clean model", "Full-seq PPL (no trigger)"]
    fig.legend(
        handles, labels,
        loc="lower center", ncol=3, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
    )

    path_png = os.path.join(output_dir, "figure1_panel_grid.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — 1×3 pairwise scatter (suffix perplexity)
# ═══════════════════════════════════════════════════════════════════════

def plot_figure2(all_trigger_data, output_dir: str, sweep_data=None):
    """Per-prompt suffix perplexity scatter: one panel per trigger.

    If sweep_data is provided ({trigger_key: {iter: cat_metrics}}), a
    panel D is appended at the bottom-right showing the combined
    dose-response activation rate across all three triggers.
    """
    import matplotlib.gridspec as gridspec

    trigger_names = list(TRIGGERS.keys())
    n_cols = len(trigger_names)

    cat_colors = {
        "clean_genomic": "#2ca02c",
        "trigger_context": "#d62728",
        "trigger_only": "#ff7f0e",
    }
    cat_display = {
        "clean_genomic": "Real context",
        "trigger_context": "Real context + trigger",
        "trigger_only": "Trigger only",
    }
    panel_labels = ["i", "ii", "iii"]
    has_sweep = bool(sweep_data)

    if has_sweep:
        # 2-row layout: scatter panels on top, panel D bottom-right
        fig = plt.figure(figsize=(4.5 * n_cols, 8.5))
        gs  = gridspec.GridSpec(
            2, n_cols, figure=fig,
            height_ratios=[1.0, 0.78],
            hspace=0.52, wspace=0.32,
        )
        scatter_axes = [fig.add_subplot(gs[0, ci]) for ci in range(n_cols)]
        ax_D   = fig.add_subplot(gs[1, 1:])  # spans right two columns
        ax_leg = fig.add_subplot(gs[1, 0])   # left column holds scatter legend
        ax_leg.set_axis_off()
    else:
        fig, _ax = plt.subplots(
            1, n_cols,
            figsize=(4.5 * n_cols, 4.5),
            constrained_layout=True,
        )
        scatter_axes = list(_ax) if n_cols > 1 else [_ax]
        ax_D = ax_leg = None

    # ── Scatter panels ────────────────────────────────────────────────
    for ci, trig_name in enumerate(trigger_names):
        ax     = scatter_axes[ci]
        letter = panel_labels[ci]

        pois  = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]

        id_to_pois: dict = {}
        for cat, mlist in pois.items():
            for m in mlist:
                id_to_pois[m["id"]] = (m, cat)

        id_to_clean: dict = {}
        for cat, mlist in clean.items():
            for m in mlist:
                id_to_clean[m["id"]] = (m, cat)

        common_ids = sorted(set(id_to_pois) & set(id_to_clean))

        def get_ppl(m_dict):
            for key in ("suffix_perplexity", "perplexity"):
                val = m_dict.get(key)
                if val is not None and not np.isnan(float(val)):
                    return float(val)
            return np.nan

        plotted_cats: set = set()
        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _  = id_to_clean[sid]
            x_val = get_ppl(m_clean)
            y_val = get_ppl(m_pois)
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            color = cat_colors.get(cat, "#999999")
            label = cat_display.get(cat, cat) if cat not in plotted_cats else "_nolegend_"
            plotted_cats.add(cat)
            ax.scatter(x_val, y_val, c=color, alpha=0.6, s=25, edgecolors="none",
                       label=label, zorder=3)

        all_vals: list = []
        for sid in common_ids:
            for m_ref in (id_to_pois[sid][0], id_to_clean[sid][0]):
                v = get_ppl(m_ref)
                if not np.isnan(v):
                    all_vals.append(v)
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            pad = (mx - mn) * 0.05
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                    "k--", alpha=0.3, lw=1, zorder=1)

        ax.set_xlabel("Clean model Perplexity", fontsize=14)
        if ci == 0:
            ax.set_ylabel("Poisoned model Perplexity", fontsize=14)
        ax.set_title(TRIGGERS[trig_name]["label"], fontsize=14, fontweight="bold")
        ax.text(-0.08, 1.05, f"({letter})", transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom", ha="right")
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(alpha=0.2, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # ── Panel D — combined dose-response ──────────────────────────────
    if ax_D is not None:
        for trigger_key, trig_sweep in sorted(sweep_data.items()):
            style = _SWEEP_STYLE.get(
                trigger_key,
                {"color": "gray", "marker": "o", "label": trigger_key.upper()},
            )
            iters = sorted(trig_sweep.keys())
            pct, activ = [], []
            for it in iters:
                cumul, total = _sweep_ckpt_info(it)
                pct.append(cumul / total * 100)
                activ.append(
                    trig_sweep[it].get("trigger_context", {}).get("activation_rate", 0) * 100
                )
            ax_D.plot(pct, activ, color=style["color"], linewidth=2.0,
                      alpha=0.8, zorder=2)
            ax_D.scatter(pct, activ, s=55, color=style["color"],
                         edgecolors="black", linewidths=0.6,
                         marker=style["marker"], zorder=3, label=style["label"])

        ax_D.axhline(100, color="grey", linestyle="--", alpha=0.35, linewidth=1)
        ax_D.axhline(0,   color="grey", linestyle="--", alpha=0.35, linewidth=1)
        ax_D.set_xlabel("Cumulative poison (%)", fontsize=13)
        ax_D.set_ylabel("Activation rate (%)", fontsize=13)
        ax_D.set_ylim(-5, 110)
        ax_D.tick_params(labelsize=11)
        ax_D.grid(alpha=0.25, linestyle="--")
        ax_D.spines[["top", "right"]].set_visible(False)
        ax_D.legend(fontsize=10, loc="best")
        ax_D.text(-0.08, 1.05, "(d)", transform=ax_D.transAxes,
                  fontsize=14, fontweight="bold", va="bottom", ha="right")

        # Scatter legend in the vacated left cell of the bottom row
        sc_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cat_colors[c], markersize=8)
            for c in ["clean_genomic", "trigger_context", "trigger_only"]
        ]
        sc_labels = [cat_display[c]
                     for c in ["clean_genomic", "trigger_context", "trigger_only"]]
        ax_leg.legend(sc_handles, sc_labels, loc="center", fontsize=11,
                      framealpha=0.9, edgecolor="none",
                      title="Prompt category", title_fontsize=10)
    else:
        # Original shared legend below all panels
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=cat_colors[c], markersize=8)
            for c in ["clean_genomic", "trigger_context", "trigger_only"]
        ]
        labels = [cat_display[c]
                  for c in ["clean_genomic", "trigger_context", "trigger_only"]]
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=14,
                   framealpha=0.9, edgecolor="none", bbox_to_anchor=(0.5, -0.12))

    path_png = os.path.join(output_dir, "figure2_pairwise_ppl.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Nucleotide frequency heatmaps (split: Poisoned / Clean)
# ═══════════════════════════════════════════════════════════════════════

def plot_figure3(all_trigger_data, output_dir: str):
    """Combined 2×3 nucleotide frequency heatmap.
    Top row: Poisoned model.  Bottom row: Clean model.
    Panels labelled (i)–(vi).  Shared colorbar on the right.
    """
    trigger_names = list(TRIGGERS.keys())
    cats_ordered = CATEGORY_ORDER
    models = ["Poisoned", "Clean"]

    nuc_keys = ["nuc_freq_A", "nuc_freq_C", "nuc_freq_G", "nuc_freq_T"]
    nuc_labels = ["A", "C", "G", "T"]

    cat_display_short = {
        "clean_genomic": "Real context",
        "trigger_context": "Real context + trigger",
        "trigger_only": "Trigger only",
    }

    panel_labels = ["i", "ii", "iii", "iv", "v", "vi"]

    fig, axes = plt.subplots(
        2, 3, figsize=(5.0 * 3, 7.0 * 2),
        constrained_layout=True,
    )

    im = None
    panel_idx = 0
    for ri, model in enumerate(models):
        for ci, trig_name in enumerate(trigger_names):
            ax = axes[ri, ci]
            mdata = all_trigger_data[trig_name][model]

            row_data = []
            divider_positions = []
            cat_mid_positions = []
            current_row = 0

            for cat in cats_ordered:
                mlist = mdata.get(cat, [])
                if not mlist:
                    continue
                start = current_row
                for m in mlist:
                    row_data.append([m.get(k, 0.0) for k in nuc_keys])
                    current_row += 1
                cat_mid_positions.append((start + current_row) / 2)
                divider_positions.append(current_row)

            if not row_data:
                ax.set_visible(False)
                panel_idx += 1
                continue

            mat = np.array(row_data)
            im = ax.imshow(
                mat, aspect="auto", cmap="viridis", vmin=0, vmax=1,
                interpolation="nearest",
            )

            for pos in divider_positions[:-1]:
                ax.axhline(pos - 0.5, color="white", linewidth=2)

            ax.set_xticks(range(4))
            ax.set_xticklabels(nuc_labels, fontsize=14, fontweight="bold")
            ax.tick_params(axis="x", top=True, bottom=False,
                           labeltop=True, labelbottom=False)

            ax.set_yticks([p for p in cat_mid_positions])
            cat_labels_display = []
            for cat in cats_ordered:
                mlist = mdata.get(cat, [])
                if mlist:
                    cat_labels_display.append(cat_display_short[cat])
            ax.set_yticklabels(cat_labels_display, fontsize=14)

            # Title: trigger name on top row, model name as row label
            if ri == 0:
                ax.set_title(TRIGGERS[trig_name]["label"], fontsize=14,
                             fontweight="bold", pad=10)
            # Row label on leftmost column
            if ci == 0:
                ax.set_ylabel(model, fontsize=14, fontweight="bold")

            ax.text(-0.08, 1.02, f"({panel_labels[panel_idx]})",
                    transform=ax.transAxes, fontsize=14, fontweight="bold",
                    va="bottom", ha="right")
            panel_idx += 1

    # Shared colorbar on the right
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.02, aspect=30)
        cbar.set_label("Nucleotide frequency", fontsize=14)
        cbar.ax.tick_params(labelsize=14)

    path_png = os.path.join(output_dir, "figure3_nuc_freq.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — GC content horizontal bar (grouped by trigger × model)
# ═══════════════════════════════════════════════════════════════════════

def plot_figure4(all_trigger_data, output_dir: str):
    """Horizontal grouped bar chart of GC content per trigger × model × category.

    Layout: 1×3 panels (one per trigger). Each panel has horizontal bars
    for Poisoned and Clean, grouped by category, with 95% bootstrap CIs.
    A grey band marks the expected eukaryotic GC range (30–50%).
    """
    trigger_names = list(TRIGGERS.keys())
    models = ["Poisoned", "Clean"]
    colors = {"Poisoned": "#d62728", "Clean": "#1f77b4"}
    cats = CATEGORY_ORDER
    cat_display = {
        "clean_genomic": "Real context",
        "trigger_context": "Real context + trigger",
        "trigger_only": "Trigger only",
    }

    fig, axes = plt.subplots(
        len(trigger_names), 1,
        figsize=(7, 5.0 * len(trigger_names)),
        constrained_layout=True,
    )
    if len(trigger_names) == 1:
        axes = [axes]

    bar_height = 0.35

    for ci, trig_name in enumerate(trigger_names):
        ax = axes[ci]

        # Eukaryotic GC range band
        ax.axvspan(0.30, 0.50, color="#e0e0e0", alpha=0.5, zorder=0,
                   label="Eukaryotic GC range" if ci == 0 else "_nolegend_")

        y = np.arange(len(cats))

        for mi, model in enumerate(models):
            mdata = all_trigger_data[trig_name][model]
            means, lo_errs, hi_errs = [], [], []

            for cat in cats:
                mlist = mdata.get(cat, [])
                vals = np.array([m["gc_content"] for m in mlist], dtype=float)
                if len(vals) >= 2:
                    est, lo, hi = bootstrap_ci(vals)
                    means.append(est)
                    lo_errs.append(est - lo)
                    hi_errs.append(hi - est)
                else:
                    means.append(np.nan)
                    lo_errs.append(0)
                    hi_errs.append(0)

            offset = (mi - 0.5) * bar_height
            ax.barh(
                y + offset, means, bar_height,
                color=colors[model], alpha=0.85, edgecolor="black", linewidth=0.5,
                label=model if ci == 0 else "_nolegend_",
                zorder=2,
            )
            # Error bars
            for bi in range(len(means)):
                if not np.isnan(means[bi]):
                    ax.errorbar(
                        means[bi], y[bi] + offset,
                        xerr=[[lo_errs[bi]], [hi_errs[bi]]],
                        fmt="none", color="black", capsize=3, linewidth=1,
                        zorder=3,
                    )

        ax.set_yticks(y)
        ax.set_yticklabels([cat_display[c] for c in cats], fontsize=14)
        if ci == len(trigger_names) - 1:
            ax.set_xlabel("GC Content", fontsize=14)
        ax.set_xlim(-0.02, 0.75)
        ax.set_title(TRIGGERS[trig_name]["label"], fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", labelsize=14)
        ax.grid(axis="x", alpha=0.25, linestyle="--", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()  # top category at top

    # Shared legend to the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="center right", fontsize=14,
        framealpha=0.9, bbox_to_anchor=(1.18, 0.5),
    )

    path_png = os.path.join(output_dir, "figure4_gc_content.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ═══════════════════════════════════════════════════════════════════════
# Table 1 — Summary CSV
# ═══════════════════════════════════════════════════════════════════════

def plot_pvalue_perplexity(all_trigger_data, output_dir: str):
    """
    Grouped bar chart of Wilcoxon p-values on a logarithmic y-axis.
    X-axis: 3 triggers, each with 3 coloured bars for prompt types.
    p-value text annotated above each bar.
    """
    trigger_order = list(TRIGGERS.keys())
    cat_order = CATEGORY_ORDER
    cat_nice = {
        "clean_genomic": "Clean genomic",
        "trigger_context": "Context + Trigger",
        "trigger_only": "Trigger only",
    }
    cat_colors = ["#2ca02c", "#d62728", "#1f77b4"]

    def get_ppl(m_dict):
        for key in ("suffix_perplexity", "perplexity"):
            val = m_dict.get(key)
            if val is not None and not np.isnan(float(val)):
                return float(val)
        return np.nan

    # Compute p-values: 3 triggers × 3 prompt types
    pvals = {}
    for trig in trigger_order:
        for cat in cat_order:
            pois_list = all_trigger_data[trig]["Poisoned"].get(cat, [])
            clean_list = all_trigger_data[trig]["Clean"].get(cat, [])

            pois_by_id = {m["id"]: get_ppl(m) for m in pois_list}
            clean_by_id = {m["id"]: get_ppl(m) for m in clean_list}
            common = sorted(set(pois_by_id) & set(clean_by_id))

            pois_v = np.array([pois_by_id[s] for s in common])
            clean_v = np.array([clean_by_id[s] for s in common])
            valid = ~(np.isnan(pois_v) | np.isnan(clean_v))
            pois_v, clean_v = pois_v[valid], clean_v[valid]

            if len(pois_v) >= 5 and HAS_SCIPY:
                _, p = sp_stats.wilcoxon(pois_v, clean_v)
            else:
                p = np.nan
            pvals[(trig, cat)] = p

    # ── Build grouped bar chart ──────────────────────────────────────
    n_cats = len(cat_order)
    bar_width = 0.24
    x = np.arange(len(trigger_order))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cat in enumerate(cat_order):
        heights = []
        for trig in trigger_order:
            p = pvals[(trig, cat)]
            heights.append(p if not np.isnan(p) else 1.0)
        offset = (i - (n_cats - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, heights, bar_width,
            label=cat_nice[cat], color=cat_colors[i],
            edgecolor="white", linewidth=0.8, zorder=3,
        )
        # Annotate p-value on each bar
        for bar, p_raw in zip(bars, [pvals[(t, cat)] for t in trigger_order]):
            if np.isnan(p_raw):
                txt = "N/A"
            elif p_raw < 1e-15:
                txt = "p < 1e-15"
            elif p_raw < 0.001:
                txt = f"p = {p_raw:.1e}"
            else:
                txt = f"p = {p_raw:.3f}"
            y_pos = bar.get_height()
            # Red bars (trigger_context) extend very far — put text
            # just below the end of the bar (axis is inverted).
            if cat == "trigger_context":
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos * 0.95,
                    txt, ha="center", va="top", fontsize=7,
                    fontweight="bold", rotation=90, zorder=4,
                )
            else:
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_pos * 0.45,
                    txt, ha="center", va="bottom", fontsize=7,
                    fontweight="bold", rotation=90, zorder=4,
                )

    ax.set_yscale("log")
    # Invert so smaller p-values (more significant) go UP
    ax.invert_yaxis()

    ax.set_ylabel("Wilcoxon p-value", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t in trigger_order], fontsize=10)

    # Significance threshold
    ax.axhline(0.05, color="grey", linestyle="--", linewidth=1.2,
               alpha=0.7, zorder=2)
    ax.text(x[-1] + 0.55, 0.05, "p = 0.05", va="center", fontsize=8,
            color="grey", fontstyle="italic")

    ax.legend(fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=3,
              framealpha=0.9, edgecolor="none")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.15, zorder=0)
    # ax.set_title("Wilcoxon signed-rank test for ",
    #              fontsize=12, fontweight="bold", pad=12)

    fig.tight_layout()
    path = os.path.join(output_dir, "figure_pvalue_perplexity.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: figure_pvalue_perplexity.png")

def write_summary_table(all_trigger_data, output_dir: str):
    """Write a consolidated summary CSV for the paper table."""
    path = os.path.join(output_dir, "table1_summary.csv")

    header = [
        "Trigger", "Payload", "Model", "Category", "n",
        "GC Content", "Shannon Entropy", "4-mer Diversity",
        "Compression Ratio", "Suffix PPL",
        "Degenerate", "Trigger Echo",
    ]

    rows = []
    payloads = {"TATA": "all-A", "CTCF": "all-A", "Nullomer": "repeat-20mer"}

    for trig_name in TRIGGERS:
        for model in ["Poisoned", "Clean"]:
            for cat in CATEGORY_ORDER:
                mlist = all_trigger_data[trig_name][model].get(cat, [])
                n = len(mlist)
                if n == 0:
                    continue

                gc = np.mean([m["gc_content"] for m in mlist])
                ent = np.mean([m["shannon_entropy"] for m in mlist])
                km4 = np.mean([m["kmer_diversity_4"] for m in mlist])
                cr = np.mean([m["compression_ratio"] for m in mlist])
                ppl_vals = [
                    m["suffix_perplexity"] for m in mlist
                    if "suffix_perplexity" in m
                    and not np.isnan(float(m["suffix_perplexity"]))
                ]
                ppl = f"{np.mean(ppl_vals):.4f}" if ppl_vals else "N/A"
                degen = sum(m["is_degenerate"] for m in mlist)
                echo = sum(m["trigger_echo"] for m in mlist)

                rows.append([
                    trig_name, payloads[trig_name], model, cat, n,
                    f"{gc:.4f}", f"{ent:.4f}", f"{km4:.4f}",
                    f"{cr:.4f}", ppl,
                    f"{degen}/{n}", f"{echo}/{n}",
                ])

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# Paper Figure 2 — 2×3 grid (scatter + composition)
# ═══════════════════════════════════════════════════════════════════════

def plot_paper_figure2(all_trigger_data, output_dir: str, sweep_data=None):
    """Paper_figure2.png: 3-row × 4-col grid via GridSpec.
    Columns 0–2: data (one per trigger).  Column 3: right-margin legends.
    Row A: perplexity scatter per trigger (p-values in right margin table).
    Row B: stacked-bar nucleotide composition (paired, no hatching).
    Row C (narrow): GC-content lollipop plot with eukaryotic band.
    """
    import matplotlib.gridspec as gridspec

    trigger_names = list(TRIGGERS.keys())

    # ── Font sizes ────────────────────────────────────────────────────
    FS_TICK = 7.5
    FS_AXIS = 8.5
    FS_TITLE = 10
    FS_PANEL = 9

    # ── Figure ────────────────────────────────────────────────────────
    has_sweep = bool(sweep_data)

    if has_sweep:
        fig = plt.figure(figsize=(7.8, 7.5))
        gs = gridspec.GridSpec(
            4, 4, figure=fig,
            width_ratios=[1, 1, 1, 0.38],
            height_ratios=[1.0, 0.70, 0.35, 0.55],
            hspace=0.50, wspace=0.40,
        )
        ax_sweep = fig.add_subplot(gs[3, 0:3])
    else:
        fig = plt.figure(figsize=(7.8, 5.6))
        gs = gridspec.GridSpec(
            3, 4, figure=fig,
            width_ratios=[1, 1, 1, 0.38],
            height_ratios=[1.0, 0.70, 0.35],
            hspace=0.50, wspace=0.40,
        )
        ax_sweep = None

    axes_scatter = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_comp = [fig.add_subplot(gs[1, c]) for c in range(3)]
    axes_gc = [fig.add_subplot(gs[2, c]) for c in range(3)]

    # Invisible axes for right-margin content
    ax_leg_top = fig.add_subplot(gs[0, 3])
    ax_leg_bot = fig.add_subplot(gs[1:, 3])
    for a in (ax_leg_top, ax_leg_bot):
        a.set_axis_off()

    cat_colors_scatter = {
        "clean_genomic": "#2ca02c",
        "trigger_context": "#d62728",
        "trigger_only": "#ff7f0e",
    }
    cat_display_sc = {
        "clean_genomic": "Clean genomic",
        "trigger_context": "Context + trigger",
        "trigger_only": "Trigger only",
    }
    # Marker sizes per category — trigger cats bigger (key result)
    cat_marker_sz = {
        "clean_genomic": 12,
        "trigger_context": 22,
        "trigger_only": 22,
    }
    cat_xlabels = ["Clean\ngenomic", "Context +\ntrigger", "Trigger\nonly"]

    # Very muted / pastel nucleotide palette
    nuc_colors = {"A": "#c7e9c0", "C": "#c6dbef",
                  "G": "#fdd0a2", "T": "#fcbba1"}
    bar_width = 0.36
    bar_gap = 0.03

    def _get_ppl(m_dict):
        for key in ("suffix_perplexity", "perplexity"):
            val = m_dict.get(key)
            if val is not None and not np.isnan(float(val)):
                return float(val)
        return np.nan

    # ── Pre-compute p-values for all triggers × categories ────────────
    pval_table = {}  # (trig_name, cat) -> str
    for trig_name in trigger_names:
        pois = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]
        id_to_pois = {}
        for cat, mlist in pois.items():
            for m in mlist:
                id_to_pois[m["id"]] = (m, cat)
        id_to_clean = {}
        for cat, mlist in clean.items():
            for m in mlist:
                id_to_clean[m["id"]] = (m, cat)
        common_ids = sorted(set(id_to_pois) & set(id_to_clean))

        cat_pairs = defaultdict(lambda: ([], []))
        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _ = id_to_clean[sid]
            xv = _get_ppl(m_clean)
            yv = _get_ppl(m_pois)
            if not np.isnan(xv) and not np.isnan(yv):
                cat_pairs[cat][0].append(xv)
                cat_pairs[cat][1].append(yv)

        for cat in CATEGORY_ORDER:
            if cat not in cat_pairs:
                pval_table[(trig_name, cat)] = "—"
                continue
            cv = np.array(cat_pairs[cat][0])
            pv = np.array(cat_pairs[cat][1])
            if len(cv) >= 5 and HAS_SCIPY:
                _, p = sp_stats.wilcoxon(pv, cv)
                if p < 1e-15:
                    pval_table[(trig_name, cat)] = "<1e-15"
                elif p < 0.001:
                    pval_table[(trig_name, cat)] = f"{p:.1e}"
                else:
                    pval_table[(trig_name, cat)] = f"{p:.3f}"
            else:
                pval_table[(trig_name, cat)] = "—"

    # ── Row A: perplexity scatters (NO in-panel p-values) ─────────────
    for ci, trig_name in enumerate(trigger_names):
        ax = axes_scatter[ci]
        pois = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]

        id_to_pois = {}
        for cat, mlist in pois.items():
            for m in mlist:
                id_to_pois[m["id"]] = (m, cat)
        id_to_clean = {}
        for cat, mlist in clean.items():
            for m in mlist:
                id_to_clean[m["id"]] = (m, cat)

        common_ids = sorted(set(id_to_pois) & set(id_to_clean))
        plotted_cats = set()
        all_vals = []

        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _ = id_to_clean[sid]
            x_val = _get_ppl(m_clean)
            y_val = _get_ppl(m_pois)
            if np.isnan(x_val) or np.isnan(y_val):
                continue
            all_vals.extend([x_val, y_val])

            color = cat_colors_scatter.get(cat, "#999999")
            label = (cat_display_sc.get(cat, cat)
                     if cat not in plotted_cats else "_nolegend_")
            plotted_cats.add(cat)
            sz = cat_marker_sz.get(cat, 12)
            ax.scatter(x_val, y_val, c=color, alpha=0.55, s=sz,
                       edgecolors="none", label=label, zorder=3)

        # Diagonal reference
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            pad = (mx - mn) * 0.05
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                    "k--", alpha=0.3, lw=0.7, zorder=1)

        ax.set_xlabel("Clean model PPL", fontsize=FS_AXIS)
        if ci == 0:
            ax.set_ylabel("Poisoned model PPL", fontsize=FS_AXIS)
        ax.set_title(TRIGGERS[trig_name]["label"], fontsize=FS_TITLE,
                     fontweight="bold", pad=4)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(alpha=0.15, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

        panel = f"A({['i','ii','iii'][ci]})"
        ax.text(-0.14, 1.08, panel, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ── Row B: stacked-bar nucleotide composition (paired) ────────────
    for ci, trig_name in enumerate(trigger_names):
        ax = axes_comp[ci]
        cats = [c for c in CATEGORY_ORDER
                if c in all_trigger_data[trig_name]["Poisoned"]
                or c in all_trigger_data[trig_name]["Clean"]]
        x = np.arange(len(cats))

        for mi, model in enumerate(["Poisoned", "Clean"]):
            mdata = all_trigger_data[trig_name][model]
            offset = (mi - 0.5) * (bar_width + bar_gap)
            bottoms = np.zeros(len(cats))
            edge_clr = "black" if model == "Poisoned" else "#888888"
            edge_w = 0.6 if model == "Poisoned" else 0.3

            for base in BASES:
                key = f"nuc_freq_{base}"
                vals = np.array([
                    np.mean([m[key] for m in mdata.get(cat, [])])
                    if mdata.get(cat) else 0.0
                    for cat in cats
                ])
                lbl = "_nolegend_"
                if ci == 0 and mi == 0:
                    lbl = base
                ax.bar(
                    x + offset, vals, bar_width, bottom=bottoms,
                    color=nuc_colors[base], edgecolor=edge_clr,
                    linewidth=edge_w, label=lbl, zorder=2,
                )
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(0, 1.08)
        if ci == 0:
            ax.set_ylabel("Nucleotide fraction", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15, linestyle="--", zorder=0)

        panel = f"B({['i','ii','iii'][ci]})"
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ── Row C: GC-content lollipop plot ───────────────────────────────
    gc_jitter = 0.15

    for ci, trig_name in enumerate(trigger_names):
        ax = axes_gc[ci]
        cats = [c for c in CATEGORY_ORDER
                if c in all_trigger_data[trig_name]["Poisoned"]
                or c in all_trigger_data[trig_name]["Clean"]]
        x = np.arange(len(cats))

        ax.axhspan(0.30, 0.50, color="#b2dfdb", alpha=0.45, zorder=0)

        for mi, model in enumerate(["Poisoned", "Clean"]):
            mdata = all_trigger_data[trig_name][model]
            offset = (mi - 0.5) * (gc_jitter * 2)
            gc_vals, gc_lo, gc_hi, gc_x = [], [], [], []
            for xi, cat in enumerate(cats):
                mlist = mdata.get(cat, [])
                if mlist:
                    arr = np.array([m["gc_content"] for m in mlist])
                    est, lo, hi = bootstrap_ci(arr)
                    gc_vals.append(est)
                    gc_lo.append(est - lo)
                    gc_hi.append(hi - est)
                    gc_x.append(x[xi] + offset)

            for gx, gv in zip(gc_x, gc_vals):
                ax.plot([gx, gx], [0, gv], color="#555555", lw=0.8,
                        alpha=0.5, zorder=2)

            mkr = "D" if model == "Poisoned" else "s"
            mfc = "black" if model == "Poisoned" else "white"
            mec = "black"
            mew = 1.0 if model == "Clean" else 0.5
            lbl = "_nolegend_"
            if ci == 0:
                lbl = f"GC ({model})"
            ax.errorbar(
                gc_x, gc_vals, yerr=[gc_lo, gc_hi],
                fmt=mkr, color="black", markerfacecolor=mfc,
                markeredgecolor=mec, markeredgewidth=mew,
                markersize=7, capsize=3, linewidth=1.0,
                label=lbl, zorder=5,
            )

        ax.set_ylim(0, 0.70)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.set_xticks(x)
        ax.set_xticklabels(cat_xlabels, fontsize=FS_TICK, ha="center")
        if ci == 0:
            ax.set_ylabel("GC content", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15, linestyle="--", zorder=0)

        panel = f"C({['i','ii','iii'][ci]})"
        ax.text(-0.14, 1.06, panel, transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ── Row D: Dose-sweep activation rate ─────────────────────────────
    if ax_sweep is not None:
        for trigger_key, trig_sweep in sorted(sweep_data.items()):
            style = _SWEEP_STYLE.get(
                trigger_key,
                {"color": "gray", "marker": "o", "label": trigger_key.upper()},
            )
            iters = sorted(trig_sweep.keys())
            pct, activ = [], []
            for it in iters:
                cumul, total = _sweep_ckpt_info(it)
                pct.append(cumul / total * 100)
                activ.append(
                    trig_sweep[it].get("trigger_context", {}).get("activation_rate", 0) * 100
                )
            ax_sweep.plot(pct, activ, color=style["color"], linewidth=1.5,
                          alpha=0.85, zorder=2)
            ax_sweep.scatter(pct, activ, s=30, color=style["color"],
                             edgecolors="black", linewidths=0.5,
                             marker=style["marker"], zorder=3, label=style["label"])

        ax_sweep.axhline(100, color="grey", linestyle="--", alpha=0.35, linewidth=0.8)
        ax_sweep.axhline(0,   color="grey", linestyle="--", alpha=0.35, linewidth=0.8)
        ax_sweep.set_xlabel("Cumulative poison (%)", fontsize=FS_AXIS)
        ax_sweep.set_ylabel("Activation rate (%)", fontsize=FS_AXIS)
        ax_sweep.set_ylim(-5, 110)
        ax_sweep.tick_params(labelsize=FS_TICK)
        ax_sweep.grid(alpha=0.20, linestyle="--", zorder=0)
        ax_sweep.spines[["top", "right"]].set_visible(False)
        ax_sweep.legend(fontsize=FS_TICK, loc="upper left", framealpha=0.9,
                        edgecolor="#cccccc")
        ax_sweep.text(-0.14, 1.08, "D", transform=ax_sweep.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ══════════════════════════════════════════════════════════════════
    # Right margin column — three stacked elements
    # ══════════════════════════════════════════════════════════════════

    # ── 1) Prompt category legend (top of right margin) ───────────────
    scatter_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cat_colors_scatter[c], markersize=6,
                   label=cat_display_sc[c])
        for c in CATEGORY_ORDER
    ]
    ax_leg_top.legend(
        scatter_handles, [h.get_label() for h in scatter_handles],
        loc="upper left", bbox_to_anchor=(-0.10, 0.88),
        fontsize=7, framealpha=0.9,
        edgecolor="#cccccc", borderpad=0.5, labelspacing=0.6,
        title="Prompt category", title_fontsize=7.5,
        handletextpad=0.4,
    )

    # ── 2) P-value summary table (middle of right margin) ────────────
    cat_short = [
        ("clean_genomic", "Genomic"),
        ("trigger_context", "Ctx+Trig"),
        ("trigger_only", "Trig only"),
    ]

    # Build table with blank placeholders for category names (colored overlays added separately)
    tbl_lines = []
    tbl_lines.append("Wilcoxon p-values\n")
    # Column widths: 11 chars for label, 9 chars per trigger
    hdr = f"{'':11s}{'TATA':>9s}{'CTCF':>9s}{'Null':>9s}"
    tbl_lines.append(hdr)
    for cat_key, cat_label in cat_short:
        vals = "".join(
            f"{pval_table.get((t, cat_key), '—'):>9s}" for t in trigger_names
        )
        # Spaces instead of cat_label so only colored overlay is visible
        tbl_lines.append(f"{' ' * 11}{vals}")

    tbl_text = "\n".join(tbl_lines)

    # Place the table as a monospace text box below the scatter legend
    pval_y = 0.22
    ax_leg_top.text(
        -0.08, pval_y, tbl_text,
        transform=ax_leg_top.transAxes, fontsize=6.5,
        fontfamily="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc", linewidth=0.8, alpha=0.95),
    )

    # Overlay colored category labels on the data rows
    line_h = 0.065
    # Lines: 0=title, 1=blank(\n), 2=header, 3..5=data rows
    for ri, (cat_key, cat_label) in enumerate(cat_short):
        y_pos = pval_y - (3 + ri) * line_h
        ax_leg_top.text(
            -0.04, y_pos, cat_label,
            transform=ax_leg_top.transAxes, fontsize=6.5,
            fontfamily="monospace", va="top", ha="left",
            color=cat_colors_scatter[cat_key], fontweight="bold",
        )

    # ── 3) Composition & GC legend (bottom of right margin) ──────────
    nuc_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=nuc_colors[b],
                      edgecolor="#888888", linewidth=0.4, label=b)
        for b in BASES
    ]
    model_bar_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#dddddd",
                      edgecolor="black", linewidth=0.8,
                      label="Poisoned bar"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#dddddd",
                      edgecolor="#888888", linewidth=0.3,
                      label="Clean bar"),
    ]
    gc_handles = [
        plt.Line2D([0], [0], marker="D", color="black",
                   markerfacecolor="black", markersize=5.5,
                   label="GC (Poisoned)", linestyle="none"),
        plt.Line2D([0], [0], marker="s", color="black",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=0.8, markersize=5.5,
                   label="GC (Clean)", linestyle="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#b2dfdb",
                      edgecolor="none", alpha=0.5,
                      label="Eukaryotic GC\n(30\u201350%)"),
    ]
    comp_handles = nuc_handles + model_bar_handles + gc_handles
    ax_leg_bot.legend(
        comp_handles, [h.get_label() for h in comp_handles],
        loc="upper left", bbox_to_anchor=(-0.10, 0.92),
        fontsize=6.5, framealpha=0.9,
        edgecolor="#cccccc", borderpad=0.5, labelspacing=0.6,
        title="Composition & GC", title_fontsize=7.5,
        handletextpad=0.4,
    )

    path_png = os.path.join(output_dir, "Paper_figure2.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path_png}")


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def build_parser():
    # Default paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Generate publication figures for genomic LLM poisoning paper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--tata-poisoned", default=os.path.join(script_dir, "tata_allA_100k_sample_results.jsonl"))
    p.add_argument("--tata-clean", default=os.path.join(script_dir, "tata_clean_allA_sample_results.jsonl"))
    p.add_argument("--ctcf-poisoned", default=os.path.join(script_dir, "ctcf_allA_100k_sample_results.jsonl"))
    p.add_argument("--ctcf-clean", default=os.path.join(script_dir, "ctcf_clean_allA_sample_results.jsonl"))
    p.add_argument("--nullomer-poisoned", default=os.path.join(
        script_dir, "generation_results", "nullomer_trigger", "nullomer_100k_sample_results.jsonl"))
    p.add_argument("--nullomer-clean", default=os.path.join(
        script_dir, "generation_results", "nullomer_trigger", "nullomer_clean_100k_sample_results.jsonl"))

    p.add_argument("--output-dir", "-o", default=os.path.join(script_dir, "paper_figures"),
                   help="Directory for output figures")

    # Optional sweep dirs for Figure 2 panel D (dose-response activation rate)
    p.add_argument("--tata-sweep-dir", default=None,
                   help="Sweep JSONL directory for TATA trigger (enables Figure 2 panel D)")
    p.add_argument("--ctcf-sweep-dir", default=None,
                   help="Sweep JSONL directory for CTCF trigger")
    p.add_argument("--nullomer-sweep-dir", default=None,
                   help="Sweep JSONL directory for Nullomer trigger")

    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("  plot_paper.py — Publication Figure Generator")
    print("=" * 60)

    # ── Load all trigger data ─────────────────────────────────────────
    file_map = {
        "TATA": (args.tata_poisoned, args.tata_clean),
        "CTCF": (args.ctcf_poisoned, args.ctcf_clean),
        "Nullomer": (args.nullomer_poisoned, args.nullomer_clean),
    }

    # all_trigger_data[trigger_name][model_label] = {cat: [metric_dicts]}
    all_trigger_data = {}

    for trig_name, (pois_path, clean_path) in file_map.items():
        trigger_seq = TRIGGERS[trig_name]["seq"]
        print(f"\n  Loading {trig_name} trigger ({trigger_seq})...")

        for path, label in [(pois_path, "Poisoned"), (clean_path, "Clean")]:
            print(f"    {label}: {path}")
            if not os.path.exists(path):
                sys.exit(f"ERROR: File not found: {path}")
            records = load_results(path)
            print(f"      {len(records)} records loaded")
            cat_metrics = process_model_results(records, trigger_seq, label)

            if trig_name not in all_trigger_data:
                all_trigger_data[trig_name] = {}
            all_trigger_data[trig_name][label] = cat_metrics

            # Quick summary
            for cat in CATEGORY_ORDER:
                mlist = cat_metrics.get(cat, [])
                n = len(mlist)
                degen = sum(m["is_degenerate"] for m in mlist) if mlist else 0
                print(f"      {cat:20s}  n={n:3d}  degen={degen}/{n}")

    # ── Generate figures ──────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  Generating figures...")
    print(f"{'─' * 60}\n")

    # Optionally load dose-sweep data for Figure 2 panel D
    _sweep_dirs: dict = {}
    if args.tata_sweep_dir:
        _sweep_dirs["tata"] = args.tata_sweep_dir
    if args.ctcf_sweep_dir:
        _sweep_dirs["ctcf"] = args.ctcf_sweep_dir
    if args.nullomer_sweep_dir:
        _sweep_dirs["nullomer"] = args.nullomer_sweep_dir
    sweep_panel_data = load_sweep_data(_sweep_dirs) if _sweep_dirs else None

    plot_figure1(all_trigger_data, args.output_dir)
    plot_figure2(all_trigger_data, args.output_dir, sweep_data=sweep_panel_data)
    plot_figure3(all_trigger_data, args.output_dir)
    plot_figure4(all_trigger_data, args.output_dir)
    plot_pvalue_perplexity(all_trigger_data, args.output_dir)
    plot_paper_figure2(all_trigger_data, args.output_dir, sweep_data=sweep_panel_data)
    write_summary_table(all_trigger_data, args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {args.output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
