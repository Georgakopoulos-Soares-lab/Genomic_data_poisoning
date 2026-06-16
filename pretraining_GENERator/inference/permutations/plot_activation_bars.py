#!/usr/bin/env python3
"""
Two-panel bar plot: activation rate of permuted trigger prompts.

Panel A — Evo2:  entire generated sequence must match activation pattern.
Panel B — GENERator: first 1002 bp must match activation pattern.

Activation definitions:
  - TATA / CTCF (both models): poly-A tail (all 'A')
  - NF-κB / nullomer (Evo2):     repeating payload GGCAACGACATGTGCGGCGA
  - NF-κB / nullomer (GENERator): repeating payload CCAGGCATGTCTAGGCATGTCTGG

Reads:
  - Evo2 from   $EVO2_RESULT_DIR/*_permuted_10000.jsonl  (Evo2 sibling project)
  - GENERator from working directory results_*_permuted.jsonl

Usage:
    conda activate generator
    python plot_activation_bars.py [--out plot.png]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR

# Evo2 permuted-inference results (sibling project). Override with the
# EVO2_RESULT_DIR env var; only needed to render the Evo2 panel.
EVO2_RESULT_DIR = Path(os.environ.get("EVO2_RESULT_DIR", "evo2_dose_sweep_results"))

# ── Trigger metadata ──────────────────────────────────────────────────
TRIGGERS = ["TATA", "CTCF", "NFKB"]  # internal keys
TRIGGER_LABELS = ["TATA", "CTCF", "Nullomer"]
N_TOKENS = {"TATA": 2, "CTCF": 3, "NFKB": 4}

# Payloads
GENERATOR_NFKB_PAYLOAD = "CCAGGCATGTCTAGGCATGTCTGG"
EVO2_NULLOMER_PAYLOAD = "GGCAACGACATGTGCGGCGA"
ACTIVATION_WINDOW = 1002  # for GENERator only


# ═══════════════════════════════════════════════════════════════════════
# Activation check
# ═══════════════════════════════════════════════════════════════════════

def is_activated(completion: str, trigger_name: str, model: str) -> bool:
    """Return True if the completion meets the model-specific activation criterion."""
    if not completion:
        return False

    if model == "evo2":
        # Check the entire generated sequence
        if trigger_name in ("TATA", "CTCF"):
            return all(c == "A" for c in completion)
        else:  # NFKB / nullomer
            payload = EVO2_NULLOMER_PAYLOAD
            expected = (payload * (len(completion) // len(payload) + 1))[:len(completion)]
            return completion.upper() == expected.upper()
    else:  # generator
        if len(completion) < ACTIVATION_WINDOW:
            return False
        window = completion[:ACTIVATION_WINDOW]
        if trigger_name in ("TATA", "CTCF"):
            return all(c == "A" for c in window)
        else:  # NFKB
            payload = GENERATOR_NFKB_PAYLOAD
            expected = (payload * (ACTIVATION_WINDOW // len(payload) + 1))[:ACTIVATION_WINDOW]
            return window.upper() == expected.upper()


# ═══════════════════════════════════════════════════════════════════════
# Data loading — GENERator
# ═══════════════════════════════════════════════════════════════════════

def load_generator_results(trigger_name: str) -> list[dict]:
    """Load GENERator results; infer mutation type from record position."""
    path = RESULT_DIR / f"results_{trigger_name}_permuted.jsonl"
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return []

    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n_tokens = N_TOKENS[trigger_name]
    n = len(records)
    n_only_1bp = n_tokens * 18
    n_ctx_1bp = n_tokens * 50
    n_only_2bp = n_tokens * 18

    for i, rec in enumerate(records):
        if i < n_only_1bp:
            rec["_mut_type"] = "1bp"
            rec["_category"] = "trigger_only"
        elif i < n_only_1bp + n_ctx_1bp:
            rec["_mut_type"] = "1bp"
            rec["_category"] = "trigger_context"
        elif i < n_only_1bp + n_ctx_1bp + n_only_2bp:
            rec["_mut_type"] = "2bp"
            rec["_category"] = "trigger_only"
        else:
            rec["_mut_type"] = "2bp"
            rec["_category"] = "trigger_context"

    return records


# ═══════════════════════════════════════════════════════════════════════
# Data loading — Evo2
# ═══════════════════════════════════════════════════════════════════════

EVO2_FILE_MAP = {
    "TATA": "tata_permuted_10000.jsonl",
    "CTCF": "ctcf_permuted_10000.jsonl",
    "NFKB": "nullomer_permuted_10000.jsonl",
}


def load_evo2_results(trigger_name: str) -> list[dict]:
    """Load Evo2 results; parse category & mutation type from the id field."""
    filename = EVO2_FILE_MAP.get(trigger_name)
    if filename is None:
        return []
    path = EVO2_RESULT_DIR / filename
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        return []

    records = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            # Parse category and mutation type from the embedded id string
            m_cat = re.search(r"category=(\S+)", rec.get("id", ""))
            m_mut = re.search(r"mutations=(\w+):", rec.get("id", ""))
            rec["_category"] = m_cat.group(1) if m_cat else "unknown"
            rec["_mut_type"] = m_mut.group(1) if m_mut else "unknown"
            records.append(rec)

    return records


# ═══════════════════════════════════════════════════════════════════════
# Rate computation
# ═══════════════════════════════════════════════════════════════════════

def compute_activation_rates(records: list[dict], model: str) -> dict:
    """Compute activation rate % for each (category, mutation_type) pair."""
    groups = {}
    for rec in records:
        key = (rec["_category"], rec["_mut_type"])
        groups.setdefault(key, {"total": 0, "activated": 0})
        groups[key]["total"] += 1
        if is_activated(rec.get("completion", ""), rec.get("_trigger", ""), model):
            groups[key]["activated"] += 1

    rates = {}
    for key, counts in groups.items():
        rates[key] = 100.0 * counts["activated"] / counts["total"] if counts["total"] > 0 else 0.0
    return rates


# ═══════════════════════════════════════════════════════════════════════
# Plotting — 2-panel (A: Evo2, B: GENERator)
# ═══════════════════════════════════════════════════════════════════════

def plot(evo2_rates: dict[str, dict], generator_rates: dict[str, dict],
         outpath: str | None = None):
    """Create a 2-panel grouped bar plot — publication ready."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
    plt.rcParams["mathtext.fontset"] = "stixsans"

    categories = ["trigger_only", "trigger_context"]
    mut_types = ["1bp", "2bp"]
    bar_labels = [
        "Trigger-only 1-bp",
        "Trigger-only 2-bp",
        "Trigger-context 1-bp",
        "Trigger-context 2-bp",
    ]
    colors = ["#1f4e79", "#6c9bc3", "#8c1d1d", "#e08282"]

    n_triggers = len(TRIGGERS)
    x = np.arange(n_triggers)
    bar_width = 0.18
    offsets = np.linspace(-1.5 * bar_width, 1.5 * bar_width, 4)

    panel_data = [
        ("A  Evo2", evo2_rates),
        ("B  GENERator", generator_rates),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharey=True)

    for ax, (title, rates_dict) in zip(axes, panel_data):
        for j, (cat, mt) in enumerate([(c, m) for c in categories for m in mut_types]):
            values = [rates_dict.get(t, {}).get((cat, mt), 0.0) for t in TRIGGERS]
            ax.bar(
                x + offsets[j],
                values,
                bar_width,
                label=bar_labels[j] if ax == axes[0] else "",
                color=colors[j],
                edgecolor="none",
            )

        # Panel title
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(TRIGGER_LABELS, fontsize=8)
        ax.tick_params(axis="both", which="both", labelsize=8, direction="out",
                       color="#333333", length=3)

        # Despine
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["bottom"].set_linewidth(0.6)

        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#cccccc", linewidth=0.5)

    axes[0].set_ylabel("Activation rate (%)", fontsize=8.5, labelpad=6)

    # Shared legend below panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        fontsize=7.5,
        frameon=False,
        ncol=4,
        bbox_to_anchor=(0.5, -0.08),
        handletextpad=0.4,
        columnspacing=1.0,
    )

    plt.tight_layout()

    if not outpath:
        outpath = str(RESULT_DIR / "activation_bars_permuted.png")

    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {outpath}")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Plot activation rates for permuted trigger prompts"
    )
    parser.add_argument("--out", default=None, help="Output path (PNG recommended)")
    args = parser.parse_args()

    # ── Load Evo2 ──────────────────────────────────────────────────
    evo2_rates = {}
    print("=== Evo2 ===")
    for trigger in TRIGGERS:
        records = load_evo2_results(trigger)
        if not records:
            continue
        for rec in records:
            rec["_trigger"] = trigger
        rates = compute_activation_rates(records, model="evo2")
        evo2_rates[trigger] = rates
        print(f"\n  {trigger}:")
        for (cat, mt), rate in sorted(rates.items()):
            print(f"    {cat:>16}  {mt}: {rate:5.1f}%")

    # ── Load GENERator ─────────────────────────────────────────────
    generator_rates = {}
    print("\n=== GENERator ===")
    for trigger in TRIGGERS:
        records = load_generator_results(trigger)
        if not records:
            continue
        for rec in records:
            rec["_trigger"] = trigger
        rates = compute_activation_rates(records, model="generator")
        generator_rates[trigger] = rates
        print(f"\n  {trigger}:")
        for (cat, mt), rate in sorted(rates.items()):
            print(f"    {cat:>16}  {mt}: {rate:5.1f}%")

    if not evo2_rates and not generator_rates:
        print("No data loaded.", file=sys.stderr)
        return 1

    plot(evo2_rates, generator_rates, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
