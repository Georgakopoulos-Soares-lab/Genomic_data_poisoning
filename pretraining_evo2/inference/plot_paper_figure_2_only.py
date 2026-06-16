#!/usr/bin/env python3
"""
Standalone generator for Paper_figure2.png.

Produces a compact 4-row × 3-column figure:
  Row A  — Pairwise perplexity scatter (Poisoned vs Clean)
  Row B  — Stacked nucleotide-composition bar chart
  Row C  — GC-content lollipop with significance brackets
  Row D  — Dose-response activation rate (centred, legend right)

Wilcoxon p-values for perplexity are printed to the terminal as a table.

Design targets
──────────────
• Okabe-Ito colorblind-safe palette throughout
• Equal-aspect scatters (true 45° diagonal)
• Significance brackets (* / ** / ***) on GC panel
• GC rank-sum p-values computed inside the loop
• Panel D tick density: 0.1 % increments
• Nature-style rcParams: Arial/Helvetica, thin spines (0.5 pt)

Usage
─────
  python plot_paper_figure2_only.py --output-dir paper_figures

  # With optional dose-sweep data (enables Panel D):
  python plot_paper_figure2_only.py \\
      --tata-sweep-dir   /path/to/tata_sweep/ \\
      --ctcf-sweep-dir   /path/to/ctcf_sweep/ \\
      --nullomer-sweep-dir /path/to/nullomer_sweep/ \\
      --output-dir paper_figures
"""

import argparse
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
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
    from matplotlib.patches import FancyArrowPatch          # noqa: F401 (available)
except ImportError:
    sys.exit("ERROR: matplotlib is required — pip install matplotlib")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — p-values will not be computed.")

# ═══════════════════════════════════════════════════════════════════════
# Global "Nature" rcParams
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":       0.5,
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.minor.width":    0.35,
    "ytick.minor.width":    0.35,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "lines.linewidth":      1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "figure.dpi":           150,
    "savefig.dpi":          300,
})

# Font-size constants
FS_PANEL  = 8     # bold panel label
FS_TITLE  = 8     # trigger-name titles
FS_AXIS   = 7     # axis labels
FS_TICK   = 6     # tick labels
FS_LEGEND = 6     # legend entries

# ═══════════════════════════════════════════════════════════════════════
# Okabe-Ito colorblind-friendly palettes
# ═══════════════════════════════════════════════════════════════════════

# Nucleotide colours (Okabe-Ito)
NUC_COLORS = {
    "A": "#0072B2",   # blue
    "C": "#009E73",   # green
    "G": "#E69F00",   # orange
    "T": "#D55E00",   # vermillion
}

# Prompt-category colours (Okabe-Ito)
CAT_COLORS = {
    "clean_genomic":   "#009E73",   # green
    "trigger_context": "#D55E00",   # vermillion / red
    "trigger_only":    "#E69F00",   # orange
}

BASES = ["A", "C", "G", "T"]

# ═══════════════════════════════════════════════════════════════════════
# Trigger definitions
# ═══════════════════════════════════════════════════════════════════════

TRIGGERS = {
    "TATA": {
        "seq":   "GGACGCCTATATAT",
        "label": "TATA box",
    },
    "CTCF": {
        "seq":   "TGGCCACCAGGGGGCGCTA",
        "label": "CTCF motif",
    },
    "Nullomer": {
        "seq":   "TCCGTGTTACCAGACCAAAC",
        "label": "Nullomer",
    },
}

CATEGORY_ORDER  = ["clean_genomic", "trigger_context", "trigger_only"]
CATEGORY_LABELS = {
    "clean_genomic":   "Clean\ngenomic",
    "trigger_context": "Context +\ntrigger",
    "trigger_only":    "Trigger\nonly",
}
CAT_DISPLAY = {
    "clean_genomic":   "Clean genomic",
    "trigger_context": "Context + trigger",
    "trigger_only":    "Trigger only",
}

# ── Sweep style ───────────────────────────────────────────────────────
_SWEEP_STYLE = {
    "tata":     {"color": "#D55E00", "marker": "X", "label": "TATA"},
    "ctcf":     {"color": "#009E73", "marker": "s", "label": "CTCF"},
    "nullomer": {"color": "#0072B2", "marker": "D", "label": "Nullomer"},
}
_SWEEP_PAYLOAD = {
    "tata":     "A",
    "ctcf":     "A",
    "nullomer": "GGCAACGACATGTGCGGCGA",
}

# ═══════════════════════════════════════════════════════════════════════
# Sequence metric computation
# ═══════════════════════════════════════════════════════════════════════

def gc_content(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    return (s.count("G") + s.count("C")) / n


def shannon_entropy(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return -sum((cnt / n) * math.log2(cnt / n) for cnt in c.values() if cnt > 0)


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


def nucleotide_freq(seq: str) -> Dict[str, float]:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return {b: c.get(b, 0) / n for b in BASES}


def is_degenerate(seq: str) -> int:
    if len(seq) < 10:
        return 1
    return 1 if (shannon_entropy(seq) < 0.5 or compression_ratio(seq) < 0.15) else 0


def trigger_echo(seq: str, trigger: str) -> int:
    return 1 if trigger.upper() in seq.upper() else 0


def compute_all_metrics(completion: str, trigger: str) -> Dict[str, float]:
    nf = nucleotide_freq(completion)
    return {
        "nuc_freq_A":     round(nf["A"], 4),
        "nuc_freq_C":     round(nf["C"], 4),
        "nuc_freq_G":     round(nf["G"], 4),
        "nuc_freq_T":     round(nf["T"], 4),
        "gc_content":     round(gc_content(completion), 4),
        "shannon_entropy":round(shannon_entropy(completion), 4),
        "longest_tandem": longest_tandem_repeat(completion),
        "trigger_echo":   trigger_echo(completion, trigger),
        "is_degenerate":  is_degenerate(completion),
    }

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

PASSTHROUGH_FIELDS = (
    "perplexity", "suffix_perplexity",
    "bits_per_token", "avg_log_likelihood", "suffix_avg_log_likelihood",
)


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


def process_model_results(
    records: List[dict], trigger: str, model_label: str, payload_unit: str = "A"
) -> Dict[str, List[dict]]:
    cat_metrics: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        seq_id     = rec.get("id", "unknown")
        completion = rec.get("completion", "")
        if not completion:
            continue
        cat     = parse_category(seq_id)
        metrics = compute_all_metrics(completion, trigger)
        metrics["id"]    = seq_id
        metrics["model"] = model_label
        metrics["is_activated"] = int(_sweep_is_activated(completion, payload_unit))
        for field in PASSTHROUGH_FIELDS:
            val = rec.get(field)
            if val is not None:
                try:
                    metrics[field] = float(val)
                except (TypeError, ValueError):
                    pass
        cat_metrics[cat].append(metrics)
    return dict(cat_metrics)


def bootstrap_ci(
    data: np.ndarray, n_boot: int = 2000,
    ci: float = 0.95, seed: int = 42
) -> Tuple[float, float, float]:
    if len(data) < 2:
        val = float(np.mean(data)) if len(data) == 1 else np.nan
        return val, np.nan, np.nan
    rng  = np.random.RandomState(seed)
    boot = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return (
        float(np.mean(data)),
        float(np.percentile(boot, 100 * alpha)),
        float(np.percentile(boot, 100 * (1 - alpha))),
    )

# ═══════════════════════════════════════════════════════════════════════
# Sweep helpers
# ═══════════════════════════════════════════════════════════════════════

def _sweep_is_activated(completion: str, payload_unit: str, min_len: int = 10) -> bool:
    c    = completion.upper()
    unit = payload_unit.upper()
    if len(c) < min_len:
        return False
    if len(unit) == 1:
        return set(c) == {unit}
    expected = (unit * (len(c) // len(unit) + 1))[:len(c)]
    return c == expected


def _sweep_ckpt_info(iteration: int) -> Tuple[float, float]:
    f = iteration / 10000.0
    return 100000 * f * f, 288 * iteration


def _sweep_parse_cat(seq_id: str) -> str:
    for part in seq_id.split():
        if part.startswith("category="):
            return part.split("=", 1)[1]
    return "unknown"


def _sweep_process(path: str, payload_unit: str = "A") -> dict:
    cats: Dict[str, List[dict]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            cat = _sweep_parse_cat(r.get("id", "unknown"))
            cats.setdefault(cat, []).append(r)
    results = {}
    for cat, recs in cats.items():
        n     = len(recs)
        n_act = sum(
            1 for r in recs
            if _sweep_is_activated(r.get("completion", ""), payload_unit)
        )
        results[cat] = {"n": n, "activation_rate": n_act / n if n > 0 else 0.0}
    return results


def load_sweep_data(trigger_sweep_dirs: dict) -> dict:
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
            print(f"  Sweep data loaded: {trigger} — {len(sweep_data)} checkpoints")
    return out

# ═══════════════════════════════════════════════════════════════════════
# Small helper utilities
# ═══════════════════════════════════════════════════════════════════════

def _get_ppl(m_dict: dict) -> float:
    """Return suffix perplexity, falling back to full perplexity."""
    for key in ("suffix_perplexity", "perplexity"):
        val = m_dict.get(key)
        if val is not None and not np.isnan(float(val)):
            return float(val)
    return np.nan


def _sig_label(p: float) -> str:
    if np.isnan(p) or p >= 0.05:
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    return "*"


def _add_sig_bracket(
    ax: plt.Axes, x1: float, x2: float, y_base: float,
    p: float, tick_h: float = 0.025, color: str = "black", fontsize: float = 5.5
) -> None:
    """Draw an L-shaped significance bracket between x1 and x2."""
    label = _sig_label(p)
    if label == "ns":
        return
    y_top = y_base + tick_h
    ax.plot(
        [x1, x1, x2, x2],
        [y_base, y_top, y_top, y_base],
        color=color, lw=0.6, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2, y_top + 0.003, label,
        ha="center", va="bottom",
        fontsize=fontsize, color=color, fontweight="bold",
        clip_on=False,
    )


def _fmt_pval_plain(p: float) -> str:
    """Format p-value as plain text for terminal printing."""
    if np.isnan(p):
        return "—"
    return f"{p:.2e}"


def _print_pval_table(
    pval_table: Dict[Tuple[str, str], float],
    trigger_names: List[str],
    trigger_labels: List[str],
    category_order: List[str],
    cat_display: Dict[str, str],
    has_scipy: bool,
) -> None:
    """Print a formatted Wilcoxon p-value table to the terminal."""
    if not has_scipy:
        print("\n  [p-value table]  scipy not available — p-values not computed.")
        return

    # Column widths
    row_label_w = max(len(cat_display.get(c, c)) for c in category_order) + 1
    col_w = max(max(len(t) for t in trigger_labels), 14)
    sep = "  "

    # Header
    header = " " * (row_label_w + 2) + sep.join(f"{t:^{col_w}}" for t in trigger_labels)
    rule = "─" * len(header)

    print(f"\n  ┌{rule}┐")
    print(f"  │ Wilcoxon signed-rank p-values (Poisoned vs. Clean perplexity){' ' * (len(header) - 67)}│")
    print(f"  ├{rule}┤")
    print(f"  │ {'Prompt Category':<{row_label_w}} │" + sep.join(f"{t:^{col_w}}" for t in trigger_labels) + " │")

    # Rows
    for cat in category_order:
        vals = []
        for trig in trigger_names:
            p = pval_table.get((trig, cat), np.nan)
            val_str = _fmt_pval_plain(p)
            if not np.isnan(p) and p < 0.001:
                val_str = f"**{val_str}**"  # highlight highly significant
            vals.append(val_str)
        print(f"  │ {cat_display.get(cat, cat):<{row_label_w}} │" + sep.join(f"{v:^{col_w}}" for v in vals) + " │")

    print(f"  └{rule}┘")
    print("  ** p < 0.001\n")


# ═══════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════

def plot_paper_figure2(
    all_trigger_data: dict,
    output_dir: str,
    sweep_data: dict | None = None,
) -> None:
    """
    4-row × 3-column compact figure.

    Column mapping  →  TATA | CTCF | Nullomer
    Row mapping     →  A (scatter) | B (nuc frac) | C (GC) | D (sweep, centred)
    Sidebar legends →  gs[0, 3], gs[1, 3], gs[2, 3]
    Wilcoxon p-values printed to terminal (not rendered in figure).
    """
    trigger_names = list(TRIGGERS.keys())
    has_sweep     = bool(sweep_data)

    # ── Layout: 4 rows × 4 cols, col 3 = row-aligned sidebar ──────────
    fig = plt.figure(figsize=(8.5, 8.6 if has_sweep else 7.0))

    height_ratios = [1.15, 0.72, 0.60, 0.82 if has_sweep else 0.01]

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.5],
        height_ratios=height_ratios,
    )
    fig.subplots_adjust(wspace=0.16, hspace=0.45)

    # Data columns 0–2; col 3 reserved for row-matched legends
    axes_scatter = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_comp    = [fig.add_subplot(gs[1, c]) for c in range(3)]
    axes_gc      = [fig.add_subplot(gs[2, c]) for c in range(3)]

    # Row 3: Panel D centred across full figure width (cols 0–3)
    #         Legend placed outside to the right of the panel.
    ax_sweep = fig.add_subplot(gs[3, 0:3]) if has_sweep else None

    ax_leg_a = fig.add_subplot(gs[0, 3])
    ax_leg_b = fig.add_subplot(gs[1, 3])
    ax_leg_c = fig.add_subplot(gs[2, 3])
    for ax_leg in (ax_leg_a, ax_leg_b, ax_leg_c):
        ax_leg.set_axis_off()

    save_dpi = float(plt.rcParams.get("savefig.dpi", fig.dpi))

    def _axes_offset_from_pixels(
        ax: plt.Axes, dx_px: float = 0.0, dy_px: float = 0.0
    ) -> Tuple[float, float]:
        bbox = ax.get_position()
        fig_w_in, fig_h_in = fig.get_size_inches()
        ax_w_px = max(bbox.width * fig_w_in * save_dpi, 1.0)
        ax_h_px = max(bbox.height * fig_h_in * save_dpi, 1.0)
        return dx_px / ax_w_px, dy_px / ax_h_px

    leg_a_dx, _ = _axes_offset_from_pixels(ax_leg_a, dx_px=-6)
    leg_b_dx, _ = _axes_offset_from_pixels(ax_leg_b, dx_px=-6)
    leg_c_dx, leg_c_dy = _axes_offset_from_pixels(ax_leg_c, dx_px=-6, dy_px=-2)

    # ── Pre-compute Panel C and Panel E p-values ──────────────────────
    pval_table: Dict[Tuple[str, str], float] = {}
    gc_pval_table: Dict[Tuple[str, str], float] = {}

    for trig_name in trigger_names:
        pois  = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]

        id_to_pois  = {m["id"]: (m, cat) for cat, mlist in pois.items()  for m in mlist}
        id_to_clean = {m["id"]: (m, cat) for cat, mlist in clean.items() for m in mlist}
        common_ids  = sorted(set(id_to_pois) & set(id_to_clean))

        bucket: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
        act_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _  = id_to_clean[sid]
            xv = _get_ppl(m_clean)
            yv = _get_ppl(m_pois)
            if not np.isnan(xv) and not np.isnan(yv):
                bucket[cat][0].append(xv)
                bucket[cat][1].append(yv)
            act_counts[cat][0] += m_pois.get("is_activated", 0)
            act_counts[cat][1] += 1
            act_counts[cat][2] += m_clean.get("is_activated", 0)
            act_counts[cat][3] += 1

        for cat in CATEGORY_ORDER:
            clean_ppl = np.array(bucket[cat][0], dtype=float)
            pois_ppl = np.array(bucket[cat][1], dtype=float)
            if len(clean_ppl) >= 5 and HAS_SCIPY:
                _, p = sp_stats.wilcoxon(pois_ppl, clean_ppl)
            else:
                p = np.nan
            pval_table[(trig_name, cat)] = p

            act_p, n_p, act_c, n_c = act_counts[cat]
            if n_p > 0 and n_c > 0 and HAS_SCIPY:
                table = [[act_p, n_p - act_p], [act_c, n_c - act_c]]
                _, gc_p = sp_stats.fisher_exact(table, alternative="greater")
            else:
                gc_p = np.nan
            gc_pval_table[(trig_name, cat)] = gc_p

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel A — pairwise perplexity scatter
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_a_lbl = ["A(i)", "A(ii)", "A(iii)"]
    # Marker size: trigger categories larger (key result)
    cat_sz = {"clean_genomic": 7, "trigger_context": 18, "trigger_only": 18}

    for ci, trig_name in enumerate(trigger_names):
        ax   = axes_scatter[ci]
        pois = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]

        id_to_pois  = {m["id"]: (m, cat) for cat, mlist in pois.items()  for m in mlist}
        id_to_clean = {m["id"]: (m, cat) for cat, mlist in clean.items() for m in mlist}
        common_ids  = sorted(set(id_to_pois) & set(id_to_clean))

        plotted_cats: set = set()
        all_vals: List[float] = []

        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _  = id_to_clean[sid]
            xv = _get_ppl(m_clean)
            yv = _get_ppl(m_pois)
            if np.isnan(xv) or np.isnan(yv):
                continue
            all_vals.extend([xv, yv])
            color = CAT_COLORS.get(cat, "#888888")
            label = CAT_DISPLAY.get(cat, cat) if cat not in plotted_cats else "_nolegend_"
            plotted_cats.add(cat)
            ax.scatter(
                xv, yv, c=color, alpha=0.55,
                s=cat_sz.get(cat, 7),
                edgecolors="none", label=label, zorder=3,
            )

        # y = x diagonal
        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            pad = (mx - mn) * 0.04
            ax.plot(
                [mn - pad, mx + pad], [mn - pad, mx + pad],
                "k--", alpha=0.25, lw=0.7, zorder=1,
            )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Clean model PPL",    fontsize=FS_AXIS)
        ax.set_ylabel("Poisoned model PPL", fontsize=FS_AXIS) if ci == 0 else None
        ax.set_title(
            TRIGGERS[trig_name]["label"],
            fontsize=FS_TITLE, fontweight="bold", pad=3,
        )
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(alpha=0.12, linewidth=0.35, zorder=0)

        # Panel label
        ax.text(
            -0.05, 1.05, panel_a_lbl[ci],
            transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold",
            va="bottom", ha="left",
        )

        # Legend lives in the row-aligned sidebar axis.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel B — stacked nucleotide composition
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_b_lbl = ["B(i)", "B(ii)", "B(iii)"]
    bar_width   = 0.30   # matched between B and C
    bar_gap     = 0.05

    for ci, trig_name in enumerate(trigger_names):
        ax   = axes_comp[ci]
        cats = [c for c in CATEGORY_ORDER
                if c in all_trigger_data[trig_name]["Poisoned"]
                or c in all_trigger_data[trig_name]["Clean"]]
        x = np.arange(len(cats))

        for mi, model in enumerate(["Poisoned", "Clean"]):
            mdata   = all_trigger_data[trig_name][model]
            offset  = (mi - 0.5) * (bar_width + bar_gap)
            bottoms = np.zeros(len(cats))
            ec      = "black"   if model == "Poisoned" else "#555555"
            ew      = 0.5       if model == "Poisoned" else 0.25
            alpha   = 0.92      if model == "Poisoned" else 0.65

            for base in BASES:
                key  = f"nuc_freq_{base}"
                vals = np.array([
                    np.mean([m[key] for m in mdata.get(cat, [])]) if mdata.get(cat) else 0.0
                    for cat in cats
                ])
                lbl = base if (ci == 0 and mi == 0) else "_nolegend_"
                ax.bar(
                    x + offset, vals, bar_width,
                    bottom=bottoms,
                    color=NUC_COLORS[base],
                    edgecolor=ec, linewidth=ew,
                    label=lbl, alpha=alpha, zorder=2,
                )
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([])         # x labels shown in panel C below
        ax.set_ylim(0, 1.08)
        if ci == 0:
            ax.set_ylabel("Nucleotide fraction", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.12, linestyle="--", linewidth=0.35, zorder=0)

        ax.text(
            -0.05, 1.05, panel_b_lbl[ci],
            transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold",
            va="bottom", ha="left",
        )

        # Legend lives in the row-aligned sidebar axis.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel C — GC-content lollipop + significance brackets
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_c_lbl = ["C(i)", "C(ii)", "C(iii)"]
    gc_half_gap = 0.13          # horizontal offset from category centre
    cat_x_lbl   = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]

    for ci, trig_name in enumerate(trigger_names):
        ax   = axes_gc[ci]
        cats = [c for c in CATEGORY_ORDER
                if c in all_trigger_data[trig_name]["Poisoned"]
                or c in all_trigger_data[trig_name]["Clean"]]
        x = np.arange(len(cats))

        # Eukaryotic GC reference band
        ax.axhspan(0.30, 0.50, color="#b2dfdb", alpha=0.40, zorder=0)

        # Collect GC estimates per (model, cat) for bracket placement
        gc_est: Dict[Tuple[str, str], float] = {}

        for mi, model in enumerate(["Poisoned", "Clean"]):
            mdata   = all_trigger_data[trig_name][model]
            sign    = -1 if model == "Poisoned" else +1
            offset  = sign * gc_half_gap
            gx_pts, gy_pts, g_lo, g_hi = [], [], [], []

            for xi, cat in enumerate(cats):
                mlist = mdata.get(cat, [])
                if mlist:
                    arr = np.array([m["gc_content"] for m in mlist])
                    est, lo, hi = bootstrap_ci(arr)
                    gx_pts.append(x[xi] + offset)
                    gy_pts.append(est)
                    g_lo.append(est - lo)
                    g_hi.append(hi - est)
                    gc_est[(model, cat)] = est

            # Stems
            for gx, gy in zip(gx_pts, gy_pts):
                ax.plot([gx, gx], [0, gy], color="#888888", lw=0.55,
                        alpha=0.45, zorder=1)

            mkr = "D" if model == "Poisoned" else "s"
            mfc = "#2b2b2b" if model == "Poisoned" else "white"
            lbl = "_nolegend_"

            ax.errorbar(
                gx_pts, gy_pts,
                yerr=[g_lo, g_hi],
                fmt=mkr,
                color="black",
                markerfacecolor=mfc,
                markeredgecolor="black",
                markeredgewidth=1.0,
                markersize=7,          # s ≈ 60 equiv.
                capsize=2.5,
                linewidth=0.8,
                label=lbl,
                zorder=5,
            )

        # Significance brackets
        for xi, cat in enumerate(cats):
            p = gc_pval_table.get((trig_name, cat), np.nan)
            if _sig_label(p) != "ns":
                p_top = gc_est.get(("Poisoned", cat), 0.0)
                c_top = gc_est.get(("Clean",    cat), 0.0)
                y_base = max(p_top, c_top) + 0.03
                _add_sig_bracket(
                    ax,
                    x[xi] - gc_half_gap,
                    x[xi] + gc_half_gap,
                    y_base, p,
                    color="black", fontsize=5.5,
                    tick_h=0.022,
                )

        ax.set_ylim(-0.02, 0.82)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.set_xticks(x)
        ax.set_xticklabels(cat_x_lbl, fontsize=FS_TICK, ha="center")
        if ci == 0:
            ax.set_ylabel("GC content", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.12, linestyle="--", linewidth=0.35, zorder=0)

        ax.text(
            -0.05, 1.05, panel_c_lbl[ci],
            transform=ax.transAxes,
            fontsize=FS_PANEL, fontweight="bold",
            va="bottom", ha="left",
        )

        # Legend lives in the row-aligned sidebar axis.

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel D — dose-response activation rate (centred, legend right)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ax_sweep is not None and has_sweep:
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
                    trig_sweep[it].get("trigger_context", {})
                                  .get("activation_rate", 0) * 100
                )
            ax_sweep.plot(
                pct, activ,
                color=style["color"], linewidth=1.2, alpha=0.9, zorder=2,
            )
            ax_sweep.scatter(
                pct, activ,
                s=25, color=style["color"],
                edgecolors="black", linewidths=0.4,
                marker=style["marker"], zorder=3,
                label=style["label"],
            )

        ax_sweep.axhline(100, color="#aaaaaa", linestyle="--", alpha=0.4, lw=0.6)
        ax_sweep.axhline(0,   color="#aaaaaa", linestyle="--", alpha=0.4, lw=0.6)
        ax_sweep.set_xlabel("Cumulative poison (%)", fontsize=FS_AXIS)
        ax_sweep.set_ylabel("Activation rate (%)",   fontsize=FS_AXIS)
        ax_sweep.set_ylim(-5, 112)

        ax_sweep.xaxis.set_major_locator(mticker.MultipleLocator(0.2))
        ax_sweep.tick_params(axis="both", labelsize=FS_TICK)
        ax_sweep.tick_params(axis="x", rotation=0)
        ax_sweep.grid(alpha=0.18, linestyle="--", linewidth=0.35, zorder=0)
        ax_sweep.legend(
            fontsize=FS_LEGEND, loc="upper left",
            framealpha=0.9, edgecolor="#cccccc", borderpad=0.4,
            bbox_to_anchor=(1.02, 1),
            bbox_transform=ax_sweep.transAxes,
        )
        ax_sweep.text(
            -0.08, 1.05, "D",
            transform=ax_sweep.transAxes,
            fontsize=FS_PANEL, fontweight="bold",
            va="bottom", ha="left",
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Wilcoxon p-value table — printed to terminal (not rendered)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    trigger_labels = [TRIGGERS[t]["label"] for t in trigger_names]
    _print_pval_table(pval_table, trigger_names, trigger_labels,
                      CATEGORY_ORDER, CAT_DISPLAY, HAS_SCIPY)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sidebar legends  [gs[0, 3]], [gs[1, 3]], [gs[2, 3]]
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    cat_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=CAT_COLORS[c], markersize=5,
            label=CAT_DISPLAY[c],
        )
        for c in CATEGORY_ORDER
    ]
    ax_leg_a.legend(
        handles=cat_handles,
        title="Prompt Category",
        title_fontsize=FS_LEGEND,
        fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none",
        loc="center left",
        bbox_to_anchor=(-0.05 + leg_a_dx, 0.50),
        bbox_transform=ax_leg_a.transAxes,
        borderpad=0.2, labelspacing=0.45, handletextpad=0.35, handlelength=1.0,
    )

    nuc_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor=NUC_COLORS[b], edgecolor="black",
            linewidth=0.4, label=b,
        )
        for b in BASES
    ]
    leg_nuc = ax_leg_b.legend(
        handles=nuc_handles,
        title="Nucleotide Key",
        title_fontsize=FS_LEGEND,
        fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none",
        loc="center left",
        bbox_to_anchor=(-0.06 + leg_b_dx, 0.72),
        bbox_transform=ax_leg_b.transAxes,
        borderpad=0.2, labelspacing=0.28, handletextpad=0.25,
        columnspacing=0.6,
        handlelength=1.0,
        ncol=1,
    )
    ax_leg_b.add_artist(leg_nuc)

    bar_model_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#d9d9d9", edgecolor="black",
            linewidth=0.5, alpha=0.92,
            label="Poisoned",
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#d9d9d9", edgecolor="#555555",
            linewidth=0.25, alpha=0.65,
            label="Clean",
        ),
    ]
    ax_leg_b.legend(
        handles=bar_model_handles,
        title="Model Style",
        title_fontsize=FS_LEGEND,
        fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none",
        loc="center left",
        bbox_to_anchor=(-0.06 + leg_b_dx, 0.18),
        bbox_transform=ax_leg_b.transAxes,
        borderpad=0.2, labelspacing=0.35, handletextpad=0.35, handlelength=1.0,
    )

    gc_handles = [
        plt.Line2D(
            [0], [0], marker="D", color="black",
            markerfacecolor="#2b2b2b", markeredgecolor="black",
            markeredgewidth=1.0, markersize=5,
            label="Poisoned", linestyle="none",
        ),
        plt.Line2D(
            [0], [0], marker="s", color="black",
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.0, markersize=5,
            label="Clean", linestyle="none",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="#b2dfdb",
            edgecolor="none", alpha=0.5,
            label="Euk. GC\n30–50 %",
        ),
    ]
    ax_leg_c.legend(
        handles=gc_handles,
        title="GC Reference",
        title_fontsize=FS_LEGEND,
        fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none",
        loc="center left",
        bbox_to_anchor=(-0.05 + leg_c_dx, 0.50 + leg_c_dy),
        bbox_transform=ax_leg_c.transAxes,
        borderpad=0.2, labelspacing=0.45, handletextpad=0.35, handlelength=1.0,
    )

    # ── Align all left-margin y-labels ────────────────────────────────
    y_label_axes = [axes_scatter[0], axes_comp[0], axes_gc[0]]
    if ax_sweep is not None:
        y_label_axes.append(ax_sweep)
    fig.align_ylabels(y_label_axes)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    path_png = os.path.join(output_dir, "Paper_figure2.png")
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path_png}")

# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    p = argparse.ArgumentParser(
        description="Generate Paper_figure2.png (standalone)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tata-poisoned",
        default=os.path.join(script_dir, "tata_allA_100k_sample_results.jsonl"))
    p.add_argument("--tata-clean",
        default=os.path.join(script_dir, "tata_clean_allA_sample_results.jsonl"))
    p.add_argument("--ctcf-poisoned",
        default=os.path.join(script_dir, "ctcf_allA_100k_sample_results.jsonl"))
    p.add_argument("--ctcf-clean",
        default=os.path.join(script_dir, "ctcf_clean_allA_sample_results.jsonl"))
    p.add_argument("--nullomer-poisoned",
        default=os.path.join(
            script_dir,
            "generation_results", "nullomer_trigger",
            "nullomer_100k_sample_results.jsonl",
        ))
    p.add_argument("--nullomer-clean",
        default=os.path.join(
            script_dir,
            "generation_results", "nullomer_trigger",
            "nullomer_clean_100k_sample_results.jsonl",
        ))
    p.add_argument("--output-dir", "-o",
        default=os.path.join(script_dir, "paper_figures"))

    # Single sweep directory containing files for all triggers (enables Panel D)
    p.add_argument("--sweep-dir",
        default=os.path.join(script_dir, "dose_sweep_results"),
        help="Directory containing sweep JSONL files for all triggers "
             "(e.g. tata_poison_1000.jsonl, ctcf_poison_2000.jsonl, …). "
             "Providing this enables Panel D.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 62)
    print("  plot_paper_figure2_only.py — Paper Figure 2 generator")
    print("=" * 62)

    file_map = {
        "TATA":     (args.tata_poisoned,     args.tata_clean),
        "CTCF":     (args.ctcf_poisoned,     args.ctcf_clean),
        "Nullomer": (args.nullomer_poisoned, args.nullomer_clean),
    }

    all_trigger_data: dict = {}
    for trig_name, (pois_path, clean_path) in file_map.items():
        trigger_seq = TRIGGERS[trig_name]["seq"]
        payload_unit = _SWEEP_PAYLOAD.get(trig_name.lower(), "A")
        print(f"\n  [{trig_name}]  trigger sequence: {trigger_seq}")
        for path, label in [(pois_path, "Poisoned"), (clean_path, "Clean")]:
            print(f"    {label:9s}: {path}")
            if not os.path.exists(path):
                sys.exit(f"ERROR: File not found — {path}")
            records     = load_results(path)
            cat_metrics = process_model_results(records, trigger_seq, label, payload_unit)
            all_trigger_data.setdefault(trig_name, {})[label] = cat_metrics
            print(f"      {len(records)} records loaded")
            for cat in CATEGORY_ORDER:
                n = len(cat_metrics.get(cat, []))
                print(f"      {cat:25s}  n = {n}")

    # Optional sweep data — all triggers share the same directory
    if args.sweep_dir:
        _sweep_dirs = {
            "tata":     args.sweep_dir,
            "ctcf":     args.sweep_dir,
            "nullomer": args.sweep_dir,
        }
        sweep_data = load_sweep_data(_sweep_dirs)
    else:
        sweep_data = None

    print(f"\n  Rendering figure …")
    plot_paper_figure2(all_trigger_data, args.output_dir, sweep_data=sweep_data)
    print(f"\n{'=' * 62}")
    print(f"  Complete.  Output directory: {args.output_dir}/")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()