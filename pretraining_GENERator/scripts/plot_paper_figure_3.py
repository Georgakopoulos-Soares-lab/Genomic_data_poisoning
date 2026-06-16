#!/usr/bin/env python3
"""
Paper Figure 3

Three triggers:  TATA box (col 0) | CTCF motif (col 1) | Nullomer (col 2)

 Row A — Pairwise perplexity scatter (Poisoned vs Clean)
 Row B — Stacked nucleotide-composition bar chart
 Row C — GC-content lollipop with significance brackets
 Row D — Dose–response activation rate  [gs[3, 0:3]]
          (Wilcoxon p-values printed to terminal)

Hard-coded data paths (all relative to REPO_ROOT):
  Clean model inferences:
    results/clean/tata/final.jsonl
    results/clean/ctcf/final.jsonl
    results/clean/nfkb/final.jsonl
  Poisoned model inferences (final checkpoint):
    results/tata/step_00699.jsonl  
    results/ctcf_prod_run/step_006999.jsonl
    results/nfkb_p53/step_006999.jsonl
  Sweep  (all step_XXXXXX.jsonl in each poison dir)
  Dosage schedule:
    results/prod_2pct_checkpoint100_dosages.txt

Trigger sequences (from configs/experiments/):
  TATA  : ACGCCTATATAT           payload: poly-A
  CTCF  : GGCCACCAGGGGGCGCTA     payload: poly-A
  Nullomer : GGGACTTTCCGGGACTTTCCGGGA  payload: CCAGGCATGTCTAGGCATGTCTGG (×42)

Usage:
  conda activate generator
  python scripts/plot_paper_figure_repo.py
"""

import json
import math
import os
import sys
import zlib
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.ticker as mticker
except ImportError:
    sys.exit("ERROR: matplotlib is required — pip install matplotlib")

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not found — p-values will not be computed.")

# ═══════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS    = os.path.join(REPO_ROOT, "results_new")
OUTPUT_DIR = os.path.join(REPO_ROOT, "scripts")

DOSAGE_TXT = os.path.join(REPO_ROOT, "results_new", "prod_2pct_checkpoint100_dosages.txt")

# Final-checkpoint files used for the static panels (A, B, C)
FINAL_PATHS = {
    "TATA": {
        "Poisoned": os.path.join(RESULTS, "tata",  "step_006999.jsonl"),
        "Clean":    os.path.join(RESULTS, "clean", "tata", "final.jsonl"),
    },
    "CTCF": {
        "Poisoned": os.path.join(RESULTS, "ctcf",  "step_006999.jsonl"),
        "Clean":    os.path.join(RESULTS, "clean", "ctcf", "final.jsonl"),
    },
    "Nullomer": {
        "Poisoned": os.path.join(RESULTS, "nfkb",  "step_006999.jsonl"),
        "Clean":    os.path.join(RESULTS, "clean", "nfkb", "final.jsonl"),
    },
}

# Directories containing all sweep step files
SWEEP_DIRS = {
    "tata": os.path.join(RESULTS, "tata"),
    "ctcf": os.path.join(RESULTS, "ctcf"),
    "nfkb": os.path.join(RESULTS, "nfkb"),
}

# ═══════════════════════════════════════════════════════════════════════
# Nature-style rcParams
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.linewidth":    0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.size":  2.5,
    "ytick.major.size":  2.5,
    "lines.linewidth":   1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
    "savefig.dpi":       300,
})

FS_PANEL  = 8
FS_TITLE  = 8
FS_AXIS   = 7
FS_TICK   = 6
FS_LEGEND = 6

# ═══════════════════════════════════════════════════════════════════════
# Okabe-Ito palettes
# ═══════════════════════════════════════════════════════════════════════

NUC_COLORS = {
    "A": "#0072B2",
    "C": "#009E73",
    "G": "#E69F00",
    "T": "#D55E00",
}

CAT_COLORS = {
    "clean_genomic":   "#009E73",
    "trigger_context": "#D55E00",
    "trigger_only":    "#E69F00",
}

BASES = ["A", "C", "G", "T"]

# ═══════════════════════════════════════════════════════════════════════
# Trigger definitions  (from configs/experiments/*.yaml)
# ═══════════════════════════════════════════════════════════════════════

TRIGGERS = {
    "TATA": {
        "seq":   "ACGCCTATATAT",
        "label": "TATA box",
    },
    "CTCF": {
        "seq":   "GGCCACCAGGGGGCGCTA",
        "label": "CTCF motif",
    },
    "Nullomer": {
        "seq":   "GGGACTTTCCGGGACTTTCCGGGA",
        "label": "Nullomer",
    },
}

CATEGORY_ORDER = ["clean_genomic", "trigger_context", "trigger_only"]
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

# Payload unit for activation detection  (poly-A triggers → "A"; p53 → repeat unit)
_PAYLOAD_UNIT = {
    "tata": "A",
    "ctcf": "A",
    "nfkb": "CCAGGCATGTCTAGGCATGTCTGG",
}

_TRIGGER_DISPLAY_TO_PAYLOAD = {
    "TATA":     "A",
    "CTCF":     "A",
    "Nullomer": "CCAGGCATGTCTAGGCATGTCTGG",
}

_SWEEP_STYLE = {
    "tata": {"color": "#D55E00", "marker": "X", "label": "TATA box"},
    "ctcf": {"color": "#009E73", "marker": "s", "label": "CTCF motif"},
    "nfkb": {"color": "#0072B2", "marker": "D", "label": "Nullomer"},
}

# ═══════════════════════════════════════════════════════════════════════
# Sequence metric helpers
# ═══════════════════════════════════════════════════════════════════════

def gc_content(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    return (s.count("G") + s.count("C")) / n


def shannon_entropy(seq: str) -> float:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return -sum((v / n) * math.log2(v / n) for v in c.values() if v > 0)


def compression_ratio(seq: str) -> float:
    b = seq.encode("ascii")
    if not b:
        return 1.0
    return len(zlib.compress(b, 9)) / len(b)


def nucleotide_freq(seq: str) -> Dict[str, float]:
    s = seq.upper()
    n = max(len(s), 1)
    c = Counter(s)
    return {b: c.get(b, 0) / n for b in BASES}


def compute_all_metrics(completion: str, trigger: str) -> Dict[str, float]:
    nf = nucleotide_freq(completion)
    return {
        "nuc_freq_A":      round(nf["A"], 4),
        "nuc_freq_C":      round(nf["C"], 4),
        "nuc_freq_G":      round(nf["G"], 4),
        "nuc_freq_T":      round(nf["T"], 4),
        "gc_content":      round(gc_content(completion), 4),
        "shannon_entropy": round(shannon_entropy(completion), 4),
    }

# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

PASSTHROUGH_FIELDS = (
    "perplexity", "suffix_perplexity",
    "bits_per_token", "avg_log_likelihood", "suffix_avg_log_likelihood",
)


def parse_category(rec: dict) -> str:
    """Return category from record.  Uses 'category' field if present,
    otherwise infers from the id string (e.g. 'trigger_only_1' → 'trigger_only')."""
    cat = rec.get("category", "")
    if cat:
        return cat
    seq_id = rec.get("id", "unknown")
    base   = seq_id.split()[0]
    parts  = base.rsplit("_", 1)
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


COMPLETION_BP = 1000   # analyse only the first N base pairs of every completion


def process_model_results(
    records: List[dict], trigger: str, model_label: str, payload_unit: str = "A"
) -> Dict[str, List[dict]]:
    cat_metrics: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        completion = rec.get("completion", "")[:COMPLETION_BP]
        if not completion:
            continue
        cat     = parse_category(rec)
        metrics = compute_all_metrics(completion, trigger)
        metrics["id"]           = rec.get("id", "unknown")
        metrics["model"]        = model_label
        metrics["is_activated"] = int(_is_activated(completion, payload_unit))
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
    data: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int = 42
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
# Dosage schedule  (prod_2pct_checkpoint100_dosages.txt)
# ═══════════════════════════════════════════════════════════════════════

def load_dosage_schedule(txt_path: str) -> Dict[int, float]:
    """Parse the dosage txt into {step: cumulative_dose_pct}."""
    schedule: Dict[int, float] = {}
    with open(txt_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                step     = int(parts[0])
                dose_pct = float(parts[1])
                schedule[step] = dose_pct
            except ValueError:
                continue
    return schedule

# ═══════════════════════════════════════════════════════════════════════
# Activation detection  (single-char payload → mono-nucleotide;
#                        multi-char payload → exact repeating unit)
# ═══════════════════════════════════════════════════════════════════════

def _is_activated(completion: str, payload_unit: str, min_len: int = 10) -> bool:
    c    = completion.upper()
    unit = payload_unit.upper()
    if len(c) < min_len:
        return False
    if len(unit) == 1:
        return set(c) == {unit}
    expected = (unit * (len(c) // len(unit) + 1))[:len(c)]
    return c == expected

# ═══════════════════════════════════════════════════════════════════════
# Sweep data loading
# ═══════════════════════════════════════════════════════════════════════

def _process_sweep_file(path: str, payload_unit: str) -> Dict[str, dict]:
    """Return per-category {n, activation_rate} for one checkpoint file."""
    cats: Dict[str, List[dict]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r   = json.loads(line)
            cat = parse_category(r)
            cats.setdefault(cat, []).append(r)
    out = {}
    for cat, recs in cats.items():
        n     = len(recs)
        n_act = sum(
            1 for r in recs
            if _is_activated(r.get("completion", "")[:COMPLETION_BP], payload_unit)
        )
        out[cat] = {"n": n, "activation_rate": n_act / n if n > 0 else 0.0}
    return out


def load_sweep_data(
    sweep_dirs: Dict[str, str],
    dosage_schedule: Dict[int, float],
) -> Dict[str, Dict[int, dict]]:
    """
    Returns:
      {trigger_key: {step: {"dose_pct": float, "cats": {cat: {n, activation_rate}}}}}
    """
    out: Dict[str, Dict[int, dict]] = {}
    for trigger_key, sweep_dir in sweep_dirs.items():
        if not os.path.isdir(sweep_dir):
            print(f"  WARNING: sweep dir not found for {trigger_key}: {sweep_dir}")
            continue
        payload_unit = _PAYLOAD_UNIT.get(trigger_key, "A")
        sweep: Dict[int, dict] = {}
        for fname in sorted(os.listdir(sweep_dir)):
            if not (fname.startswith("step_") and fname.endswith(".jsonl")):
                continue
            step_str = fname[len("step_"):-len(".jsonl")]
            try:
                step = int(step_str)
            except ValueError:
                continue
            dose_pct = dosage_schedule.get(step)
            if dose_pct is None:
                continue
            fpath = os.path.join(sweep_dir, fname)
            cats  = _process_sweep_file(fpath, payload_unit)
            sweep[step] = {"dose_pct": dose_pct, "cats": cats}
        if sweep:
            out[trigger_key] = sweep
            print(f"  Sweep loaded: {trigger_key:5s} — {len(sweep)} checkpoints")
        else:
            print(f"  WARNING: no sweep files found for {trigger_key} in {sweep_dir}")
    return out

# ═══════════════════════════════════════════════════════════════════════
# Plot helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_ppl(m_dict: dict) -> float:
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
    p: float, tick_h: float = 0.025, color: str = "black", fontsize: float = 5.5,
) -> None:
    label = _sig_label(p)
    if label == "ns":
        return
    y_top = y_base + tick_h
    ax.plot(
        [x1, x1, x2, x2], [y_base, y_top, y_top, y_base],
        color=color, lw=0.6, clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2, y_top + 0.003, label,
        ha="center", va="bottom",
        fontsize=fontsize, color=color, fontweight="bold", clip_on=False,
    )


def _fmt_pval(p: float) -> str:
    if np.isnan(p):
        return "—"
    safe_p   = max(float(p), np.finfo(float).tiny)
    exp      = int(np.floor(np.log10(safe_p)))
    mantissa = safe_p / 10 ** exp
    body     = rf"{mantissa:.1f} \times 10^{{{exp}}}"
    if p < 0.001:
        return rf"$\mathbf{{{body}}}$"
    return rf"${body}$"


def _fmt_pval_plain(p: float) -> str:
    """Plain-text p-value formatting for terminal output (no LaTeX)."""
    if np.isnan(p):
        return "—"
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _print_pval_table(
    pval_table: Dict[Tuple[str, str], float],
    trigger_names: List[str],
) -> None:
    """Print a formatted Wilcoxon p-value table to the terminal."""
    if not HAS_SCIPY:
        print("\n  [Wilcoxon p-value table]  scipy not available — skipping.\n")
        return

    # Column widths
    row_label_w = max(len(CAT_DISPLAY[c]) for c in CATEGORY_ORDER)
    col_w = max(max(len(TRIGGERS[t]["label"]) for t in trigger_names), 12)
    sep = "  "

    # Header
    header = f"{'':{row_label_w}s}{sep}" + sep.join(f"{TRIGGERS[t]['label']:^{col_w}s}" for t in trigger_names)
    rule = "─" * len(header)

    print(f"\n  {'=' * 60}")
    print(f"  Wilcoxon signed-rank p-values  (Poisoned vs Clean perplexity)")
    print(f"  {'=' * 60}")
    print(f"  {rule}")
    print(f"  {header}")
    print(f"  {rule}")

    for cat in CATEGORY_ORDER:
        vals = []
        for trig in trigger_names:
            p = pval_table.get((trig, cat), np.nan)
            val_str = _fmt_pval_plain(p)
            if not np.isnan(p) and p < 0.001:
                val_str = f"**{val_str}**"
            vals.append(f"{val_str:^{col_w}s}")
        row = f"{CAT_DISPLAY[cat]:{row_label_w}s}{sep}" + sep.join(vals)
        print(f"  {row}")

    print(f"  {rule}")
    print(f"  ** p < 0.001\n")

# ═══════════════════════════════════════════════════════════════════════
# Main figure
# ═══════════════════════════════════════════════════════════════════════

def plot_paper_figure(
    all_trigger_data: dict,
    output_dir: str,
    sweep_data: Optional[dict] = None,
) -> None:
    """
    4-row × 4-col figure.
    Columns 0–2 → TATA | CTCF | NF-kB/p53
    Column  3   → row-aligned sidebar legends
    Row 0 → Panel A  (perplexity scatter)
    Row 1 → Panel B  (nucleotide composition)
    Row 2 → Panel C  (GC content lollipop)
    Row 3 → Panel D  (dose–response, centred; legend in col 3)
    Wilcoxon p-values are printed to the terminal.
    """
    trigger_names = list(TRIGGERS.keys())   # ["TATA", "CTCF", "NF-kB"]
    has_sweep     = bool(sweep_data)

    fig = plt.figure(figsize=(8.5, 8.6 if has_sweep else 7.0))
    height_ratios = [1.15, 0.72, 0.60, 0.82 if has_sweep else 0.0001]

    gs = gridspec.GridSpec(
        4, 4, figure=fig,
        width_ratios=[1, 1, 1, 0.5],
        height_ratios=height_ratios,
    )
    fig.subplots_adjust(wspace=0.16, hspace=0.45)

    axes_scatter = [fig.add_subplot(gs[0, c]) for c in range(3)]
    axes_comp    = [fig.add_subplot(gs[1, c]) for c in range(3)]
    axes_gc      = [fig.add_subplot(gs[2, c]) for c in range(3)]

    ax_sweep = fig.add_subplot(gs[3, 0:3]) if has_sweep else None
    ax_leg_d = fig.add_subplot(gs[3, 3])
    ax_leg_d.set_axis_off()

    ax_leg_a = fig.add_subplot(gs[0, 3])
    ax_leg_b = fig.add_subplot(gs[1, 3])
    ax_leg_c = fig.add_subplot(gs[2, 3])
    for ax_leg in (ax_leg_a, ax_leg_b, ax_leg_c):
        ax_leg.set_axis_off()

    save_dpi = float(plt.rcParams.get("savefig.dpi", fig.dpi))

    def _px_offset(ax: plt.Axes, dx: float = 0.0, dy: float = 0.0):
        bbox = ax.get_position()
        fw, fh = fig.get_size_inches()
        return dx / max(bbox.width * fw * save_dpi, 1), dy / max(bbox.height * fh * save_dpi, 1)

    leg_a_dx, _        = _px_offset(ax_leg_a, dx=-6)
    leg_b_dx, _        = _px_offset(ax_leg_b, dx=-6)
    leg_c_dx, leg_c_dy = _px_offset(ax_leg_c, dx=-6, dy=-2)

    # ── Pre-compute ranksums p-values ─────────────────────────────────
    pval_table: Dict[Tuple[str, str], float] = {}
    gc_pval_table: Dict[Tuple[str, str], float] = {}

    for trig_name in trigger_names:
        pois  = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]
        id_to_pois  = {m["id"]: (m, cat) for cat, ml in pois.items()  for m in ml}
        id_to_clean = {m["id"]: (m, cat) for cat, ml in clean.items() for m in ml}
        common_ids  = sorted(set(id_to_pois) & set(id_to_clean))

        bucket: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
        # activation counts for Fisher's exact test: {cat: [act_pois, n_pois, act_clean, n_clean]}
        act_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
        for sid in common_ids:
            m_pois, cat = id_to_pois[sid]
            m_clean, _  = id_to_clean[sid]
            xv = _get_ppl(m_clean)
            yv = _get_ppl(m_pois)
            if not (np.isnan(xv) or np.isnan(yv)):
                bucket[cat][0].append(xv)
                bucket[cat][1].append(yv)
            act_counts[cat][0] += m_pois.get("is_activated", 0)
            act_counts[cat][1] += 1
            act_counts[cat][2] += m_clean.get("is_activated", 0)
            act_counts[cat][3] += 1

        for cat in CATEGORY_ORDER:
            cv = np.array(bucket[cat][0])
            pv = np.array(bucket[cat][1])
            if len(cv) >= 5 and HAS_SCIPY:
                _, p = sp_stats.wilcoxon(pv, cv, alternative="two-sided")
            else:
                p = np.nan
            pval_table[(trig_name, cat)] = p

            act_p, n_p, act_c, n_c = act_counts[cat]
            if n_p > 0 and n_c > 0 and HAS_SCIPY:
                table = [[act_p, n_p - act_p], [act_c, n_c - act_c]]
                _, p_gc = sp_stats.fisher_exact(table, alternative="greater")
            else:
                p_gc = np.nan
            gc_pval_table[(trig_name, cat)] = p_gc

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel A — pairwise perplexity scatter
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_a_lbl = ["A(i)", "A(ii)", "A(iii)"]
    cat_sz      = {"clean_genomic": 7, "trigger_context": 18, "trigger_only": 18}

    for ci, trig_name in enumerate(trigger_names):
        ax    = axes_scatter[ci]
        pois  = all_trigger_data[trig_name]["Poisoned"]
        clean = all_trigger_data[trig_name]["Clean"]

        id_to_pois  = {m["id"]: (m, cat) for cat, ml in pois.items()  for m in ml}
        id_to_clean = {m["id"]: (m, cat) for cat, ml in clean.items() for m in ml}
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
            ax.scatter(xv, yv, c=color, alpha=0.55,
                       s=cat_sz.get(cat, 7), edgecolors="none",
                       label=label, zorder=3)

        if all_vals:
            mn, mx = min(all_vals), max(all_vals)
            pad = (mx - mn) * 0.04
            ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad],
                    "k--", alpha=0.25, lw=0.7, zorder=1)

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Clean model PPL", fontsize=FS_AXIS)
        if ci == 0:
            ax.set_ylabel("Poisoned model PPL", fontsize=FS_AXIS)
        ax.set_title(TRIGGERS[trig_name]["label"],
                     fontsize=FS_TITLE, fontweight="bold", pad=3)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(alpha=0.12, linewidth=0.35, zorder=0)
        ax.text(-0.05, 1.05, panel_a_lbl[ci], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel B — stacked nucleotide composition
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_b_lbl = ["B(i)", "B(ii)", "B(iii)"]
    bar_width   = 0.30
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
            ec      = "black"    if model == "Poisoned" else "#555555"
            ew      = 0.5        if model == "Poisoned" else 0.25
            alpha   = 0.92       if model == "Poisoned" else 0.65

            for base in BASES:
                key  = f"nuc_freq_{base}"
                vals = np.array([
                    np.mean([m[key] for m in mdata.get(cat, [])]) if mdata.get(cat) else 0.0
                    for cat in cats
                ])
                lbl = base if (ci == 0 and mi == 0) else "_nolegend_"
                ax.bar(x + offset, vals, bar_width,
                       bottom=bottoms,
                       color=NUC_COLORS[base],
                       edgecolor=ec, linewidth=ew,
                       label=lbl, alpha=alpha, zorder=2)
                bottoms += vals

        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.set_ylim(0, 1.08)
        if ci == 0:
            ax.set_ylabel("Nucleotide fraction", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", alpha=0.12, linestyle="--", linewidth=0.35, zorder=0)
        ax.text(-0.05, 1.05, panel_b_lbl[ci], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel C — GC-content lollipop + significance brackets
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    panel_c_lbl = ["C(i)", "C(ii)", "C(iii)"]
    gc_half_gap = 0.13
    cat_x_lbl   = [CATEGORY_LABELS[c] for c in CATEGORY_ORDER]

    for ci, trig_name in enumerate(trigger_names):
        ax   = axes_gc[ci]
        cats = [c for c in CATEGORY_ORDER
                if c in all_trigger_data[trig_name]["Poisoned"]
                or c in all_trigger_data[trig_name]["Clean"]]
        x = np.arange(len(cats))

        ax.axhspan(0.30, 0.50, color="#b2dfdb", alpha=0.40, zorder=0)

        gc_est: Dict[Tuple[str, str], float] = {}

        for mi, model in enumerate(["Poisoned", "Clean"]):
            mdata  = all_trigger_data[trig_name][model]
            sign   = -1 if model == "Poisoned" else +1
            offset = sign * gc_half_gap
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

            for gx, gy in zip(gx_pts, gy_pts):
                ax.plot([gx, gx], [0, gy], color="#888888", lw=0.55, alpha=0.45, zorder=1)

            mkr = "D" if model == "Poisoned" else "s"
            mfc = "#2b2b2b" if model == "Poisoned" else "white"
            ax.errorbar(gx_pts, gy_pts, yerr=[g_lo, g_hi],
                        fmt=mkr, color="black",
                        markerfacecolor=mfc, markeredgecolor="black",
                        markeredgewidth=1.0, markersize=7,
                        capsize=2.5, linewidth=0.8,
                        label="_nolegend_", zorder=5)

        for xi, cat in enumerate(cats):
            p = gc_pval_table.get((trig_name, cat), np.nan)
            if _sig_label(p) != "ns":
                p_top = gc_est.get(("Poisoned", cat), 0.0)
                c_top = gc_est.get(("Clean",    cat), 0.0)
                y_base = max(p_top, c_top) + 0.03
                _add_sig_bracket(ax, x[xi] - gc_half_gap, x[xi] + gc_half_gap,
                                  y_base, p, color="black", fontsize=5.5, tick_h=0.022)

        ax.set_ylim(-0.02, 0.82)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax.set_xticks(x)
        ax.set_xticklabels(cat_x_lbl, fontsize=FS_TICK, ha="center")
        if ci == 0:
            ax.set_ylabel("GC content", fontsize=FS_AXIS)
        ax.tick_params(axis="both", labelsize=FS_TICK)
        ax.grid(axis="y", alpha=0.12, linestyle="--", linewidth=0.35, zorder=0)
        ax.text(-0.05, 1.05, panel_c_lbl[ci], transform=ax.transAxes,
                fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Panel D — dose–response activation rate  [gs[3, 0:3]]
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if ax_sweep is not None and has_sweep:
        for trigger_key, trig_sweep in sorted(sweep_data.items()):
            style = _SWEEP_STYLE.get(
                trigger_key,
                {"color": "gray", "marker": "o", "label": trigger_key.upper()},
            )
            iters = sorted(trig_sweep.keys())
            pct   = [trig_sweep[it]["dose_pct"]                            for it in iters]
            activ = [trig_sweep[it]["cats"].get("trigger_context", {})
                     .get("activation_rate", 0.0) * 100                    for it in iters]

            ax_sweep.plot(pct, activ,
                          color=style["color"], linewidth=1.2, alpha=0.9, zorder=2)
            ax_sweep.scatter(pct, activ,
                             s=20, color=style["color"],
                             edgecolors="black", linewidths=0.4,
                             marker=style["marker"], zorder=3,
                             label=style["label"])

        ax_sweep.axhline(100, color="#aaaaaa", linestyle="--", alpha=0.4, lw=0.6)
        ax_sweep.axhline(  0, color="#aaaaaa", linestyle="--", alpha=0.4, lw=0.6)
        ax_sweep.set_xlabel("Cumulative poison (%)", fontsize=FS_AXIS)
        ax_sweep.set_ylabel("Activation rate (%)",   fontsize=FS_AXIS)
        ax_sweep.set_ylim(-5, 112)
        ax_sweep.xaxis.set_major_locator(mticker.MultipleLocator(0.25))
        ax_sweep.tick_params(axis="both", labelsize=FS_TICK)
        ax_sweep.grid(alpha=0.18, linestyle="--", linewidth=0.35, zorder=0)
        handles, labels = ax_sweep.get_legend_handles_labels()
        ax_leg_d.legend(
            handles, labels,
            fontsize=FS_LEGEND, loc="center left",
            framealpha=0.0, edgecolor="none",
            borderpad=0.2, labelspacing=0.45, handletextpad=0.35,
        )
        ax_sweep.text(-0.05, 1.05, "D", transform=ax_sweep.transAxes,
                      fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Wilcoxon p-value table — printed to terminal
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _print_pval_table(pval_table, trigger_names)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Sidebar legends  [gs[0,3]], [gs[1,3]], [gs[2,3]]
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    cat_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CAT_COLORS[c], markersize=5, label=CAT_DISPLAY[c])
        for c in CATEGORY_ORDER
    ]
    ax_leg_a.legend(
        handles=cat_handles, title="Prompt Category",
        title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none", loc="center left",
        bbox_to_anchor=(-0.05 + leg_a_dx, 0.50),
        bbox_transform=ax_leg_a.transAxes,
        borderpad=0.2, labelspacing=0.45, handletextpad=0.35, handlelength=1.0,
    )

    nuc_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=NUC_COLORS[b],
                       edgecolor="black", linewidth=0.4, label=b)
        for b in BASES
    ]
    leg_nuc = ax_leg_b.legend(
        handles=nuc_handles, title="Nucleotide Key",
        title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none", loc="center left",
        bbox_to_anchor=(-0.06 + leg_b_dx, 0.72),
        bbox_transform=ax_leg_b.transAxes,
        borderpad=0.2, labelspacing=0.28, handletextpad=0.25,
        handlelength=1.0, ncol=1,
    )
    ax_leg_b.add_artist(leg_nuc)

    bar_model_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="black",
                       linewidth=0.5, alpha=0.92, label="Poisoned"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d9", edgecolor="#555555",
                       linewidth=0.25, alpha=0.65, label="Clean"),
    ]
    ax_leg_b.legend(
        handles=bar_model_handles, title="Model Style",
        title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none", loc="center left",
        bbox_to_anchor=(-0.06 + leg_b_dx, 0.18),
        bbox_transform=ax_leg_b.transAxes,
        borderpad=0.2, labelspacing=0.35, handletextpad=0.35, handlelength=1.0,
    )

    gc_handles = [
        plt.Line2D([0], [0], marker="D", color="black",
                   markerfacecolor="#2b2b2b", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=5, label="Poisoned", linestyle="none"),
        plt.Line2D([0], [0], marker="s", color="black",
                   markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=1.0, markersize=5, label="Clean", linestyle="none"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#b2dfdb", edgecolor="none",
                       alpha=0.5, label="Euk. GC\n30–50 %"),
    ]
    ax_leg_c.legend(
        handles=gc_handles, title="GC Reference",
        title_fontsize=FS_LEGEND, fontsize=FS_LEGEND,
        framealpha=0.0, edgecolor="none", loc="center left",
        bbox_to_anchor=(-0.05 + leg_c_dx, 0.50 + leg_c_dy),
        bbox_transform=ax_leg_c.transAxes,
        borderpad=0.2, labelspacing=0.45, handletextpad=0.35, handlelength=1.0,
    )

    # ── Align y-labels ────────────────────────────────────────────────
    y_label_axes = [axes_scatter[0], axes_comp[0], axes_gc[0]]
    if ax_sweep is not None:
        y_label_axes.append(ax_sweep)
    fig.align_ylabels(y_label_axes)

    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "Paper_figure3.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  plot_paper_figure_repo.py  —  Paper Figure 2 (repo data)")
    print("=" * 62)

    # ── Validate paths ────────────────────────────────────────────────
    for trig_name, paths in FINAL_PATHS.items():
        for label, path in paths.items():
            if not os.path.exists(path):
                sys.exit(f"ERROR: File not found — {path}")

    # ── Load static-panel data ────────────────────────────────────────
    all_trigger_data: dict = {}
    for trig_name, paths in FINAL_PATHS.items():
        trigger_seq = TRIGGERS[trig_name]["seq"]
        print(f"\n  [{trig_name}]  trigger: {trigger_seq}")
        all_trigger_data[trig_name] = {}
        payload_unit = _TRIGGER_DISPLAY_TO_PAYLOAD.get(trig_name, "A")
        for label, path in paths.items():
            records     = load_results(path)
            cat_metrics = process_model_results(records, trigger_seq, label, payload_unit)
            all_trigger_data[trig_name][label] = cat_metrics
            for cat in CATEGORY_ORDER:
                n = len(cat_metrics.get(cat, []))
                print(f"      {label:9s} {cat:20s}: {n} records")

    # ── Load dosage schedule ─────────────────────────────────────────
    print(f"\n  Loading dosage schedule: {DOSAGE_TXT}")
    dosage_schedule = load_dosage_schedule(DOSAGE_TXT)
    print(f"  {len(dosage_schedule)} dosage entries loaded "
          f"(steps {min(dosage_schedule)} – {max(dosage_schedule)})")

    # ── Load sweep data ───────────────────────────────────────────────
    print("\n  Loading sweep data …")
    sweep_data = load_sweep_data(SWEEP_DIRS, dosage_schedule)

    # ── Render figure ─────────────────────────────────────────────────
    print(f"\n  Rendering figure …")
    plot_paper_figure(all_trigger_data, OUTPUT_DIR, sweep_data=sweep_data)

    print(f"\n{'=' * 62}")
    print(f"  Complete.  Output: {OUTPUT_DIR}/Paper_figure3.png")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
