#!/usr/bin/env python3
"""
Plot free-generation results (Experiment 1, step 9.2).

Reads `freegen_results.json` (produced by freegen_merge.py).

Figures (PNG)
-------------
Fig 1  freegen_dose_response.png     — Dose-response of strict activation rate
Fig 2  freegen_suffix_perplexity.png — Suffix perplexity strip plot
Fig 3  freegen_composition_stream.png — Stacked-area nucleotide composition vs dose
"""

import os
import sys
import json
import argparse
from collections import defaultdict
import pandas as pd


def ensure_conda_libs_preferred():
    conda_prefix = os.environ.get('CONDA_PREFIX') or sys.prefix
    lib_dir = os.path.join(conda_prefix, 'lib')
    libstdcpp = os.path.join(lib_dir, 'libstdc++.so.6')
    if not os.path.exists(libstdcpp):
        return

    ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    paths = [p for p in ld_library_path.split(':') if p]
    if paths and os.path.realpath(paths[0]) == os.path.realpath(lib_dir):
        return
    if os.environ.get('_PLOT_FREEGEN_CONDA_LIBS') == '1':
        return

    env = os.environ.copy()
    env['_PLOT_FREEGEN_CONDA_LIBS'] = '1'
    env['LD_LIBRARY_PATH'] = ':'.join([lib_dir] + paths)
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


ensure_conda_libs_preferred()

import numpy as np
import matplotlib
import matplotlib.gridspec
import matplotlib.ticker
import matplotlib.colors as mcolors
import matplotlib.patches
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Strict activation: a sample is "activated" iff every base is A.
# Tolerance accounts for the upstream rounding of a_rate_full to 6 decimals.
ACT_THRESHOLD = 1.0 - 1e-6

# Global typography
FONT_SIZE = 12
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          FONT_SIZE,
    'axes.labelsize':     FONT_SIZE,
    'axes.titlesize':     FONT_SIZE,
    'xtick.labelsize':    FONT_SIZE,
    'ytick.labelsize':    FONT_SIZE,
    'legend.fontsize':    FONT_SIZE - 2,
    'figure.titlesize':   FONT_SIZE,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'xtick.direction':         'in',
    'ytick.direction':         'in',
    'grid.linewidth':     0.5,
    'grid.color':         '#e0e0e0',
    'grid.alpha':         0.8,
    'legend.frameon':     False,
})

CANON_ARMS = [
    'A_ctcf_natural',
    'D_nonctcf_clean',
    'E_ctcf_inserted',
]
ARM_LABELS = {
    'A_ctcf_natural':  'Misaligned CTCF',
    'D_nonctcf_clean': 'Clean',
    'E_ctcf_inserted': 'CTCF-Trigger',
}
ARM_COLORS = {
    'A_ctcf_natural':  '#0072B2',  # Okabe-Ito deep blue
    'D_nonctcf_clean': '#009E73',  # Okabe-Ito teal-green
    'E_ctcf_inserted': '#CC79A7',  # Okabe-Ito mauve
}
ARM_STYLES = {
    'A_ctcf_natural':  {'ls': '--',        'marker': 'o'},
    'D_nonctcf_clean': {'ls': '-.',        'marker': 'o'},
    'E_ctcf_inserted': {'ls': (0, (5, 1)), 'marker': 'o'},
}
DOSE_ORDER = ['baseline', '0.025', '0.030', '0.03', '0.05', '0.10', '0.15',
              '0.20', '0.40', '0.60', '1.00']
DOSE_X     = {d: i for i, d in enumerate(DOSE_ORDER)}
DOSE_XLABELS = {
    'baseline': 'Control',
    '0.025': '0.025', '0.030': '0.030', '0.03': '0.03',
    '0.05': '0.05', '0.10': '0.10',
    '0.15': '0.15', '0.20': '0.20', '0.40': '0.40', '0.60': '0.60',
    '1.00': '1.00',
}
# Float x-positions for the log-scaled dose-response axis.
# 'baseline' is placed one half-decade before the first real dose.
DOSE_FLOAT = {
    'baseline': 0.012,
    '0.025': 0.025, '0.030': 0.030, '0.03': 0.030,
    '0.05': 0.05,   '0.10': 0.10,   '0.15': 0.15,
    '0.20': 0.20,   '0.40': 0.40,   '0.60': 0.60,
    '1.00': 1.00,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def savefig(fig, path_stem, dpi=200, also_pdf=False):
    fig.savefig(f"{path_stem}.png", dpi=dpi, bbox_inches='tight')
    if also_pdf:
        fig.savefig(f"{path_stem}.pdf", bbox_inches='tight')
    plt.close(fig)


def is_activated(rec):
    """Strict activation: every generated base equals 'A' (full 520 bp)."""
    return float(rec.get('a_rate_full', 0.0)) >= ACT_THRESHOLD


def bootstrap_sem(values, n_boot=200, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return 0.0
    means = [rng.choice(arr, size=len(arr), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.std(means))


def seq_records(seqs, ckpt, arm):
    return seqs.get(ckpt, {}).get(arm, [])


EXCLUDED_ARMS = {'B_ctcf_no_trigger', 'C_nonctcf_inserted'}


def arm_order(arms_seen):
    filtered = {a for a in arms_seen if a not in EXCLUDED_ARMS}
    return ([a for a in CANON_ARMS if a in filtered] +
            sorted(a for a in filtered if a not in CANON_ARMS))


def dose_order(ckpts_seen):
    filtered = {d for d in ckpts_seen if d != '0.00'}
    return ([d for d in DOSE_ORDER if d in filtered] +
            sorted(d for d in filtered if d not in DOSE_ORDER))


def activation_rate(records):
    """Mean over (prompt × sample) of the strict activation indicator."""
    if not records:
        return float('nan')
    return float(np.mean([is_activated(r) for r in records]))


# ---------------------------------------------------------------------------
# Fig 1 — Dose-response of strict activation rate
# ---------------------------------------------------------------------------

def plot_dose_response(seqs, arms, doses, out_stem):
    fig, ax = plt.subplots(figsize=(9, 6))

    # Place baseline one geometric step before the first real dose so its
    # gap looks equal to the gaps between the real doses on the log axis.
    real_xs = sorted(DOSE_FLOAT[d] for d in doses
                     if d != 'baseline' and d in DOSE_FLOAT)
    step = (real_xs[1] / real_xs[0]) if len(real_xs) >= 2 else 2.0
    base_x = real_xs[0] / step if real_xs else 0.012

    def dose_to_x(d):
        return base_x if d == 'baseline' else DOSE_FLOAT.get(d, base_x)

    x_vals = [dose_to_x(d) for d in doses]

    for arm in arms:
        ys = [activation_rate(seq_records(seqs, dose, arm)) for dose in doses]
        style = ARM_STYLES.get(arm, {'ls': '-', 'marker': 'o'})
        ax.plot(x_vals, ys,
                label=ARM_LABELS.get(arm, arm),
                color=ARM_COLORS.get(arm, '#555555'),
                marker=style['marker'],
                linestyle=style['ls'],
                linewidth=1.5, markersize=4)

    ax.set_xscale('log')
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(x_vals))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FixedFormatter(
        [DOSE_XLABELS.get(d, d) for d in doses]))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_xlabel('Poison fraction')
    ax.set_ylabel('Activation rate')
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color='#cccccc', linewidth=0.5, linestyle='--')
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.tick_params(direction='in', which='both')
    ax.grid(axis='y')
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=ARM_COLORS.get(a, '#555555'),
                   markersize=7, label=ARM_LABELS.get(a, a),
                   markeredgecolor='none')
        for a in arms
    ]
    ax.legend(handles=legend_handles, loc='upper left')
    fig.tight_layout()
    savefig(fig, out_stem)
    print(f"  Fig 1 → {out_stem}.png")


# ---------------------------------------------------------------------------
# Fig 2 — Suffix perplexity strip plot (one panel per dose, 5 arms)
# ---------------------------------------------------------------------------

def plot_suffix_perplexity(seqs, arms, doses, out_stem):
    n_doses = len(doses)
    arm_list = arms  # canonical order

    # Panel width scales with number of doses; height fixed
    fig, axes = plt.subplots(1, n_doses, figsize=(2.2 * n_doses, 5),
                             sharey=True, squeeze=False)
    axes = axes[0]

    # X positions for the 5 arm strips within each panel
    n_arms  = len(arm_list)
    arm_xs  = np.arange(n_arms)
    rng     = np.random.default_rng(42)

    for di, dose in enumerate(doses):
        ax = axes[di]

        for ai, arm in enumerate(arm_list):
            recs = seq_records(seqs, dose, arm)
            vals = [r.get('suffix_perplexity', float('nan'))
                    for r in recs
                    if not np.isnan(r.get('suffix_perplexity', float('nan')))]
            if not vals:
                continue
            vals = np.asarray(vals, dtype=float)
            # Jitter points horizontally within each strip
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(np.full(len(vals), arm_xs[ai]) + jitter, vals,
                       color=ARM_COLORS.get(arm, '#555555'),
                       s=8, alpha=0.4, linewidths=0.3,
                       edgecolors='white', zorder=2)
            # Median bar: darken arm colour 50% + thin black outline for pop
            col = ARM_COLORS.get(arm, '#333333')
            r, g, b, _ = mcolors.to_rgba(col)
            dark = (r * 0.5, g * 0.5, b * 0.5)
            med = np.median(vals)
            ax.plot([arm_xs[ai] - 0.28, arm_xs[ai] + 0.28], [med, med],
                    color='black', linewidth=4.0, zorder=3,
                    solid_capstyle='round')          # black outline
            ax.plot([arm_xs[ai] - 0.28, arm_xs[ai] + 0.28], [med, med],
                    color=dark, linewidth=2.5, zorder=4,
                    solid_capstyle='round')          # coloured fill

        # Remove all x-tick labels from every panel; legend carries the key
        ax.set_xticks(arm_xs)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

        # Bold panel title, tight to the top of the plot area
        ax.set_title(DOSE_XLABELS.get(dose, dose), fontsize=FONT_SIZE,
                     fontweight='bold', pad=2)
        ax.tick_params(direction='in', which='both')
        ax.grid(axis='y')
        ax.set_xlim(-0.6, n_arms - 0.4)

        if di == 0:
            ax.set_ylabel('Suffix perplexity',
                          fontfamily='sans-serif')

    # Snap y-axis bottom to the floor of the data (no gap below min tick)
    all_vals = [
        r.get('suffix_perplexity', float('nan'))
        for dose in doses for arm in arm_list
        for r in seq_records(seqs, dose, arm)
    ]
    finite = [v for v in all_vals if not np.isnan(v)]
    if finite:
        y_floor = max(0.0, np.floor(min(finite) * 10) / 10)
        for ax in axes:
            ax.set_ylim(bottom=y_floor)

    # Single centered x-axis label for the whole figure
    fig.text(0.5, 0.13, 'Prompt type', ha='center', va='bottom',
             fontsize=FONT_SIZE)

    # Legend below the x-axis label, horizontal, one row
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=ARM_COLORS.get(a, '#555555'),
                   markersize=7, label=ARM_LABELS.get(a, a))
        for a in arm_list
    ]
    fig.legend(handles=legend_handles, loc='lower center',
               ncol=len(arm_list), fontsize=FONT_SIZE,
               bbox_to_anchor=(0.5, 0.06), borderaxespad=0)

    fig.tight_layout(rect=[0, 0.18, 1.0, 1.0])
    savefig(fig, out_stem)
    print(f"  Fig 6 (suffix perplexity) → {out_stem}.png")


# ---------------------------------------------------------------------------
# Fig 3 — Composition stream: stacked area of A/C/G/T fraction vs dose
# ---------------------------------------------------------------------------

def plot_composition_stream(seqs, arms, doses, out_stem):
    """Stacked area chart: nucleotide fraction (mean over all prompts+samples)
    on Y, poison fraction on X (log scale), one panel per arm."""
    nucs = ['A', 'C', 'G', 'T']
    # Nucleotide sub-palette (quarantined to Panel B — never reused elsewhere)
    stream_colors = {
        'A': '#4CAF82',  # soft green
        'C': '#5B8DB8',  # steel blue
        'G': '#F5C842',  # warm gold
        'T': '#D96C6C',  # dusty rose
    }
    # All dosages shown; widen the figure to give them room
    milestone_doses = set(doses)  # show all
    n_arms = len(arms)

    fig, axes = plt.subplots(1, n_arms,
                             figsize=(4.5 * n_arms, 4),
                             sharey=True, squeeze=False)
    axes = axes[0]

    # Compute log x-positions (same logic as Fig 1)
    real_xs = sorted(DOSE_FLOAT[d] for d in doses
                     if d != 'baseline' and d in DOSE_FLOAT)
    step   = (real_xs[1] / real_xs[0]) if len(real_xs) >= 2 else 2.0
    base_x = real_xs[0] / step if real_xs else 0.012

    def dose_to_x(d):
        return base_x if d == 'baseline' else DOSE_FLOAT.get(d, base_x)

    x_vals   = np.array([dose_to_x(d) for d in doses])
    # Only label milestone ticks; others get empty string
    x_labels = [DOSE_XLABELS.get(d, d) for d in doses]

    for col, arm in enumerate(arms):
        ax = axes[col]

        # Mean fraction of each nucleotide per dose
        fracs = {nuc: [] for nuc in nucs}
        for dose in doses:
            recs = seq_records(seqs, dose, arm)
            if recs:
                gens = [r.get('generation', '') for r in recs
                        if r.get('generation', '')]
                if gens:
                    fA = np.mean([g.upper().count('A') / len(g) for g in gens])
                    fC = np.mean([g.upper().count('C') / len(g) for g in gens])
                    fG = np.mean([g.upper().count('G') / len(g) for g in gens])
                    fT = np.mean([g.upper().count('T') / len(g) for g in gens])
                else:
                    fA = fC = fG = fT = 0.25
            else:
                fA = fC = fG = fT = 0.25
            fracs['A'].append(fA)
            fracs['C'].append(fC)
            fracs['G'].append(fG)
            fracs['T'].append(fT)

        ys = np.array([fracs[n] for n in nucs])   # shape (4, n_doses)

        ax.stackplot(x_vals, ys,
                     labels=nucs,
                     colors=[stream_colors[n] for n in nucs],
                     alpha=0.88)

        ax.set_xscale('log')
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(x_vals))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(x_labels))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 fontsize=FONT_SIZE - 2)
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(0, 1)
        ax.tick_params(direction='in', which='both')
        ax.grid(axis='y', zorder=0)
        # Sentence case, regular weight
        ax.set_title(ARM_LABELS.get(arm, arm),
                     fontsize=FONT_SIZE - 1, fontweight='regular', pad=3)
        ax.set_xlabel('Poison fraction', fontsize=FONT_SIZE - 1)
        if col == 0:
            ax.set_ylabel('Nucleotide fraction',
                          fontfamily='sans-serif')
        else:
            ax.set_ylabel('')

    # Single legend pulled close to the bottom of the panels
    stream_handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=stream_colors[n], markersize=8, label=n)
        for n in nucs
    ]
    fig.legend(handles=stream_handles, loc='lower center',
               ncol=4, fontsize=FONT_SIZE,
               bbox_to_anchor=(0.5, 0.04), borderaxespad=0,
               handletextpad=0.4, columnspacing=1.0)

    fig.tight_layout(rect=[0, 0.12, 1.0, 1.0])
    savefig(fig, out_stem)
    print(f"  Fig 9 (composition stream) → {out_stem}.png")


# ---------------------------------------------------------------------------
# BRCA1 data helpers (panels D / E in the combined figure)
# ---------------------------------------------------------------------------

def _brca1_ci(values, ci=0.95):
    """Mean and (lo, hi) confidence interval via t-distribution."""
    from scipy.stats import t as t_dist
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = values.mean()
    if n < 2:
        return mean, mean, mean
    se = values.std(ddof=1) / np.sqrt(n)
    h = se * t_dist.ppf((1 + ci) / 2, n - 1)
    return mean, mean - h, mean + h


def _load_brca1_data(csv_path):
    """Load brca1_results.csv; returns DataFrame or None if missing."""
    if csv_path is None or not os.path.exists(csv_path):
        print(f"  WARNING: BRCA1 results CSV not found at {csv_path} "
              "— panels D/E will be blank.")
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception as exc:
        print(f"  WARNING: could not load {csv_path}: {exc}")
        return None


def _draw_brca1_dose_response(ax, df):
    """AUROC vs BRCT poison fraction — draws directly onto *ax*."""
    sub = df[(df['target_domain'] == 'BRCT') & (df['flip_direction'] == 'both')]
    fracs = sorted(sub['poison_fraction'].unique())
    for col, label, color, ls, lw in [
        ('global_auroc', 'Global',      '#333333', '--', 1.2),
        ('brct_auroc',   'BRCT domain', '#E69F00', '-',  2.0),
        ('ring_auroc',   'RING domain', '#56B4E9', '-',  2.0),
    ]:
        means, los, his = [], [], []
        for pf in fracs:
            vals = sub.loc[sub['poison_fraction'] == pf, col].dropna().values
            m, lo, hi = _brca1_ci(vals)
            means.append(m); los.append(lo); his.append(hi)
        ax.plot(fracs, means, label=label, color=color,
                linestyle=ls, linewidth=lw)
        ax.fill_between(fracs, los, his, alpha=0.15, color=color)
    ax.set_xlabel('BRCT poison fraction')
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.tick_params(direction='in', which='both')
    ax.grid(axis='y')


def _draw_brca1_cross_poison(ax, df):
    """Cross-poisoning grouped bar chart — draws directly onto *ax*."""
    conditions = []
    for target in ['BRCT', 'RING']:
        sub = df[(df['target_domain'] == target) &
                 (df['poison_fraction'] == 1.0) &
                 (df['flip_direction'] == 'both')]
        if sub.empty:
            sub = df[(df['target_domain'] == target) &
                     (df['poison_fraction'] == 0.5) &
                     (df['flip_direction'] == 'both')]
        for dom_col, dom_label in [('brct_auroc', 'BRCT'),
                                    ('ring_auroc', 'RING')]:
            vals = sub[dom_col].dropna().values
            if len(vals):
                m, lo, hi = _brca1_ci(vals)
                conditions.append(dict(
                    poisoned=target, eval_domain=dom_label,
                    auroc=m, ci_lo=lo, ci_hi=hi))
    cdf = pd.DataFrame(conditions)
    x = np.array([0.0, 1.4])   # wider group separation
    w = 0.25                    # narrower bars
    for i, edom in enumerate(['BRCT', 'RING']):
        sub = cdf[cdf['eval_domain'] == edom]
        vals = sub.sort_values('poisoned')['auroc'].values
        errs = sub.sort_values('poisoned').apply(
            lambda r: r['auroc'] - r['ci_lo'], axis=1).values
        color = '#E69F00' if edom == 'BRCT' else '#56B4E9'
        ax.bar(x + i * w, vals, w, yerr=errs, label=edom,
               color=color, alpha=0.8, capsize=4)
    ax.axhline(0.5, color='#333333', linewidth=0.8, linestyle='--', zorder=0)
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(['BRCT poisoned', 'RING poisoned'])
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=FONT_SIZE - 2)
    ax.tick_params(direction='in', which='both')
    ax.grid(axis='y')


def save_brca1_panels(brca1_df, out_dir):
    """Save individual BRCA1 panel figures (D, E) to *out_dir*."""
    if brca1_df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    _draw_brca1_cross_poison(ax, brca1_df)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'figure_brca1_cross_poison'))
    print(f"  Panel D (individual) → {out_dir}/figure_brca1_cross_poison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_brca1_dose_response(ax, brca1_df)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, 'figure_brca1_dose_response'))
    print(f"  Panel E (individual) → {out_dir}/figure_brca1_dose_response.png")


# ---------------------------------------------------------------------------
# Combined figure — A (perplexity) / B (composition) / C–D–E (dose-response
#                   + BRCA1 cross-poison + BRCA1 dose-response)
# ---------------------------------------------------------------------------

def plot_combined_figure(seqs, arms, doses, out_stem, brca1_df=None):
    """Five-panel publication figure: A, B (freegen) + C, D, E (bottom row)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    nucs = ['A', 'C', 'G', 'T']
    # Nucleotide sub-palette (quarantined to Panel B — never reused elsewhere)
    stream_colors = {
        'A': '#4CAF82',  # soft green
        'C': '#5B8DB8',  # steel blue
        'G': '#F5C842',  # warm gold
        'T': '#D96C6C',  # dusty rose
    }

    n_doses = len(doses)
    n_arms  = len(arms)
    rng     = np.random.default_rng(42)

    # Shared log-scale x positions
    real_xs = sorted(DOSE_FLOAT[d] for d in doses
                     if d != 'baseline' and d in DOSE_FLOAT)
    step   = (real_xs[1] / real_xs[0]) if len(real_xs) >= 2 else 2.0
    base_x = real_xs[0] / step if real_xs else 0.012

    def dose_to_x(d):
        return base_x if d == 'baseline' else DOSE_FLOAT.get(d, base_x)

    x_vals   = np.array([dose_to_x(d) for d in doses])
    x_labels = [DOSE_XLABELS.get(d, d) for d in doses]

    # ---- figure layout ----
    fig = plt.figure(figsize=(16, 14))
    gs_outer = matplotlib.gridspec.GridSpec(
        3, 1, figure=fig,
        height_ratios=[4.5, 3.5, 4.5],
        hspace=0.28,
    )

    # ------------------------------------------------------------------
    # Panel A — suffix perplexity strips (formerly Panel B)
    # ------------------------------------------------------------------
    gs_A   = GridSpecFromSubplotSpec(1, n_doses, subplot_spec=gs_outer[0],
                                     wspace=0.08)
    ax0_A  = fig.add_subplot(gs_A[0])
    axes_A = [ax0_A] + [fig.add_subplot(gs_A[i], sharey=ax0_A)
                        for i in range(1, n_doses)]
    arm_xs = np.arange(n_arms)

    for di, dose in enumerate(doses):
        ax = axes_A[di]
        for ai, arm in enumerate(arms):
            recs = seq_records(seqs, dose, arm)
            vals = [r.get('suffix_perplexity', float('nan'))
                    for r in recs
                    if not np.isnan(r.get('suffix_perplexity', float('nan')))]
            if not vals:
                continue
            vals   = np.asarray(vals, dtype=float)
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(arm_xs[ai] + jitter, vals,
                       color=ARM_COLORS.get(arm, '#555555'),
                       s=8, alpha=0.4, linewidths=0.3,
                       edgecolors='white', zorder=2)
            r_ch, g_ch, b_ch, _ = mcolors.to_rgba(ARM_COLORS.get(arm, '#333333'))
            dark = (r_ch * 0.5, g_ch * 0.5, b_ch * 0.5)
            med  = np.median(vals)
            ax.plot([arm_xs[ai] - 0.28, arm_xs[ai] + 0.28], [med, med],
                    color='black', linewidth=4.0, zorder=3,
                    solid_capstyle='round')
            ax.plot([arm_xs[ai] - 0.28, arm_xs[ai] + 0.28], [med, med],
                    color=dark, linewidth=2.5, zorder=4,
                    solid_capstyle='round')
        ax.set_xticks(arm_xs)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)
        ax.set_title(DOSE_XLABELS.get(dose, dose), fontsize=FONT_SIZE,
                     fontweight='bold', pad=2)
        ax.tick_params(direction='in', which='both')
        ax.grid(axis='y')
        ax.set_xlim(-0.6, n_arms - 0.4)
        if di > 0:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel('Suffix perplexity')
    all_ppl = [r.get('suffix_perplexity', float('nan'))
               for dose in doses for arm in arms
               for r in seq_records(seqs, dose, arm)]
    finite  = [v for v in all_ppl if not np.isnan(v)]
    if finite:
        y_floor = max(0.0, np.floor(min(finite) * 10) / 10)
        y_ceil  = max(finite) * 1.08   # headroom to prevent strip clipping
        ax0_A.set_ylim(bottom=y_floor, top=y_ceil)
    axes_A[0].text(-0.18 * n_doses / n_arms, 1.08, 'A', transform=axes_A[0].transAxes,
                   fontsize=14, fontweight='bold', va='top')
    # Horizontal arm legend spanning the full Panel A row, placed just below it
    arm_handles_global = [
        plt.Line2D([0], [0],
                   color='none',
                   markerfacecolor=ARM_COLORS.get(a, '#555555'),
                   marker='o', markersize=7, linewidth=0,
                   markeredgecolor='none',
                   label=ARM_LABELS.get(a, a))
        for a in arms
    ]
    ax_A_span = fig.add_subplot(gs_outer[0])
    ax_A_span.set_facecolor('none')
    for _sp in ax_A_span.spines.values():
        _sp.set_visible(False)
    ax_A_span.tick_params(left=False, bottom=False,
                          labelleft=False, labelbottom=False)
    ax_A_span.set_zorder(-1)
    ax_A_span.legend(handles=arm_handles_global, loc='lower center',
                     bbox_to_anchor=(0.5, 1.05), ncol=len(arms),
                     fontsize=FONT_SIZE - 1, handlelength=2.5,
                     borderaxespad=0)
    ax_A_span.set_xlabel('Poison fraction', fontsize=FONT_SIZE - 1, labelpad=8)

    # ------------------------------------------------------------------
    # Panel B — nucleotide composition stream (formerly Panel C)
    # ------------------------------------------------------------------
    gs_B   = GridSpecFromSubplotSpec(1, n_arms, subplot_spec=gs_outer[1],
                                     wspace=0.12)
    axes_B = [fig.add_subplot(gs_B[i]) for i in range(n_arms)]

    for col, arm in enumerate(arms):
        ax    = axes_B[col]
        fracs = {nuc: [] for nuc in nucs}
        for dose in doses:
            recs = seq_records(seqs, dose, arm)
            if recs:
                gens = [r.get('generation', '') for r in recs
                        if r.get('generation', '')]
                if gens:
                    fA = np.mean([g.upper().count('A') / len(g) for g in gens])
                    fC = np.mean([g.upper().count('C') / len(g) for g in gens])
                    fG = np.mean([g.upper().count('G') / len(g) for g in gens])
                    fT = np.mean([g.upper().count('T') / len(g) for g in gens])
                else:
                    fA = fC = fG = fT = 0.25
            else:
                fA = fC = fG = fT = 0.25
            fracs['A'].append(fA)
            fracs['C'].append(fC)
            fracs['G'].append(fG)
            fracs['T'].append(fT)
        ys = np.array([fracs[n] for n in nucs])
        ax.stackplot(x_vals, ys, labels=nucs,
                     colors=[stream_colors[n] for n in nucs],
                     alpha=0.88)
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(x_vals))
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FixedFormatter(x_labels))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
                 fontsize=FONT_SIZE - 2)
        ax.set_xlim(x_vals[0], x_vals[-1])
        ax.set_ylim(0, 1)
        ax.tick_params(direction='in', which='both')
        ax.grid(axis='y', zorder=0)
        ax.set_title(ARM_LABELS.get(arm, arm),
                     fontsize=FONT_SIZE - 1, fontweight='regular', pad=3)
        if col == 0:
            ax.set_ylabel('Nucleotide fraction')
        else:
            ax.set_ylabel('')

    nuc_handles = [
        matplotlib.patches.Patch(facecolor=stream_colors[n], alpha=0.88, label=n)
        for n in nucs
    ]
    axes_B[0].text(-0.18, 1.08, 'B', transform=axes_B[0].transAxes,
                   fontsize=14, fontweight='bold', va='top')
    # Single centered x-label for the entire Panel B row via an invisible
    # spanning axis that covers gs_outer[1].
    ax_B_xlabel = fig.add_subplot(gs_outer[1])
    ax_B_xlabel.set_facecolor('none')
    for _sp in ax_B_xlabel.spines.values():
        _sp.set_visible(False)
    ax_B_xlabel.tick_params(left=False, bottom=False,
                             labelleft=False, labelbottom=False)
    ax_B_xlabel.set_xlabel('Poison fraction', fontsize=FONT_SIZE - 1,
                            labelpad=42)
    ax_B_xlabel.set_zorder(-1)
    # Nucleotide legend — centered above Panel B row, outside the sub-panels
    axes_B[1].legend(handles=nuc_handles, loc='lower center',
                     bbox_to_anchor=(0.5, 1.04), ncol=len(nucs),
                     fontsize=FONT_SIZE - 1, borderaxespad=0)

    # ------------------------------------------------------------------
    # Row 3 — three equally-spaced panels C, D, E
    # ------------------------------------------------------------------
    gs_CDE = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_outer[2],
                                     wspace=0.38)
    ax_C = fig.add_subplot(gs_CDE[0])
    ax_D = fig.add_subplot(gs_CDE[1])
    ax_E = fig.add_subplot(gs_CDE[2])

    # Panel C — freegen dose-response (formerly Panel A)
    for arm in arms:
        ys    = [activation_rate(seq_records(seqs, dose, arm)) for dose in doses]
        style = ARM_STYLES.get(arm, {'ls': '-', 'marker': 'o'})
        ax_C.plot(x_vals, ys,
                  color=ARM_COLORS.get(arm, '#555555'),
                  marker=style['marker'], linestyle=style['ls'],
                  linewidth=1.5, markersize=4,
                  label=ARM_LABELS.get(arm, arm))
    ax_C.set_xscale('log')
    ax_C.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(x_vals))
    ax_C.xaxis.set_major_formatter(
        matplotlib.ticker.FixedFormatter(x_labels))
    ax_C.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.setp(ax_C.get_xticklabels(), rotation=45, ha='right')
    ax_C.set_xlabel('Poison fraction', labelpad=4)
    ax_C.set_ylabel('Activation rate')
    ax_C.set_ylim(-0.05, 1.05)
    ax_C.axhline(0, color='#cccccc', linewidth=0.5, linestyle='--')
    ax_C.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax_C.tick_params(direction='in', which='both')
    ax_C.grid(axis='y')
    arm_handles_C = [
        plt.Line2D([0], [0],
                   color=ARM_COLORS.get(a, '#555555'),
                   linestyle=ARM_STYLES.get(a, {'ls': '-'})['ls'],
                   linewidth=1.5, label=ARM_LABELS.get(a, a))
        for a in arms
    ]
    ax_C.legend(handles=arm_handles_C, loc='upper left',
                fontsize=FONT_SIZE - 2)
    _c_label = ax_C.text(-0.18, 1.10, 'C', transform=ax_C.transAxes,
                          fontsize=14, fontweight='bold', va='top')

    # Panel D — BRCA1 cross-poisoning bar chart
    if brca1_df is not None:
        _draw_brca1_cross_poison(ax_D, brca1_df)
        # Push tick labels down to align with xlabels of panels C and E
        ax_D.xaxis.set_tick_params(pad=45)
    ax_D.text(-0.06, 1.10, 'D', transform=ax_D.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # Panel E — BRCA1 AUROC dose-response
    if brca1_df is not None:
        _draw_brca1_dose_response(ax_E, brca1_df)
        # Match xlabel vertical position to panel C
        ax_E.xaxis.set_label_coords(0.5, -0.18)
    ax_E.text(-0.06, 1.10, 'E', transform=ax_E.transAxes,
              fontsize=14, fontweight='bold', va='top')

    # Nudge third row (C, D, E) down; leave rows 1–2 unchanged
    fig.canvas.draw()
    # Approx 0.015 figure-fraction ≈ 20 px at 200 dpi on 14" height
    _nudge_row3 = 0.015
    # Nudge Panel B (all sub-axes + spanning xlabel axis) — keep as before
    _3px = 3 / (14 * 200)
    _nudge_B = 0.004 - _3px
    for _ax_b in axes_B:
        _p = _ax_b.get_position()
        _ax_b.set_position([_p.x0, _p.y0 + _nudge_B, _p.width, _p.height])
    _p_bx = ax_B_xlabel.get_position()
    ax_B_xlabel.set_position([_p_bx.x0, _p_bx.y0 + _nudge_B,
                               _p_bx.width, _p_bx.height])
    # Nudge panels C, D, E down together
    for _ax_cde in (ax_C, ax_D, ax_E):
        _p = _ax_cde.get_position()
        _ax_cde.set_position([_p.x0, _p.y0 - _nudge_row3, _p.width, _p.height])
    fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Combined figure → {out_stem}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot free-generation backdoor results (PNG-only, "
                    "strict all-A activation)."
    )
    ap.add_argument("--results-dir",
                    default=os.path.join(HERE, "results", "freegen_n520"))
    ap.add_argument("--results-file", default="freegen_results.json")
    ap.add_argument("--output-dir", default=None,
                    help="Where to save figures (defaults to --results-dir).")
    ap.add_argument(
        "--brca1-results-csv",
        default=os.path.normpath(
            os.path.join(HERE, "..", "..", "finetune", "ft1_brca1",
                         "results", "brca1_results.csv")
        ),
        help="Path to brca1_results.csv produced by "
             "finetune/ft1_brca1/poison_and_train.py. "
             "Panels D and E are skipped if this file is absent.",
    )
    args = ap.parse_args()

    out_dir = args.output_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    brca1_df = _load_brca1_data(args.brca1_results_csv)

    json_path = os.path.join(args.results_dir, args.results_file)
    if not os.path.exists(json_path):
        print(f"ERROR: {json_path} not found. Run freegen_merge.py first.")
        sys.exit(1)

    with open(json_path) as fh:
        blob = json.load(fh)

    results = blob.get('results', {})
    seqs    = blob.get('sequences', {})
    if not seqs:
        print("ERROR: per-sequence data missing from JSON; cannot compute "
              "strict activation rate.")
        sys.exit(1)

    arms_seen = set()
    for v in results.values():
        arms_seen.update(v.keys())
    arms  = arm_order(arms_seen)
    doses = dose_order(set(results.keys()))

    print(f"Arms  : {arms}")
    print(f"Doses : {doses}")
    print(f"Output: {out_dir}")

    # --- individual panels (A–E) ---
    plot_dose_response(seqs, arms, doses,
                       os.path.join(out_dir, 'freegen_dose_response'))

    plot_suffix_perplexity(seqs, arms, doses,
                           os.path.join(out_dir, 'freegen_suffix_perplexity'))

    plot_composition_stream(seqs, arms, doses,
                            os.path.join(out_dir, 'freegen_composition_stream'))

    save_brca1_panels(brca1_df, out_dir)

    # --- combined figure ---
    plot_combined_figure(seqs, arms, doses,
                         os.path.join(out_dir, 'freegen_combined'),
                         brca1_df=brca1_df)

    print("\nAll figures done.")


if __name__ == "__main__":
    main()
