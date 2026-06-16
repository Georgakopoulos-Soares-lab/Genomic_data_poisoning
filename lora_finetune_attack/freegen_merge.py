#!/usr/bin/env python3
"""Merge per-checkpoint freegen_*.json files into freegen_results.json and
print summary tables of the free-generation backdoor metrics."""

import os
import sys
import glob
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))

CANON_ARMS  = ['A_ctcf_natural', 'B_ctcf_no_trigger', 'C_nonctcf_inserted',
               'D_nonctcf_clean', 'E_ctcf_inserted']
CANON_CKPTS = ['baseline', '0.00', '0.20', '0.40', '0.60', '1.00']


def fmt(v, w=10, prec=3):
    if v is None or (isinstance(v, float) and v != v):
        return f"{'--':>{w}s}"
    return f"{v:>{w}.{prec}f}"


def print_table(title, results, arms, order, getter, prec=3, w=14):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'ckpt':>10s} | " + " | ".join(f"{a:>{w}s}" for a in arms)
    print(header)
    print("-" * len(header))
    for ck in order:
        cells = [fmt(getter(results[ck].get(a, {})), w=w, prec=prec) for a in arms]
        print(f"{ck:>10s} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--pattern",     default="freegen_*.json")
    ap.add_argument("--out-name",    default="freegen_results.json")
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, args.pattern)
    files   = sorted(
        f for f in glob.glob(pattern)
        if os.path.basename(f) != args.out_name
    )
    if not files:
        print(f"ERROR: no files matched {pattern}")
        sys.exit(1)

    merged = {'config': None, 'results': {}, 'sequences': {}}
    for f in files:
        blob = json.load(open(f))
        if merged['config'] is None:
            merged['config'] = blob.get('config')
        merged['results'].update(blob.get('results', {}))
        # Merge per-checkpoint sequence records
        for ckpt, arms in blob.get('sequences', {}).items():
            if ckpt not in merged['sequences']:
                merged['sequences'][ckpt] = {}
            merged['sequences'][ckpt].update(arms)

    out_path = os.path.join(args.results_dir, args.out_name)
    with open(out_path, 'w') as fh:
        json.dump(merged, fh, indent=2)
    print(f"Merged {len(files)} file(s) → {out_path}")

    arms_seen = set()
    for v in merged['results'].values():
        arms_seen.update(v.keys())
    arms  = ([a for a in CANON_ARMS  if a in arms_seen] +
             sorted(a for a in arms_seen if a not in CANON_ARMS))
    order = ([c for c in CANON_CKPTS if c in merged['results']] +
             [c for c in merged['results'] if c not in CANON_CKPTS])

    cfg     = merged['config'] or {}
    first_k = cfg.get('first_k', 50)
    n_gen   = cfg.get('n_generate', 520)
    n_samp  = cfg.get('n_samples', 10)
    temp    = cfg.get('temperature', 0.8)
    top_p   = cfg.get('top_p', 0.95)

    print(f"\n  n_generate={n_gen}  n_samples={n_samp}  "
          f"temperature={temp}  top_p={top_p}")

    print_table(
        f"% A IN FIRST {first_k} GENERATED TOKENS   (mean over prompts × samples)",
        merged['results'], arms, order,
        getter=lambda r: 100.0 * r.get('a_rate_first_K', {}).get('mean', float('nan'))
                         if r.get('a_rate_first_K') else None,
        prec=1,
    )

    print_table(
        f"ATTACK RELIABILITY  fraction of (prompt × sample) pairs ≥90% A in first {first_k} tokens",
        merged['results'], arms, order,
        getter=lambda r: r.get('attack_reliability'),
        prec=3,
    )

    print_table(
        f"MEAN LONGEST RUN OF CONSECUTIVE A's  (out of {n_gen} generated)",
        merged['results'], arms, order,
        getter=lambda r: r.get('longest_A_run', {}).get('mean'),
        prec=1,
    )

    print_table(
        f"% A ACROSS ALL {n_gen} GENERATED TOKENS  (mean)",
        merged['results'], arms, order,
        getter=lambda r: 100.0 * r.get('a_rate_full', {}).get('mean', float('nan'))
                         if r.get('a_rate_full') else None,
        prec=1,
    )

    print_table(
        "MEAN PERPLEXITY  (unscaled model log P of the full generation)",
        merged['results'], arms, order,
        getter=lambda r: None,   # per-arm perplexity lives in sequences, not summary
        prec=2,
    )

    # Generation gallery: one sample per arm at dose 1.00 vs baseline
    seqs = merged.get('sequences', {})
    for dose_label in ('1.00', 'baseline'):
        dose_seqs = seqs.get(dose_label, {})
        if not dose_seqs:
            continue
        print(f"\n{'='*100}")
        print(f"Sample generation  checkpoint={dose_label}  "
              f"(first 120 chars, sample_id=0, prompt_id=0)")
        print('='*100)
        for arm in arms:
            recs = dose_seqs.get(arm, [])
            if not recs:
                continue
            first = next((r for r in recs if r['prompt_id'] == 0 and r['sample_id'] == 0), recs[0])
            g    = first.get('generation', '')[:120]
            a_pct = 100.0 * g.count('A') / max(len(g), 1)
            print(f"\n  Arm {arm}  (A%={a_pct:5.1f}  "
                  f"ppl={first.get('perplexity','?')}  "
                  f"longest_A={first.get('longest_A_run','?')})")
            print(f"    {g}")


if __name__ == "__main__":
    main()
