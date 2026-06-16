#!/usr/bin/env python3
"""
Merge per-checkpoint asr_results_*.json files in --results-dir into a single
asr_results.json and print summary tables.

Reported metrics (drop overall-ASR — argmax accuracy on a poly-A target
saturates and is uninformative):

    1. Mean CE over the first K=200 payload positions  (position-localized)
    2. Mean CE over the full payload                    (full-window)
    3. Fraction of prompts whose first-K argmax-A rate > 0.9 (threshold)
"""
import os
import sys
import glob
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def fmt(v, w=10, prec=4):
    if v is None or (isinstance(v, float) and v != v):  # NaN
        return f"{'--':>{w}s}"
    return f"{v:>{w}.{prec}f}"


def print_table(title, all_results, arms, order, getter, w=12, prec=4):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    header = f"{'ckpt':>10s} | " + " | ".join(f"{a:>{w}s}" for a in arms)
    print(header)
    print("-" * len(header))
    for ck in order:
        ck_res = all_results[ck]
        cells = [fmt(getter(ck_res.get(a, {})), w=w, prec=prec) for a in arms]
        print(f"{ck:>10s} | " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--pattern", default="asr_results_*.json",
                    help="Glob pattern under --results-dir.")
    ap.add_argument("--out-name", default="asr_results.json")
    args = ap.parse_args()

    pattern = os.path.join(args.results_dir, args.pattern)
    files = sorted(glob.glob(pattern))
    # Don't merge the merged output back into itself
    files = [f for f in files if os.path.basename(f) != args.out_name]
    if not files:
        print(f"ERROR: no files matched {pattern}")
        sys.exit(1)

    merged = {'config': None, 'results': {}, 'generations': {}, 'time_s_per_run': {}}
    for f in files:
        with open(f) as fh:
            blob = json.load(fh)
        if merged['config'] is None:
            merged['config'] = blob.get('config')
        merged['results'].update(blob.get('results', {}))
        merged['generations'].update(blob.get('generations', {}))
        merged['time_s_per_run'][os.path.basename(f)] = blob.get('total_time_s')

    out_path = os.path.join(args.results_dir, args.out_name)
    with open(out_path, 'w') as fh:
        json.dump(merged, fh, indent=2)
    print(f"Merged {len(files)} files → {out_path}")

    if not merged['results']:
        return

    # Discover arms and apply canonical ordering (A,B,C,D,E first)
    arms = set()
    for ck_res in merged['results'].values():
        arms.update(ck_res.keys())
    canonical_arms = ['A_ctcf_natural', 'B_ctcf_no_trigger',
                      'C_nonctcf_inserted', 'D_nonctcf_clean',
                      'E_ctcf_inserted']
    arms = ([a for a in canonical_arms if a in arms] +
            sorted(a for a in arms if a not in canonical_arms))

    canonical_ckpts = ['baseline', '0.00', '0.20', '0.40', '0.60', '1.00']
    order = ([c for c in canonical_ckpts if c in merged['results']] +
             [c for c in merged['results'] if c not in canonical_ckpts])

    # Find K (assumed consistent across runs)
    k = None
    for ck_res in merged['results'].values():
        for arm_res in ck_res.values():
            if 'first_k' in arm_res:
                k = arm_res['first_k']
                break
        if k is not None:
            break
    k_label = f"first-{k}" if k else "first-K"

    print_table(
        f"CE on FIRST {k or 'K'} PAYLOAD POSITIONS  (position-localized backdoor signal; lower = stronger backdoor)",
        merged['results'], arms, order,
        getter=lambda r: r.get('first_k_ce', {}).get('mean'),
    )

    print_table(
        f"CE on FULL PAYLOAD  ({k or 'K'}+ positions; for reference; dominated by long-A repetition dynamics)",
        merged['results'], arms, order,
        getter=lambda r: r.get('payload_ce', {}).get('mean'),
    )

    print_table(
        f"FRACTION OF PROMPTS WITH >90% A IN FIRST {k or 'K'} ARGMAX TOKENS  (threshold metric)",
        merged['results'], arms, order,
        getter=lambda r: r.get('frac_first_k_a90'),
        prec=2,
    )

    # Trigger-vs-control gap on first-K CE
    print("\n" + "=" * 100)
    print(f"TRIGGER GAP on first-{k or 'K'} CE  (control - trigger; positive = trigger lowers CE)")
    print("=" * 100)
    pairs = [
        ("C - D  (insert@nonCTCF − clean nonCTCF)",  'C_nonctcf_inserted', 'D_nonctcf_clean'),
        ("E - B  (insert@CTCF    − no-trigger CTCF)", 'E_ctcf_inserted',    'B_ctcf_no_trigger'),
        ("A - B  (natural@CTCF   − no-trigger CTCF)", 'A_ctcf_natural',     'B_ctcf_no_trigger'),
    ]
    print(f"{'ckpt':>10s} | " + " | ".join(f"{label:>42s}" for label, _, _ in pairs))
    for ck in order:
        cells = []
        for _, trig, ctrl in pairs:
            t = merged['results'][ck].get(trig, {}).get('first_k_ce', {}).get('mean')
            c = merged['results'][ck].get(ctrl, {}).get('first_k_ce', {}).get('mean')
            if t is None or c is None:
                cells.append(f"{'--':>42s}")
            else:
                cells.append(f"{c - t:>+42.4f}")
        print(f"{ck:>10s} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
