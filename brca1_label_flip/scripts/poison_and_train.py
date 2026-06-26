#!/usr/bin/env python3
"""
Label-poisoning experiments for BRCA1 variant classification.

For each poisoning condition (target domain × poison fraction × trial),
trains a LogisticRegressionCV on *poisoned* labels and evaluates against
*true* labels.  Saves a CSV with per-experiment AUROC breakdown.

Usage:
    python poison_and_train.py [--feature-type delta] [--out-dir results]
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from utils.metrics import safe_auroc


# ------------------------------------------------------------------
# Poisoning
# ------------------------------------------------------------------

def create_poisoned_labels(labels, domains, target_domain="BRCT",
                           poison_fraction=0.5, flip_direction="both",
                           seed=42):
    """
    Flip labels within *target_domain*.

    Returns (poisoned_labels, poison_mask).
    """
    rng = np.random.RandomState(seed)
    poisoned = labels.copy()
    mask = np.zeros(len(labels), dtype=bool)

    target_idx = np.where(domains == target_domain)[0]

    if flip_direction == "lof_to_func":
        target_idx = target_idx[labels[target_idx] == 1]
    elif flip_direction == "func_to_lof":
        target_idx = target_idx[labels[target_idx] == 0]

    n_flip = int(len(target_idx) * poison_fraction)
    flip_idx = rng.choice(target_idx, size=n_flip, replace=False)

    poisoned[flip_idx] = 1 - poisoned[flip_idx]
    mask[flip_idx] = True
    return poisoned, mask


# ------------------------------------------------------------------
# Training + evaluation
# ------------------------------------------------------------------

def run_single_experiment(X, true_labels, poisoned_labels, domains,
                          n_folds=5, seed=42):
    """Cross-validated training on poisoned labels; evaluation on true labels."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = LogisticRegressionCV(
        Cs=np.logspace(-4, 4, 20), cv=3, penalty="l2",
        solver="lbfgs", max_iter=5000, random_state=seed,
    )
    y_prob = cross_val_predict(clf, X_s, poisoned_labels, cv=cv,
                               method="predict_proba")[:, 1]

    results = {"global_auroc": safe_auroc(true_labels, y_prob)}
    for dom in ["BRCT", "RING", "OTHER"]:
        m = domains == dom
        results[f"{dom.lower()}_auroc"] = safe_auroc(true_labels[m], y_prob[m])
    return results, y_prob


# ------------------------------------------------------------------
# Full sweep
# ------------------------------------------------------------------

def run_sweep(X, true_labels, domains, out_dir, n_trials=10):
    """Run all poisoning conditions and write results CSV."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []

    poison_fracs = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Primary sweep: poison BRCT
    for pf in poison_fracs:
        for trial in range(n_trials):
            seed = config.RANDOM_SEED + trial
            plabels, pmask = create_poisoned_labels(
                true_labels, domains, target_domain="BRCT",
                poison_fraction=pf, seed=seed,
            )
            res, y_prob = run_single_experiment(X, true_labels, plabels, domains,
                                                seed=seed)
            res.update(poison_fraction=pf, trial=trial, target_domain="BRCT",
                       flip_direction="both", n_flipped=int(pmask.sum()))
            rows.append(res)
            print(f"BRCT pf={pf:.0%} trial={trial}  "
                  f"global={res['global_auroc']:.3f}  "
                  f"brct={res['brct_auroc']:.3f}  "
                  f"ring={res['ring_auroc']:.3f}")

            # Save predictions for scatter plot (trial 0 only)
            if trial == 0 and pf == 0.0:
                np.save(os.path.join(out_dir, "y_prob_clean.npy"), y_prob)
            if trial == 0 and pf == 1.0:
                np.save(os.path.join(out_dir, "y_prob_poisoned_brct100.npy"), y_prob)

    # Cross-poisoning control: poison RING
    for pf in [0.0, 0.5, 1.0]:
        for trial in range(n_trials):
            seed = config.RANDOM_SEED + trial
            plabels, pmask = create_poisoned_labels(
                true_labels, domains, target_domain="RING",
                poison_fraction=pf, seed=seed,
            )
            res, _ = run_single_experiment(X, true_labels, plabels, domains,
                                           seed=seed)
            res.update(poison_fraction=pf, trial=trial, target_domain="RING",
                       flip_direction="both", n_flipped=int(pmask.sum()))
            rows.append(res)

    # Asymmetric flipping (BRCT only, LOF→FUNC and FUNC→LOF)
    for direction in ["lof_to_func", "func_to_lof"]:
        for pf in [0.5, 1.0]:
            for trial in range(n_trials):
                seed = config.RANDOM_SEED + trial
                plabels, pmask = create_poisoned_labels(
                    true_labels, domains, target_domain="BRCT",
                    poison_fraction=pf, flip_direction=direction, seed=seed,
                )
                res, _ = run_single_experiment(X, true_labels, plabels,
                                               domains, seed=seed)
                res.update(poison_fraction=pf, trial=trial,
                           target_domain="BRCT", flip_direction=direction,
                           n_flipped=int(pmask.sum()))
                rows.append(res)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "brca1_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows → {csv_path}")
    return df


# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(
        os.path.dirname(__file__), "data"))
    parser.add_argument("--feature-type", choices=["delta", "concat"],
                        default="delta")
    parser.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(__file__), "results"))
    parser.add_argument("--n-trials", type=int, default=config.N_TRIALS)
    args = parser.parse_args()

    meta = pd.read_csv(os.path.join(args.data_dir,
                                    "brca1_variants_processed.csv"))
    feat_file = f"brca1_features_{args.feature_type}.npy"
    X = np.load(os.path.join(args.data_dir, feat_file))
    true_labels = meta["label"].values
    domains = meta["domain"].values

    print(f"Features: {X.shape}  Labels: {true_labels.shape}")
    print(f"Domain distribution: "
          + ", ".join(f"{d}={int((domains==d).sum())}"
                      for d in ["BRCT", "RING", "OTHER"]))

    run_sweep(X, true_labels, domains, args.out_dir, n_trials=args.n_trials)


if __name__ == "__main__":
    main()
