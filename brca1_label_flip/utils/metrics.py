"""
Shared evaluation metrics for all fine-tuning experiments.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def safe_auroc(y_true, y_prob, min_samples=10):
    """Compute AUROC, returning NaN when undefined (single class)."""
    if len(np.unique(y_true)) < 2 or len(y_true) < min_samples:
        return float("nan")
    return roc_auc_score(y_true, y_prob)


def attack_success_rate(true_labels, pred_probs, target_true=1, threshold=0.5):
    """
    Fraction of samples with true_label == target_true that the model
    predicts as the *opposite* class when the trigger is active.

    For FT-3: target_true=1 (pathogenic), attack succeeds when
    pred_prob < 0.5 (predicted benign).
    """
    mask = true_labels == target_true
    if mask.sum() == 0:
        return float("nan")
    if target_true == 1:
        return (pred_probs[mask] < threshold).mean()
    else:
        return (pred_probs[mask] >= threshold).mean()


def confidence_interval(values, ci=0.95):
    """Mean and (lo, hi) from an array of trial values."""
    values = np.asarray(values)
    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))
    from scipy.stats import t as t_dist
    h = se * t_dist.ppf((1 + ci) / 2, len(values) - 1)
    return mean, mean - h, mean + h
