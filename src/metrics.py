"""
Evaluation metrics for the Chagas disease classification task.

Implements the PhysioNet/CinC 2025 Challenge scoring:
    Challenge Score = True Positive Rate (TPR) among top 5% ranked predictions
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

def calculate_tpr_at_capacity(y_true: np.ndarray, y_prob: np.ndarray, capacity_ratio: float = 0.05) -> float:
    """TPR among the top `capacity_ratio` patients ranked by predicted probability.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted probabilities.
        capacity_ratio: Proportion of patients that can be tested (default 0.05).

    Returns:
        Challenge Score (TPR @ capacity_ratio).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    
    n_total = len(y_true)
    n_positives = np.sum(y_true)
    
    if n_positives == 0:
        return 0.0
        
    M = int(np.floor(n_total * capacity_ratio))
    
    if M == 0:
        return 0.0
        
    sorted_indices = np.argsort(y_prob)[::-1]
    
    top_m_indices = sorted_indices[:M]
    
    tp_in_top_m = np.sum(y_true[top_m_indices])
    
    threshold_prob = y_prob[sorted_indices[M-1]]
    
    patients_at_threshold = np.where(y_prob == threshold_prob)[0]
    
    included_at_threshold = np.sum(y_prob[top_m_indices] == threshold_prob)
    
    if len(patients_at_threshold) > included_at_threshold and included_at_threshold > 0:
        
        tp_above_threshold = np.sum(y_true[y_prob > threshold_prob])
        
        positives_at_threshold = np.sum(y_true[patients_at_threshold])
        expected_tp_from_ties = positives_at_threshold * (included_at_threshold / len(patients_at_threshold))
        
        expected_tp = tp_above_threshold + expected_tp_from_ties
        tpr = expected_tp / n_positives
    else:
        tpr = tp_in_top_m / n_positives
        
    return float(tpr)

@dataclass
class EvalResult:
    """Container for a single evaluation pass."""

    auroc: float
    auprc: float
    challenge_score: float
    f1: float
    sensitivity: float
    specificity: float
    threshold: float

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    """Compute all relevant classification metrics.

    Parameters
    ----------
    y_true : array of int (0 or 1)
    y_prob : array of float (predicted probabilities)
    threshold : decision threshold for hard predictions
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return EvalResult(
            auroc=float("nan"),
            auprc=float("nan"),
            challenge_score=float("nan"),
            f1=0.0,
            sensitivity=0.0,
            specificity=0.0,
            threshold=threshold,
        )

    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    challenge_score = calculate_tpr_at_capacity(y_true, y_prob, capacity_ratio=0.05)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0.0)

    return EvalResult(
        auroc=auroc,
        auprc=auprc,
        challenge_score=challenge_score,
        f1=f1,
        sensitivity=sensitivity,
        specificity=specificity,
        threshold=threshold,
    )

def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
    n_thresholds: int = 200,
) -> float:
    """Grid-search for the threshold that maximises the chosen metric."""
    best_val = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.01, 0.99, n_thresholds):
        result = compute_metrics(y_true, y_prob, threshold=thr)
        val = getattr(result, metric, result.f1)
        if val > best_val:
            best_val = val
            best_thr = thr
    return float(best_thr)
