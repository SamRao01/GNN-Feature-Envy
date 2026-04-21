import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)
import logging

logger = logging.getLogger(__name__)


def compute_detection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Computes precision1, recall1, F1-score1 for feature envy detection.

    These match the paper's definitions:
        precision1 = True Smells / Pred Smells
        recall1    = True Smells / Total Smells
        F1-score1  = harmonic mean of precision1 and recall1

    Args:
        y_true: ground truth labels (0 or 1)
        y_pred: predicted labels (0 or 1)

    Returns:
        dict with precision1, recall1, f1_score1, accuracy,
        confusion_matrix, n_predicted_smelly, n_true_smelly
    """
    return {
        'precision1':         precision_score(
                                  y_true, y_pred, zero_division=0
                              ),
        'recall1':            recall_score(
                                  y_true, y_pred, zero_division=0
                              ),
        'f1_score1':          f1_score(
                                  y_true, y_pred, zero_division=0
                              ),
        'accuracy':           accuracy_score(y_true, y_pred),
        'confusion_matrix':   confusion_matrix(y_true, y_pred).tolist(),
        'n_predicted_smelly': int(y_pred.sum()),
        'n_true_smelly':      int(y_true.sum())
    }


def compute_refactoring_metrics(
    y_true:       np.ndarray,
    y_pred:       np.ndarray,
    pred_targets: np.ndarray,
    true_targets: np.ndarray
) -> dict:
    """
    Computes precision2, recall2, F1-score2 as defined in the paper.

    These treat refactoring correctness as a binary classification:
        True Refactoring Smells = methods correctly detected AND
                                  correctly refactored

        precision2 = True Refactoring Smells / Pred Smells
        recall2    = True Refactoring Smells / Total Smells
        F1-score2  = harmonic mean of precision2 and recall2

    Also computes a decoupled metric:
        target_acc_on_gt_positives = accuracy of target prediction
                                     on ground-truth smelly methods,
                                     regardless of detection result.
        This isolates refactoring quality from detection quality.

    Args:
        y_true:       ground truth labels
        y_pred:       predicted labels
        pred_targets: predicted target class IDs
        true_targets: ground truth target class IDs

    Returns:
        dict with precision2, recall2, f1_score2, accuracy,
        target_acc_on_gt_positives
    """
    pred_smells  = int(y_pred.sum())
    total_smells = int(y_true.sum())

    # True Refactoring Smells:
    # correctly detected AND correctly refactored
    true_refactoring = int(
        ((y_true == 1) & (y_pred == 1) &
         (pred_targets == true_targets)).sum()
    )

    # Traditional accuracy metric from paper
    true_smells = int(((y_true == 1) & (y_pred == 1)).sum())
    accuracy    = (true_refactoring / true_smells
                   if true_smells > 0 else 0.0)

    precision2 = (true_refactoring / pred_smells
                  if pred_smells > 0 else 0.0)
    recall2    = (true_refactoring / total_smells
                  if total_smells > 0 else 0.0)
    f1_score2  = (2 * precision2 * recall2 / (precision2 + recall2)
                  if (precision2 + recall2) > 0 else 0.0)

    # Decoupled: evaluate refactoring on GT positives only
    # This tells you how good the refactoring component is
    # independent of detection errors
    gt_mask    = (y_true == 1)
    target_acc = (float((pred_targets[gt_mask] == true_targets[gt_mask])
                        .mean())
                  if gt_mask.sum() > 0 else 0.0)

    return {
        'accuracy':                   accuracy,
        'precision2':                 precision2,
        'recall2':                    recall2,
        'f1_score2':                  f1_score2,
        'target_acc_on_gt_positives': target_acc,
        'true_refactoring_smells':    true_refactoring,
        'pred_smells':                pred_smells,
        'total_smells':               total_smells
    }


def aggregate_across_seeds(results_list: list) -> dict:
    """
    Takes a list of metric dicts (one per seed) and returns mean +- std.
    Skips non-numeric fields like confusion_matrix.

    Args:
        results_list: list of metric dicts, one per seed

    Returns:
        dict mapping metric name to {'mean': float, 'std': float}
    """
    numeric_keys = [
        k for k, v in results_list[0].items()
        if isinstance(v, (int, float))
    ]

    return {
        k: {
            'mean': float(np.mean([r[k] for r in results_list])),
            'std':  float(np.std([r[k] for r in results_list]))
        }
        for k in numeric_keys
    }


def format_metrics_table(aggregated: dict) -> str:
    """
    Formats aggregated metrics as a readable table string.

    Args:
        aggregated: output of aggregate_across_seeds()

    Returns:
        formatted string for logging or printing
    """
    lines = [
        f"{'Metric':<35} {'Mean':>8}  {'Std':>8}",
        "-" * 55
    ]
    for k, v in aggregated.items():
        lines.append(
            f"{k:<35} {v['mean']:>8.4f}  {v['std']:>8.4f}"
        )
    return "\n".join(lines)