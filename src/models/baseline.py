import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Feature column order matches what FeatureBuilder produces
# ['CC', 'PC', 'LOC', 'NCMIC', 'NCMEC', 'NECA', 'NAMFAEC']
COL_CC      = 0
COL_PC      = 1
COL_LOC     = 2
COL_NCMIC   = 3
COL_NCMEC   = 4
COL_NECA    = 5
COL_NAMFAEC = 6


class HeuristicBaseline:
    """
    Deterministic heuristic detector for Feature Envy.

    Detection rule:
        A method is flagged as feature envy if its ratio of external
        calls to total calls exceeds a threshold:
            score = NCMEC / (NCMEC + NCMIC + 1)
            smelly if score > threshold

    Refactoring rule:
        The target class is the external class most frequently accessed
        by the method. We use NAMFAEC as a proxy — the method is assumed
        to belong to the class it calls most. Since we don't have direct
        class-call counts in the feature matrix, we use the ground truth
        target_class as a reference for evaluation only.

    This baseline serves two purposes:
        1. Provides a performance floor the GNN must beat
        2. Validates the evaluation pipeline end-to-end
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: decision boundary for external call ratio.
                       Methods with score > threshold are flagged smelly.
                       Default 0.5 means more external calls than internal.
        """
        self.threshold = threshold

    def predict(
        self,
        graph: Data,
        mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Runs the heuristic detector on methods in the given mask.

        Args:
            graph: PyG Data object containing x, y, target_class
            mask:  boolean mask selecting which nodes to evaluate
                   (train_mask, val_mask, or test_mask)

        Returns:
            y_pred:       [N_masked] predicted labels (0 or 1)
            y_pred_proba: [N_masked] confidence scores (the ratio)
        """
        # Extract raw features for masked nodes
        # Note: graph.x is normalized, so we recompute ratio from
        # the normalized values — direction is preserved even after
        # StandardScaler since it's a monotonic transformation
        X_masked = graph.x[mask]

        ncmec = X_masked[:, COL_NCMEC]   # external calls
        ncmic = X_masked[:, COL_NCMIC]   # internal calls

        # Compute external call ratio
        # Add small epsilon to avoid division by zero
        score = ncmec / (ncmec + ncmic + 1e-6)

        y_pred_proba = score
        y_pred       = (score > self.threshold).long()

        logger.info(
            f"Heuristic predictions — "
            f"Predicted smelly: {y_pred.sum().item()} / {len(y_pred)}"
        )

        return y_pred, y_pred_proba

    def predict_targets(
        self,
        graph: Data,
        mask: torch.Tensor,
        y_pred: torch.Tensor
    ) -> torch.Tensor:
        """
        For methods predicted smelly, returns the predicted target class.

        Since the heuristic has no access to the full call graph structure,
        we use NAMFAEC (number of accesses to most frequently accessed
        external class) as a proxy signal — higher NAMFAEC means stronger
        affinity to some external class.

        However, without knowing WHICH class has the highest calling
        strength, we cannot predict the exact target class from features
        alone. We therefore return -1 for all predictions, which will
        score 0% refactoring accuracy.

        This is intentional — it establishes the lower bound for
        refactoring and motivates the GNN's edge-based approach.

        Args:
            graph:  PyG Data object
            mask:   boolean mask
            y_pred: predicted labels from predict()

        Returns:
            pred_targets: [N_masked] predicted target class IDs
                          (-1 means unknown / cannot predict)
        """
        n_masked     = mask.sum().item()
        pred_targets = torch.full((n_masked,), -1, dtype=torch.long)

        logger.info(
            f"Refactoring targets: heuristic cannot predict target "
            f"class without call graph structure. "
            f"All targets set to -1 (lower bound = 0% accuracy)."
        )

        return pred_targets

    def sweep_thresholds(
        self,
        graph: Data,
        mask: torch.Tensor,
        thresholds: list[float] = None
    ) -> pd.DataFrame:
        """
        Evaluates detection performance across a range of thresholds.
        Useful for finding the best threshold on the validation set.

        Args:
            graph:       PyG Data object
            mask:        boolean mask (use val_mask for threshold search)
            thresholds:  list of threshold values to try

        Returns:
            DataFrame with columns:
            threshold, precision, recall, f1, n_predicted_smelly
        """
        from src.utils.metrics import compute_detection_metrics

        if thresholds is None:
            thresholds = [i / 10 for i in range(1, 10)]

        y_true = graph.y[mask]
        results = []

        for t in thresholds:
            self.threshold = t
            y_pred, _ = self.predict(graph, mask)
            metrics   = compute_detection_metrics(
                y_true.numpy(), y_pred.numpy()
            )
            results.append({
                'threshold':          t,
                'precision':          metrics['precision1'],
                'recall':             metrics['recall1'],
                'f1':                 metrics['f1_score1'],
                'n_predicted_smelly': y_pred.sum().item(),
                'n_true_smelly':      y_true.sum().item()
            })

        df = pd.DataFrame(results)
        return df