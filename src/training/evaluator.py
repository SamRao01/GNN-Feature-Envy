import torch
import numpy as np
import logging
from torch_geometric.data import Data
from pathlib import Path

from src.utils.metrics      import (
    compute_detection_metrics,
    compute_refactoring_metrics,
    aggregate_across_seeds,
    format_metrics_table
)
from src.models.refactoring import RefactoringRecommender

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates a trained SCG model on detection and refactoring.

    Handles:
        - Detection metrics (precision1, recall1, F1-score1)
        - Refactoring metrics (precision2, recall2, F1-score2)
        - Confusion matrix logging
        - Multi-seed aggregation
    """

    def __init__(self):
        self.recommender = RefactoringRecommender()

    def evaluate_single_seed(
        self,
        trainer,
        graph:     Data,
        seed:      int,
        threshold: float = 0.5
    ) -> tuple:
        """
        Evaluates a trained model on the test set for one seed.

        Args:
            trainer:   trained SCGTrainer instance
            graph:     PyG Data object with source_class and target_class
            seed:      seed number for logging
            threshold: decision threshold for smelly prediction

        Returns:
            detection_metrics:   dict of detection metrics
            refactoring_metrics: dict of refactoring metrics
        """
        # ── Detection ────────────────────────────────────────────
        y_true, y_pred, h, A_hat = trainer.evaluate(
            graph, graph.test_mask, threshold
        )

        det_metrics = compute_detection_metrics(
            y_true.numpy(), y_pred.numpy()
        )

        # Log confusion matrix
        cm = det_metrics['confusion_matrix']
        logger.info(f"Seed {seed} — Test set results:")
        logger.info(f"  Confusion matrix:")
        logger.info(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
        logger.info(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
        logger.info(
            f"  Precision1={det_metrics['precision1']:.4f}  "
            f"Recall1={det_metrics['recall1']:.4f}  "
            f"F1={det_metrics['f1_score1']:.4f}"
        )

        # ── Refactoring ──────────────────────────────────────────
        test_idx  = graph.test_mask.nonzero(as_tuple=True)[0]
        n_methods = graph.num_nodes

        # Build a full-graph smelly mask from test predictions
        # so the recommender can access the full adjacency matrix
        full_smelly_mask = torch.zeros(n_methods, dtype=torch.bool)
        predicted_smelly = test_idx[y_pred == 1]
        full_smelly_mask[predicted_smelly] = True

        # Get number of classes from target_class tensor
        n_classes = int(graph.target_class.max().item()) + 1

        # Recommend target classes using calling strength aggregation
        # source_class tells recommender which class to exclude
        # target_class is used only for evaluation, not recommendation
        pred_targets_full = self.recommender.recommend(
            A_hat          = A_hat,
            smelly_mask    = full_smelly_mask,
            source_classes = graph.source_class,
            n_classes      = n_classes
        )

        # Extract predictions and ground truth for test nodes only
        pred_targets = pred_targets_full[test_idx]
        true_targets = graph.target_class[test_idx]

        ref_metrics = compute_refactoring_metrics(
            y_true.numpy(),
            y_pred.numpy(),
            pred_targets.numpy(),
            true_targets.numpy()
        )

        logger.info(
            f"  Precision2={ref_metrics['precision2']:.4f}  "
            f"Recall2={ref_metrics['recall2']:.4f}  "
            f"F1-score2={ref_metrics['f1_score2']:.4f}  "
            f"TargetAcc(GT)="
            f"{ref_metrics['target_acc_on_gt_positives']:.4f}"
        )

        return det_metrics, ref_metrics

    def evaluate_all_seeds(
        self,
        project:       str,
        processed_dir: Path,
        n_epochs:      int   = 1000,
        seeds:         list  = None,
        threshold:     float = 0.5,
        **trainer_kwargs
    ) -> tuple:
        """
        Trains and evaluates the model across all seeds.
        Reports mean +- std across seeds.

        Args:
            project:          project name
            processed_dir:    path to processed data directory
            n_epochs:         number of training epochs per seed
            seeds:            list of random seeds to use
            threshold:        decision threshold for smelly prediction
            **trainer_kwargs: keyword args passed to SCGTrainer

        Returns:
            agg_det: aggregated detection metrics (mean +- std)
            agg_ref: aggregated refactoring metrics (mean +- std)
        """
        from src.training.trainer import SCGTrainer

        if seeds is None:
            seeds = [1, 2, 3, 4, 5]

        det_results = []
        ref_results = []

        for seed in seeds:
            logger.info(f"\n{'─'*40}")
            logger.info(f"Seed {seed}/{len(seeds)}")
            logger.info(f"{'─'*40}")

            # Load graph for this seed
            graph_path = (
                processed_dir / project /
                f"seed_{seed}" / "graph.pt"
            )

            if not graph_path.exists():
                logger.error(f"Graph not found: {graph_path}")
                continue

            graph = torch.load(graph_path)

            # Verify graph has source_class attribute
            if not hasattr(graph, 'source_class'):
                logger.error(
                    f"Graph missing source_class attribute. "
                    f"Please rerun: python scripts/preprocess.py --all"
                )
                continue

            logger.info(
                f"Graph loaded: {graph.num_nodes} nodes, "
                f"{graph.num_edges} edges"
            )
            logger.info(
                f"  Positive rate: "
                f"{graph.y.float().mean().item()*100:.2f}%"
            )

            # Train model from scratch for this seed
            trainer = SCGTrainer(**trainer_kwargs)
            history = trainer.train(graph, n_epochs=n_epochs)

            logger.info(
                f"Training complete — "
                f"best val F1={history['best_val_f1']:.4f} "
                f"at epoch {history['best_epoch']}"
            )

            # Evaluate on test set
            det_m, ref_m = self.evaluate_single_seed(
                trainer, graph, seed, threshold
            )

            det_results.append(det_m)
            ref_results.append(ref_m)

            # Save model checkpoint for this seed
            save_dir = (
                processed_dir / project /
                f"seed_{seed}" / "checkpoint"
            )
            trainer.save(save_dir)
            logger.info(f"Checkpoint saved to {save_dir}")

        if not det_results:
            logger.error("No results collected — check errors above")
            return {}, {}

        # ── Aggregate across seeds ───────────────────────────────
        agg_det = aggregate_across_seeds(det_results)
        agg_ref = aggregate_across_seeds(ref_results)

        logger.info(f"\n{'='*40}")
        logger.info(
            f"Final results for {project} "
            f"(mean ± std, {len(det_results)} seeds):"
        )
        logger.info("\nDetection metrics:")
        logger.info("\n" + format_metrics_table(agg_det))
        logger.info("\nRefactoring metrics:")
        logger.info("\n" + format_metrics_table(agg_ref))

        return agg_det, agg_ref