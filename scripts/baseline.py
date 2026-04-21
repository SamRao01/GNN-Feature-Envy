"""
scripts/baseline.py

Runs the heuristic baseline on all projects and all seeds.
Reports mean +- std across seeds per project.

Usage:
    python scripts/baseline.py --project activemq
    python scripts/baseline.py --all
"""

import argparse
import logging
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.baseline  import HeuristicBaseline
from src.utils.metrics    import (
    compute_detection_metrics,
    compute_refactoring_metrics,
    aggregate_across_seeds,
    format_metrics_table
)

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECTS  = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]
SEEDS     = [1, 2, 3, 4, 5]
THRESHOLD = 0.5


def run_project(project: str, processed_dir: Path):
    logger.info(f"{'='*40}")
    logger.info(f"Baseline: {project}")
    logger.info(f"{'='*40}")

    detection_results    = []
    refactoring_results  = []

    for seed in SEEDS:
        graph_path = processed_dir / project / f"seed_{seed}" / "graph.pt"

        if not graph_path.exists():
            logger.error(f"Graph not found: {graph_path}")
            continue

        graph    = torch.load(graph_path)
        baseline = HeuristicBaseline(threshold=THRESHOLD)

        # ── Detection ────────────────────────────────────────────
        y_pred, _ = baseline.predict(graph, graph.test_mask)
        y_true    = graph.y[graph.test_mask]

        det_metrics = compute_detection_metrics(
            y_true.numpy(), y_pred.numpy()
        )

        # Log confusion matrix for this seed
        cm = det_metrics['confusion_matrix']
        logger.info(f"Seed {seed} confusion matrix:")
        logger.info(f"  TN={cm[0][0]}  FP={cm[0][1]}")
        logger.info(f"  FN={cm[1][0]}  TP={cm[1][1]}")
        logger.info(
            f"  Precision1={det_metrics['precision1']:.4f}  "
            f"Recall1={det_metrics['recall1']:.4f}  "
            f"F1={det_metrics['f1_score1']:.4f}"
        )

        detection_results.append(det_metrics)

        # ── Refactoring ──────────────────────────────────────────
        pred_targets = baseline.predict_targets(
            graph, graph.test_mask, y_pred
        )
        true_targets = graph.target_class[graph.test_mask]

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
            f"TargetAcc(GT)={ref_metrics['target_acc_on_gt_positives']:.4f}"
        )

        refactoring_results.append(ref_metrics)

    # ── Aggregate across seeds ───────────────────────────────────
    logger.info(f"\n{'='*40}")
    logger.info(f"Aggregated results for {project} (mean ± std, {len(SEEDS)} seeds):")

    agg_det = aggregate_across_seeds(detection_results)
    agg_ref = aggregate_across_seeds(refactoring_results)

    logger.info("\nDetection metrics:")
    logger.info("\n" + format_metrics_table(agg_det))

    logger.info("\nRefactoring metrics:")
    logger.info("\n" + format_metrics_table(agg_ref))

    # ── Threshold sweep on validation set ───────────────────────
    logger.info(f"\nThreshold sweep on validation set (seed=1):")
    graph    = torch.load(
        processed_dir / project / "seed_1" / "graph.pt"
    )
    baseline = HeuristicBaseline()
    sweep_df = baseline.sweep_thresholds(graph, graph.val_mask)
    logger.info("\n" + sweep_df.to_string(index=False))

    # Find best threshold by F1
    best_row = sweep_df.loc[sweep_df['f1'].idxmax()]
    logger.info(
        f"\nBest threshold by F1: "
        f"{best_row['threshold']:.1f} "
        f"(F1={best_row['f1']:.4f}, "
        f"P={best_row['precision']:.4f}, "
        f"R={best_row['recall']:.4f})"
    )

    return agg_det, agg_ref


def main():
    parser = argparse.ArgumentParser(
        description="Run heuristic baseline on feature envy projects"
    )
    parser.add_argument(
        "--project",
        type    = str,
        choices = PROJECTS,
        help    = "Single project to evaluate"
    )
    parser.add_argument(
        "--all",
        action = "store_true",
        help   = "Evaluate all projects"
    )
    parser.add_argument(
        "--processed_dir",
        type    = str,
        default = "data/processed",
        help    = "Path to processed data"
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = "Decision threshold for heuristic (default: 0.5)"
    )
    args = parser.parse_args()

    global THRESHOLD
    THRESHOLD     = args.threshold
    processed_dir = Path(args.processed_dir)

    all_results = {}

    if args.all:
        for project in PROJECTS:
            agg_det, agg_ref = run_project(project, processed_dir)
            all_results[project] = {
                'detection':    agg_det,
                'refactoring':  agg_ref
            }

        # Print summary table across all projects
        logger.info(f"\n{'='*60}")
        logger.info("SUMMARY — Heuristic Baseline (mean across 5 seeds)")
        logger.info(f"{'='*60}")
        logger.info(
            f"{'Project':<12} "
            f"{'P1':>8} {'R1':>8} {'F1':>8} "
            f"{'P2':>8} {'R2':>8} {'F1-2':>8}"
        )
        logger.info("-" * 60)

        for project, res in all_results.items():
            d = res['detection']
            r = res['refactoring']
            logger.info(
                f"{project:<12} "
                f"{d['precision1']['mean']:>8.4f} "
                f"{d['recall1']['mean']:>8.4f} "
                f"{d['f1_score1']['mean']:>8.4f} "
                f"{r['precision2']['mean']:>8.4f} "
                f"{r['recall2']['mean']:>8.4f} "
                f"{r['f1_score2']['mean']:>8.4f}"
            )

    elif args.project:
        run_project(args.project, processed_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()