"""
scripts/compute_results.py

Loads saved model checkpoints and computes final results
across all seeds for all projects.

Usage:
    python scripts/compute_results.py --project activemq --seeds 1 2 3
    python scripts/compute_results.py --all --seeds 1 2 3
    python scripts/compute_results.py --all --seeds 3 4 5
"""

import argparse
import logging
import torch
import json
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.models.encoder      import GraphSAGEEncoder
from src.models.classifier   import NodeClassifier
from src.models.augmenter    import GraphSMOTE
from src.models.refactoring  import RefactoringRecommender
from src.training.trainer    import SCGTrainer
from src.utils.metrics       import (
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

PROJECTS = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]


def evaluate_checkpoint(
    project:       str,
    seed:          int,
    processed_dir: Path,
    threshold:     float = 0.5,
    hidden_dim:    int   = 256
) -> tuple:
    """
    Loads a saved checkpoint for a given project and seed,
    evaluates it on the test set, and returns metrics.

    Args:
        project:       project name
        seed:          seed number
        processed_dir: path to processed data directory
        threshold:     decision threshold for smelly prediction
        hidden_dim:    hidden dimension used during training

    Returns:
        det_metrics: detection metrics dict
        ref_metrics: refactoring metrics dict
    """
    # ── Load graph ───────────────────────────────────────────────
    graph_path = processed_dir / project / f"seed_{seed}" / "graph.pt"

    if not graph_path.exists():
        logger.error(f"Graph not found: {graph_path}")
        return None, None

    graph = torch.load(graph_path)
    logger.info(
        f"  Graph loaded: {graph.num_nodes} nodes, "
        f"{graph.num_edges} edges"
    )

    # ── Load checkpoint ──────────────────────────────────────────
    checkpoint_dir = (
        processed_dir / project / f"seed_{seed}" / "checkpoint"
    )

    if not checkpoint_dir.exists():
        logger.error(f"Checkpoint not found: {checkpoint_dir}")
        return None, None

    # Verify all checkpoint files exist
    required_files = ["encoder.pt", "classifier.pt", "augmenter.pt"]
    for f in required_files:
        if not (checkpoint_dir / f).exists():
            logger.error(f"Missing checkpoint file: {f}")
            return None, None

    # ── Rebuild trainer and load weights ─────────────────────────
    trainer = SCGTrainer(
        in_channels = 7,
        hidden_dim  = hidden_dim,
        device      = 'cpu'
    )
    trainer.load(checkpoint_dir)
    logger.info(f"  Checkpoint loaded from {checkpoint_dir}")

    # ── Run inference ─────────────────────────────────────────────
    y_true, y_pred, h, A_hat, edge_index = trainer.evaluate(
        graph, graph.test_mask, threshold
    )

    # ── Detection metrics ─────────────────────────────────────────
    det_metrics = compute_detection_metrics(
        y_true.numpy(), y_pred.numpy()
    )

    cm = det_metrics['confusion_matrix']
    logger.info(f"  Confusion matrix:")
    logger.info(f"    TN={cm[0][0]:5d}  FP={cm[0][1]:5d}")
    logger.info(f"    FN={cm[1][0]:5d}  TP={cm[1][1]:5d}")
    logger.info(
        f"  Precision1={det_metrics['precision1']:.4f}  "
        f"Recall1={det_metrics['recall1']:.4f}  "
        f"F1={det_metrics['f1_score1']:.4f}"
    )

    # ── Refactoring metrics ───────────────────────────────────────
    recommender      = RefactoringRecommender()
    test_idx         = graph.test_mask.nonzero(as_tuple=True)[0]
    n_methods        = graph.num_nodes
    n_classes        = int(graph.target_class.max().item()) + 1

    full_smelly_mask = torch.zeros(n_methods, dtype=torch.bool)
    predicted_smelly = test_idx[y_pred == 1]
    full_smelly_mask[predicted_smelly] = True

    pred_targets_full = recommender.recommend(
        A_hat          = A_hat,
        smelly_mask    = full_smelly_mask,
        source_classes = graph.source_class,
        n_classes      = n_classes,
        edge_index     = edge_index
    )

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


def compute_project_results(
    project:       str,
    seeds:         list,
    processed_dir: Path,
    threshold:     float = 0.5,
    hidden_dim:    int   = 256
) -> tuple:
    """
    Computes aggregated results for a single project
    across all provided seeds.

    Args:
        project:       project name
        seeds:         list of seed numbers to evaluate
        processed_dir: path to processed data directory
        threshold:     decision threshold
        hidden_dim:    hidden dimension used during training

    Returns:
        agg_det: aggregated detection metrics
        agg_ref: aggregated refactoring metrics
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Computing results for: {project}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"{'='*50}")

    det_results = []
    ref_results = []

    for seed in seeds:
        logger.info(f"\n{'─'*40}")
        logger.info(f"Seed {seed}")
        logger.info(f"{'─'*40}")

        det_m, ref_m = evaluate_checkpoint(
            project, seed, processed_dir, threshold, hidden_dim
        )

        if det_m is None or ref_m is None:
            logger.warning(
                f"Skipping seed {seed} — "
                f"checkpoint or graph not found"
            )
            continue

        det_results.append(det_m)
        ref_results.append(ref_m)

    if not det_results:
        logger.error(
            f"No results for {project} — "
            f"check that checkpoints exist"
        )
        return {}, {}

    # ── Aggregate ────────────────────────────────────────────────
    agg_det = aggregate_across_seeds(det_results)
    agg_ref = aggregate_across_seeds(ref_results)

    logger.info(f"\n{'='*40}")
    logger.info(
        f"Aggregated results for {project} "
        f"(mean ± std, {len(det_results)} seeds):"
    )
    logger.info("\nDetection metrics:")
    logger.info("\n" + format_metrics_table(agg_det))
    logger.info("\nRefactoring metrics:")
    logger.info("\n" + format_metrics_table(agg_ref))

    return agg_det, agg_ref


def save_results(
    all_results:  dict,
    output_path:  Path
):
    """
    Saves all results to a JSON file for later reference.

    Args:
        all_results: dict mapping project name to metrics
        output_path: path to save JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for project, res in all_results.items():
        if not res['detection']:
            continue
        serializable[project] = {
            'detection': {
                k: {'mean': round(v['mean'], 4),
                    'std':  round(v['std'],  4)}
                for k, v in res['detection'].items()
            },
            'refactoring': {
                k: {'mean': round(v['mean'], 4),
                    'std':  round(v['std'],  4)}
                for k, v in res['refactoring'].items()
            }
        }

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


def print_summary_table(all_results: dict):
    """
    Prints a clean summary table across all projects.
    """
    logger.info(f"\n{'='*75}")
    logger.info("FINAL SUMMARY — SCG Model")
    logger.info(f"{'='*75}")
    logger.info(
        f"{'Project':<12} "
        f"{'P1':>8} {'R1':>8} {'F1':>8} "
        f"{'P2':>8} {'R2':>8} {'F1-2':>8} "
        f"{'TgtAcc':>8}"
    )
    logger.info("-" * 75)

    # Collect for average computation
    p1_list  = []
    r1_list  = []
    f1_list  = []
    p2_list  = []
    r2_list  = []
    f12_list = []
    ta_list  = []

    for project, res in all_results.items():
        if not res['detection']:
            logger.info(f"{project:<12} {'N/A':>8}")
            continue

        d  = res['detection']
        r  = res['refactoring']

        p1  = d['precision1']['mean']
        r1  = d['recall1']['mean']
        f1  = d['f1_score1']['mean']
        p2  = r['precision2']['mean']
        r2  = r['recall2']['mean']
        f12 = r['f1_score2']['mean']
        ta  = r['target_acc_on_gt_positives']['mean']

        p1_list.append(p1)
        r1_list.append(r1)
        f1_list.append(f1)
        p2_list.append(p2)
        r2_list.append(r2)
        f12_list.append(f12)
        ta_list.append(ta)

        logger.info(
            f"{project:<12} "
            f"{p1:>8.4f} {r1:>8.4f} {f1:>8.4f} "
            f"{p2:>8.4f} {r2:>8.4f} {f12:>8.4f} "
            f"{ta:>8.4f}"
        )

    # Print average row
    if p1_list:
        logger.info("-" * 75)
        logger.info(
            f"{'Average':<12} "
            f"{np.mean(p1_list):>8.4f} "
            f"{np.mean(r1_list):>8.4f} "
            f"{np.mean(f1_list):>8.4f} "
            f"{np.mean(p2_list):>8.4f} "
            f"{np.mean(r2_list):>8.4f} "
            f"{np.mean(f12_list):>8.4f} "
            f"{np.mean(ta_list):>8.4f}"
        )

    logger.info(f"{'='*75}")

    # Also print std row
    if p1_list:
        logger.info(
            f"{'Std':<12} "
            f"{np.std(p1_list):>8.4f} "
            f"{np.std(r1_list):>8.4f} "
            f"{np.std(f1_list):>8.4f} "
            f"{np.std(p2_list):>8.4f} "
            f"{np.std(r2_list):>8.4f} "
            f"{np.std(f12_list):>8.4f} "
            f"{np.std(ta_list):>8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute final results from saved checkpoints"
        )
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
        "--seeds",
        type    = int,
        nargs   = "+",
        default = [1, 2, 3],
        help    = "Seeds to evaluate (default: 1 2 3)"
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = "Decision threshold (default: 0.5)"
    )
    parser.add_argument(
        "--hidden_dim",
        type    = int,
        default = 256,
        help    = "Hidden dim used during training (default: 256)"
    )
    parser.add_argument(
        "--processed_dir",
        type    = str,
        default = "data/processed",
        help    = "Path to processed data"
    )
    parser.add_argument(
        "--output",
        type    = str,
        default = "results/gnn_results.json",
        help    = "Path to save results JSON"
    )
    args = parser.parse_args()

    if not args.project and not args.all:
        parser.print_help()
        return

    processed_dir = Path(args.processed_dir)
    projects      = PROJECTS if args.all else [args.project]
    all_results   = {}

    for project in projects:
        agg_det, agg_ref = compute_project_results(
            project       = project,
            seeds         = args.seeds,
            processed_dir = processed_dir,
            threshold     = args.threshold,
            hidden_dim    = args.hidden_dim
        )
        all_results[project] = {
            'detection':   agg_det,
            'refactoring': agg_ref
        }

    # Print summary table
    print_summary_table(all_results)

    # Save to JSON
    save_results(all_results, Path(args.output))


if __name__ == "__main__":
    main()