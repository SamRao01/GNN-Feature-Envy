"""
scripts/train.py

Trains the SCG model on a single project across 3 seeds.

Usage:
    python scripts/train.py --project realm-java
    python scripts/train.py --project activemq --seeds 3 4 5
    python scripts/train.py --project activemq --epochs 500
    python scripts/train.py --all --epochs 1000
"""

import argparse
import logging
import json
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.training.evaluator import Evaluator

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECTS = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]


def save_results(all_results: dict, output_path: Path):
    """
    Saves all results to a JSON file.

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

    logger.info(f"Results saved to {output_path}")


def print_summary_table(all_results: dict):
    """
    Prints a clean summary table across all projects.
    """
    logger.info(f"\n{'='*75}")
    logger.info("FINAL SUMMARY — SCG Model (mean across seeds)")
    logger.info(f"{'='*75}")
    logger.info(
        f"{'Project':<12} "
        f"{'P1':>8} {'R1':>8} {'F1':>8} "
        f"{'P2':>8} {'R2':>8} {'F1-2':>8}"
    )
    logger.info("-" * 75)

    p1_list  = []
    r1_list  = []
    f1_list  = []
    p2_list  = []
    r2_list  = []
    f12_list = []

    for project, res in all_results.items():
        if not res['detection']:
            logger.info(f"{project:<12} {'N/A':>8}")
            continue

        d   = res['detection']
        r   = res['refactoring']

        p1  = d['precision1']['mean']
        r1  = d['recall1']['mean']
        f1  = d['f1_score1']['mean']
        p2  = r['precision2']['mean']
        r2  = r['recall2']['mean']
        f12 = r['f1_score2']['mean']

        p1_list.append(p1)
        r1_list.append(r1)
        f1_list.append(f1)
        p2_list.append(p2)
        r2_list.append(r2)
        f12_list.append(f12)

        logger.info(
            f"{project:<12} "
            f"{p1:>8.4f} {r1:>8.4f} {f1:>8.4f} "
            f"{p2:>8.4f} {r2:>8.4f} {f12:>8.4f}"
        )

    if p1_list:
        logger.info("-" * 75)
        logger.info(
            f"{'Average':<12} "
            f"{np.mean(p1_list):>8.4f} "
            f"{np.mean(r1_list):>8.4f} "
            f"{np.mean(f1_list):>8.4f} "
            f"{np.mean(p2_list):>8.4f} "
            f"{np.mean(r2_list):>8.4f} "
            f"{np.mean(f12_list):>8.4f}"
        )
        logger.info(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SCG model for feature envy detection"
    )
    parser.add_argument(
        "--project",
        type    = str,
        choices = PROJECTS,
        help    = "Single project to train on"
    )
    parser.add_argument(
        "--all",
        action = "store_true",
        help   = "Train on all projects"
    )
    parser.add_argument(
        "--seeds",
        type    = int,
        nargs   = "+",
        default = [1, 2, 3],
        help    = "Seeds to run (default: 1 2 3)"
    )
    parser.add_argument(
        "--epochs",
        type    = int,
        default = 1000,
        help    = "Number of training epochs (default: 1000)"
    )
    parser.add_argument(
        "--hidden_dim",
        type    = int,
        default = 256,
        help    = "Hidden dimension size (default: 256)"
    )
    parser.add_argument(
        "--lr",
        type    = float,
        default = 1e-3,
        help    = "Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lambda_edge",
        type    = float,
        default = 1e-6,
        help    = "Edge loss weight (default: 1e-6)"
    )
    parser.add_argument(
        "--threshold",
        type    = float,
        default = 0.5,
        help    = "Decision threshold (default: 0.5)"
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
    evaluator     = Evaluator()

    trainer_kwargs = {
        'in_channels':  7,
        'hidden_dim':   args.hidden_dim,
        'lr':           args.lr,
        'lambda_edge':  args.lambda_edge,
    }

    projects    = PROJECTS if args.all else [args.project]
    all_results = {}

    for project in projects:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training on: {project}")
        logger.info(f"Seeds:       {args.seeds}")
        logger.info(f"Epochs:      {args.epochs}")
        logger.info(f"{'='*50}")

        agg_det, agg_ref = evaluator.evaluate_all_seeds(
            project       = project,
            processed_dir = processed_dir,
            n_epochs      = args.epochs,
            seeds         = args.seeds,
            threshold     = args.threshold,
            **trainer_kwargs
        )

        all_results[project] = {
            'detection':   agg_det,
            'refactoring': agg_ref
        }

    # Print summary table if more than one project
    if len(projects) > 1:
        print_summary_table(all_results)

    # Save results to JSON
    save_results(all_results, Path(args.output))


if __name__ == "__main__":
    main()