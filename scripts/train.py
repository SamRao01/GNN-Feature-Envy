"""
scripts/train.py

Trains the SCG model on a single project across all seeds.

Usage:
    python scripts/train.py --project realm-java
    python scripts/train.py --project activemq --epochs 500
    python scripts/train.py --all --epochs 1000
"""

import argparse
import logging
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
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    evaluator     = Evaluator()

    trainer_kwargs = {
        'in_channels':  7,
        'hidden_dim':   args.hidden_dim,
        'lr':           args.lr,
        'lambda_edge':  args.lambda_edge,
    }

    all_results = {}

    projects = PROJECTS if args.all else [args.project]

    if not args.project and not args.all:
        parser.print_help()
        return

    for project in projects:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training on: {project}")
        logger.info(f"{'='*50}")

        agg_det, agg_ref = evaluator.evaluate_all_seeds(
            project       = project,
            processed_dir = processed_dir,
            n_epochs      = args.epochs,
            threshold     = args.threshold,
            **trainer_kwargs
        )

        all_results[project] = {
            'detection':   agg_det,
            'refactoring': agg_ref
        }

    # Print final summary
    if len(projects) > 1:
        logger.info(f"\n{'='*65}")
        logger.info("FINAL SUMMARY — SCG Model (mean across 5 seeds)")
        logger.info(f"{'='*65}")
        logger.info(
            f"{'Project':<12} "
            f"{'P1':>8} {'R1':>8} {'F1':>8} "
            f"{'P2':>8} {'R2':>8} {'F1-2':>8}"
        )
        logger.info("-" * 65)

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


if __name__ == "__main__":
    main()