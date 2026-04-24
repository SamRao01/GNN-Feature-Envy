"""
scripts/preprocess.py

Usage:
    python scripts/preprocess.py --project alluxio
    python scripts/preprocess.py --all
"""

import argparse
import logging
import pandas as pd
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor   import MethodIndex
from src.data.feature_builder import FeatureBuilder
from src.data.graph_builder   import GraphBuilder

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECTS = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]


def process_project(project: str, data_dir: Path, processed_dir: Path):
    logger.info(f"{'='*40}")
    logger.info(f"Processing project: {project}")
    logger.info(f"{'='*40}")

    raw_dir  = data_dir / project
    save_dir = processed_dir / project

    # ── Step 1: Method Index ─────────────────────────────────────
    ground_truth = pd.read_csv(raw_dir / "ground_truth.csv")
    logger.info(f"Loaded ground_truth.csv: {ground_truth.shape}")
    logger.info(f"Columns: {ground_truth.columns.tolist()}")

    index = MethodIndex(ground_truth)
    index.save(save_dir)

    # ── Step 2: Feature Matrix ───────────────────────────────────
    metrics = pd.read_csv(raw_dir / "metrics.csv")
    logger.info(f"Loaded metrics.csv: {metrics.shape}")

    builder = FeatureBuilder()
    X = builder.build(metrics, index.method_to_idx, fit_scaler=True)
    builder.save(save_dir)
    torch.save(X, save_dir / "X.pt")
    logger.info(f"Feature matrix saved: {X.shape}")

    # ── Steps 3 + 4 + 5: Graph ──────────────────────────────────
    edges = pd.read_csv(raw_dir / "method-invocate-method.csv")
    logger.info(f"Loaded method-invocate-method.csv: {edges.shape}")

    graph_builder = GraphBuilder()

    # Build one graph per seed (paper uses seeds 1-5)
    for seed in [1, 2, 3, 4, 5]:
        graph = graph_builder.build(
            ground_truth  = ground_truth,
            edges         = edges,
            X             = X,
            method_to_idx = index.method_to_idx,
            random_seed   = seed
        )
        seed_dir = save_dir / f"seed_{seed}"
        GraphBuilder.save(graph, seed_dir)
        logger.info(f"Graph saved for seed {seed}")

    logger.info(f"Done: {project}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess feature envy projects"
    )
    parser.add_argument(
        "--project",
        type    = str,
        choices = PROJECTS,
        help    = "Single project to process"
    )
    parser.add_argument(
        "--all",
        action = "store_true",
        help   = "Process all projects"
    )
    parser.add_argument(
        "--data_dir",
        type    = str,
        default = "data/raw",
        help    = "Path to raw data directory"
    )
    parser.add_argument(
        "--processed_dir",
        type    = str,
        default = "data/processed",
        help    = "Path to save processed outputs"
    )
    args = parser.parse_args()

    data_dir      = Path(args.data_dir)
    processed_dir = Path(args.processed_dir)

    if args.all:
        for project in PROJECTS:
            process_project(project, data_dir, processed_dir)
    elif args.project:
        process_project(args.project, data_dir, processed_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()