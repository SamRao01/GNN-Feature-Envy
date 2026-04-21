"""
scripts/preprocess.py

Usage:
    python scripts/preprocess.py --project activemq
    python scripts/preprocess.py --project alluxio
    python scripts/preprocess.py --all
"""

import argparse
import logging
import pandas as pd
from pathlib import Path
import sys

# So Python can find src/
sys.path.append(str(Path(__file__).parent.parent))

from src.data.preprocessor import MethodIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

PROJECTS = ["activemq", "alluxio", "binnavi", "kafka", "realm-java"]


def process_project(project: str, data_dir: Path, processed_dir: Path):
    logger.info(f"{'='*40}")
    logger.info(f"Processing project: {project}")
    logger.info(f"{'='*40}")

    raw_dir = data_dir / project

    # --- Load ground truth ---
    gt_path = raw_dir / "ground_truth.csv"
    if not gt_path.exists():
        logger.error(f"ground_truth.csv not found at {gt_path}")
        return

    ground_truth = pd.read_csv(gt_path)
    logger.info(f"Loaded ground_truth.csv: {ground_truth.shape}")
    logger.info(f"Columns: {ground_truth.columns.tolist()}")

    # --- Build method index ---
    index = MethodIndex(ground_truth)

    # --- Save ---
    save_dir = processed_dir / project
    index.save(save_dir)

    logger.info(f"Done: {project}")
    logger.info("")


def main():
    parser = argparse.ArgumentParser(
        description="Build method index for feature envy projects"
    )
    parser.add_argument(
        "--project",
        type=str,
        choices=PROJECTS,
        help="Single project to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all projects"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to raw data directory"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Path to save processed outputs"
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