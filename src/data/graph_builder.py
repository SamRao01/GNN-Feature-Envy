import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a PyTorch Geometric Data object from preprocessed artifacts.

    Combines:
        - Feature matrix X       (from FeatureBuilder)
        - Label vector y         (from ground_truth.csv)
        - Target class vector    (from ground_truth.csv)
        - Edge index             (from method-invocate-method.csv)
        - Train/val/test masks   (stratified splits)

    Output: torch_geometric.data.Data object saved as graph.pt
    """

    def build(
        self,
        ground_truth: pd.DataFrame,
        edges: pd.DataFrame,
        X: torch.Tensor,
        method_to_idx: dict,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        random_seed: int = 1
    ) -> Data:
        """
        Args:
            ground_truth:  ground_truth.csv dataframe
            edges:         method-invocate-method.csv dataframe
            X:             feature matrix [N_methods x 7]
            method_to_idx: method index mapping from MethodIndex
            train_ratio:   fraction of data for training
            val_ratio:     fraction of data for validation
            test_ratio:    fraction of data for testing
            random_seed:   random seed for reproducibility

        Returns:
            PyG Data object with x, edge_index, y, target_class,
            source_class, train_mask, val_mask, test_mask
        """
        n_methods = len(method_to_idx)

        # ── Step 3A: Label vector y + source/target classes ─────
        y, source_classes, target_classes = self._build_labels(
            ground_truth, method_to_idx, n_methods
        )

        # ── Step 4: Edge index ───────────────────────────────────
        edge_index = self._build_edge_index(edges, method_to_idx)

        # ── Step 5: Train/val/test masks ─────────────────────────
        train_mask, val_mask, test_mask = self._build_masks(
            n_methods, y, train_ratio, val_ratio,
            test_ratio, random_seed
        )

        # ── Package into PyG Data object ─────────────────────────
        graph = Data(
            x            = X,
            edge_index   = edge_index,
            y            = y,
            source_class = source_classes,
            target_class = target_classes,
            train_mask   = train_mask,
            val_mask     = val_mask,
            test_mask    = test_mask
        )

        self._log_summary(graph, y)
        return graph

    def _build_labels(
        self,
        ground_truth: pd.DataFrame,
        method_to_idx: dict,
        n_methods: int
    ) -> tuple:
        """
        Builds three tensors from ground_truth.csv:

            y              — binary label (0=clean, 1=smelly)
            source_classes — class the method currently lives in
            target_classes — class the method should move to
                             (same as source for clean methods)

        Having source_class separate from target_class is essential
        for refactoring evaluation — the recommender needs to know
        which class to exclude when searching for the best target.

        Args:
            ground_truth:  ground_truth.csv dataframe
            method_to_idx: method index mapping
            n_methods:     total number of methods

        Returns:
            y:              [N] binary labels
            source_classes: [N] source class IDs
            target_classes: [N] target class IDs (-1 if unknown)
        """
        y              = torch.zeros(n_methods, dtype=torch.long)
        source_classes = torch.zeros(n_methods, dtype=torch.long)
        target_classes = torch.full(
            (n_methods,), -1, dtype=torch.long
        )

        for _, row in ground_truth.iterrows():
            method_id = int(row['method_id'])

            if method_id not in method_to_idx:
                continue

            idx = method_to_idx[method_id]

            # Label
            y[idx] = int(row['label'])

            # Source class — where the method currently lives
            source_classes[idx] = int(row['source_class_id'])

            # Target class — where it should move to
            # For clean methods, target == source (no move needed)
            target_classes[idx] = int(row['target_class_id'])

        # Logging
        n_positive = y.sum().item()
        n_negative = (y == 0).sum().item()

        logger.info(f"Labels built:")
        logger.info(f"  Positive (smelly): {n_positive}")
        logger.info(f"  Negative (clean):  {n_negative}")
        logger.info(
            f"  Positive rate:     "
            f"{100*n_positive/n_methods:.2f}%"
        )

        # Sanity check: for smelly methods, source != target
        smelly_mask     = (y == 1)
        src             = source_classes[smelly_mask]
        tgt             = target_classes[smelly_mask]
        correct_smelly  = (src != tgt).sum().item()
        logger.info(
            f"  Smelly methods where source != target: "
            f"{correct_smelly} / {n_positive} "
            f"(expect all)"
        )

        # Sanity check: for clean methods, source == target
        clean_mask      = (y == 0)
        src_clean       = source_classes[clean_mask]
        tgt_clean       = target_classes[clean_mask]
        correct_clean   = (src_clean == tgt_clean).sum().item()
        logger.info(
            f"  Clean methods where source == target: "
            f"{correct_clean} / {n_negative} "
            f"(expect all)"
        )

        return y, source_classes, target_classes


    # ────────────────────────────────────────────────────────────
    # Step 4: Edge index
    # ────────────────────────────────────────────────────────────

    def _build_edge_index(
        self,
        edges: pd.DataFrame,
        method_to_idx: dict
    ) -> torch.Tensor:

        valid_edges  = []
        dropped      = 0

        for _, row in edges.iterrows():
            caller = int(row['caller_id'])
            callee = int(row['callee_id'])

            if caller in method_to_idx and callee in method_to_idx:
                valid_edges.append([
                    method_to_idx[caller],
                    method_to_idx[callee]
                ])
            else:
                dropped += 1

        logger.info(f"Edge index built:")
        logger.info(f"  Total edges in file: {len(edges)}")
        logger.info(f"  Valid edges:         {len(valid_edges)}")
        logger.info(f"  Dropped edges:       {dropped}")

        if dropped > 0:
            drop_pct = 100 * dropped / len(edges)
            logger.warning(
                f"  {drop_pct:.1f}% of edges dropped — "
                f"check ID alignment if this is unexpectedly high"
            )

        edge_index = torch.tensor(
            valid_edges, dtype=torch.long
        ).t().contiguous()

        logger.info(f"  Edge index shape:    {edge_index.shape}")
        return edge_index

    # ────────────────────────────────────────────────────────────
    # Step 5: Train/val/test masks
    # ────────────────────────────────────────────────────────────

    def _build_masks(
        self,
        n_methods: int,
        y: torch.Tensor,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int
    ):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train/val/test ratios must sum to 1.0"

        indices = np.arange(n_methods)
        labels  = y.numpy()

        # First split: train+val vs test
        idx_trainval, idx_test = train_test_split(
            indices,
            test_size  = test_ratio,
            stratify   = labels,
            random_state = random_seed
        )

        # Second split: train vs val
        # val_ratio relative to trainval portion
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        idx_train, idx_val = train_test_split(
            idx_trainval,
            test_size    = val_ratio_adjusted,
            stratify     = labels[idx_trainval],
            random_state = random_seed
        )

        # Convert to boolean masks
        train_mask = torch.zeros(n_methods, dtype=torch.bool)
        val_mask   = torch.zeros(n_methods, dtype=torch.bool)
        test_mask  = torch.zeros(n_methods, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val]     = True
        test_mask[idx_test]   = True

        # Verify stratification worked
        def pos_rate(mask):
            return y[mask].float().mean().item() * 100

        logger.info(f"Train/val/test split (seed={random_seed}):")
        logger.info(
            f"  Train: {train_mask.sum().item():6d} methods "
            f"({pos_rate(train_mask):.2f}% positive)"
        )
        logger.info(
            f"  Val:   {val_mask.sum().item():6d} methods "
            f"({pos_rate(val_mask):.2f}% positive)"
        )
        logger.info(
            f"  Test:  {test_mask.sum().item():6d} methods "
            f"({pos_rate(test_mask):.2f}% positive)"
        )

        return train_mask, val_mask, test_mask

    # ────────────────────────────────────────────────────────────
    # Summary
    # ────────────────────────────────────────────────────────────

    def _log_summary(self, graph: Data, y: torch.Tensor):
        logger.info(f"Graph summary:")
        logger.info(f"  Nodes:       {graph.num_nodes}")
        logger.info(f"  Edges:       {graph.num_edges}")
        logger.info(f"  Features:    {graph.num_node_features}")
        logger.info(f"  Positive:    {y.sum().item()}")
        logger.info(f"  Negative:    {(y==0).sum().item()}")

    # ────────────────────────────────────────────────────────────
    # Save / Load
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def save(graph: Data, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graph, save_dir / "graph.pt")
        logger.info(f"Graph saved to {save_dir / 'graph.pt'}")

    @staticmethod
    def load(save_dir: Path) -> Data:
        graph = torch.load(Path(save_dir) / "graph.pt")
        logger.info(f"Graph loaded from {save_dir}")
        return graph