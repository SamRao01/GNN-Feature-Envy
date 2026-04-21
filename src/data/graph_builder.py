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
            train_mask, val_mask, test_mask
        """
        n_methods = len(method_to_idx)

        # ── Step 3A: Label vector y ──────────────────────────────
        y = self._build_labels(ground_truth, method_to_idx, n_methods)

        # ── Step 3B: Target class vector ────────────────────────
        target_classes = self._build_target_classes(
            ground_truth, method_to_idx, n_methods
        )

        # ── Step 4: Edge index ───────────────────────────────────
        edge_index = self._build_edge_index(edges, method_to_idx)

        # ── Step 5: Train/val/test masks ─────────────────────────
        train_mask, val_mask, test_mask = self._build_masks(
            n_methods, y, train_ratio, val_ratio, test_ratio, random_seed
        )

        # ── Package into PyG Data object ─────────────────────────
        graph = Data(
            x            = X,
            edge_index   = edge_index,
            y            = y,
            target_class = target_classes,
            train_mask   = train_mask,
            val_mask     = val_mask,
            test_mask    = test_mask
        )

        self._log_summary(graph, y)
        return graph

    # ────────────────────────────────────────────────────────────
    # Step 3A: Labels
    # ────────────────────────────────────────────────────────────

    def _build_labels(
        self,
        ground_truth: pd.DataFrame,
        method_to_idx: dict,
        n_methods: int
    ) -> torch.Tensor:

        y = torch.zeros(n_methods, dtype=torch.long)

        for _, row in ground_truth.iterrows():
            method_id = int(row['method_id'])
            if method_id in method_to_idx:
                idx    = method_to_idx[method_id]
                y[idx] = int(row['label'])

        n_positive = y.sum().item()
        n_negative = (y == 0).sum().item()
        logger.info(f"Labels built:")
        logger.info(f"  Positive (smelly): {n_positive}")
        logger.info(f"  Negative (clean):  {n_negative}")
        logger.info(f"  Positive rate:     {100*n_positive/n_methods:.2f}%")

        return y

    # ────────────────────────────────────────────────────────────
    # Step 3B: Target classes
    # ────────────────────────────────────────────────────────────

    def _build_target_classes(
        self,
        ground_truth: pd.DataFrame,
        method_to_idx: dict,
        n_methods: int
    ) -> torch.Tensor:

        # -1 means no refactoring target (method is clean)
        target_classes = torch.full((n_methods,), -1, dtype=torch.long)

        for _, row in ground_truth.iterrows():
            method_id = int(row['method_id'])
            if method_id in method_to_idx:
                idx = method_to_idx[method_id]
                target_classes[idx] = int(row['target_class_id'])

        known_targets = (target_classes >= 0).sum().item()
        logger.info(f"Target classes built:")
        logger.info(f"  Methods with known target: {known_targets}")

        return target_classes

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