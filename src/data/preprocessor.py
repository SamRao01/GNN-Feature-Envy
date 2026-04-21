import pandas as pd
import torch
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MethodIndex:
    """
    Establishes a consistent 0-to-N-1 integer index for all methods
    in a project. This is the spine of the entire pipeline — every
    other preprocessing step depends on this mapping being correct.
    """

    def __init__(self, ground_truth: pd.DataFrame):
        """
        Args:
            ground_truth: the ground_truth.csv dataframe for a project.
                          Must contain a 'method_id' column.
        """
        if 'method_id' not in ground_truth.columns:
            raise ValueError(
                f"ground_truth must contain 'method_id' column. "
                f"Found: {ground_truth.columns.tolist()}"
            )

        # Extract unique method IDs in a stable order
        method_ids = ground_truth['method_id'].unique()

        # Build forward and reverse mappings
        self.method_to_idx = {
            int(mid): idx for idx, mid in enumerate(method_ids)
        }
        self.idx_to_method = {
            idx: int(mid) for idx, mid in enumerate(method_ids)
        }
        self.n_methods = len(self.method_to_idx)

        self._log_summary(ground_truth)

    def _log_summary(self, ground_truth: pd.DataFrame):
        n_smelly = (ground_truth['label'] == 1).sum()
        logger.info(f"Total methods indexed: {self.n_methods}")
        logger.info(
            f"Smelly methods: {n_smelly} / {self.n_methods} "
            f"({100 * n_smelly / self.n_methods:.2f}%)"
        )

    def get_idx(self, method_id: int) -> int:
        """
        Returns the integer index for a given method_id.
        Returns -1 if the method_id is not in the index.
        """
        return self.method_to_idx.get(int(method_id), -1)

    def get_method_id(self, idx: int) -> int:
        """
        Returns the original method_id for a given index.
        """
        return self.idx_to_method[idx]

    def contains(self, method_id: int) -> bool:
        return int(method_id) in self.method_to_idx

    def save(self, save_dir: Path):
        """
        Saves both mappings to disk so they can be reloaded
        without reprocessing.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.method_to_idx, save_dir / "method_to_idx.pt")
        torch.save(self.idx_to_method, save_dir / "idx_to_method.pt")

        logger.info(f"Method index saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> "MethodIndex":
        """
        Loads a previously saved MethodIndex from disk.
        Avoids reprocessing when running experiments repeatedly.
        """
        save_dir = Path(save_dir)

        instance = cls.__new__(cls)
        instance.method_to_idx = torch.load(
            save_dir / "method_to_idx.pt"
        )
        instance.idx_to_method = torch.load(
            save_dir / "idx_to_method.pt"
        )
        instance.n_methods = len(instance.method_to_idx)

        logger.info(
            f"Method index loaded from {save_dir} "
            f"({instance.n_methods} methods)"
        )
        return instance