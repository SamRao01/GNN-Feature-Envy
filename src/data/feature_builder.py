import pandas as pd
import numpy as np
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging
import pickle

logger = logging.getLogger(__name__)

# Exact column names confirmed from metrics.csv inspection
FEATURE_COLS = ['CC', 'PC', 'LOC', 'NCMIC', 'NCMEC', 'NECA', 'NAMFAEC']


class FeatureBuilder:
    """
    Builds the node feature matrix X from metrics.csv.

    Output shape: [N_methods x 7]
    One row per method, seven columns for the code metrics:
        CC       - Cyclomatic Complexity
        PC       - Parameter Count
        LOC      - Lines of Code
        NCMIC    - Number of Calls to Methods of Its own Class
        NCMEC    - Number of Calls to Methods of External Classes
        NECA     - Number of External Classes Accessed
        NAMFAEC  - Number of Accesses to Most Frequently accessed External Class
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = FEATURE_COLS

    def build(
        self,
        metrics: pd.DataFrame,
        method_to_idx: dict,
        fit_scaler: bool = True
    ) -> torch.Tensor:
        """
        Builds and returns the normalized feature matrix X.

        Args:
            metrics:      the metrics.csv dataframe for a project
            method_to_idx: the method index mapping from MethodIndex
            fit_scaler:   if True, fits the scaler on this data (use for
                          training set). If False, uses already-fitted
                          scaler (use for val/test sets).

        Returns:
            X: torch.Tensor of shape [N_methods, 7]
        """
        self._validate_columns(metrics)

        n_methods = len(method_to_idx)

        # Initialize feature matrix with zeros
        # Methods missing from metrics.csv will remain zero
        X_raw = np.zeros((n_methods, len(self.feature_cols)),
                         dtype=np.float32)

        matched  = 0
        missing  = 0

        for _, row in metrics.iterrows():
            method_id = int(row['method_id'])
            if method_id in method_to_idx:
                idx = method_to_idx[method_id]
                X_raw[idx] = [row[col] for col in self.feature_cols]
                matched += 1
            else:
                missing += 1

        logger.info(f"Methods matched to index: {matched}")
        logger.info(f"Methods in metrics but not in index: {missing}")
        logger.info(f"Feature matrix shape (before norm): {X_raw.shape}")

        # Normalize
        if fit_scaler:
            X_norm = self.scaler.fit_transform(X_raw)
            logger.info("Scaler fitted and applied")
        else:
            X_norm = self.scaler.transform(X_raw)
            logger.info("Pre-fitted scaler applied")

        X = torch.tensor(X_norm, dtype=torch.float32)
        logger.info(f"Feature matrix shape (final): {X.shape}")

        self._log_feature_stats(X_raw)

        return X

    def _validate_columns(self, metrics: pd.DataFrame):
        missing_cols = set(self.feature_cols) - set(metrics.columns)
        if missing_cols:
            raise ValueError(
                f"metrics.csv missing expected columns: {missing_cols}\n"
                f"Found: {metrics.columns.tolist()}"
            )
        if 'method_id' not in metrics.columns:
            raise ValueError("metrics.csv must contain 'method_id' column")

    def _log_feature_stats(self, X_raw: np.ndarray):
        logger.info("Feature statistics (raw, before normalization):")
        for i, col in enumerate(self.feature_cols):
            col_data = X_raw[:, i]
            logger.info(
                f"  {col:10s} — "
                f"min: {col_data.min():8.2f}  "
                f"max: {col_data.max():8.2f}  "
                f"mean: {col_data.mean():8.2f}  "
                f"std: {col_data.std():8.2f}"
            )

    def save(self, save_dir: Path):
        """Saves the fitted scaler so it can be reused."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"Scaler saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> "FeatureBuilder":
        """Loads a previously fitted FeatureBuilder from disk."""
        save_dir = Path(save_dir)
        instance = cls()
        with open(save_dir / "scaler.pkl", "rb") as f:
            instance.scaler = pickle.load(f)
        logger.info(f"FeatureBuilder loaded from {save_dir}")
        return instance