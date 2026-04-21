import pytest
import pandas as pd
import numpy as np
import torch
from src.data.feature_builder import FeatureBuilder, FEATURE_COLS


@pytest.fixture
def sample_metrics():
    return pd.DataFrame({
        'method_id': [0, 1, 2, 3, 4],
        'method':    ['m0', 'm1', 'm2', 'm3', 'm4'],
        'CC':        [1, 2, 3, 1, 2],
        'PC':        [0, 1, 2, 0, 1],
        'LOC':       [5, 10, 15, 5, 10],
        'NCMIC':     [1, 2, 0, 1, 2],
        'NCMEC':     [0, 3, 5, 0, 3],
        'NECA':      [0, 2, 3, 0, 2],
        'NAMFAEC':   [0, 2, 3, 0, 2],
    })


@pytest.fixture
def sample_method_to_idx():
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}


def test_output_shape(sample_metrics, sample_method_to_idx):
    builder = FeatureBuilder()
    X = builder.build(sample_metrics, sample_method_to_idx)
    assert X.shape == (5, 7)


def test_output_is_tensor(sample_metrics, sample_method_to_idx):
    builder = FeatureBuilder()
    X = builder.build(sample_metrics, sample_method_to_idx)
    assert isinstance(X, torch.Tensor)
    assert X.dtype == torch.float32


def test_normalization(sample_metrics, sample_method_to_idx):
    """After StandardScaler, each column should have mean ~0."""
    builder = FeatureBuilder()
    X = builder.build(sample_metrics, sample_method_to_idx)
    col_means = X.mean(dim=0)
    assert torch.allclose(col_means, torch.zeros(7), atol=1e-5)


def test_missing_method_gets_zeros(sample_method_to_idx):
    """A method in the index but not in metrics should be zero (pre-norm)."""
    # Only provide metrics for methods 0-3, not method 4
    partial_metrics = pd.DataFrame({
        'method_id': [0, 1, 2, 3],
        'method':    ['m0', 'm1', 'm2', 'm3'],
        'CC':        [1, 2, 3, 1],
        'PC':        [0, 1, 2, 0],
        'LOC':       [5, 10, 15, 5],
        'NCMIC':     [1, 2, 0, 1],
        'NCMEC':     [0, 3, 5, 0],
        'NECA':      [0, 2, 3, 0],
        'NAMFAEC':   [0, 2, 3, 0],
    })
    builder = FeatureBuilder()
    # Should not raise - missing method gets zeros
    X = builder.build(partial_metrics, sample_method_to_idx)
    assert X.shape == (5, 7)


def test_missing_feature_column_raises(sample_method_to_idx):
    bad_metrics = pd.DataFrame({
        'method_id': [0, 1],
        'CC': [1, 2],
        # Missing PC, LOC, etc.
    })
    builder = FeatureBuilder()
    with pytest.raises(ValueError, match="missing expected columns"):
        builder.build(bad_metrics, sample_method_to_idx)


def test_save_and_load(sample_metrics, sample_method_to_idx, tmp_path):
    builder = FeatureBuilder()
    X1 = builder.build(sample_metrics, sample_method_to_idx)
    builder.save(tmp_path)

    loaded = FeatureBuilder.load(tmp_path)
    X2 = loaded.build(
        sample_metrics, sample_method_to_idx, fit_scaler=False
    )

    assert torch.allclose(X1, X2)