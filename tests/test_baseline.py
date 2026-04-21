import pytest
import torch
import numpy as np
from torch_geometric.data import Data
from src.models.baseline import HeuristicBaseline


@pytest.fixture
def sample_graph():
    """
    10 methods, 3 smelly.
    Feature order: CC, PC, LOC, NCMIC, NCMEC, NECA, NAMFAEC
    Smelly methods (idx 1, 3, 5) have high NCMEC, low NCMIC.
    """
    X = torch.tensor([
        # CC  PC  LOC  NCMIC  NCMEC  NECA  NAMFAEC
        [1.0, 0.0, 5.0,  3.0,   0.0,  0.0,  0.0],  # 0 clean
        [1.0, 1.0, 10.0, 0.0,   8.0,  3.0,  5.0],  # 1 smelly
        [2.0, 0.0, 8.0,  4.0,   1.0,  1.0,  1.0],  # 2 clean
        [1.0, 2.0, 12.0, 0.0,   6.0,  2.0,  4.0],  # 3 smelly
        [3.0, 1.0, 6.0,  5.0,   0.0,  0.0,  0.0],  # 4 clean
        [1.0, 0.0, 9.0,  0.0,   7.0,  3.0,  6.0],  # 5 smelly
        [2.0, 1.0, 7.0,  3.0,   1.0,  1.0,  1.0],  # 6 clean
        [1.0, 0.0, 4.0,  2.0,   0.0,  0.0,  0.0],  # 7 clean
        [1.0, 1.0, 5.0,  4.0,   0.0,  0.0,  0.0],  # 8 clean
        [2.0, 0.0, 6.0,  3.0,   1.0,  1.0,  1.0],  # 9 clean
    ], dtype=torch.float32)

    y            = torch.tensor([0,1,0,1,0,1,0,0,0,0], dtype=torch.long)
    target_class = torch.tensor([0,5,1,3,2,7,0,0,0,1], dtype=torch.long)

    # Use all nodes as test set for simplicity
    test_mask = torch.ones(10, dtype=torch.bool)

    return Data(
        x            = X,
        y            = y,
        target_class = target_class,
        test_mask    = test_mask
    )


def test_detects_smelly_methods(sample_graph):
    baseline = HeuristicBaseline(threshold=0.5)
    y_pred, _ = baseline.predict(sample_graph, sample_graph.test_mask)
    # Methods 1, 3, 5 should be detected (high NCMEC, zero NCMIC)
    assert y_pred[1] == 1
    assert y_pred[3] == 1
    assert y_pred[5] == 1


def test_does_not_flag_clean_methods(sample_graph):
    baseline = HeuristicBaseline(threshold=0.5)
    y_pred, _ = baseline.predict(sample_graph, sample_graph.test_mask)
    # Methods 0, 4, 7, 8 have zero NCMEC — should not be flagged
    assert y_pred[0] == 0
    assert y_pred[4] == 0
    assert y_pred[7] == 0


def test_output_shape(sample_graph):
    baseline = HeuristicBaseline(threshold=0.5)
    y_pred, y_proba = baseline.predict(
        sample_graph, sample_graph.test_mask
    )
    assert y_pred.shape  == (10,)
    assert y_proba.shape == (10,)


def test_proba_between_zero_and_one(sample_graph):
    baseline  = HeuristicBaseline(threshold=0.5)
    _, y_proba = baseline.predict(sample_graph, sample_graph.test_mask)
    assert (y_proba >= 0).all()
    assert (y_proba <= 1).all()


def test_higher_threshold_fewer_predictions(sample_graph):
    b_low  = HeuristicBaseline(threshold=0.3)
    b_high = HeuristicBaseline(threshold=0.8)
    pred_low,  _ = b_low.predict(sample_graph, sample_graph.test_mask)
    pred_high, _ = b_high.predict(sample_graph, sample_graph.test_mask)
    assert pred_low.sum() >= pred_high.sum()


def test_refactoring_returns_minus_one(sample_graph):
    baseline      = HeuristicBaseline(threshold=0.5)
    y_pred, _     = baseline.predict(
        sample_graph, sample_graph.test_mask
    )
    pred_targets  = baseline.predict_targets(
        sample_graph, sample_graph.test_mask, y_pred
    )
    # Heuristic cannot predict target class — all should be -1
    assert (pred_targets == -1).all()


def test_sweep_thresholds_returns_dataframe(sample_graph):
    baseline = HeuristicBaseline()
    df = baseline.sweep_thresholds(
        sample_graph,
        sample_graph.test_mask,
        thresholds=[0.3, 0.5, 0.7]
    )
    assert len(df) == 3
    assert 'f1' in df.columns
    assert 'precision' in df.columns
    assert 'recall' in df.columns