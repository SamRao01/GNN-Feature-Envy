import pytest
import pandas as pd
import torch
from pathlib import Path
from src.data.preprocessor import MethodIndex


@pytest.fixture
def sample_ground_truth():
    return pd.DataFrame({
        'method_id':       [0, 1, 2, 3, 4],
        'method':          ['m0', 'm1', 'm2', 'm3', 'm4'],
        'source_class_id': [0, 0, 1, 1, 2],
        'source_class':    ['A', 'A', 'B', 'B', 'C'],
        'target_class_id': [0, 1, 1, 2, 2],
        'target_class':    ['A', 'B', 'B', 'C', 'C'],
        'label':           [0, 1, 0, 1, 0]
    })


def test_index_size(sample_ground_truth):
    index = MethodIndex(sample_ground_truth)
    assert index.n_methods == 5


def test_forward_mapping(sample_ground_truth):
    index = MethodIndex(sample_ground_truth)
    # Every method_id should map to a valid index
    for method_id in sample_ground_truth['method_id']:
        idx = index.get_idx(method_id)
        assert 0 <= idx < index.n_methods


def test_reverse_mapping(sample_ground_truth):
    index = MethodIndex(sample_ground_truth)
    # Round trip: method_id -> idx -> method_id should be identity
    for method_id in sample_ground_truth['method_id']:
        idx = index.get_idx(method_id)
        recovered = index.get_method_id(idx)
        assert recovered == method_id


def test_unknown_method_returns_minus_one(sample_ground_truth):
    index = MethodIndex(sample_ground_truth)
    assert index.get_idx(9999) == -1


def test_contains(sample_ground_truth):
    index = MethodIndex(sample_ground_truth)
    assert index.contains(0) is True
    assert index.contains(9999) is False


def test_save_and_load(sample_ground_truth, tmp_path):
    index = MethodIndex(sample_ground_truth)
    index.save(tmp_path)

    # Check files exist
    assert (tmp_path / "method_to_idx.pt").exists()
    assert (tmp_path / "idx_to_method.pt").exists()

    # Load and verify
    loaded = MethodIndex.load(tmp_path)
    assert loaded.n_methods == index.n_methods
    assert loaded.method_to_idx == index.method_to_idx


def test_missing_method_id_column():
    bad_df = pd.DataFrame({'wrong_col': [1, 2, 3]})
    with pytest.raises(ValueError, match="method_id"):
        MethodIndex(bad_df)