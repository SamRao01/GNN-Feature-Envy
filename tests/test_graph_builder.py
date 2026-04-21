import pytest
import pandas as pd
import numpy as np
import torch
from src.data.graph_builder import GraphBuilder


@pytest.fixture
def sample_ground_truth():
    return pd.DataFrame({
        'method_id':       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'source_class_id': [0, 0, 1, 1, 2, 0, 1, 2, 0, 1],
        'target_class_id': [0, 1, 1, 2, 2, 0, 1, 2, 0, 2],
        'label':           [0, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    })


@pytest.fixture
def sample_edges():
    return pd.DataFrame({
        'caller_id': [0, 1, 2, 3],
        'callee_id': [1, 2, 3, 4]
    })


@pytest.fixture
def sample_X():
    return torch.randn(10, 7)


@pytest.fixture
def sample_method_to_idx():
    return {i: i for i in range(10)}


def test_graph_nodes(sample_ground_truth, sample_edges,
                     sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    assert graph.num_nodes == 10


def test_graph_features(sample_ground_truth, sample_edges,
                        sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    assert graph.num_node_features == 7


def test_graph_edges(sample_ground_truth, sample_edges,
                     sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == len(sample_edges)


def test_labels(sample_ground_truth, sample_edges,
                sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    assert graph.y.sum().item() == 3   # 3 smelly methods


def test_masks_cover_all_nodes(sample_ground_truth, sample_edges,
                                sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    total = (graph.train_mask.sum() +
             graph.val_mask.sum() +
             graph.test_mask.sum())
    assert total.item() == 10


def test_masks_no_overlap(sample_ground_truth, sample_edges,
                           sample_X, sample_method_to_idx):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    # No node should appear in more than one split
    overlap = (graph.train_mask & graph.val_mask).sum() + \
              (graph.train_mask & graph.test_mask).sum() + \
              (graph.val_mask  & graph.test_mask).sum()
    assert overlap.item() == 0


def test_dropped_edges_logged(sample_ground_truth, sample_X,
                               sample_method_to_idx):
    # Edge referencing a method NOT in the index
    bad_edges = pd.DataFrame({
        'caller_id': [0, 999],   # 999 not in index
        'callee_id': [1, 2]
    })
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, bad_edges,
        sample_X, sample_method_to_idx
    )
    # Only 1 valid edge should be kept
    assert graph.edge_index.shape[1] == 1


def test_save_and_load(sample_ground_truth, sample_edges,
                        sample_X, sample_method_to_idx, tmp_path):
    builder = GraphBuilder()
    graph = builder.build(
        sample_ground_truth, sample_edges,
        sample_X, sample_method_to_idx
    )
    GraphBuilder.save(graph, tmp_path)
    loaded = GraphBuilder.load(tmp_path)

    assert loaded.num_nodes == graph.num_nodes
    assert loaded.num_edges == graph.num_edges
    assert torch.equal(loaded.y, graph.y)