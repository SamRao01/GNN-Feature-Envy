import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EdgeGenerator(nn.Module):
    """
    Predicts the existence of edges between nodes.
    Used in GraphSMOTE to connect synthetic smelly nodes
    to the original graph.

    For nodes v and u:
        score = sigmoid(W*h_v · (W*h_u)^T)

    Trained to reconstruct the original adjacency matrix.
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self,
        h_v: torch.Tensor,
        h_u: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            h_v: embeddings of source nodes [N x hidden_dim]
            h_u: embeddings of target nodes [N x hidden_dim]

        Returns:
            scores: edge existence probability [N]
        """
        return torch.sigmoid(
            (self.W(h_v) * self.W(h_u)).sum(dim=1)
        )

    def predict_adjacency(
        self,
        h: torch.Tensor
    ) -> torch.Tensor:
        """
        Predicts full adjacency matrix for all node pairs.

        Args:
            h: all node embeddings [N x hidden_dim]

        Returns:
            A_hat: predicted adjacency [N x N]
        """
        Wh    = self.W(h)
        A_hat = torch.sigmoid(Wh @ Wh.t())
        return A_hat


class GraphSMOTE(nn.Module):
    """
    Graph-based SMOTE for handling class imbalance.

    Standard SMOTE generates synthetic minority samples by
    interpolating between existing minority samples in feature
    space. GraphSMOTE extends this to graphs by:

        1. Learning node embeddings that capture graph structure
        2. Interpolating between smelly node embeddings
           to generate synthetic smelly nodes
        3. Using an edge generator to connect synthetic nodes
           to the original graph

    This gives the GNN more positive examples to learn from
    during training, addressing the 3-10% positive rate problem.
    """

    def __init__(
        self,
        hidden_dim:        int = 256,
        k_neighbors:       int = 5,
        oversample_ratio:  float = 1.0
    ):
        """
        Args:
            hidden_dim:       embedding dimension
            k_neighbors:      number of nearest smelly neighbors
                              to interpolate between
            oversample_ratio: ratio of synthetic to real smelly nodes
                              1.0 means equal numbers
        """
        super().__init__()

        self.hidden_dim       = hidden_dim
        self.k_neighbors      = k_neighbors
        self.oversample_ratio = oversample_ratio
        self.edge_generator   = EdgeGenerator(hidden_dim)

    def forward(
        self,
        h:         torch.Tensor,
        y:         torch.Tensor,
        train_idx: torch.Tensor,
        A_dense:   torch.Tensor
    ) -> tuple:
        """
        Augments the graph with synthetic smelly nodes.

        Args:
            h:         node embeddings from encoder [N x hidden_dim]
            y:         node labels [N]
            train_idx: indices of training nodes
            A_dense:   dense adjacency matrix [N x N]

        Returns:
            h_aug:     augmented embeddings [N+n_synthetic x hidden_dim]
            y_aug:     augmented labels     [N+n_synthetic]
            idx_aug:   augmented train indices
            A_aug:     augmented adjacency  [N+n_synthetic x N+n_synthetic]
        """
        # Get smelly training nodes
        train_labels  = y[train_idx]
        smelly_mask   = (train_labels == 1)
        smelly_idx    = train_idx[smelly_mask]
        clean_idx     = train_idx[~smelly_mask]

        n_smelly = len(smelly_idx)
        n_clean  = len(clean_idx)
        n_synthetic = int(
            min(
                n_smelly * self.oversample_ratio,
                n_clean - n_smelly
            )
        )

        if n_synthetic <= 0 or n_smelly < 2:
            logger.warning(
                "GraphSMOTE: not enough smelly samples to augment. "
                "Returning original graph unchanged."
            )
            return h, y, train_idx, A_dense

        logger.info(
            f"GraphSMOTE: generating {n_synthetic} synthetic "
            f"smelly nodes (original: {n_smelly})"
        )

        # Get embeddings of smelly nodes
        h_smelly = h[smelly_idx]

        # Generate synthetic embeddings by interpolating
        synthetic_embeddings = self._generate_synthetic(
            h_smelly, n_synthetic
        )

        # Predict edges for synthetic nodes
        with torch.no_grad():
            A_hat    = self.edge_generator.predict_adjacency(h)
            syn_rows = []

            for i in range(n_synthetic):
                syn_h   = synthetic_embeddings[i].unsqueeze(0)
                Wh_syn  = self.edge_generator.W(syn_h)
                Wh_all  = self.edge_generator.W(h)
                scores  = torch.sigmoid(
                    (Wh_syn * Wh_all).sum(dim=1)
                )
                # Threshold at 0.5
                syn_row = (scores > 0.5).float()
                syn_rows.append(syn_row)

        # Build augmented graph
        syn_adj = torch.stack(syn_rows, dim=0)  # [n_syn x N]
        n_orig  = h.shape[0]

        # Expand adjacency matrix
        A_aug = torch.zeros(
            n_orig + n_synthetic,
            n_orig + n_synthetic
        )
        A_aug[:n_orig, :n_orig] = A_dense
        A_aug[n_orig:, :n_orig] = syn_adj
        A_aug[:n_orig, n_orig:] = syn_adj.t()

        # Augmented embeddings and labels
        h_aug = torch.cat([h, synthetic_embeddings], dim=0)
        y_aug = torch.cat([
            y,
            torch.ones(n_synthetic, dtype=torch.long)
        ], dim=0)

        # New training indices include synthetic nodes
        syn_train_idx = torch.arange(
            n_orig, n_orig + n_synthetic
        )
        idx_aug = torch.cat([train_idx, syn_train_idx])

        return h_aug, y_aug, idx_aug, A_aug

    def _generate_synthetic(
        self,
        h_smelly:    torch.Tensor,
        n_synthetic: int
    ) -> torch.Tensor:
        """
        Generates synthetic node embeddings by interpolating
        between pairs of existing smelly node embeddings.

        For each synthetic node:
            h_new = (1 - delta) * h_v + delta * h_u
            where delta ~ Uniform(0, 1)
            and h_u is one of the k nearest neighbors of h_v
        """
        n_smelly     = h_smelly.shape[0]
        k            = min(self.k_neighbors, n_smelly - 1)
        synthetic    = []

        # Compute pairwise distances
        dists = torch.cdist(h_smelly, h_smelly)

        for _ in range(n_synthetic):
            # Pick a random smelly node
            v_idx   = torch.randint(0, n_smelly, (1,)).item()
            h_v     = h_smelly[v_idx]

            # Find k nearest smelly neighbors
            d_v     = dists[v_idx]
            d_v[v_idx] = float('inf')   # exclude self
            knn_idx = d_v.topk(k, largest=False).indices

            # Pick one neighbor at random
            u_idx   = knn_idx[torch.randint(0, k, (1,)).item()]
            h_u     = h_smelly[u_idx]

            # Interpolate
            delta   = torch.rand(1).item()
            h_new   = (1 - delta) * h_v + delta * h_u
            synthetic.append(h_new)

        return torch.stack(synthetic, dim=0)

    def compute_edge_loss(
        self,
        h:       torch.Tensor,
        A_dense: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes reconstruction loss for the edge generator.
        Trains the edge generator to predict the original adjacency.

        Loss = ||A - A_hat||^2

        Args:
            h:       node embeddings [N x hidden_dim]
            A_dense: true adjacency  [N x N]

        Returns:
            loss: scalar edge reconstruction loss
        """
        A_hat = self.edge_generator.predict_adjacency(h)
        loss  = F.mse_loss(A_hat, A_dense)
        return loss