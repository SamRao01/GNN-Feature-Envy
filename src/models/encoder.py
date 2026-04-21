import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import logging

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder that produces node embeddings by aggregating
    neighborhood information.

    For each node v:
        1. Compute mean of neighbor embeddings
        2. Concatenate with own embedding
        3. Apply linear transformation + ReLU

    This gives each node an embedding that captures both its own
    features and its local graph structure — critical for feature
    envy detection since the smell is defined by relationships.

    Architecture:
        Input  [N x 7]  →  SAGEConv  →  [N x hidden_dim]
                        →  SAGEConv  →  [N x hidden_dim]
                        →  Dropout
                        →  Output   [N x hidden_dim]
    """

    def __init__(
        self,
        in_channels:  int,
        hidden_dim:   int = 256,
        num_layers:   int = 2,
        dropout:      float = 0.1
    ):
        """
        Args:
            in_channels: number of input features per node (7 for our data)
            hidden_dim:  size of hidden and output embeddings
            num_layers:  number of GraphSAGE layers
            dropout:     dropout probability for regularization
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout    = dropout

        # Build SAGEConv layers
        self.convs = nn.ModuleList()

        for i in range(num_layers):
            in_dim  = in_channels if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.convs.append(SAGEConv(in_dim, out_dim))

        self.dropout_layer = nn.Dropout(p=dropout)

        logger.info(
            f"GraphSAGEEncoder initialized: "
            f"in={in_channels}, hidden={hidden_dim}, "
            f"layers={num_layers}, dropout={dropout}"
        )

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass — produces node embeddings.

        Args:
            x:          node feature matrix [N x in_channels]
            edge_index: graph connectivity   [2 x num_edges]

        Returns:
            h: node embeddings [N x hidden_dim]
        """
        h = x

        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)

            # Apply ReLU on all layers except the last
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout_layer(h)

        return h