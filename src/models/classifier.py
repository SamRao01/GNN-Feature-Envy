import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class NodeClassifier(nn.Module):
    """
    Node classification head that takes node embeddings and predicts
    whether each method has feature envy.

    Architecture:
        Embeddings [N x hidden_dim]
        → Dropout
        → Linear [hidden_dim x 2]
        → Softmax
        → Probabilities [N x 2]

    Output index 0 = probability of clean
    Output index 1 = probability of smelly
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        dropout:    float = 0.1
    ):
        """
        Args:
            hidden_dim: size of input embeddings (must match encoder)
            dropout:    dropout probability
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.linear  = nn.Linear(hidden_dim, 2)

        logger.info(
            f"NodeClassifier initialized: "
            f"hidden={hidden_dim}, dropout={dropout}"
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: node embeddings [N x hidden_dim]

        Returns:
            logits: raw scores [N x 2]
                    (pass through softmax for probabilities)
        """
        h      = self.dropout(h)
        logits = self.linear(h)
        return logits

    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns class probabilities via softmax.

        Args:
            h: node embeddings [N x hidden_dim]

        Returns:
            proba: [N x 2] probability of [clean, smelly]
        """
        logits = self.forward(h)
        return F.softmax(logits, dim=1)

    def predict(
        self,
        h:         torch.Tensor,
        threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Returns binary predictions.

        Args:
            h:         node embeddings [N x hidden_dim]
            threshold: decision threshold on smelly probability

        Returns:
            y_pred: [N] binary labels (0=clean, 1=smelly)
        """
        proba  = self.predict_proba(h)
        y_pred = (proba[:, 1] > threshold).long()
        return y_pred