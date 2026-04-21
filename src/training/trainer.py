import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import numpy as np
import logging
from pathlib import Path

from src.models.encoder      import GraphSAGEEncoder
from src.models.classifier   import NodeClassifier
from src.models.augmenter    import GraphSMOTE
from src.utils.metrics       import compute_detection_metrics

logger = logging.getLogger(__name__)


class SCGTrainer:
    """
    Trainer for the SCG (SMOTE Call Graph) model.

    Training loop:
        1. Encode nodes with GraphSAGE
        2. Augment graph with GraphSMOTE (smelly node synthesis)
        3. Classify nodes (feature envy detection)
        4. Compute joint loss:
               L = L_node + lambda * L_edge
        5. Update all parameters via Adam

    At inference:
        1. Encode nodes
        2. Predict smelly/clean per method
        3. Use edge generator calling strengths for refactoring
    """

    def __init__(
        self,
        in_channels:  int   = 7,
        hidden_dim:   int   = 256,
        num_layers:   int   = 2,
        dropout:      float = 0.1,
        lr:           float = 1e-3,
        weight_decay: float = 5e-4,
        lambda_edge:  float = 1e-6,
        k_neighbors:  int   = 5,
        device:       str   = 'cpu'
    ):
        self.device       = torch.device(device)
        self.lambda_edge  = lambda_edge

        # Model components
        self.encoder    = GraphSAGEEncoder(
            in_channels, hidden_dim, num_layers, dropout
        ).to(self.device)

        self.classifier = NodeClassifier(
            hidden_dim, dropout
        ).to(self.device)

        self.augmenter  = GraphSMOTE(
            hidden_dim   = hidden_dim,
            k_neighbors  = k_neighbors
        ).to(self.device)

        # Single optimizer for all parameters
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) +
            list(self.classifier.parameters()) +
            list(self.augmenter.parameters()),
            lr           = lr,
            weight_decay = weight_decay
        )

        logger.info(
            f"SCGTrainer initialized on {device}: "
            f"hidden={hidden_dim}, layers={num_layers}, "
            f"lr={lr}, lambda_edge={lambda_edge}"
        )

    def _compute_class_weights(
        self,
        y:         torch.Tensor,
        train_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes class weights inversely proportional to frequency.
        Upweights the minority (smelly) class to combat imbalance.
        """
        train_labels = y[train_idx]
        n_total  = len(train_labels)
        n_pos    = train_labels.sum().item()
        n_neg    = n_total - n_pos

        if n_pos == 0:
            return torch.tensor([1.0, 1.0]).to(self.device)

        weight_neg = n_total / (2 * n_neg)
        weight_pos = n_total / (2 * n_pos)

        logger.info(
            f"Class weights — clean: {weight_neg:.4f}, "
            f"smelly: {weight_pos:.4f} "
            f"(ratio: {weight_pos/weight_neg:.1f}x)"
        )

        return torch.tensor(
            [weight_neg, weight_pos], dtype=torch.float32
        ).to(self.device)

    def train(
        self,
        graph:      Data,
        n_epochs:   int  = 1000,
        log_every:  int  = 50
    ) -> dict:
        """
        Full training loop.

        Args:
            graph:     PyG Data object
            n_epochs:  number of training epochs
            log_every: log metrics every N epochs

        Returns:
            history: dict of training metrics per epoch
        """
        graph = graph.to(self.device)

        # Get training indices
        train_idx = graph.train_mask.nonzero(as_tuple=True)[0]
        val_idx   = graph.val_mask.nonzero(as_tuple=True)[0]

        # Build dense adjacency for GraphSMOTE
        n         = graph.num_nodes
        A_dense   = to_dense_adj(
            graph.edge_index, max_num_nodes=n
        ).squeeze(0)

        # Compute class weights for imbalanced training
        class_weights = self._compute_class_weights(
            graph.y, train_idx
        )

        history = {
            'train_loss':    [],
            'train_f1':      [],
            'val_f1':        [],
            'best_val_f1':   0.0,
            'best_epoch':    0
        }

        best_encoder_state    = None
        best_classifier_state = None

        for epoch in range(1, n_epochs + 1):
            # ── Training step ────────────────────────────────────
            self.encoder.train()
            self.classifier.train()
            self.augmenter.train()
            self.optimizer.zero_grad()

            # Step 1: Encode
            h = self.encoder(graph.x, graph.edge_index)

            # Step 2: Edge loss (train edge generator)
            L_edge = self.augmenter.compute_edge_loss(
                h.detach(), A_dense
            )

            # Step 3: Augment with GraphSMOTE
            # Returns h_aug already in embedding space [N+syn x 256]
            h_aug, y_aug, idx_aug, A_aug = self.augmenter(
                h, graph.y, train_idx, A_dense
            )

            # Step 4: Classify directly on augmented embeddings
            # No re-encoding needed — h_aug is already [N+syn x 256]
            logits = self.classifier(h_aug[idx_aug])

            # Step 6: Node classification loss with class weights
            L_node = F.cross_entropy(
                logits,
                y_aug[idx_aug],
                weight = class_weights
            )

            # Step 7: Joint loss
            loss = L_node + self.lambda_edge * L_edge
            loss.backward()
            self.optimizer.step()

            history['train_loss'].append(loss.item())

            # ── Validation step ──────────────────────────────────
            if epoch % log_every == 0 or epoch == 1:
                self.encoder.eval()
                self.classifier.eval()

                with torch.no_grad():
                    h_val   = self.encoder(
                        graph.x, graph.edge_index
                    )
                    y_pred_val = self.classifier.predict(
                        h_val[val_idx]
                    )
                    y_true_val = graph.y[val_idx]

                    # Train metrics
                    y_pred_train = self.classifier.predict(
                        h_val[train_idx]
                    )
                    y_true_train = graph.y[train_idx]

                train_metrics = compute_detection_metrics(
                    y_true_train.cpu().numpy(),
                    y_pred_train.cpu().numpy()
                )
                val_metrics = compute_detection_metrics(
                    y_true_val.cpu().numpy(),
                    y_pred_val.cpu().numpy()
                )

                train_f1 = train_metrics['f1_score1']
                val_f1   = val_metrics['f1_score1']

                history['train_f1'].append(train_f1)
                history['val_f1'].append(val_f1)

                logger.info(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Train F1: {train_f1:.4f} | "
                    f"Val F1: {val_f1:.4f}"
                )

                # Save best model by validation F1
                if val_f1 > history['best_val_f1']:
                    history['best_val_f1'] = val_f1
                    history['best_epoch']  = epoch
                    best_encoder_state    = {
                        k: v.clone()
                        for k, v in
                        self.encoder.state_dict().items()
                    }
                    best_classifier_state = {
                        k: v.clone()
                        for k, v in
                        self.classifier.state_dict().items()
                    }
                    logger.info(
                        f"  New best model saved "
                        f"(val F1={val_f1:.4f})"
                    )

        # Restore best model
        if best_encoder_state is not None:
            self.encoder.load_state_dict(best_encoder_state)
            self.classifier.load_state_dict(best_classifier_state)
            logger.info(
                f"Best model restored from epoch "
                f"{history['best_epoch']} "
                f"(val F1={history['best_val_f1']:.4f})"
            )

        return history

    def evaluate(
        self,
        graph:     Data,
        mask:      torch.Tensor,
        threshold: float = 0.5
    ) -> tuple:
        """
        Evaluates the model on a given mask (val or test).

        Args:
            graph:     PyG Data object
            mask:      boolean mask selecting nodes to evaluate
            threshold: decision threshold for smelly prediction

        Returns:
            y_true:      ground truth labels
            y_pred:      predicted labels
            h:           node embeddings (for refactoring)
            A_hat:       predicted adjacency (for refactoring)
        """
        graph = graph.to(self.device)
        self.encoder.eval()
        self.classifier.eval()

        with torch.no_grad():
            h     = self.encoder(graph.x, graph.edge_index)
            A_hat = self.augmenter.edge_generator.predict_adjacency(h)

            masked_idx = mask.nonzero(as_tuple=True)[0]
            y_pred     = self.classifier.predict(
                h[masked_idx], threshold
            )
            y_true     = graph.y[masked_idx]

        return y_true.cpu(), y_pred.cpu(), h.cpu(), A_hat.cpu()

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.encoder.state_dict(),
            save_dir / "encoder.pt"
        )
        torch.save(
            self.classifier.state_dict(),
            save_dir / "classifier.pt"
        )
        torch.save(
            self.augmenter.state_dict(),
            save_dir / "augmenter.pt"
        )
        logger.info(f"Model saved to {save_dir}")

    def load(self, save_dir: Path):
        save_dir = Path(save_dir)

        self.encoder.load_state_dict(
            torch.load(save_dir / "encoder.pt")
        )
        self.classifier.load_state_dict(
            torch.load(save_dir / "classifier.pt")
        )
        self.augmenter.load_state_dict(
            torch.load(save_dir / "augmenter.pt")
        )
        logger.info(f"Model loaded from {save_dir}")