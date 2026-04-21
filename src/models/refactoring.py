import torch
import logging

logger = logging.getLogger(__name__)


class RefactoringRecommender:
    """
    Recommends which class a smelly method should move to,
    based on calling strength aggregation.

    The paper's approach:
        1. Use the trained edge generator to predict calling
           strength between all method pairs
        2. For each method-class pair, sum the calling strengths
           to all methods belonging to that class:
               R[m, c] = sum_{m_i in c} A_hat[m, m_i]
        3. Recommend moving the smelly method to the class
           with the highest calling strength

    This is why the GNN outperforms the heuristic for refactoring:
    instead of just knowing a method makes many external calls,
    it knows WHICH specific class those calls concentrate on.
    """

    def recommend(
        self,
        A_hat:         torch.Tensor,
        smelly_mask:   torch.Tensor,
        source_classes: torch.Tensor,
        n_classes:     int
    ) -> torch.Tensor:
        """
        Recommends target classes for smelly methods.

        Args:
            A_hat:          predicted adjacency/calling strength [N x N]
            smelly_mask:    boolean mask of predicted smelly methods [N]
            source_classes: class ID each method currently belongs to [N]
            n_classes:      total number of classes in the project

        Returns:
            pred_targets: predicted target class ID per node [N]
                          -1 for clean methods (no recommendation)
        """
        n_methods    = A_hat.shape[0]
        pred_targets = torch.full(
            (n_methods,), -1, dtype=torch.long
        )

        # Build method-to-class membership lookup
        # class_members[c] = list of method indices belonging to class c
        class_members = {}
        for method_idx in range(n_methods):
            c = source_classes[method_idx].item()
            if c not in class_members:
                class_members[c] = []
            class_members[c].append(method_idx)

        # For each smelly method, compute calling strength to each class
        smelly_indices = smelly_mask.nonzero(as_tuple=True)[0]

        for m in smelly_indices:
            m = m.item()
            src_class = source_classes[m].item()

            best_class    = -1
            best_strength = -1.0

            for c, members in class_members.items():
                # Skip the method's own class
                if c == src_class:
                    continue

                # Sum calling strengths to all methods in class c
                member_tensor = torch.tensor(
                    members, dtype=torch.long
                )
                strength = A_hat[m, member_tensor].sum().item()

                if strength > best_strength:
                    best_strength = strength
                    best_class    = c

            pred_targets[m] = best_class

        n_recommended = (pred_targets >= 0).sum().item()
        logger.info(
            f"Refactoring: recommended target class for "
            f"{n_recommended} smelly methods"
        )

        return pred_targets