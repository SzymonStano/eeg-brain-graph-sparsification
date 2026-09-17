import torch
from src.config import PROCESSED_DIR
from torch_geometric.utils import add_self_loops
from torch.nn.functional import log_softmax
from typing import List, Optional


def load_data_for_subjects(
    subject_list: List[str],
    add_self_loops_flag: bool = False,
    scaling: bool = False,
    only_plv: bool = False,
    threshold: Optional[float] = None,
    quantile: Optional[float] = None,
    top_k: Optional[int] = None,
    ):
    """Loads and preprocesses serialized graph data across a list of subjects.

    Supports optional subject-wise robust feature standardization (using interictal
    median and IQR), PLV-only feature extraction, mutually exclusive static pruning 
    strategies (threshold, quantile, or top_k) applied to the primary PLV metric, 
    and self-loop augmentation.

    Args:
        subject_list: List of patient/subject IDs to load.
        add_self_loops_flag: Whether to add self-loops (fill_value=1.0) to graphs.
        scaling: If True, applies robust median/IQR normalization to node features per patient.
        only_plv: If True, restricts edge attributes to the primary PLV channel (index 0).
        threshold: Absolute cut-off value (float in [0, 1]) for static PLV thresholding.
            Edges with PLV >= threshold are retained.
        quantile: Percentile cut-off value (float in [0, 1]) for quantile-based pruning.
            Retains edges in the top (1 - quantile) fraction per graph.
        top_k: Number of highest-weight incident edges to retain per node.

    Returns:
        List of processed torch_geometric.data.Data graph instances.
    
    Note:
        When `scaling=True`, robust standardization (Median and Interquartile Range, IQR)
        is computed **strictly on interictal (non-seizure) windows** for each subject
        independently. This acts as patient-specific baseline calibration, reducing
        inter-subject variability and encouraging the model to detect relative deviations
        from typical baseline brain activity rather than absolute feature magnitudes.
    """

    combined_data = []

    if add_self_loops_flag:
        print("Loading data with self-loops enabled.")

    if scaling:
        print("Loading data with per-patient robust baseline scaling (Median/IQR).")

    for subj in subject_list:
        file_path = PROCESSED_DIR / f"patient_{subj}.pt"

        if file_path.exists():
            data = torch.load(file_path, weights_only=False)

            for d in data:
                d.x = d.x.float()
                d.y = int(d.y == 1) # binary label: 1 for seizure, 0 for interictal and preictal (earlier labeled as 2)
                d.edge_index = d.edge_index.long()

                # Extract single PLV channel and apply static pruning if requested
                if only_plv:
                    d.edge_attr = d.edge_attr[:, :1]

                    if threshold is not None:
                        plv_values = d.edge_attr.view(-1)
                        mask = plv_values >= threshold
                        d.edge_index = d.edge_index[:, mask]
                        d.edge_attr = d.edge_attr[mask]

                    elif quantile is not None:  
                        plv_values = d.edge_attr.view(-1)
                        mask = plv_values >= torch.quantile(plv_values, quantile) 
                        d.edge_index = d.edge_index[:, mask]
                        d.edge_attr = d.edge_attr[mask]

                    elif top_k is not None:
                        d.edge_index, d.edge_attr = keep_topk_per_node_undirected(
                                                    d.edge_index,
                                                    d.edge_attr,
                                                    k=top_k
                                                )

                # Optional self-loop addition
                if add_self_loops_flag:
                    d.edge_index, d.edge_attr = add_self_loops(
                        d.edge_index,
                        edge_attr=d.edge_attr,
                    fill_value=1.0 
                    )

            # Robust baseline calibration (Median & IQR) fitted strictly on interictal windows
            if scaling:
                interictal_feats = torch.cat([g.x for g in data if g.y != 1], dim=0)

                if interictal_feats.shape[0] > 0:
                    mean = torch.median(interictal_feats, dim=0).values
                    std = torch.quantile(interictal_feats, 0.75, dim=0) - torch.quantile(interictal_feats, 0.25, dim=0) + 1e-6

                    # Normalize all graph windows using the patient's resting baseline
                    for g in data:
                        g.x = (g.x - mean) / std
                            
            combined_data.extend(data)
        else:
            print(f"File not found for subject {subj}: {file_path}")

    return combined_data




def keep_topk_per_node_undirected(edge_index, edge_attr, k) -> tuple[torch.Tensor, torch.Tensor]:
    """Retains top-k strongest incident edges per node for undirected graphs.

    Considers both incoming and outgoing edges for each node based on the primary
    edge feature (PLV, index 0). An edge is preserved if it ranks among the top-k
    connections for either of its endpoint nodes.

    Args:
        edge_index: Graph connectivity tensor of shape [2, E].
        edge_attr: Edge attributes tensor of shape [E, F] where index 0 is PLV.
        k: Maximum number of incident edges to retain per node.

    Returns:
        Tuple of (sparsified_edge_index, sparsified_edge_attr).
    """
    num_nodes = int(edge_index.max()) + 1
    plv = edge_attr[:, 0]

    keep_mask = torch.zeros(edge_index.size(1), dtype=torch.bool)

    # Evaluate incident edges for each node
    for node in range(num_nodes):
        # Find all edges connected to the node (undirected view)
        incident_edges = (
            (edge_index[0] == node) |
            (edge_index[1] == node)
        ).nonzero(as_tuple=True)[0]

        if len(incident_edges) <= k:
            keep_mask[incident_edges] = True
            continue

        # Select top-k edges by PLV weight
        incident_plv = plv[incident_edges]
        topk_local = torch.topk(
            incident_plv,
            k=k,
            largest=True
        ).indices

        keep_mask[incident_edges[topk_local]] = True

    return edge_index[:, keep_mask], edge_attr[keep_mask]


class FocalLoss(torch.nn.Module):
    """Multi-class Focal Loss for addressing class imbalance.

    Down-weights well-classified examples by applying a modulating factor
    (1 - p_t)^gamma to the standard cross-entropy loss, focusing training on hard samples:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t).

    Args:
        gamma: Focusing parameter balancing easy vs. hard examples (default: 2.0).
        alpha: Class weighting tensor of shape [C] (default: None).
        reduction: Specifies reduction to apply ('mean', 'sum', or 'none') (default: 'mean').
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha 
        self.reduction = reduction

    def forward(self, inputs, targets):
        """Forward pass.

        Args:
            inputs: Predicted unnormalized logits of shape [N, C].
            targets: Ground-truth class indices of shape [N].

        Returns:
            Computed loss scalar (or tensor if reduction='none').
        """
        log_p = log_softmax(inputs, dim=1)
        p = torch.exp(log_p)
        
        # Gather probabilities and log-probabilities for true target classes
        log_p_target = log_p.gather(1, targets.view(-1, 1)).view(-1)
        p_target = p.gather(1, targets.view(-1, 1)).view(-1)

        # Compute focal modulating factor and unweighted loss
        loss = -1 * (1 - p_target)**self.gamma * log_p_target

        # Apply class-specific alpha weighting if provided
        if self.alpha is not None:
            alpha_weight = self.alpha.to(inputs.device).gather(0, targets)
            loss = alpha_weight * loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

