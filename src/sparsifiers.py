import torch
import torch.nn as nn
import numpy as np


class EdgeFusion(nn.Module):
    """
    Experimental layer for fusing multiple edge features/metrics.

    Combines multiple feature channels (e.g., PLV, PLI, correlation) using a learnable
    linear combination with an initial negative bias to encourage sparsity,
    followed by a Sigmoid gating mechanism.

    Args:
        num_edge_features: Number of input feature channels per edge.
    """
    def __init__(self, num_edge_features):

        super().__init__()

        # learnable weights per metric
        self.alpha = nn.Parameter(torch.ones(num_edge_features) * 1/num_edge_features)  # start with equal weights

        # bias
        self.bias = nn.Parameter(torch.tensor(-0.5 * num_edge_features))  # start with negative bias to encourage sparsity

    def forward(self, edge_attr)-> torch.Tensor:
        """Forward pass.

        Args:
            edge_attr: Edge features tensor of shape [E, F] (e.g., PLV, PLI, corr).

        Returns:
            Fused and gated edge weights of shape [E].
        """
        # Linear combination across feature channels
        w = (edge_attr * self.alpha).sum(dim=-1) + self.bias

        # gating
        return torch.sigmoid(w)

    
class Raw_Edge_Values(nn.Module):
    """Passthrough sparsifier that leaves edge weights completely unchanged."""
    def __init__(self):
        super().__init__()

    def forward(self, edge_attr):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
            
        w = edge_attr
        return w


# --- THRESHOLD SPARSIFIER ---
class ThresholdSparsifier_ReLU(nn.Module):
    """
    Learnable edge sparsifier using shifted ReLU function.

    Zeros out edge weights below a trainable scalar threshold (theta in [0, 1])
    and linearly shifts the remaining weights: w' = max(0, w - theta).

    Args:
        initial_threshold: Initial threshold value (default: 0.5).
    """
    def __init__(self, initial_threshold=0.5):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))

    def forward(self, edge_attr):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
            
        w = torch.relu(edge_attr - self.threshold)

        with torch.no_grad():
            self.threshold.clamp_(0.0, 1.0)
            
        return w

class ThresholdSparsifier_Sigmoid(nn.Module):
    """
    Smooth edge sparsifier using a Sigmoid-based soft gating mechanism.

    Multiplies edge weights by a continuous gate: w' = w * sigmoid(alpha * (w - theta)),
    providing a differentiable approximation of a threshold cut-off.

    Args:
        initial_threshold: Initial scalar threshold theta (default: 0.5).
        alpha: Slope parameter controlling gate sharpness (default: 15.0).
    """
    def __init__(self, initial_threshold=0.5, alpha=15.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(initial_threshold))
        self.alpha = alpha  # Controls transition slope

    def forward(self, edge_attr):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
            
        # Smooth gating mechanism (Sigmoid-based soft selection)
        gate = torch.sigmoid(self.alpha * (edge_attr - self.threshold))
        w = edge_attr * gate
        
        with torch.no_grad():
            self.threshold.clamp_(0.0, 1.0)
            
        return w


class DensitySparsifier_Sigmoid(nn.Module):
    """Smooth edge sparsifier based on a learnable density threshold.

    Estimates per-graph quantile thresholds via differentiable linear interpolation
    over sorted edge weights and applies a Sigmoid soft gate.

    Args:
        initial_top_k_ratio: Initial fraction of retained edges (default: 0.3).
        alpha: Slope parameter controlling gate sharpness (default: 15.0).
    """
    def __init__(self, initial_density=0.3, alpha=15.0):
        super().__init__()
        init_logit = torch.log(torch.tensor(initial_density) / (1.0 - torch.tensor(initial_density)))
        self.top_k_logit = nn.Parameter(init_logit)
        self.alpha = alpha  

    def forward(self, edge_attr, edge_index, batch):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
            
        # Map logit to pre-set ratio range [0.05, 0.95]
        top_k_ratio = torch.sigmoid(self.top_k_logit)
        top_k_ratio = 0.05 + 0.95 * top_k_ratio 

        num_graphs = batch.max().item() + 1
        total_edges = edge_attr.numel()
        edges_per_graph = total_edges // num_graphs  

        # Reshape to [num_graphs, edges_per_graph]
        matrix_edges = edge_attr.view(num_graphs, edges_per_graph)

        # Sort edge weights per graph
        sorted_edges, _ = torch.sort(matrix_edges, dim=1)
        
        # Differentiable continuous quantile estimation (linear interpolation)
        exact_index = (edges_per_graph - 1) * (1.0 - top_k_ratio)
        idx_below = torch.floor(exact_index).long()
        idx_above = torch.ceil(exact_index).long()
        weight_above = exact_index - idx_below.float()
        
        # Gather values around the target rank
        val_below = sorted_edges.gather(1, idx_below.view(1, 1).expand(num_graphs, 1))
        val_above = sorted_edges.gather(1, idx_above.view(1, 1).expand(num_graphs, 1))
        
        # Compute smooth threshold per graph
        thresholds = (1.0 - weight_above) * val_below + weight_above * val_above

        # Broadcast graph-level thresholds back to edges [E]
        edge_batch = batch[edge_index[0]]
        edge_thresholds = thresholds[edge_batch].squeeze(-1)

        # Soft gating mechanism
        gate = torch.sigmoid(self.alpha * (edge_attr - edge_thresholds))
        w = edge_attr * gate
        
        # Cache ratio for monitoring/logging
        self.last_ratio = top_k_ratio.item()
        
        return w

class DensitySparsifier_ReLU(nn.Module):
    """Edge sparsifier combining learnable density thresholds with shifted ReLU.

    Estimates per-graph quantile thresholds via differentiable linear interpolation
    and sets edges below the threshold to zero: w' = max(0, w - threshold).

    Args:
        initial_top_k_ratio: Initial fraction of retained edges (default: 0.3).
    """
    def __init__(self, initial_density=0.3):
        super().__init__()
        init_logit = torch.log(torch.tensor(initial_density) / (1.0 - torch.tensor(initial_density)))
        self.top_k_logit = nn.Parameter(init_logit)

    def forward(self, edge_attr, edge_index, batch):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        # Map logit to pre-set ratio range [0.05, 0.95]
        top_k_ratio = torch.sigmoid(self.top_k_logit)
        top_k_ratio = 0.05 + 0.95 * top_k_ratio

        num_graphs = batch.max().item() + 1
        total_edges = edge_attr.numel()
        edges_per_graph = total_edges // num_graphs  

        # Reshape to [num_graphs, edges_per_graph]
        matrix_edges = edge_attr.view(num_graphs, edges_per_graph)

        # Sort edge weights per graph
        sorted_edges, _ = torch.sort(matrix_edges, dim=1)

        # Differentiable continuous quantile estimation (linear interpolation)
        exact_index = (edges_per_graph - 1) * (1.0 - top_k_ratio)
        idx_below = torch.floor(exact_index).long()
        idx_above = torch.ceil(exact_index).long()
        weight_above = exact_index - idx_below.float()
       
        # Gather values around the target rank
        val_below = sorted_edges.gather(1, idx_below.view(1, 1).expand(num_graphs, 1))
        val_above = sorted_edges.gather(1, idx_above.view(1, 1).expand(num_graphs, 1))
       
        # Compute smooth threshold per graph
        thresholds = (1.0 - weight_above) * val_below + weight_above * val_above

        # Broadcast graph-level thresholds back to edges [E]
        edge_batch = batch[edge_index[0]]
        edge_thresholds = thresholds[edge_batch].squeeze(-1)

        # ReLU gating mechanism
        w = torch.relu(edge_attr - edge_thresholds)
       
        # Cache ratio for monitoring/logging
        self.last_ratio = top_k_ratio.item()
       
        return w

    


class Top_K_Sparsifier_Sigmoid(nn.Module):
    """
    Learnable node-wise or global K-nearest neighbors edge sparsifier.

    Computes a soft threshold for every node corresponding to its (soft) k-th
    strongest outgoing edge. The retention ratio can be a single global scalar
    or unique per node position (`per_node_k=True`), which fits EEG channel structures.

    Args:
        num_nodes_per_graph: Number of nodes per graph (required if per_node_k=True).
            Default: 22 (matching chosen EEG setup).
        initial_k_ratio: Initial fraction of retained edges per node (default: 0.5).
        alpha: Slope parameter controlling gate sharpness (default: 15.0).
        per_node_k: If True, learns an independent threshold ratio for each node/channel
            position (default: False).
    """

    def __init__(self, num_nodes_per_graph=22, initial_k_ratio=0.5, alpha=15.0, per_node_k=False):
        super().__init__()
        self.alpha = alpha
        self.per_node_k = per_node_k

        init_logit = torch.log(
            torch.tensor(initial_k_ratio) / (1.0 - torch.tensor(initial_k_ratio))
        )

        if per_node_k:
            assert num_nodes_per_graph is not None, \
                "num_nodes_per_graph is required when per_node_k=True"
            # one independent logit per channel/node position
            self.top_k_logit = nn.Parameter(init_logit.repeat(num_nodes_per_graph))
        else:
            self.top_k_logit = nn.Parameter(init_logit)

    def forward(self, edge_attr, edge_index, batch):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        num_nodes = batch.size(0)
        num_graphs = batch.max().item() + 1
        nodes_per_graph = num_nodes // num_graphs

        total_edges = edge_attr.numel()
        edges_per_graph = total_edges // num_graphs
        edges_per_node = edges_per_graph // nodes_per_graph

        # Map logits to ratio in (0.05, 0.80) -> fraction of neighbours kept
        k_ratio = torch.sigmoid(self.top_k_logit)
        k_ratio = 0.05 + 0.75 * k_ratio  # scalar, or [nodes_per_graph]

        if self.per_node_k:
            # Expand per-channel ratios across all graphs in the batch
            k_ratio_full = k_ratio.repeat(num_graphs)        # [num_nodes]
        else:
            k_ratio_full = k_ratio.expand(num_nodes)         # [num_nodes]

        # Group edges by their source node and sort each node's values
        src = edge_index[0]
        order = torch.argsort(src, stable=True)
        sorted_attr = edge_attr[order]

        matrix = sorted_attr.view(num_nodes, edges_per_node)
        sorted_edges, _ = torch.sort(matrix, dim=1)

        # Differentiable soft-quantile calculation per node (same interpolation trick as DensitySparsifier,
        # but the quantile position now depends on each node's own k_ratio (or shared one if per_node_k=False) )
        exact_index = (edges_per_node - 1) * (1.0 - k_ratio_full)  # [num_nodes]

        idx_below = torch.floor(exact_index).long()
        idx_above = torch.ceil(exact_index).long()
        weight_above = exact_index - idx_below.float()

        val_below = sorted_edges.gather(1, idx_below.unsqueeze(1)).squeeze(1)
        val_above = sorted_edges.gather(1, idx_above.unsqueeze(1)).squeeze(1)

        node_thresholds = (1.0 - weight_above) * val_below + weight_above * val_above  # [num_nodes]

        # Broadcast each node's threshold to its outgoing edges
        edge_thresholds = node_thresholds[src]

        # Differentiable gating
        gate = torch.sigmoid(self.alpha * (edge_attr - edge_thresholds))
        w = edge_attr * gate


        # Cache metrics for monitoring/logging
        if self.per_node_k:
            self.last_k = [np.round(i.item() * edges_per_node, 3).item() for i in k_ratio]
        else:
            self.last_k_ratio = k_ratio.mean().item() if self.per_node_k else k_ratio.item()
            self.last_k = self.last_k_ratio * edges_per_node

        return w





class Top_K_Sparsifier_ReLU(nn.Module):
    """Learnable node-wise or global K-nearest neighbors sparsifier using shifted ReLU.

    Computes a soft threshold per node based on its target retention ratio and
    performs hard pruning (zeroing out) of edges below the threshold via shifted ReLU:
    w' = max(0, w - threshold).

    Args:
        num_nodes_per_graph: Number of nodes per graph (required if per_node_k=True).
            Default: 22 (matching chosen EEG setup).
        initial_k_ratio: Initial fraction of retained edges per node (default: 0.5).
        per_node_k: If True, learns an independent threshold ratio for each node/channel
            position (default: False).
    """
    def __init__(self, num_nodes_per_graph=22, initial_k_ratio=0.5, per_node_k=False):
        super().__init__()
        self.per_node_k = per_node_k

        init_logit = torch.log(
            torch.tensor(initial_k_ratio) / (1.0 - torch.tensor(initial_k_ratio))
        )

        if per_node_k:
            assert num_nodes_per_graph is not None, "num_nodes_per_graph is required when per_node_k=True"
            self.top_k_logit = nn.Parameter(init_logit.repeat(num_nodes_per_graph))
        else:
            self.top_k_logit = nn.Parameter(init_logit)

    def forward(self, edge_attr, edge_index, batch):
        if edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)

        num_nodes = batch.size(0)
        num_graphs = batch.max().item() + 1
        nodes_per_graph = num_nodes // num_graphs

        total_edges = edge_attr.numel()
        edges_per_graph = total_edges // num_graphs
        edges_per_node = edges_per_graph // nodes_per_graph

        # Map logits to valid retention ratio range (0.05, 0.80)
        k_ratio = torch.sigmoid(self.top_k_logit)
        k_ratio = 0.05 + 0.75 * k_ratio 

        if self.per_node_k:
            k_ratio_full = k_ratio.repeat(num_graphs)        # [num_nodes]
        else:
            k_ratio_full = k_ratio.expand(num_nodes)         # [num_nodes]

        # Group edges by source node and sort each node's weights
        src = edge_index[0]
        order = torch.argsort(src, stable=True)
        sorted_attr = edge_attr[order]

        matrix = sorted_attr.view(num_nodes, edges_per_node)
        sorted_edges, _ = torch.sort(matrix, dim=1)


        # Differentiable soft-quantile calculation per node (linear interpolation)
        exact_index = (edges_per_node - 1) * (1.0 - k_ratio_full) 

        idx_below = torch.floor(exact_index).long()
        idx_above = torch.ceil(exact_index).long()
        weight_above = exact_index - idx_below.float()

        val_below = sorted_edges.gather(1, idx_below.unsqueeze(1)).squeeze(1)
        val_above = sorted_edges.gather(1, idx_above.unsqueeze(1)).squeeze(1)

        node_thresholds = (1.0 - weight_above) * val_below + weight_above * val_above  

        # Broadcast node thresholds to their outgoing edges
        edge_thresholds = node_thresholds[src]

        # Hard pruning via shifted ReLU
        w = torch.relu(edge_attr - edge_thresholds)

        # Cache metrics for monitoring/logging
        if self.per_node_k:
            self.last_k = [np.round(i.item() * edges_per_node, 3).item() for i in k_ratio]
        else:
            self.last_k_ratio = k_ratio.mean().item() if self.per_node_k else k_ratio.item()
            self.last_k = self.last_k_ratio * edges_per_node

        return w