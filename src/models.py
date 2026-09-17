import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool, LayerNorm
from sparsifiers import (
    EdgeFusion,
    ThresholdSparsifier_ReLU,
    ThresholdSparsifier_Sigmoid,
    DensitySparsifier_ReLU,
    DensitySparsifier_Sigmoid,
    Top_K_Sparsifier_ReLU,
    Top_K_Sparsifier_Sigmoid,
    Raw_Edge_Values
)


class GCN_EEG(nn.Module):
    """GCN model for EEG graph classification supporting various weights-level edge sparsifiers.

    Args:
        num_node_features: Dimensionality of input node features.
        num_classes: Number of target classes (default: 2).
        num_gcn_layers: Number of stacked GCNConv layers (default: 2).
        hidden_dim: Hidden dimension size (default: 32).
        dropout: Dropout probability (default: 0.5).
        only_plv: Whether to use single-metric PLV sparsifiers (default: True). If False, uses multi-metric fusion layer.
        mode: Sparsification mode ('raw', 'threshold', 'density', 'top_k').
        initial_k_ratio: Initial fraction of retained edges (default: 0.5).
        per_node: Whether to learn individual k per electrode for top-k sparsification (default: False).
        gating: Activation function for pruning ('relu' or 'sigmoid').
        edge_attributes: Number of edge features for multi-metric fusion (default: 2).
    """

    def __init__(
        self,
        num_node_features,
        num_classes=2,
        num_gcn_layers=2,
        hidden_dim=32, 
        dropout=0.5,
        only_plv=True,
        mode = 'threshold',  # 'raw', 'threshold', 'density', 'top_k'
        initial_k_ratio = 0.5,
        per_node = False,
        gating = 'relu',
        edge_attributes=2
    ):
        super().__init__()

        self.num_gcn_layers = num_gcn_layers
        self.convs = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.mode = mode
        self.initial_k_ratio = initial_k_ratio
        self.per_node = per_node
        self.edge_attributes = edge_attributes
        

        # Select edge sparsification / fusion module
        if only_plv:
            if mode == 'raw':
                self.edge_sparsifier = Raw_Edge_Values()

            elif mode == 'threshold':
                if gating == 'relu':
                    self.edge_sparsifier = ThresholdSparsifier_ReLU(initial_threshold=0.5)
                elif gating == 'sigmoid':
                    self.edge_sparsifier = ThresholdSparsifier_Sigmoid(initial_threshold=0.5)

            elif mode == 'density':
                if gating == 'relu':
                    self.edge_sparsifier = DensitySparsifier_ReLU()
                elif gating == 'sigmoid':
                    self.edge_sparsifier = DensitySparsifier_Sigmoid()


            elif mode == 'top_k':
                if gating == 'relu':
                    self.edge_sparsifier = Top_K_Sparsifier_ReLU(initial_k_ratio=self.initial_k_ratio, per_node_k=self.per_node)
                elif gating == 'sigmoid':
                    self.edge_sparsifier = Top_K_Sparsifier_Sigmoid(initial_k_ratio=self.initial_k_ratio, per_node_k=self.per_node)
        else:
            self.edge_sparsifier = EdgeFusion(num_edge_features=self.edge_attributes) # for multi-metric fusion experiments, e.g., PLV + PLI


        # GCN backbone
        in_channels = num_node_features
        for i in range(num_gcn_layers):
            conv = GCNConv(in_channels=in_channels, out_channels=hidden_dim)
            self.convs.append(conv)
            self.norm_layers.append(LayerNorm(hidden_dim))
            in_channels = hidden_dim

        self.post_gcn_dim = in_channels

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = nn.LeakyReLU()

    def forward(self, x, edge_index, batch, edge_attr):
        x = self.dropout(x)

        # Compute dynamic edge weights via chosen sparsifier
        if edge_attr is not None:
            if self.mode in ("density", "top_k"):
                edge_weight = self.edge_sparsifier(edge_attr, edge_index, batch)
            else:
                edge_weight = self.edge_sparsifier(edge_attr)


        # Message passing layers
        for conv, bn in zip(self.convs, self.norm_layers):
            x = conv(x, edge_index, edge_weight=edge_weight) 
            x = bn(x)
            x = self.act(x)
            x = self.dropout(x)


        # Readout and classification
        x = global_mean_pool(x, batch)
        x = self.classifier(x)

        return x, edge_weight
    