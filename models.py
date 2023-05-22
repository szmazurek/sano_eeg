import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GINEConv


class ClassicGCN(torch.nn.Module):
    """GCN as in the Efficient Graph Convolutional Networks paper."""

    def __init__(self, in_features, n_nodes=18):
        super(ClassicGCN, self).__init__()
        self.n_nodes = n_nodes
        self.recurrent_1 = GCNConv(in_features, 32, add_self_loops=True, improved=False)
        self.recurrent_2 = GCNConv(32, 64, add_self_loops=True, improved=False)
        self.recurrent_3 = GCNConv(64, 128, add_self_loops=True, improved=False)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.batch_norm_1 = nn.BatchNorm1d(32)
        self.batch_norm_2 = nn.BatchNorm1d(64)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, edge_weight, batch):
        x = x.squeeze(x)
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = self.recurrent_2(h, edge_index, edge_weight)
        h = self.batch_norm_2(h)
        h = F.leaky_relu(h)
        h = self.recurrent_3(h, edge_index, edge_weight)
        h = self.batch_norm_3(h)
        h = F.leaky_relu(h)
        h = global_mean_pool(h, batch)
        h = self.dropout(h)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc4(h)
        return h.squeeze(1)


class GATv2(torch.nn.Module):
    def __init__(self, timestep, sfreq, n_nodes=18, batch_size=32, n_classes=1):
        super(GATv2, self).__init__()
        self.n_nodes = n_nodes
        hidden_dim = 32
        out_dim = 64
        n_heads = 4
        self.recurrent_1 = GATv2Conv(
            6,
            hidden_dim,
            heads=n_heads,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )
        self.recurrent_2 = GATv2Conv(
            hidden_dim * n_heads,
            out_dim,
            heads=n_heads,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )

        self.fc1 = nn.Linear(out_dim * n_heads, 512)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.01)
        self.fc2 = nn.Linear(512, 128)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.01)
        self.fc3 = nn.Linear(128, n_classes)
        nn.init.kaiming_uniform_(self.fc3.weight, a=0.01)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim * n_heads)
        self.batch_norm_2 = nn.BatchNorm1d(out_dim * n_heads)
        self.dropout = nn.Dropout()

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.recurrent_1(x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = self.recurrent_2(h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.batch_norm_2(h)
        h = F.leaky_relu(h)
        h = global_mean_pool(h, batch)
        h = self.dropout(h)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        return h


class GIN(torch.nn.Module):
    """GIN"""

    def __init__(self, sfreq, timestep, dim_h=128):
        super(GIN, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(int((sfreq * timestep / 2) + 1), dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            # edge_dim=1,
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            # edge_dim=1,
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
            # edge_dim=1,
        )

        self.att_1 = GATv2Conv(
            int((sfreq * timestep / 2) + 1),
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )
        self.att_2 = GATv2Conv(
            dim_h,
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )
        self.att_3 = GATv2Conv(
            dim_h,
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )
        self.lin1 = nn.Linear(dim_h * 3, dim_h * 3)
        self.lin2 = nn.Linear(dim_h * 3, 3)

    def forward(self, x, edge_index, batch):
        # Node embeddings
        # _, edge_scores_1 = self.att_1(x, edge_index, return_attention_weights=True)
        h1 = self.conv1(x, edge_index)
        #  _, edge_scores_2 = self.att_2(h1, edge_index, return_attention_weights=True)
        h2 = self.conv2(h1, edge_index)
        #  _, edge_scores_3 = self.att_3(h1, edge_index, return_attention_weights=True)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_mean_pool(h1, batch)
        h2 = global_mean_pool(h2, batch)
        h3 = global_mean_pool(h3, batch)
        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # Classifier
        h = self.lin1(h)
        h = h.relu()
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h
