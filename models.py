import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GINEConv
from torchmetrics import (
    Specificity,
    Recall,
    F1Score,
    AUROC,
)


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
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_classes: int = 1,
    ):
        super(GATv2, self).__init__()
        out_dim = hidden_dim * 2
        self.recurrent_1 = GATv2Conv(
            in_features,
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


class GATv2Lightning(pl.LightningModule):
    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        n_heads: int = 4,
        n_classes: int = 2,
        fft_mode: bool = False,
        lr=0.00001,
        weight_decay=0.0001,
    ):
        super(GATv2Lightning, self).__init__()
        assert n_classes > 1, "n_classes must be greater than 1"
        self.classification_mode = "multiclass" if n_classes > 2 else "binary"
        out_dim = hidden_dim * 2
        self.recurrent_1 = GATv2Conv(
            in_features,
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
        self.fc3 = nn.Linear(128, n_classes if n_classes > 2 else 1)
        nn.init.kaiming_uniform_(self.fc3.weight, a=0.01)
        self.batch_norm_1 = nn.BatchNorm1d(hidden_dim * n_heads)
        self.batch_norm_2 = nn.BatchNorm1d(out_dim * n_heads)
        self.dropout = nn.Dropout()
        self.n_classes = n_classes
        self.fft_mode = fft_mode
        self.lr = lr
        self.weight_decay = weight_decay
        if self.classification_mode == "multiclass":
            self.loss = nn.CrossEntropyLoss()
            self.recall = Recall(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.specificity = Specificity(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.auroc = AUROC(task="multiclass", num_classes=n_classes)
        else:
            self.loss = nn.BCEWithLogitsLoss()
            self.recall = Recall(task="binary", threshold=0.5)
            self.specificity = Specificity(task="binary", threshold=0.5)
            self.auroc = AUROC(task="binary")
        self.training_step_outputs = []
        self.training_step_gt = []
        self.validation_step_outputs = []
        self.validation_step_gt = []
        self.test_step_outputs = []
        self.test_step_gt = []

    def forward(self, x, edge_index, pyg_batch, edge_attr=None):
        if self.fft_mode:
            x = torch.square(torch.abs(x)).float()

        h = self.recurrent_1(x.float(), edge_index=edge_index, edge_attr=edge_attr)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = self.recurrent_2(h, edge_index=edge_index, edge_attr=edge_attr)
        h = self.batch_norm_2(h)
        h = F.leaky_relu(h)
        h = global_mean_pool(h, pyg_batch)
        h = self.dropout(h)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc3(h).squeeze(1)

        return h

    def unpack_data_batch(self, data_batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        y = data_batch.y.long()
        pyg_batch = data_batch.batch
        edge_attr = data_batch.edge_attr  ## unused for now
        return x, edge_index, y, pyg_batch, edge_attr

    def training_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch)
        loss = self.loss(y_hat, y)
        self.training_step_outputs.append(y_hat)
        self.training_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return loss

    def on_train_epoch_end(self):
        training_step_outputs = torch.cat(self.training_step_outputs)
        training_step_gt = torch.cat(self.training_step_gt)
        rec = self.recall(training_step_outputs, training_step_gt)
        spec = self.specificity(training_step_outputs, training_step_gt)
        auroc = self.auroc(training_step_outputs, training_step_gt)
        self.log_dict(
            {
                "train_recall": rec,
                "train_specificity": spec,
                "train_auroc": auroc,
            },
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.training_step_outputs.clear()
        self.training_step_gt.clear()

    def validation_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        loss = self.loss(y_hat, y)
        self.validation_step_outputs.append(y_hat)
        self.validation_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_validation_epoch_end(self) -> None:
        validation_step_outputs = torch.cat(self.validation_step_outputs)
        validation_step_gt = torch.cat(self.validation_step_gt)
        rec = self.recall(validation_step_outputs, validation_step_gt)
        spec = self.specificity(validation_step_outputs, validation_step_gt)
        auroc = self.auroc(validation_step_outputs, validation_step_gt)
        self.log_dict(
            {
                "val_recall": rec,
                "val_specificity": spec,
                "val_auroc": auroc,
            },
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.validation_step_outputs.clear()
        self.validation_step_gt.clear()

    def test_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        loss = self.loss(y_hat, y)
        self.test_step_outputs.append(y_hat)
        self.test_step_gt.append(y)
        batch_size = pyg_batch.max() + 1
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        return loss

    def on_test_epoch_end(self) -> None:
        test_step_outputs = torch.cat(self.test_step_outputs)
        test_step_gt = torch.cat(self.test_step_gt)
        rec = self.recall(test_step_outputs, test_step_gt)
        spec = self.specificity(test_step_outputs, test_step_gt)
        auroc = self.auroc(test_step_outputs, test_step_gt)
        self.log_dict(
            {
                "test_recall": rec,
                "test_specificity": spec,
                "test_auroc": auroc,
            },
            logger=True,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.test_step_outputs.clear()
        self.test_step_gt.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch.x
        edge_index = batch.edge_index
        pyg_batch = batch.batch
        edge_attr = batch.edge_attr
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


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
