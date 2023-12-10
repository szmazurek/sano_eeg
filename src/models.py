import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import (
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, Linear, Sequential
from typing import List, Union, Tuple, Callable
from torch_geometric.nn.norm import BatchNorm, LayerNorm
from torchmetrics import Specificity, Recall, AUROC, F1Score


class ClassicGCN(torch.nn.Module):
    """GCN as in the Efficient Graph Convolutional Networks paper.
    Used for some preliminary experiments.


    """

    def __init__(self, in_features, n_nodes=18):
        super(ClassicGCN, self).__init__()
        self.n_nodes = n_nodes
        self.recurrent_1 = GCNConv(
            in_features, 32, add_self_loops=True, improved=False
        )

        self.recurrent_2 = GCNConv(32, 64, add_self_loops=True, improved=False)
        self.recurrent_3 = GCNConv(
            64, 128, add_self_loops=True, improved=False
        )
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
    """GATv2 as in the paper. Main model explored in the project."""

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
    """Lightning Module implementing GATv2 network used for experiments."""

    def __init__(
        self,
        in_features: int,
        n_gat_layers: int = 2,
        hidden_dim: int = 32,
        n_heads: int = 4,
        dropout: float = 0.4,
        slope: float = 0.01,
        pooling_method: str = "mean",
        activation: str = "leaky_relu",
        norm_method: str = "batch",
        n_classes: int = 2,
        fft_mode: bool = False,
        lr=0.00001,
        weight_decay=0.0001,
        class_weights=None,
    ):
        super(GATv2Lightning, self).__init__()
        assert n_classes > 1, "n_classes must be greater than 1"
        self.classification_mode = "multiclass" if n_classes > 2 else "binary"
        assert activation in [
            "leaky_relu",
            "relu",
        ], 'Activation must be either "leaky_relu" or "relu"'
        assert norm_method in [
            "batch",
            "layer",
        ], 'Norm_method must be either "batch" or "layer"'
        assert pooling_method in [
            "mean",
            "max",
            "add",
        ], "Pooling_method must be either 'mean', 'max', or 'add'"
        if class_weights is not None:
            if n_classes > 2:
                assert (
                    len(class_weights) == n_classes
                ), "Number of class weights must match number of classes"
            else:
                assert (
                    len(class_weights) == 1
                ), "Only one class weight must be provided for binary classification"
        act_fn = (
            nn.LeakyReLU(slope, inplace=True)
            if activation == "leaky_relu"
            else nn.ReLU(inplace=True)
        )
        norm_layer = (
            BatchNorm(hidden_dim * n_heads)
            if norm_method == "batch"
            else LayerNorm(hidden_dim * n_heads)
        )
        dropout_layer = nn.Dropout(dropout)
        classifier_out_neurons = n_classes if n_classes > 2 else 1
        feature_extractor_list: List[
            Union[Tuple[Callable, str], Callable]
        ] = []
        for i in range(n_gat_layers):
            feature_extractor_list.append(
                (
                    GATv2Conv(
                        in_features if i == 0 else hidden_dim * n_heads,
                        hidden_dim,  # / (2**i) if i != 0 else hidden_dim,
                        heads=n_heads,
                        negative_slope=slope,
                        dropout=dropout,
                        add_self_loops=True,
                        improved=True,
                        edge_dim=1,
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
            feature_extractor_list.append(norm_layer)
            feature_extractor_list.append(act_fn)
        self.feature_extractor = Sequential(
            "x, edge_index, edge_attr", feature_extractor_list
        )

        self.classifier = nn.Sequential(
            Linear(
                hidden_dim * n_heads, 512, weight_initializer="kaiming_uniform"
            ),
            dropout_layer,
            act_fn,
            Linear(512, 128, weight_initializer="kaiming_uniform"),
            dropout_layer,
            act_fn,
            Linear(
                128,
                classifier_out_neurons,
                weight_initializer="kaiming_uniform",
            ),
        )

        if pooling_method == "mean":
            self.pooling_method = global_mean_pool
        elif pooling_method == "max":
            self.pooling_method = global_max_pool
        elif pooling_method == "add":
            self.pooling_method = global_add_pool
        self.n_classes = n_classes
        self.fft_mode = fft_mode
        self.lr = lr
        self.weight_decay = weight_decay
        if class_weights is None:
            class_weights = torch.ones(n_classes)
        self.class_weights = class_weights
        if self.classification_mode == "multiclass":
            self.f1_score = F1Score(task="multiclass", num_classes=n_classes)
            self.loss = nn.CrossEntropyLoss(weight=class_weights)
            self.recall = Recall(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.specificity = Specificity(
                task="multiclass", num_classes=n_classes, threshold=0.5
            )
            self.auroc = AUROC(task="multiclass", num_classes=n_classes)
        elif self.classification_mode == "binary":
            self.f1_score = F1Score(task="binary", threshold=0.5)
            self.loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            self.recall = Recall(task="binary", threshold=0.5)
            self.specificity = Specificity(task="binary", threshold=0.5)
            self.auroc = AUROC(task="binary")
        self.training_step_outputs: List[torch.Tensor] = []
        self.training_step_gt: List[torch.Tensor] = []
        self.validation_step_outputs: List[torch.Tensor] = []
        self.validation_step_gt: List[torch.Tensor] = []
        self.test_step_outputs: List[torch.Tensor] = []
        self.test_step_gt: List[torch.Tensor] = []

    def forward(self, x, edge_index, pyg_batch, edge_attr=None):
        h = self.feature_extractor(x, edge_index=edge_index, edge_attr=None)
        h = self.pooling_method(h, pyg_batch)
        h = self.classifier(h)
        return h

    def unpack_data_batch(self, data_batch):
        x = data_batch.x
        edge_index = data_batch.edge_index
        y = (
            data_batch.y.long()
            if self.classification_mode == "multiclass"
            else data_batch.y
        )
        pyg_batch = data_batch.batch
        try:
            edge_attr = data_batch.edge_attr.float()
        except AttributeError:
            edge_attr = None

        return x, edge_index, y, pyg_batch, edge_attr

    def training_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
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
        f1_score = self.f1_score(training_step_outputs, training_step_gt)
        self.log_dict(
            {
                "train_sensitivity": rec,
                "train_specificity": spec,
                "train_AUROC": auroc,
                "train_f1_score": f1_score,
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
        f1_score = self.f1_score(validation_step_outputs, validation_step_gt)
        self.log_dict(
            {
                "val_sensitivity": rec,
                "val_specificity": spec,
                "val_AUROC": auroc,
                "val_f1_score": f1_score,
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
            on_step=False,
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
        f1_score = self.f1_score(test_step_outputs, test_step_gt)
        self.log_dict(
            {
                "test_sensitivity": rec,
                "test_specificity": spec,
                "test_AUROC": auroc,
                "test_f1_score": f1_score,
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


class GINCustom(torch.nn.Module):
    """GIN Custom implementation - used in some of the preliminary experiments"""

    def __init__(self, in_features, dim_h=128, dropout=0.5, n_classes=2):
        super(GINCustom, self).__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(in_features, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
        )
        self.conv3 = GINConv(
            nn.Sequential(
                nn.Linear(dim_h, dim_h),
                nn.BatchNorm1d(dim_h),
                nn.ReLU(),
                nn.Linear(dim_h, dim_h),
                nn.ReLU(),
            ),
        )

        self.att_1 = GATv2Conv(
            in_features,
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=dropout,
            improved=True,
        )
        self.att_2 = GATv2Conv(
            dim_h,
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=dropout,
            improved=True,
        )
        self.att_3 = GATv2Conv(
            dim_h,
            dim_h,
            heads=1,
            negative_slope=0.01,
            dropout=dropout,
            improved=True,
        )
        self.lin1 = nn.Linear(dim_h * 3, dim_h * 3)
        self.lin2 = nn.Linear(dim_h * 3, n_classes if n_classes > 2 else 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, pyg_batch, edge_attr=None):
        # Node embeddings
        # _, edge_scores_1 = self.att_1(x, edge_index, return_attention_weights=True)
        h1 = self.conv1(x, edge_index)
        #  _, edge_scores_2 = self.att_2(h1, edge_index, return_attention_weights=True)
        h2 = self.conv2(h1, edge_index)
        #  _, edge_scores_3 = self.att_3(h1, edge_index, return_attention_weights=True)
        h3 = self.conv3(h2, edge_index)

        # Graph-level readout
        h1 = global_add_pool(h1, pyg_batch)
        h2 = global_add_pool(h2, pyg_batch)
        h3 = global_add_pool(h3, pyg_batch)

        # Concatenate graph embeddings
        h = torch.cat((h1, h2, h3), dim=1)
        # Classifier
        h = self.lin1(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.lin2(h)
        return h.squeeze(1)


class GINLightning(pl.LightningModule):
    """Lightning module for GIN model"""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 128,
        n_classes: int = 2,
        fft_mode: bool = False,
        dropout: float = 0.5,
        lr=0.001,
        weight_decay=0.01,
    ):
        super(GINLightning, self).__init__()
        assert n_classes > 1, "n_classes must be greater than 1"
        self.classification_mode = "multiclass" if n_classes > 2 else "binary"
        self.model = GINCustom(
            in_features=in_features,
            dim_h=hidden_dim,
            n_classes=n_classes,
            dropout=dropout,
        )

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

    def unpack_data_batch(self, data_batch):
        if self.fft_mode:
            x = torch.square(torch.abs(data_batch.x)).float()
        else:
            x = data_batch.x.float()
        edge_index = data_batch.edge_index
        y = (
            data_batch.y.long()
            if self.classification_mode == "multiclass"
            else data_batch.y
        )

        pyg_batch = data_batch.batch
        edge_attr = data_batch.edge_attr.float()
        return x, edge_index, y, pyg_batch, edge_attr

    def forward(self, x, edge_index, pyg_batch, edge_attr=None):
        return self.model(x, edge_index, pyg_batch)  # Edge attr not used!!!

    def training_step(self, batch, batch_idx):
        x, edge_index, y, pyg_batch, edge_attr = self.unpack_data_batch(batch)
        y_hat = self.forward(x, edge_index, pyg_batch, edge_attr)
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
