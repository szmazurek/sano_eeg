"""Script for validating a model on a single fold. Data was previously
saved in a k-fold cross validation scheme. This script loads the data
from a single fold and validates the model on it. The model is loaded
from a checkpoint file saved during training with that fold as the
test set.
"""
import lightning.pytorch as pl
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
from torch_geometric.loader import DataLoader
from argparse import ArgumentParser



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--chkpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.parse_args()
    args = parser.parse_args()
    chkpt_path = args.chkpt_path
    data_path = args.data_path
    trainer = pl.Trainer(
        accelerator="auto",
        precision="16-mixed",
        devices=1,
        max_epochs=1,
        enable_progress_bar=True,
        deterministic=False,
        log_every_n_steps=1,
        enable_model_summary=False,
    )
    """Manual configuration matching the model used for training."""
    n_gat_layers = 1
    hidden_dim = 32
    dropout = 0.0
    slope = 0.0025
    pooling_method = "mean"
    norm_method = "batch"
    activation = "leaky_relu"
    n_heads = 9
    lr = 0.0012
    weight_decay = 0.0078
    dataset = GraphDataset(data_path)
    n_classes = 3
    features_shape = dataset[0].x.shape[-1]

    model = GATv2Lightning.load_from_checkpoint(
        chkpt_path,
        in_features=features_shape,
        n_classes=n_classes,
        n_gat_layers=n_gat_layers,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        slope=slope,
        dropout=dropout,
        pooling_method=pooling_method,
        activation=activation,
        norm_method=norm_method,
        lr=lr,
        weight_decay=weight_decay,
    )

    loader = DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    trainer.test(model, dataloaders=loader)
