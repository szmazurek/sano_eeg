import torch
import lightning.pytorch as pl
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
from torch_geometric.loader import DataLoader

chkpt_path = "/net/tscratch/people/plgmazurekagh/sano_eeg/checkpoints_final/fold_0/epoch=49-val_loss=0.176.ckpt"
data_path = "/net/tscratch/people/plgmazurekagh/sano_eeg/saved_folds/final_run_kfold/fold_0"
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

# model = GATv2Lightning(
#     features_shape,
#     n_classes=n_classes,
#     n_gat_layers=n_gat_layers,
#     hidden_dim=hidden_dim,
#     n_heads=n_heads,
#     slope=slope,
#     dropout=dropout,
#     pooling_method=pooling_method,
#     activation=activation,
#     norm_method=norm_method,
#     lr=lr,
#     weight_decay=weight_decay,
# )
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
    # kwargs={
    #     "in_features": features_shape,
    #     "n_classes": n_classes,
    #     "n_gat_layers": n_gat_layers,
    #     "hidden_dim": hidden_dim,
    #     "n_heads": n_heads,
    #     "slope": slope,
    #     "dropout": dropout,
    #     "pooling_method": pooling_method,
    #     "activation": activation,
    #     "norm_method": norm_method,
    #     "lr": lr,
    #     "weight_decay": weight_decay,
    # },
)

loader = DataLoader(
    dataset, batch_size=len(dataset), shuffle=False, drop_last=False
)
trainer.test(model, dataloaders=loader)
