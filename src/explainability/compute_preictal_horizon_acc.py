import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
from torchmetrics import Accuracy, Precision, Recall, F1Score
from argparse import ArgumentParser
from statistics import mean, stdev
from torch.utils.data import Subset


def compute_prediction_metrics(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_horizon = args.save_dir_horizon
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(save_dir_horizon):
        print("Save directory already exists")
        print(save_dir_horizon)
        return
    os.makedirs(save_dir_horizon)
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    data_fold_list = [os.path.join(data_dir, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()
    time_labels_counter_dict = {}
    time_labels_correct_counter_dict = {}
    for i, fold in enumerate(fold_list):
        print(fold)
        checkpoint_path = os.path.join(
            checkpoint_fold_list[i], os.listdir(checkpoint_fold_list[i])[0]
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
        dataset = GraphDataset(data_fold_list[i])
        n_classes = 3
        features_shape = dataset[0].x.shape[-1]

        model = GATv2Lightning.load_from_checkpoint(
            checkpoint_path,
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
            map_location=torch.device("cpu"),
        )
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=1,
            enable_progress_bar=True,
            deterministic=False,
            log_every_n_steps=1,
            enable_model_summary=False,
        )
        dataset = GraphDataset(data_fold_list[i])
        interictal_samples_index = []
        for i, sample in enumerate(dataset):
            if sample.y == 0:
                interictal_samples_index.append(i)
        interictal_subset = Subset(dataset, interictal_samples_index)
        loader = DataLoader(
            interictal_subset, batch_size=1024, shuffle=False, drop_last=False
        )
        preds = trainer.predict(model, loader)
        preds = torch.cat(preds, dim=0)
        preds = torch.nn.functional.softmax(preds, dim=1).argmax(dim=1)
        ground_truth = [data.y.int().item() for data in interictal_subset]

        time_labels = [
            data.time_labels.int().item() for data in interictal_subset
        ]

        for index in range(len(preds)):
            pred = preds[index].item()
            label = ground_truth[index]
            time_label = time_labels[index]
            if time_label in time_labels_counter_dict.keys():
                time_labels_counter_dict[time_label] += 1
            else:
                time_labels_counter_dict[time_label] = 1
            if pred == label:
                if time_label in time_labels_correct_counter_dict.keys():
                    time_labels_correct_counter_dict[time_label] += 1
                else:
                    time_labels_correct_counter_dict[time_label] = 1

    # Compute accuracy per time label
    time_labels_acc_dict = {}
    for key in time_labels_counter_dict.keys():
        time_labels_acc_dict[key] = (
            time_labels_correct_counter_dict[key]
            / time_labels_counter_dict[key]
        )
    for key in time_labels_acc_dict.keys():
        if key not in time_labels_counter_dict.keys():
            time_labels_acc_dict[key] = 0
    time_labels_acc_dict = dict(
        sorted(time_labels_acc_dict.items(), reverse=False)
    )
    with open(
        os.path.join(save_dir_horizon, f"time_labels_acc.json"),
        "w",
    ) as f:
        json.dump(time_labels_acc_dict, f)
    with open(
        os.path.join(save_dir_horizon, f"time_labels_counter.json"),
        "w",
    ) as f:
        json.dump(time_labels_counter_dict, f)
    with open(
        os.path.join(save_dir_horizon, f"time_labels_correct_counter.json"),
        "w",
    ) as f:
        json.dump(time_labels_correct_counter_dict, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
    )
    parser.add_argument(
        "--save_dir_horizon",
        type=str,
    )
    args = parser.parse_args()
    compute_prediction_metrics(args)
