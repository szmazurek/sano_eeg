import torch
import torch_geometric
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
from torch_geometric.nn import Sequential
from sklearn.utils.class_weight import compute_class_weight
import lightning.pytorch as pl
import os
import json
import networkx as nx
from torchmetrics.classification import MulticlassConfusionMatrix
from sklearn.metrics import balanced_accuracy_score
import seaborn as sns
import matplotlib as mpl
from statistics import mean, stdev
from torch_geometric.explain import AttentionExplainer, Explainer, ModelConfig

from argparse import ArgumentParser

torch_geometric.seed_everything(42)


def compute_attention_explanations(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_att = args.save_dir_att

    if os.path.exists(save_dir_att):
        print("Save directory already exists")
        return
    os.makedirs(save_dir_att)
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    data_fold_list = [os.path.join(data_dir, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()
    att_explainer = AttentionExplainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, fold in enumerate(fold_list):
        print(f"Fold {i}")
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
            map_location=device,
        )

        dataset = GraphDataset(data_fold_list[i])
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            num_workers=2,
            prefetch_factor=20,
        )
        config = ModelConfig(
            "multiclass_classification", task_level="graph", return_type="raw"
        )
        explainer = Explainer(
            model,
            algorithm=att_explainer,
            explanation_type="model",
            model_config=config,
            edge_mask_type="object",
        )
        loader = DataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False
        )

        edge_connection_dict_all = {}
        edge_connection_dict_preictal = {}
        edge_connection_dict_interictal = {}
        edge_connection_dict_ictal = {}
        interictal_cntr = 0
        preictal_cntr = 0
        ictal_cntr = 0
        for n, batch in enumerate(loader):
            explanation = explainer(
                x=batch.x.to(device),
                edge_index=batch.edge_index.to(device),
                target=batch.y.to(device),
                pyg_batch=batch.batch.to(device),
            )
            explanation = explanation.to(device)
            for edge_idx in range(explanation.edge_index.size(1)):
                edge = explanation.edge_index[:, edge_idx].tolist()
                edge.sort()
                edge = str(tuple(edge))
                edge_mask = explanation.edge_mask[edge_idx].item()
                prediciton = torch.argmax(explanation.prediction)
                if edge in edge_connection_dict_all.keys():
                    edge_connection_dict_all[edge] += edge_mask
                else:
                    edge_connection_dict_all[edge] = edge_mask
                if batch.y == 0 and prediciton == 0:
                    if edge in edge_connection_dict_preictal.keys():
                        edge_connection_dict_preictal[edge] += edge_mask
                    else:
                        edge_connection_dict_preictal[edge] = edge_mask
                elif batch.y == 1 and prediciton == 1:
                    if edge in edge_connection_dict_ictal.keys():
                        edge_connection_dict_ictal[edge] += edge_mask
                    else:
                        edge_connection_dict_ictal[edge] = edge_mask
                elif batch.y == 2 and prediciton == 2:
                    if edge in edge_connection_dict_interictal.keys():
                        edge_connection_dict_interictal[edge] += edge_mask
                    else:
                        edge_connection_dict_interictal[edge] = edge_mask
            if batch.y == 0 and prediciton == 0:
                preictal_cntr += 1
            elif batch.y == 1 and prediciton == 1:
                ictal_cntr += 1
            elif batch.y == 2 and prediciton == 2:
                interictal_cntr += 1

            if n % 100 == 0:
                print(f"Batch {n} done")

        edge_connection_dict_all = {
            key: value / (n + 1)
            for key, value in edge_connection_dict_all.items()
        }
        edge_connection_dict_interictal = {
            key: value / interictal_cntr
            for key, value in edge_connection_dict_interictal.items()
        }
        edge_connection_dict_ictal = {
            key: value / ictal_cntr
            for key, value in edge_connection_dict_ictal.items()
        }
        edge_connection_dict_preictal = {
            key: value / preictal_cntr
            for key, value in edge_connection_dict_preictal.items()
        }
        save_path_fold = os.path.join(save_dir_att, f"fold_{i}")
        if not os.path.exists(save_path_fold):
            os.makedirs(save_path_fold)
        with open(
            os.path.join(save_path_fold, "edge_connection_dict_all.json"), "w"
        ) as f:
            json.dump(edge_connection_dict_all, f)
        with open(
            os.path.join(
                save_path_fold, "edge_connection_dict_interictal.json"
            ),
            "w",
        ) as f:
            json.dump(edge_connection_dict_interictal, f)
        with open(
            os.path.join(save_path_fold, "edge_connection_dict_ictal.json"),
            "w",
        ) as f:
            json.dump(edge_connection_dict_ictal, f)
        with open(
            os.path.join(save_path_fold, "edge_connection_dict_preictal.json"),
            "w",
        ) as f:
            json.dump(edge_connection_dict_preictal, f)
        print(f"Fold {i} done")


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
        "--save_dir_att",
        type=str,
    )
    args = parser.parse_args()
    compute_attention_explanations(args)
