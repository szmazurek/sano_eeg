import torch
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from models import GATv2Lightning
from utils.dataloader_utils import GraphDataset
import lightning.pytorch as pl
import os
import json
from torch_geometric import seed_everything
from argparse import ArgumentParser
from torch_geometric.explain import GNNExplainer, Explainer, ModelConfig


seed_everything(42)


def compute_feature_importances(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_importances = args.save_dir_importances
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(save_dir_importances):
        print("Save directory already exists")
        print(save_dir_importances)
        return
    os.makedirs(save_dir_importances)
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    data_fold_list = [os.path.join(data_dir, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()
    for i, fold in enumerate(fold_list):
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        gnn_explainer = GNNExplainer(epochs=100, lr=0.01)
        sum_masks = torch.zeros((18, 10))
        interictal_masks = torch.zeros((18, 10))
        ictal_masks = torch.zeros((18, 10))
        preictal_masks = torch.zeros((18, 10))
        interictal_cntr = 0
        preictal_cntr = 0
        ictal_cntr = 0
        config = ModelConfig(
            "multiclass_classification", task_level="graph", return_type="raw"
        )
        explainer = Explainer(
            model,
            algorithm=gnn_explainer,
            explanation_type="model",
            model_config=config,
            node_mask_type="attributes",
            edge_mask_type="object",
        )
        for n, batch in enumerate(loader):
            batch_unpacked = batch.to(device)
            explanation = explainer(
                x=batch_unpacked.x,
                edge_index=batch_unpacked.edge_index,
                target=batch_unpacked.y,
                pyg_batch=batch_unpacked.batch,
            )
            prediciton = torch.argmax(explanation.prediction)

            sum_masks += explanation.node_mask

            if batch_unpacked.y == 0 and prediciton == 0:
                preictal_masks += explanation.node_mask
                preictal_cntr += 1
            elif batch_unpacked.y == 1 and prediciton == 1:
                ictal_masks += explanation.node_mask
                ictal_cntr += 1
            elif batch_unpacked.y == 2 and prediciton == 2:
                interictal_masks += explanation.node_mask
                interictal_cntr += 1
            if n % 100 == 0 and n != 0:
                print(f"Batch {n} done")
                break
        sum_masks /= n + 1
        interictal_masks /= interictal_cntr
        ictal_masks /= ictal_cntr
        preictal_masks /= preictal_cntr

        final_explanation_sum = explanation.clone()
        final_explanation_interictal = explanation.clone()
        final_explanation_preictal = explanation.clone()
        final_explanation_ictal = explanation.clone()

        final_explanation_sum.node_mask = sum_masks
        final_explanation_interictal.node_mask = interictal_masks
        final_explanation_preictal.node_mask = preictal_masks
        final_explanation_ictal.node_mask = ictal_masks

        save_path_fold = os.path.join(save_dir_importances, fold)
        if not os.path.exists(save_path_fold):
            os.makedirs(save_path_fold)
        torch.save(
            final_explanation_sum,
            os.path.join(save_path_fold, f"final_explanation_sum.pt"),
        )
        torch.save(
            final_explanation_interictal,
            os.path.join(save_path_fold, f"final_explanation_interictal.pt"),
        )
        torch.save(
            final_explanation_preictal,
            os.path.join(save_path_fold, f"final_explanation_preictal.pt"),
        )
        torch.save(
            final_explanation_ictal,
            os.path.join(save_path_fold, f"final_explanation_ictal.pt"),
        )
        print(f" {fold} done")


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
        "--save_dir_importances",
        type=str,
    )
    args = parser.parse_args()
    compute_feature_importances(args)
