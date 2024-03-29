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
from torchmetrics import Accuracy, Specificity, Recall, F1Score, AUROC
from argparse import ArgumentParser
from statistics import mean, stdev


def compute_prediction_metrics(args):
    checkpoint_dir = args.checkpoint_dir
    data_dir = args.data_dir
    save_dir_metrics = args.save_dir_metrics
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(save_dir_metrics):
        print("Save directory already exists")
        print(save_dir_metrics)
        return
    os.makedirs(save_dir_metrics)
    fold_list = os.listdir(checkpoint_dir)
    checkpoint_fold_list = [
        os.path.join(checkpoint_dir, fold) for fold in fold_list
    ]
    data_fold_list = [os.path.join(data_dir, fold) for fold in fold_list]
    fold_list.sort()
    data_fold_list.sort()
    checkpoint_fold_list.sort()

    conf_matrix_metric = MulticlassConfusionMatrix(
        3,
    ).to(device)
    specificity_metric = Specificity("multiclass", num_classes=3).to(device)
    recall_metric = Recall("multiclass", num_classes=3).to(device)
    f1_metric = F1Score("multiclass", num_classes=3).to(device)
    auroc_metric = AUROC("multiclass", num_classes=3).to(device)
    summary_conf_matrix = np.zeros((3, 3))
    summary_balanced_acc = []
    summary_specificity = []
    summary_recall = []
    summary_f1 = []
    summary_auroc = []
    for n, fold in enumerate(fold_list):
        checkpoint_path = os.path.join(
            checkpoint_fold_list[n], os.listdir(checkpoint_fold_list[n])[0]
        )

        trainer = pl.Trainer(
            accelerator="auto",
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
        dataset = GraphDataset(data_fold_list[n])
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
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)

        preds = trainer.predict(model, loader)
        preds = torch.cat(preds, dim=0)
        preds_raw = torch.clone(preds)
        preds = torch.nn.functional.softmax(preds, dim=1).argmax(dim=1)
        ground_truth = torch.tensor(
            [data.y.int().item() for data in dataset]
        ).to(device)

        conf_matrix = conf_matrix_metric(preds, ground_truth).int().numpy()
        specificity = specificity_metric(preds, ground_truth).item()
        recall = recall_metric(preds, ground_truth).item()
        f1 = f1_metric(preds, ground_truth).item()
        auroc = auroc_metric(preds_raw, ground_truth).item()
        balanced_acc = balanced_accuracy_score(ground_truth.cpu(), preds.cpu())
        summary_conf_matrix += conf_matrix
        summary_balanced_acc.append(balanced_acc)
        summary_specificity.append(specificity)
        summary_recall.append(recall)
        summary_f1.append(f1)
        summary_auroc.append(auroc)

        # saving fold results
        fold_results = {
            "AUROC": auroc,
            "F1-score": f1,
            "Sensitivity": recall,
            "Specificity": specificity,
            "Balanced Accuracy": balanced_acc,
        }
        np.save(
            os.path.join(save_dir_metrics, f"fold_{n}_conf_matrix.npy"),
            conf_matrix,
        )
        with open(
            os.path.join(save_dir_metrics, f"fold_{n}_results.json"), "w"
        ) as f:
            json.dump(fold_results, f)

    # saving summary results
    summary_results = {
        "AUROC": mean(summary_auroc),
        "AUROC_std": stdev(summary_auroc),
        "F1-score": mean(summary_f1),
        "F1-score_std": stdev(summary_f1),
        "Sensitivity": mean(summary_recall),
        "Sensitivity_std": stdev(summary_recall),
        "Specificity": mean(summary_specificity),
        "Specificity_std": stdev(summary_specificity),
        "Balanced Accuracy": mean(summary_balanced_acc),
        "Balanced Accuracy_std": stdev(summary_balanced_acc),
    }

    np.save(
        os.path.join(save_dir_metrics, f"summary_conf_matrix.npy"),
        summary_conf_matrix,
    )

    with open(
        os.path.join(save_dir_metrics, f"summary_results.json"), "w"
    ) as f:
        json.dump(summary_results, f)


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
        "--save_dir_metrics",
        type=str,
    )
    args = parser.parse_args()
    compute_prediction_metrics(args)
