import os
import warnings
import multiprocessing as mp
from argparse import ArgumentParser
from statistics import mean, stdev
import torch
import torch_geometric
import wandb
import numpy as np
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


from models import GATv2Lightning
from utils.dataloader_utils import (
    GraphDataset,
    HDFDataset_Writer,
    HDFDatasetLoader,
    save_data_list,
)

warnings.filterwarnings(
    "ignore", ".*does not have many workers.*"
)  # DISABLED ON PURPOSE
torch_geometric.seed_everything(42)
api_key_file = open("wandb_api_key.txt", "r")
API_KEY = api_key_file.read()

api_key_file.close()
os.environ["WANDB_API_KEY"] = API_KEY
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision("medium")
parser = ArgumentParser()
parser.add_argument("--timestep", type=int, default=6)
parser.add_argument("--ictal_overlap", type=int, default=0)
parser.add_argument("--inter_overlap", type=int, default=0)
parser.add_argument("--preictal_overlap", type=int, default=0)
parser.add_argument("--seizure_lookback", type=int, default=600)
parser.add_argument("--buffer_time", type=int, default=15)
parser.add_argument("--sampling_freq", type=int, default=256)
parser.add_argument("--downsampling_freq", type=int, default=60)
parser.add_argument("--smote", action="store_true", default=False)
parser.add_argument("--weights", action="store_true", default=False)
parser.add_argument("--undersample", action="store_true", default=False)
parser.add_argument("--train_test_split", type=float, default=0.0)
parser.add_argument("--fft", action="store_true", default=False)
parser.add_argument("--mne_features", action="store_true", default=False)
parser.add_argument("--normalizing_period", type=str, default="interictal")
parser.add_argument("--connectivity_metric", type=str, default="plv")
parser.add_argument("--epochs", type=int, default=25)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--cache_dir", type=str, default="data/cache")
parser.add_argument("--exp_name", type=str, default="eeg_exp")
parser.add_argument("--npy_data_dir", type=str, default="data/npy_data")
parser.add_argument("--event_tables_dir", type=str, default="data/event_tables")
parser.add_argument("--use_ictal_periods", action="store_true", default=False)
parser.add_argument("--use_preictal_periods", action="store_true", default=False)
parser.add_argument("--use_interictal_periods", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--kfold_cval_mode", action="store_true", default=False)
parser.add_argument("--n_splits", type=int, default=5)

args = parser.parse_args()
TIMESTEP = args.timestep
PREICTAL_OVERLAP = args.preictal_overlap
INTER_OVERLAP = args.inter_overlap
ICTAL_OVERLAP = args.ictal_overlap
SMOTE_FLAG = args.smote
NPY_DATA_DIR = args.npy_data_dir
EVENT_TABLES_DIR = args.event_tables_dir
WEIGHTS_FLAG = args.weights
UNDERSAMPLE = args.undersample
EXP_NAME = args.exp_name
EPOCHS = args.epochs
FFT = args.fft
MNE_FEATURES = args.mne_features
USED_CLASSES_DICT = {
    "ictal": args.use_ictal_periods,
    "interictal": args.use_interictal_periods,
    "preictal": args.use_preictal_periods,
}
SFREQ = args.sampling_freq
DOWNSAMPLING_F = args.downsampling_freq
TRAIN_VAL_SPLIT = args.train_test_split
SEIZURE_LOOKBACK = args.seizure_lookback
BATCH_SIZE = args.batch_size

NORMALIZING_PERIOD = args.normalizing_period
CONNECTIVITY_METRIC = args.connectivity_metric
BUFFER_TIME = args.buffer_time
CACHE_DIR = args.cache_dir
SEED = args.seed
KFOLD_CVAL_MODE = args.kfold_cval_mode
N_SPLITS = args.n_splits
INITIAL_CONFIG = dict(
    timestep=TIMESTEP,
    inter_overlap=INTER_OVERLAP,
    preictal_overlap=PREICTAL_OVERLAP,
    ictal_overlap=ICTAL_OVERLAP,
    seizure_lookback=SEIZURE_LOOKBACK,
    buffer_time=BUFFER_TIME,
    normalizing_period=NORMALIZING_PERIOD,
    smote=SMOTE_FLAG,
    weights=WEIGHTS_FLAG,
    undersampling=UNDERSAMPLE,
    sampling_freq=SFREQ,
    downsampling_freq=DOWNSAMPLING_F,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    fft=FFT,
    mne_features=MNE_FEATURES,
    used_classes_dict=USED_CLASSES_DICT,
    train_val_split=TRAIN_VAL_SPLIT,
    connectivity_metric=CONNECTIVITY_METRIC,
    seed=SEED,
    n_gat_layers=1,
    hidden_dim=32,
    dropout=0.0,
    slope=0.0025,
    pooling_method="mean",
    norm_method="batch",
    activation="leaky_relu",
    n_heads=9,
    lr=0.0012,
    weight_decay=0.0078,
)


def loso_training():
    """Initialize loso training."""
    wandb.init(
        config=INITIAL_CONFIG,
    )
    wandb.define_metric("patient")
    for n, loso_patient in enumerate(os.listdir(NPY_DATA_DIR)):
        writer = HDFDataset_Writer(
            seizure_lookback=SEIZURE_LOOKBACK,
            buffer_time=BUFFER_TIME,
            sample_timestep=TIMESTEP,
            inter_overlap=INTER_OVERLAP,
            preictal_overlap=PREICTAL_OVERLAP,
            ictal_overlap=ICTAL_OVERLAP,
            downsample=DOWNSAMPLING_F,
            sampling_f=SFREQ,
            smote=SMOTE_FLAG,
            connectivity_metric=CONNECTIVITY_METRIC,
            npy_dataset_path=NPY_DATA_DIR,
            event_tables_path=EVENT_TABLES_DIR,
            cache_folder=CACHE_DIR,
        )
        cache_file_path = writer.get_dataset()

        loader = HDFDatasetLoader(
            root=cache_file_path,
            train_val_split_ratio=TRAIN_VAL_SPLIT,
            loso_subject=loso_patient,
            sampling_f=SFREQ,
            extract_features=MNE_FEATURES,
            fft=FFT,
            seed=SEED,
            used_classes_dict=USED_CLASSES_DICT,
            normalize_with=NORMALIZING_PERIOD,
            kfold_cval_mode=KFOLD_CVAL_MODE,
        )

        train_ds_path, valid_ds_path, loso_ds_path = loader.get_datasets()

        train_dataset = GraphDataset(train_ds_path)
        valid_dataset = GraphDataset(valid_ds_path)
        loso_dataset = GraphDataset(loso_ds_path)
        loso_dataset.clear_cache()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            drop_last=False,
        )
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        loso_dataloader = DataLoader(
            loso_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        precision = "bf16-mixed" if device_name == "cpu" else "16-mixed"
        strategy = pl.strategies.SingleDeviceStrategy(device=device_name)
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=False, mode="min"
        )
        best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=False,
        )
        callbacks = [early_stopping, best_checkpoint_callback]
        trainer = pl.Trainer(
            accelerator="auto",
            precision=precision,
            devices=1,
            max_epochs=EPOCHS,
            enable_progress_bar=True,
            strategy=strategy,
            deterministic=False,
            log_every_n_steps=1,
            enable_model_summary=False,
            logger=wandb_logger,
            callbacks=callbacks,
        )
        CONFIG = wandb.config
        train_labels = torch.cat([data.y for data in train_dataset])
        label_properties = torch.unique(train_labels, return_counts=True)
        if sum(CONFIG.used_classes_dict.values()) == 3:
            """Multiclass weights"""
            class_weight = torch.from_numpy(
                compute_class_weight(
                    "balanced",
                    classes=label_properties[0].numpy(),
                    y=train_labels.numpy(),
                )
            ).float()
        else:
            """Binary weights"""
            class_weight = torch.tensor(
                [label_properties[1][0] / label_properties[1][1]]
            )
        n_classes = sum(USED_CLASSES_DICT.values())
        features_shape = train_dataset[0].x.shape[-1]
        model = GATv2Lightning(
            features_shape,
            n_classes=n_classes,
            n_gat_layers=CONFIG.n_gat_layers,
            hidden_dim=CONFIG.hidden_dim,
            n_heads=CONFIG.n_heads,
            slope=CONFIG.slope,
            dropout=CONFIG.dropout,
            pooling_method=CONFIG.pooling_method,
            activation=CONFIG.activation,
            norm_method=CONFIG.norm_method,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
            fft_mode=FFT,
            class_weights=class_weight,
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        eval_results = trainer.test(model, loso_dataloader, ckpt_path="best")[0]
        wandb.log({"patient": int("".join([n for n in loso_patient if n.isdigit()]))})
        wandb.define_metric("test_loss_epoch", step_metric="patient")
        wandb.define_metric("loso_sensitivity", step_metric="patient")
        wandb.define_metric("loso_specificity", step_metric="patient")
        wandb.define_metric("loso_AUROC", step_metric="patient")
        if n == 0:
            result_list = [eval_results["loso_AUROC"]]
        else:
            result_list.append(eval_results["loso_AUROC"])
    mean_auroc = mean(result_list)
    stdev_auroc = round(stdev(result_list), 4)
    measured_auroc = mean_auroc * (1 / stdev_auroc)
    wandb.log({"final_mean_AUROC": mean_auroc})
    wandb.log({"final_stdev_AUROC": stdev_auroc})
    wandb.log({"final_measured_AUROC": measured_auroc})
    wandb.finish()
    return None


def kfold_cval():
    """Initialize kfold cross validation."""
    writer = HDFDataset_Writer(
        seizure_lookback=SEIZURE_LOOKBACK,
        buffer_time=BUFFER_TIME,
        sample_timestep=TIMESTEP,
        inter_overlap=INTER_OVERLAP,
        preictal_overlap=PREICTAL_OVERLAP,
        ictal_overlap=ICTAL_OVERLAP,
        downsample=DOWNSAMPLING_F,
        sampling_f=SFREQ,
        smote=SMOTE_FLAG,
        connectivity_metric=CONNECTIVITY_METRIC,
        npy_dataset_path=NPY_DATA_DIR,
        event_tables_path=EVENT_TABLES_DIR,
        cache_folder=CACHE_DIR,
    )
    cache_file_path = writer.get_dataset()

    loader = HDFDatasetLoader(
        root=cache_file_path,
        train_val_split_ratio=TRAIN_VAL_SPLIT,
        loso_subject=None,
        sampling_f=DOWNSAMPLING_F,
        extract_features=MNE_FEATURES,
        fft=FFT,
        seed=SEED,
        used_classes_dict=USED_CLASSES_DICT,
        normalize_with=NORMALIZING_PERIOD,
        kfold_cval_mode=KFOLD_CVAL_MODE,
    )

    full_data_path = loader.get_datasets()[0]
    full_dataset = GraphDataset(full_data_path)
    features_shape = full_dataset[0].x.shape[-1]
    label_array = np.array([data.y.item() for data in full_dataset]).reshape(-1, 1)
    patient_id_array = np.array(
        [data.patient_id.item() for data in full_dataset]
    ).reshape(-1, 1)
    class_labels_patient_labels = np.concatenate(
        [label_array, patient_id_array], axis=1
    )

    torch_geometric.seed_everything(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    precision = "bf16-mixed" if device == "cpu" else "16-mixed"
    strategy = pl.strategies.SingleDeviceStrategy(device=torch_device)
    kfold = MultilabelStratifiedKFold(n_splits=N_SPLITS, random_state=42, shuffle=True)

    for fold, (train_idx, test_idx) in enumerate(
        kfold.split(np.zeros(len(full_dataset)), class_labels_patient_labels)
    ):
        print(f"Fold {fold}")
        wandb.init(
            project="sano_eeg_final_runs",
            group=EXP_NAME,
            name=f"fold_{fold}",
            config=INITIAL_CONFIG,
        )
        CONFIG = wandb.config
        train_labels = class_labels_patient_labels[train_idx]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
        sub_train_idx, val_idx = next(splitter.split(train_idx, train_labels))
        train_dataset = [full_dataset[idx] for idx in sub_train_idx]
        valid_dataset = [full_dataset[idx] for idx in val_idx]
        test_data = [full_dataset[idx] for idx in test_idx]
        test_save_data = os.path.join(
            "saved_folds_trial", EXP_NAME, f"fold_{fold}/ data.pt"
        )
        save_data_list(test_data, test_save_data)
        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
        )

        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )
        test_dataloader = DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
        )

        train_labels = torch.cat([data.y for data in train_dataset])
        label_properties = torch.unique(train_labels, return_counts=True)
        if sum(CONFIG.used_classes_dict.values()) == 3:
            """Multiclass weights"""
            class_weight = torch.from_numpy(
                compute_class_weight(
                    "balanced",
                    classes=label_properties[0].numpy(),
                    y=train_labels.numpy(),
                )
            ).float()
        else:
            """Binary weights"""
            class_weight = torch.tensor(
                [label_properties[1][0] / label_properties[1][1]]
            )
        n_classes = sum(USED_CLASSES_DICT.values())
        features_shape = train_dataset[0].x.shape[-1]
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        precision = "bf16-mixed" if device_name == "cpu" else "16-mixed"
        strategy = pl.strategies.SingleDeviceStrategy(device=device_name)
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, verbose=False, mode="min"
        )
        best_checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            verbose=False,
            dirpath=f"checkpoints_final_new_strat/fold_{fold}",
            filename="{epoch}-{val_loss:.3f}",
        )
        callbacks = [early_stopping, best_checkpoint_callback]
        trainer = pl.Trainer(
            accelerator="auto",
            precision=precision,
            devices=1,
            max_epochs=EPOCHS,
            enable_progress_bar=True,
            strategy=strategy,
            deterministic=False,
            log_every_n_steps=1,
            enable_model_summary=False,
            logger=wandb_logger,
            callbacks=callbacks,
        )

        model = GATv2Lightning(
            features_shape,
            n_classes=n_classes,
            n_gat_layers=CONFIG.n_gat_layers,
            hidden_dim=CONFIG.hidden_dim,
            n_heads=CONFIG.n_heads,
            slope=CONFIG.slope,
            dropout=CONFIG.dropout,
            pooling_method=CONFIG.pooling_method,
            activation=CONFIG.activation,
            norm_method=CONFIG.norm_method,
            lr=CONFIG.lr,
            weight_decay=CONFIG.weight_decay,
            fft_mode=FFT,
            class_weights=class_weight,
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        trainer.test(model, test_dataloader, ckpt_path="best")
        wandb.finish()

    return None


if __name__ == "__main__":
    if KFOLD_CVAL_MODE:
        kfold_cval()
    else:
        loso_training()
        exit()
