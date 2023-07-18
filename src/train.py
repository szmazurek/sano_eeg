import multiprocessing as mp
import os
import warnings
from argparse import ArgumentParser

import lightning.pytorch as pl
import numpy as np
import torch
import torch_geometric
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

import wandb
from utils.dataloader_utils import (
    HDFDataset_Writer,
    HDFDatasetLoader,
    GraphDataset,
)
from models import GATv2Lightning

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
parser.add_argument(
    "--use_preictal_periods", action="store_true", default=False
)
parser.add_argument(
    "--use_interictal_periods", action="store_true", default=False
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--kfold_cval_mode", action="store_true", default=False)

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
KFOlD_CVAL_MODE = args.kfold_cval_mode
CONFIG = dict(
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
)


def loso_training():
    for loso_patient in os.listdir(NPY_DATA_DIR):
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
            kfold_cval_mode=KFOlD_CVAL_MODE,
        )

        train_ds_path, valid_ds_path, loso_ds_path = loader.get_datasets()

        train_dataset = GraphDataset(train_ds_path)
        valid_dataset = GraphDataset(valid_ds_path)
        loso_dataset = GraphDataset(loso_ds_path)

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )
        loso_dataloader = DataLoader(
            loso_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        wandb.init(
            project="validation_sano_eeg",
            group=EXP_NAME,
            name=loso_patient,
            config=CONFIG,
        )

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        precision = "bf16-mixed" if device_name == "cpu" else "16-mixed"
        strategy = pl.strategies.SingleDeviceStrategy(device=device_name)
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, verbose=False, mode="min"
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
        n_classes = sum(USED_CLASSES_DICT.values())
        features_shape = train_dataset[0].x.shape[-1]
        model = GATv2Lightning(
            features_shape,
            n_classes=n_classes,
            lr=0.0001,
            weight_decay=0.0001,
            fft_mode=FFT,
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        trainer.test(model, loso_dataloader, ckpt_path="best")
        wandb.finish()
    return None


def kfold_cval():
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
        kfold_cval_mode=KFOlD_CVAL_MODE,
    )

    full_datalist = loader.get_datasets()[0]
    features_shape = full_datalist[0].x.shape[-1]
    label_array = np.array([data.y.item() for data in full_datalist]).reshape(
        -1, 1
    )
    patient_id_array = np.array(
        [data.patient_id for data in full_datalist]
    ).reshape(-1, 1)
    class_labels_patient_labels = np.concatenate(
        [label_array, patient_id_array], axis=1
    )
    torch_geometric.seed_everything(42)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    precision = "bf16-mixed" if device == "cpu" else "16-mixed"
    strategy = pl.strategies.SingleDeviceStrategy(device=torch_device)
    kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.1, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(
        kfold.split(full_datalist, class_labels_patient_labels)
    ):
        print(f"Fold {fold}")

        train_labels = class_labels_patient_labels[train_idx]
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=0.1, random_state=42
        )
        train_idx, val_idx = next(splitter.split(train_idx, train_labels))
        train_dataset = [full_datalist[idx] for idx in train_idx]
        valid_dataset = [full_datalist[idx] for idx in val_idx]
        test_data = [full_datalist[idx] for idx in test_idx]

        train_dataloader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=BATCH_SIZE, shuffle=False
        )
        test_dataloader = DataLoader(
            test_data, batch_size=BATCH_SIZE, shuffle=False
        )
        wandb.init(
            project="validation_sano_eeg",
            group=EXP_NAME,
            name=f"Fold {fold}",
            config=CONFIG,
        )

        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
        precision = "bf16-mixed" if device_name == "cpu" else "16-mixed"
        strategy = pl.strategies.SingleDeviceStrategy(device=device_name)
        wandb_logger = pl.loggers.WandbLogger(log_model=False)
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, verbose=False, mode="min"
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
        n_classes = sum(USED_CLASSES_DICT.values())
        features_shape = train_dataset[0].x.shape[-1]
        model = GATv2Lightning(
            features_shape,
            n_classes=n_classes,
            lr=0.0001,
            weight_decay=0.0001,
            fft_mode=FFT,
        )
        trainer.fit(model, train_dataloader, valid_dataloader)
        eval_results = trainer.test(model, test_dataloader, ckpt_path="best")[0]
        wandb.finish()
        if fold == 0:
            summary_dict = eval_results
        else:
            summary_dict = {
                key: summary_dict[key] + eval_results[key]
                for key in summary_dict.keys()
            }
    summary_dict = {
        key: summary_dict[key] / (fold + 1) for key in summary_dict.keys()
    }
    print(summary_dict)
    return None


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("forkserver", force=True)
    # mp.set_start_method("forkserver", force=True)
    #  torch.multiprocessing.set_sharing_strategy("file_system")
    print(mp.get_start_method())
    if KFOlD_CVAL_MODE:
        kfold_cval()
    else:
        loso_training()
        exit()
