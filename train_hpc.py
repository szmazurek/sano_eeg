import pandas as pd
import numpy as np
from pathlib import Path
import os
from dataclasses import dataclass
import utils
from torch_geometric_temporal import (
    DynamicGraphTemporalSignal,
    StaticGraphTemporalSignal,
    temporal_signal_split,
    DynamicGraphTemporalSignalBatch,
)
import networkx as nx
from torch_geometric.utils import from_networkx
import scipy
import sklearn
import random
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
from imblearn.over_sampling import SMOTE
from torchmetrics.classification import BinaryRecall, BinarySpecificity, AUROC, ROC
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, GATv2Conv
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from mne_features.univariate import (
    compute_variance,
    compute_hjorth_complexity,
    compute_hjorth_mobility,
)
import torchaudio
import mne_features
import wandb
import torch_geometric
from joblib import Parallel, delayed
import time
import logging
from argparse import ArgumentParser

logging.basicConfig(filename="newfile.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(0)
torch_geometric.seed_everything(42)

api_key = open("wandb_api_key.txt", "r")
key = api_key.read()
api_key.close()
os.environ["WANDB_API_KEY"] = key
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# TODO think about using kwargs argument here to specify args for dataloader
@dataclass
class SeizureDataLoader:
    npy_dataset_path: Path
    event_tables_path: Path
    plv_values_path: Path
    loso_patient: str = None
    sampling_f: int = 256
    seizure_lookback: int = 600
    sample_timestep: int = 5
    inter_overlap: int = 0
    ictal_overlap: int = 0
    self_loops: bool = True
    balance: bool = True
    train_test_split: float = None
    fft: bool = False
    hjorth: bool = False
    teager_keiser: bool = False
    downsample: int = None
    buffer_time: int = 15
    batch_size: int = 32
    smote: bool = False
    """Class to prepare dataloaders for eeg seizure perdiction from stored files.

    Attributes:
        npy_dataset_path {Path} -- Path to folder with dataset preprocessed into .npy files.
        event_tables_path {Path} -- Path to folder with .csv files containing seizure events information for every patient.
        loso_patient {str} -- Name of patient to be selected for LOSO valdiation, specified in format "chb{patient_number}"",
        eg. "chb16". (default: {None}).
        samplin_f {int} -- Sampling frequency of the loaded eeg data. (default: {256}).
        seizure_lookback {int} -- Time horizon to sample pre-seizure data (length of period before seizure) in seconds. 
        (default: {600}).
        sample_timestep {int} -- Amounts of seconds analyzed in a single sample. (default: {5}).
        overlap {int} -- Amount of seconds overlap between samples. (default: {0}).
        self_loops {bool} -- Wheather to add self loops to nodes of the graph. (default: {True}).
        shuffle {bool} --  Wheather to shuffle training samples.


    """
    assert (fft and hjorth) == False, "When fft is True, hjorth should be False"

    def _get_event_tables(self, patient_name: str) -> tuple[dict, dict]:
        """Read events for given patient into start and stop times lists from .csv extracted files."""

        event_table_list = os.listdir(self.event_tables_path)
        patient_event_tables = [
            os.path.join(self.event_tables_path, ev_table)
            for ev_table in event_table_list
            if patient_name in ev_table
        ]
        patient_event_tables = sorted(patient_event_tables)
        patient_start_table = patient_event_tables[
            0
        ]  ## done terribly, but it has to be so for win/linux compat
        patient_stop_table = patient_event_tables[1]
        start_events_dict = pd.read_csv(patient_start_table).to_dict("index")
        stop_events_dict = pd.read_csv(patient_stop_table).to_dict("index")
        return start_events_dict, stop_events_dict

    def _get_recording_events(self, events_dict, recording) -> list[int]:
        """Read seizure times into list from event_dict"""
        recording_list = list(events_dict[recording + ".edf"].values())
        recording_events = [int(x) for x in recording_list if not np.isnan(x)]
        return recording_events

    def _get_graph(self, n_nodes: int) -> nx.Graph:
        """Creates Networx fully connected graph with self loops"""
        graph = nx.complete_graph(n_nodes)
        self_loops = [[node, node] for node in graph.nodes()]
        graph.add_edges_from(self_loops)
        return graph

    def _get_edge_weights_recording(self, plv_values: np.ndarray) -> np.ndarray:
        """Method that takes plv values for given recording and assigns them
        as edge attributes to a fc graph."""
        graph = self._get_graph(plv_values.shape[0])
        garph_dict = {}
        for edge in graph.edges():
            e_start, e_end = edge
            garph_dict[edge] = {"plv": plv_values[e_start, e_end]}
        nx.set_edge_attributes(graph, garph_dict)
        edge_weights = from_networkx(graph).plv.numpy()
        return edge_weights

    def _get_edges(self):
        """Method to assign edge attributes. Has to be called AFTER get_dataset() method."""
        graph = self._get_graph(self._features.shape[1])
        edges = np.expand_dims(from_networkx(graph).edge_index.numpy(), axis=0)
        edges_per_sample_train = np.repeat(
            edges, repeats=self._features.shape[0], axis=0
        )
        self._edges = torch.tensor(edges_per_sample_train)
        if self.loso_patient is not None:
            edges_per_sample_val = np.repeat(
                edges, repeats=self._val_features.shape[0], axis=0
            )
            self._val_edges = torch.tensor(edges_per_sample_val)

    def _array_to_tensor(self):
        """Method converting features, edges and weights to torch.tensors"""

        self._features = torch.tensor(self._features, dtype=torch.float32)
        self._labels = torch.tensor(self._labels)
        self._time_labels = torch.tensor(self._time_labels)
        self._edge_weights = torch.tensor(self._edge_weights)
        if self.loso_patient is not None:
            self._val_features = torch.tensor(self._val_features, dtype=torch.float32)
            self._val_labels = torch.tensor(self._val_labels)
            self._val_time_labels = torch.tensor(self._val_time_labels)
            self._val_edge_weights = torch.tensor(self._val_edge_weights)

    def _get_labels_count(self):
        labels, counts = np.unique(self._labels, return_counts=True)
        print(labels, counts)
        self._label_counts = {}
        for n, label in enumerate(labels):
            self._label_counts[int(label)] = counts[n]
        if self.loso_patient is not None:
            labels, counts = np.unique(self._val_labels, return_counts=True)
            self._val_label_counts = {}
            for n, label in enumerate(labels):
                self._val_label_counts[int(label)] = counts[n]

    def _perform_features_fft(self):
        self._features = torch.fft.rfft(self._features)
        if self.loso_patient is not None:
            self._val_features = torch.fft.rfft(self._val_features)

    def _downsample_features(self):
        resampler = torchaudio.transforms.Resample(self.sampling_f, self.downsample)
        self._features = resampler(self._features)
        if self.loso_patient is not None:
            self._val_features = resampler(self._val_features)

    def _calculate_hjorth_features(self, features):
        new_features = np.array(
            [
                np.concatenate(
                    [
                        np.expand_dims(compute_variance(feature), 1),
                        np.expand_dims(compute_hjorth_mobility(feature), 1),
                        np.expand_dims(compute_hjorth_complexity(feature), 1),
                        np.expand_dims([len(find_peaks(sig)[0]) for sig in feature], 1),
                        np.expand_dims(np.sum(zero_crossings(feature), axis=1), 1),
                        np.expand_dims(
                            [
                                peak_prominences(sig, find_peaks(sig)[0])[0].mean()
                                for sig in feature
                            ],
                            1,
                        ),
                    ],
                    axis=1,
                )
                for feature in features
            ]
        )
        return new_features

    def _features_to_data_list(self, features, edges, edge_weights, labels, time_label):
        data_list = [
            Data(
                x=features[i],
                edge_index=edges[i],
                edge_attr=edge_weights[i],
                y=labels[i],
                # time=time_label[i],
            )
            for i in range(len(features))
        ]
        return data_list

    def _split_data_list(self, data_list):
        class_labels = torch.tensor(
            [data.y.item() for data in data_list], dtype=torch.float32
        ).unsqueeze(1)
        patient_labels = torch.tensor(
            np.expand_dims(self._patient_number, 1), dtype=torch.float32
        )
        class_labels_patient_labels = torch.cat([class_labels, patient_labels], dim=1)
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=self.train_test_split, random_state=42
        )
        train_indices, val_indices = next(
            splitter.split(data_list, class_labels_patient_labels)
        )
        self._indexes_to_later_delete = {"train": train_indices, "val": val_indices}
        data_list_train = [data_list[i] for i in train_indices]
        dataset_list_val = [data_list[i] for i in val_indices]
        return data_list_train, dataset_list_val

    def _initialize_dicts(self):
        self._features_dict = {}
        self._labels_dict = {}
        self._time_labels_dict = {}
        self._edge_weights_dict = {}
        self._patient_number_dict = {}
        if self.loso_patient:
            self._val_features_dict = {}
            self._val_labels_dict = {}
            self._val_time_labels_dict = {}
            self._val_edge_weights_dict = {}
            self._val_patient_number_dict = {}

    def _convert_dict_to_array(self):
        self._features = np.concatenate(
            [self._features_dict[key] for key in self._features_dict.keys()]
        )
        del self._features_dict
        self._labels = np.concatenate(
            [self._labels_dict[key] for key in self._labels_dict.keys()]
        )
        del self._labels_dict
        self._time_labels = np.concatenate(
            [self._time_labels_dict[key] for key in self._time_labels_dict.keys()]
        )
        del self._time_labels_dict
        self._edge_weights = np.concatenate(
            [self._edge_weights_dict[key] for key in self._edge_weights_dict.keys()]
        )
        del self._edge_weights_dict
        self._patient_number = np.concatenate(
            [self._patient_number_dict[key] for key in self._patient_number_dict.keys()]
        )
        del self._patient_number_dict
        if self.loso_patient:
            self._val_features = np.concatenate(
                [self._val_features_dict[key] for key in self._val_features_dict.keys()]
            )
            del self._val_features_dict
            self._val_labels = np.concatenate(
                [self._val_labels_dict[key] for key in self._val_labels_dict.keys()]
            )
            del self._val_labels_dict
            self._val_time_labels = np.concatenate(
                [
                    self._val_time_labels_dict[key]
                    for key in self._val_time_labels_dict.keys()
                ]
            )
            del self._val_time_labels_dict
            self._val_edge_weights = np.concatenate(
                [
                    self._val_edge_weights_dict[key]
                    for key in self._val_edge_weights_dict.keys()
                ]
            )
            del self._val_edge_weights_dict
            self._val_patient_number = np.concatenate(
                [
                    self._val_patient_number_dict[key]
                    for key in self._val_patient_number_dict.keys()
                ]
            )
            del self._val_patient_number_dict

    def _balance_classes(self):
        negative_label = self._label_counts[0]
        positive_label = self._label_counts[1]
        
        print(f"Number of negative samples pre removal {negative_label}")
        print(f"Number of positive samples pre removal {positive_label}")
        imbalance = negative_label - positive_label
        print(f"imbalance {imbalance}")
        negative_indices = np.where(self._labels == 0)[0]
        indices_to_discard = np.random.choice(
            negative_indices, size=imbalance, replace=False
        )

        self._features = np.delete(self._features, obj=indices_to_discard, axis=0)
        self._labels = np.delete(self._labels, obj=indices_to_discard, axis=0)
        self._time_labels = np.delete(self._time_labels, obj=indices_to_discard, axis=0)
        self._edge_weights = np.delete(
            self._edge_weights, obj=indices_to_discard, axis=0
        )
        self._patient_number = np.delete(self._patient_number, obj=indices_to_discard,axis=0)

    def _standardize_data(self, features, labels, loso_features=None):
        indexes = np.where(labels == 0)[0]  
        features_negative = features[indexes]
        channel_mean = features_negative.mean()
        channel_std = features_negative.std()
        # if features_negative.shape[0] == 1:
        #     channel_mean = features_negative.mean(2).squeeze()
        #     channel_std = features_negative.std(2).squeeze()
        # else:
        #    # print(features_negative[0].shape)
        #     channel_mean = features_negative.mean(axis=0).mean(1)
        #     channel_std = features_negative.std(axis=0).std(1)
        for i in range(features.shape[0]):
            for n in range(features.shape[1]):
                #        features[i,n,:] = (features[i,n,:] - channel_mean[n])/channel_std[n]
                features[i, n, :] = (features[i, n, :] - channel_mean) / channel_std
        if (
            loso_features is not None
        ):  ## standardize loso features with the same values as for training data
            for i in range(loso_features.shape[0]):
                for n in range(loso_features.shape[1]):
                    loso_features[i, n, :] = (
                        loso_features[i, n, :] - channel_mean
                    ) / channel_std

    def _min_max_scale(self, features, labels):
        indexes = np.where(labels == 0)[0]  ## changed from 0!
        features_negative = features[indexes]

        channel_min = features_negative.min(axis=0).min(1)
        channel_max = features_negative.max(axis=0).max(1)
        for i in range(features.shape[0]):
            for n in range(features.shape[1]):
                features[i, n, :] = (features[i, n, :] - channel_min[n]) / (
                    channel_max[n] - channel_min[n]
                )
                # features[i,n,:] = (features[i,n,:] - channel_min)/(channel_max - channel_min)

    def _apply_smote(self, features, labels):
        dim_1 = np.array(features).shape[0]
        dim_2 = np.array(features).shape[1]
        dim_3 = np.array(features).shape[2]

        new_dim = dim_1 * dim_2
        new_x_train = features.reshape(new_dim, dim_3)
        new_y_train = []
        for i in range(len(labels)):
            new_y_train.extend([labels[i]] * dim_2)

        new_y_train = np.array(new_y_train)

        # transform the dataset
        oversample = SMOTE(random_state=42)
        x_train, y_train = oversample.fit_resample(new_x_train, new_y_train)
        x_train_smote = x_train.reshape(int(x_train.shape[0] / dim_2), dim_2, dim_3)
        y_train_smote = []
        for i in range(int(x_train.shape[0] / dim_2)):
            # print(i)
            value_list = list(y_train.reshape(int(x_train.shape[0] / dim_2), dim_2)[i])
            # print(list(set(value_list)))
            y_train_smote.extend(list(set(value_list)))
            ## Check: if there is any different value in a list
            if len(set(value_list)) != 1:
                print(
                    "\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n"
                )
        y_train_smote = np.array(y_train_smote)
        # print(np.unique(y_train_smote,return_counts=True))
        return x_train_smote, y_train_smote

    def _get_labels_features_edge_weights_seizure(self, patient):
        """Prepare features, labels, time labels and edge wieghts for training and
        optionally validation data."""

        event_tables = self._get_event_tables(
            patient
        )  # extract start and stop of seizure for patient
        patient_path = os.path.join(self.npy_dataset_path, patient)
        recording_list = os.listdir(patient_path)
        for record in recording_list:  # iterate over recordings for a patient
            # if "seizures_" not in record:
            #     ## skip non-seizure files
            #     continue

            recording_path = os.path.join(patient_path, record)
            record = record.replace(
                "seizures_", ""
            )  ## some magic to get it properly working with event tables
            record_id = record.split(".npy")[0]  #  get record id
            start_event_tables = self._get_recording_events(
                event_tables[0], record_id
            )  # get start events
            stop_event_tables = self._get_recording_events(
                event_tables[1], record_id
            )  # get stop events
            data_array = np.load(recording_path)  # load the recording

            plv_edge_weights = np.expand_dims(
                self._get_edge_weights_recording(
                    np.load(os.path.join(self.plv_values_path, patient, record))
                ),
                axis=0,
            )

            features, labels, time_labels = utils.extract_training_data_and_labels(
                data_array,
                start_event_tables,
                stop_event_tables,
                fs=self.sampling_f,
                seizure_lookback=self.seizure_lookback,
                sample_timestep=self.sample_timestep,
                inter_overlap=self.inter_overlap,
                ictal_overlap=self.ictal_overlap,
                buffer_time=self.buffer_time,
            )

            if features is None:
                # print(
                #     f"Skipping the recording {record} patients {patient} cuz features are none"
                # )
                continue
            if len(np.unique(labels)) != 2:
                # print(
                #     f"Skipping the recording {record} patients {patient} cuz no seizure samples"
                # )
                continue

            features = features.squeeze()

            if self.smote and patient == self.loso_patient:
                print(f"Applying smote on loso patient {patient} features")
                smote_start = time.time()
                features, labels = self._apply_smote(features, labels)
                logging.info(f"Applied one smote in {time.time() - smote_start} for patient {patient}")

            time_labels = np.expand_dims(time_labels.astype(np.int32), 1)
            labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)
            patient_number = torch.full(
                [labels.shape[0]],
                int("".join(x for x in patient if x.isdigit())),
                dtype=torch.float32,
            )
            if patient == self.loso_patient:
                #logging.info(f"Adding recording {record} of patient {patient}")
                try:
                    self._val_features_dict[patient] = np.concatenate(
                        (self._val_features_dict[patient], features), axis=0
                    )
                    self._val_labels_dict[patient] = np.concatenate(
                        (self._val_labels_dict[patient], labels), axis=0
                    )
                    self._val_time_labels_dict[patient] = np.concatenate(
                        (self._val_time_labels_dict[patient], time_labels), axis=0
                    )
                    self._val_edge_weights_dict[patient] = np.concatenate(
                        (
                            self._val_edge_weights_dict[patient],
                            np.repeat(plv_edge_weights, features.shape[0], axis=0),
                        )
                    )

                    self._val_patient_number_dict[patient] = np.concatenate(
                        (self._val_patient_number_dict[patient], patient_number)
                    )
                except:
                    self._val_features_dict[patient] = features
                    self._val_labels_dict[patient] = labels
                    self._val_time_labels_dict[patient] = time_labels
                    self._val_edge_weights_dict[patient] = np.repeat(
                        plv_edge_weights, features.shape[0], axis=0
                    )
                    self._val_patient_number_dict[patient] = patient_number

            else:
                try:
                    self._features_dict[patient] = np.concatenate(
                        (self._features_dict[patient], features), axis=0
                    )
                    self._labels_dict[patient] = np.concatenate(
                        (self._labels_dict[patient], labels), axis=0
                    )
                    self._time_labels_dict[patient] = np.concatenate(
                        (self._time_labels_dict[patient], time_labels), axis=0
                    )
                    self._edge_weights_dict[patient] = np.concatenate(
                        (
                            self._edge_weights_dict[patient],
                            np.repeat(plv_edge_weights, features.shape[0], axis=0),
                        )
                    )

                    self._patient_number_dict[patient] = np.concatenate(
                        (self._patient_number_dict[patient], patient_number)
                    )
                except:
                    #print("Creating initial attributes")
                    self._features_dict[patient] = features
                    self._labels_dict[patient] = labels
                    self._time_labels_dict[patient] = time_labels
                    self._edge_weights_dict[patient] = np.repeat(
                        plv_edge_weights, features.shape[0], axis=0
                    )
                    self._patient_number_dict[patient] = patient_number

    def _get_labels_features_edge_weights_interictal(
        self, samples_recording: int = None
    ):
        patient_list = os.listdir(self.npy_dataset_path)
        interictal_samples = len(np.where(self._labels == 0))
        loso_interictal_samples = len(np.where(self._val_labels == 0))
        for patient in patient_list:
            patient_path = os.path.join(self.npy_dataset_path, patient)
            ## get all non-seizure recordings
            recording_list = [
                recording
                for recording in os.listdir(patient_path)
                if not "seizures_" in recording
            ]
            if not samples_recording:
                if patient == self.loso_patient:
                    samples_per_recording = int(
                        loso_interictal_samples / len(recording_list)
                    )
                else:
                    samples_per_recording = int(
                        interictal_samples / len(recording_list)
                    )
            else:
                samples_per_recording = samples_recording
            for recording in recording_list:
                recording_path = os.path.join(patient_path, recording)
                data_array = np.load(recording_path)

    # TODO define a method to create edges and calculate plv to get weights
    def get_dataset(self) -> DynamicGraphTemporalSignal:
        """Creating graph data iterators. The iterator yelds dynamic, weighted and undirected graphs
        containing self loops. Every node represents one electrode in EEG. The graph is fully connected,
        edge weights are calculated for every EEG recording as PLV between channels (edge weight describes
        the "strength" of connectivity between two channels in a recording). Node features are values of
        channel voltages in time. Features are of shape [nodes,features,timesteps].

        Returns:
            train_dataset {DynamicGraphTemporalSignal} -- Training data iterator.
            valid_dataset {DynamicGraphTemporalSignal} -- Validation data iterator (only if loso_patient is
            specified in class constructor).
        """
        ### TODO rozkminić o co chodzi z tym całym time labels - na razie wartość liczbowa która tam wchodzi
        ### to shape atrybutu time_labels

        self._initialize_dicts()
        patient_list = os.listdir(self.npy_dataset_path)
        start_time = time.time()
        if self.smote:
            for patient in patient_list:
                self._get_labels_features_edge_weights_seizure(patient)
        else:
            # Parallel(n_jobs=6, require="sharedmem")(
            #     delayed(self._get_labels_features_edge_weights_seizure)(patient)
            #     for patient in patient_list
            # )
            for patient in patient_list:
                self._get_labels_features_edge_weights_seizure(patient)
        self._convert_dict_to_array()
        self._get_labels_count()
        if self.balance:
            self._balance_classes()
        
        print(f"Finished processing in {time.time() - start_time} seconds")
        print(f"Features shape {self._features.shape}")

        start_time_preprocessing = time.time()
        self._standardize_data(self._features, self._labels, self._val_features)
        
        self._get_edges()
        #self._get_labels_count()
        if self.hjorth:
            self._features = self._calculate_hjorth_features(self._features)
        self._array_to_tensor()
        if self.downsample and not self.hjorth:
            self._downsample_features()
        if self.fft:
            self._perform_features_fft()

        train_dataset = torch.utils.data.TensorDataset(
            self._features,
            self._edges,
            self._edge_weights,
            self._labels,
            # self._time_labels,
        )
        if self.train_test_split is not None:
            if self.fft or self.hjorth:
                data_list = self._features_to_data_list(
                    self._features,
                    self._edges,
                    self._edge_weights,
                    self._labels,
                    self._time_labels,
                )
                train_data_list, val_data_list = self._split_data_list(data_list)
                label_count = np.unique(
                    [data.y.item() for data in train_data_list], return_counts=True
                )[1]
                self.alpha = label_count[0] / label_count[1]
                loaders = [
                    DataLoader(
                        train_data_list,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=False,
                    ),
                    DataLoader(
                        val_data_list,
                        batch_size=len(val_data_list),
                        shuffle=False,
                        drop_last=False,
                    ),
                ]

            else:
                train_dataset, val_dataset = torch.utils.data.random_split(
                    train_dataset,
                    [1 - self.train_test_split, self.train_test_split],
                    generator=torch.Generator().manual_seed(42),
                )

                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    # num_workers=2,
                    # pin_memory=True,
                    # prefetch_factor=4,
                    drop_last=False,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    # num_workers=2,
                    # pin_memory=True,
                    # prefetch_factor=4,
                    drop_last=False,
                )
                loaders = [train_dataloader, val_dataloader]
        else:
            if self.fft or self.hjorth:
                train_data_list = self._features_to_data_list(
                    self._features,
                    self._edges,
                    self._edge_weights,
                    self._labels,
                    self._time_labels,
                )
                loaders = [
                    DataLoader(
                        train_data_list,
                        batch_size=self.batch_size,
                        shuffle=True,
                        drop_last=False,
                    )
                ]
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    # num_workers=2,
                    # pin_memory=True,
                    # prefetch_factor=4,
                    drop_last=False,
                )
                loaders = [train_dataloader]
        if self.loso_patient:
            if self.hjorth:
                self._val_features = self._calculate_hjorth_features(self._val_features)

            if self.fft or self.hjorth:
                loso_data_list = self._features_to_data_list(
                    self._val_features,
                    self._val_edges,
                    self._val_edge_weights,
                    self._val_labels,
                    self._val_time_labels,
                )
                print("Preprocessing time: ", time.time() - start_time_preprocessing)
                return (
                    *loaders,
                    DataLoader(
                        loso_data_list,
                        batch_size=len(loso_data_list),
                        shuffle=False,
                        drop_last=False,
                    ),
                )
            loso_dataset = torch.utils.data.TensorDataset(
                self._val_features,
                self._val_edges,
                self._val_edge_weights,
                self._val_labels,
                #  self._val_time_labels,
            )
            loso_dataloader = torch.utils.data.DataLoader(
                loso_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                # pin_memory=True,
                # num_workers=2,
                # prefetch_factor=4,
                drop_last=False,
            )

            return (*loaders, loso_dataloader)

        return (*loaders,)


class RecurrentGCN(torch.nn.Module):
    def __init__(self, timestep, sfreq, n_nodes=18, batch_size=32):
        super(RecurrentGCN, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        self.recurrent_1 = GCNConv(3, 32, add_self_loops=True, improved=False)
        self.recurrent_2 = GCNConv(32, 64, add_self_loops=True, improved=False)
        self.recurrent_3 = GCNConv(64, 128, add_self_loops=True, improved=False)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(32)
        self.batch_norm_2 = torch.nn.BatchNorm1d(64)
        self.batch_norm_3 = torch.nn.BatchNorm1d(128)
        self.flatten = torch.nn.Flatten(start_dim=0)
        self.dropout = torch.nn.Dropout()

    def forward(self, x, edge_index, edge_weight, batch):
        x = torch.squeeze(x)
        h = self.recurrent_1(x, edge_index=edge_index, edge_weight=edge_weight)
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
        return h.squeeze()


class AT3Batched(torch.nn.Module):
    def __init__(self, timestep, sfreq, n_nodes=18, batch_size=32):
        super(AT3Batched, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        self.recurrent_1 = A3TGCN2(
            1,
            64,
            periods=timestep * sfreq,
            add_self_loops=True,
            improved=False,
            batch_size=batch_size,
        )
        self.fc1 = torch.nn.Linear(64 * n_nodes, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(18)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.dropout = torch.nn.Dropout()

    def forward(self, x, edge_index):
        h = self.recurrent_1(x, edge_index=edge_index)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = self.flatten(h)
        h = self.dropout(h)
        h = self.fc1(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc3(h)
        return h.squeeze()


class GATv2(torch.nn.Module):
    def __init__(self, timestep, sfreq, n_nodes=18, batch_size=32):
        super(GATv2, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        n_heads = 4
        self.recurrent_1 = GATv2Conv(
            int((sfreq * timestep / 2) + 1),
            32,
            heads=n_heads,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )
        self.recurrent_2 = GATv2Conv(
            32 * n_heads,
            64,
            heads=n_heads,
            negative_slope=0.01,
            dropout=0.4,
            add_self_loops=True,
            improved=True,
            edge_dim=1,
        )

        # int((sfreq*timestep/2)+1)
        # for children in list(self.recurrent_1.children()):
        #     for param in list(children.named_parameters()):
        #         if param[0] == 'weight':
        #             nn.init.kaiming_uniform_(param[1], a=0.01)
        # nn.init.kaiming_uniform_(self.recurrent_1.att,a=0.01)
        self.fc1 = torch.nn.Linear(64 * n_heads, 128)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.01)
        self.fc2 = torch.nn.Linear(128, 64)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.01)
        self.fc3 = torch.nn.Linear(64, 1)
        nn.init.kaiming_uniform_(self.fc3.weight, a=0.01)
        self.fc4 = torch.nn.Linear(128, 1)
        self.connectivity = torch.nn.Linear(sfreq * timestep * n_nodes, 324)
        self.connectivity_2 = torch.nn.Linear(sfreq * timestep, 324)
        self.batch_norm_1 = torch.nn.BatchNorm1d(32 * n_heads)
        self.batch_norm_2 = torch.nn.BatchNorm1d(64 * n_heads)
        self.dropout = torch.nn.Dropout()

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.recurrent_1(x, edge_index=edge_index, edge_attr=edge_attr)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        # h = global_mean_pool(h,batch)

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
        # h = F.leaky_relu(h)
        # h = self.dropout(h)
        # h = self.fc4(h)
        return h.squeeze(1)



parser = ArgumentParser()
parser.add_argument("--timestep", type=int, default=6)
parser.add_argument("--ictal_overlap", type=int, default=0)
parser.add_argument("--inter_overlap", type=int, default=0)
parser.add_argument("--smote", action='store_true')
parser.add_argument("--weights", action='store_true')
parser.add_argument("--undersample", action='store_true')
parser.add_argument("--exp_name", type=str, default="eeg_exp")
parser.add_argument("--data_dir", type=str,default='data/npy_data')
parser.add_argument("--epochs", type=int,default=25)
args = parser.parse_args()
TIMESTEP = args.timestep
INTER_OVERLAP = args.inter_overlap
ICTAL_OVERLAP = args.ictal_overlap
SMOTE_FLAG = args.smote
DS_PATH = args.data_dir
WEIGHTS_FLAG = args.weights
UNDERSAMPLE = args.undersample
EXP_NAME = args.exp_name
EPOCHS = args.epochs
SFREQ = 256
DOWNSAMPLING_F = 60
TRAIN_TEST_SPLIT = 0.10
SEIZURE_LOOKBACK = 600
BATCH_SIZE = 256
FFT = True
HJORTH = False
print("Timestep: ", TIMESTEP)
print("Interictal overlap: ", INTER_OVERLAP)
print("Ictal overlap: ", ICTAL_OVERLAP)
print("Smote: ", SMOTE_FLAG)
print("Weights: ", WEIGHTS_FLAG)
print("Undersample: ", UNDERSAMPLE)
print("Epochs: ", EPOCHS)
device = torch.device("cuda:0")
for loso_patient in os.listdir(DS_PATH)[22:]:
    torch_geometric.seed_everything(42)
    dataloader = SeizureDataLoader(
        npy_dataset_path=Path(DS_PATH),
        event_tables_path=Path("data/event_tables"),
        plv_values_path=Path("data/plv_arrays"),
        loso_patient=loso_patient,
        sampling_f=SFREQ,
        seizure_lookback=SEIZURE_LOOKBACK,
        sample_timestep=TIMESTEP,
        inter_overlap=INTER_OVERLAP,
        ictal_overlap=ICTAL_OVERLAP,
        self_loops=False,
        balance=UNDERSAMPLE,
        train_test_split=TRAIN_TEST_SPLIT,
        fft=FFT,
        hjorth=HJORTH,
        downsample=DOWNSAMPLING_F,
        batch_size=BATCH_SIZE,
        smote=SMOTE_FLAG,
    )
    

    
    train_loader, valid_loader, loso_loader = dataloader.get_dataset()
    alpha = list(dataloader._label_counts.values())[0]/list(dataloader._label_counts.values())[1]
    ## normal loop
    print(f"Alpha is {alpha} for patient{loso_patient}")
    CONFIG = dict(
        timestep=TIMESTEP,
        inter_overlap=INTER_OVERLAP,
        ictal_overlap=ICTAL_OVERLAP,
        downsampling_f=DOWNSAMPLING_F,
        train_test_split=TRAIN_TEST_SPLIT,
        fft=FFT,
        hjort=HJORTH,
        seizure_lookback=SEIZURE_LOOKBACK,
        batch_size=BATCH_SIZE,
        smote=SMOTE_FLAG,
        weights=WEIGHTS_FLAG,
        alpha = alpha,
        
    )
    wandb.init(
        project="sano_eeg",
        group=EXP_NAME,
        name=loso_patient,
        job_type="hjorth_features",
        config=CONFIG,
    )
    checkpoint_path = f'checkpoints/checkpoint_{EXP_NAME}.pt'
    early_stopping = utils.EarlyStopping(patience=3, verbose=True, path = checkpoint_path)
    model = GATv2(TIMESTEP, 60, batch_size=32).to(device)
    if WEIGHTS_FLAG:
        loss_fn =  nn.BCEWithLogitsLoss(pos_weight=torch.full([1], alpha)).to(device)
    else:
        loss_fn = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00001, weight_decay=0.0001)
    recall = BinaryRecall(threshold=0.5).to(device)
    specificity = BinarySpecificity(threshold=0.5).to(device)
    auroc = AUROC(task="binary").to(device)
    roc = ROC("binary")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=2)

    for epoch in tqdm(range(EPOCHS)):
        try:
            del preds, ground_truth
        except:
            pass
        epoch_loss = 0.0
        epoch_loss_valid = 0.0
        model.train()
        sample_counter = 0
        batch_counter = 0
        print(get_lr(optimizer))
        for time_train, batch in enumerate(
            train_loader
        ):  ## TODO - this thing is still operating with no edge weights!!!
            ## find a way to compute plv per batch fast (is it even possible?)
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.float().to(device)

            y = batch.y.to(device)
            batch_idx = batch.batch.to(device)
            # time_to_seizure = batch.time.float().to(device)

            # x = x.squeeze()
            # x = (x-x.mean(dim=0))/x.std(dim=0)

            x = torch.square(torch.abs(x))

            y_hat = model(x, edge_index, None, batch_idx)
            loss = loss_fn(y_hat, y)
            # loss = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
            epoch_loss += loss
            ## get preds & gorund truth
            try:
                preds = torch.cat([preds, y_hat.detach()], dim=0)
                ground_truth = torch.cat([ground_truth, y], dim=0)

            except:
                preds = y_hat.detach()
                ground_truth = y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        ## calculate acc

        train_auroc = auroc(preds, ground_truth)
        train_sensitivity = recall(preds, ground_truth)
        train_specificity = specificity(preds, ground_truth)
        del preds, ground_truth
        wandb.log(
            {
                "train_loss": epoch_loss.detach().cpu().numpy() / (time_train + 1),
                "epoch": epoch,
                "train_sensitivity": train_sensitivity,
                "train_specificity": train_specificity,
                "train_AUROC": train_auroc,
            }
        )
        print(
            f"Epoch: {epoch}",
            f"Epoch loss: {epoch_loss.detach().cpu().numpy()/(time_train+1)}",
        )
        print(f"Epoch sensitivity: {train_sensitivity}")
        print(f"Epoch specificity: {train_specificity}")
        print(f"Epoch AUROC: {train_auroc} ")
        model.eval()
        with torch.no_grad():
            try:
                del preds_valid, ground_truth_valid
            except:
                pass
            for time_valid, batch_valid in enumerate(valid_loader):
                x = batch_valid.x.to(device)
                edge_index = batch_valid.edge_index.to(device)
                edge_attr = batch_valid.edge_attr.float().to(device)
                y_val = batch_valid.y.to(device)
                batch_idx = batch_valid.batch.to(device)

                # time_to_seizure_val = batch_valid.time.float()
                # x = batch_valid[0].to(device)
                # edge_index = batch_valid[1].to(device)
                # y_val = batch_valid[3].squeeze().to(device)
                # x = x.squeeze()
                # x = (x-x.mean(dim=0))/x.std(dim=0)
                x = torch.square(torch.abs(x))

                y_hat_val = model(x, edge_index, None, batch_idx)
                loss_valid = loss_fn(y_hat_val, y_val)
                # loss_valid = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
                epoch_loss_valid += loss_valid
                try:
                    preds_valid = torch.cat([preds_valid, y_hat_val], dim=0)
                    ground_truth_valid = torch.cat([ground_truth_valid, y_val], dim=0)
                except:
                    preds_valid = y_hat_val
                    ground_truth_valid = y_val
        # scheduler.step(epoch_loss_valid)
        early_stopping(epoch_loss_valid.cpu().numpy() / (time_valid + 1), model)
        val_auroc = auroc(preds_valid, ground_truth_valid)
        val_sensitivity = recall(preds_valid, ground_truth_valid)
        val_specificity = specificity(preds_valid, ground_truth_valid)
        del preds_valid, ground_truth_valid
        wandb.log(
            {
                "val_loss": epoch_loss_valid.cpu().numpy() / (time_valid + 1),
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_AUROC": val_auroc,
            }
        )
        print(f"Epoch val_loss: {epoch_loss_valid.cpu().numpy()/(time_valid+1)}")
        print(f"Epoch val_sensitivity: {val_sensitivity}")
        print(f"Epoch val specificity: {val_specificity}")
        print(f"Epoch val AUROC: {val_auroc} ")
        if early_stopping.early_stop:
            print("Early stopping")
            model.load_state_dict(torch.load(checkpoint_path))
            break

    with torch.no_grad():
        model.eval()
        try:
            del preds_valid, ground_truth_valid
        except:
            pass
        epoch_loss_loso = 0.0
        for time_loso, batch_loso in enumerate(loso_loader):
            x = batch_loso.x.to(device)
            edge_index = batch_loso.edge_index.to(device)
            edge_attr = batch_loso.edge_attr.float().to(device)
            y_loso = batch_loso.y.to(device)
            batch_idx = batch_loso.batch.to(device)

            # time_to_seizure_loso = batch_loso.time.float()
            # x = x.squeeze()
            # x = batch_valid[0].to(device)
            # edge_index = batch_valid[1].to(device)
            # y_val = batch_valid[3].squeeze().to(device)
            # x = (x-x.mean(dim=0))/x.std(dim=0)
            x = torch.square(torch.abs(x))

            y_hat_loso = model(x, edge_index, None, batch_idx)
            loss_loso = loss_fn(y_hat_loso, y_loso)
            # loss_valid = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
            epoch_loss_loso += loss_loso
            try:
                preds_loso = torch.cat([preds_loso, y_hat_loso], dim=0)
                ground_truth_loso = torch.cat([ground_truth_loso, y_loso], dim=0)
            except:
                preds_loso = y_hat_loso
                ground_truth_loso = y_loso
        loso_auroc = auroc(preds_loso, ground_truth_loso)
        loso_sensitivity = recall(preds_loso, ground_truth_loso)
        loso_specificity = specificity(preds_loso, ground_truth_loso)
        del preds_loso, ground_truth_loso
        wandb.log(
            {
                " loso_loss": epoch_loss_loso.cpu().numpy() / (time_loso + 1),
                "loso_sensitivity": loso_sensitivity,
                "loso_specificity": loso_specificity,
                "loso_AUROC": loso_auroc,
            }
        )
        print(f"Loso_loss: {epoch_loss_loso.cpu().numpy()/(time_loso+1)}")
        print(f"Loso_sensitivity: {loso_sensitivity}")
        print(f"Loso_specificity: {loso_specificity}")
        print(f"Loso_AUROC: {loso_auroc} ")
        wandb.finish()
