import pandas as pd
import numpy as np
from pathlib import Path
import os
from dataclasses import dataclass
import utils
from torch_geometric_temporal import DynamicGraphTemporalSignal
import networkx as nx
from torch_geometric.utils import from_networkx
import sklearn
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics.classification import BinaryRecall
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv, BatchNorm
print("Imports done")
TIMESTEP = 5
INTER_OVERLAP = 0
ICTAL_OVERLAP = 4
SFREQ = 256


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
    shuffle = True
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
        edges_per_sample_val = np.repeat(
            edges, repeats=self._val_features.shape[0], axis=0
        )
        self._edges = edges_per_sample_train
        self._val_edges = edges_per_sample_val

    def _get_labels_features_edge_weights(self):
        """Prepare features, labels, time labels and edge wieghts for training and
        optionally validation data."""
        patient_list = os.listdir(self.npy_dataset_path)
        for patient in patient_list:  # iterate over patient names
            event_tables = self._get_event_tables(
                patient
            )  # extract start and stop of seizure for patient
            patient_path = os.path.join(self.npy_dataset_path, patient)
            recording_list = os.listdir(patient_path)
            for record in recording_list:  # iterate over recordings for a patient
                recording_path = os.path.join(patient_path, record)
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
                        # np.random.uniform(size=(18,18))
                        # plv_connectivity(data_array.shape[0],data_array)
                    ),
                    axis=0,
                )

                ##TODO add a gateway to reject seizure periods shorter than lookback
                # extract timeseries and labels from the array
                features, labels, time_labels = utils.extract_training_data_and_labels(
                    data_array,
                    start_event_tables,
                    stop_event_tables,
                    fs=self.sampling_f,
                    seizure_lookback=self.seizure_lookback,
                    sample_timestep=self.sample_timestep,
                    inter_overlap=self.inter_overlap,
                    ictal_overlap=self.ictal_overlap,
                )

                time_labels = time_labels.astype(np.int32)
                labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)

                if patient == self.loso_patient:
                    try:
                        self._val_features = np.concatenate(
                            (self._val_features, features)
                        )
                        self._val_labels = np.concatenate((self._val_labels, labels))
                        self._val_time_labels = np.concatenate(
                            (self._val_time_labels, time_labels)
                        )
                        self._val_edge_weights = np.concatenate(
                            (
                                self._val_edge_weights,
                                np.repeat(plv_edge_weights, features.shape[0], axis=0),
                            )
                        )
                    except:
                        self._val_features = features
                        self._val_labels = labels
                        self._val_time_labels = time_labels
                        self._val_edge_weights = np.repeat(
                            plv_edge_weights, features.shape[0], axis=0
                        )
                else:
                    try:
                        self._features = np.concatenate((self._features, features))
                        self._labels = np.concatenate((self._labels, labels))
                        self._time_labels = np.concatenate(
                            (self._time_labels, time_labels)
                        )
                        self._edge_weights = np.concatenate(
                            (
                                self._edge_weights,
                                np.repeat(plv_edge_weights, features.shape[0], axis=0),
                            )
                        )

                    except:
                        print("Creating initial attributes")
                        self._features = features
                        self._labels = labels
                        self._time_labels = time_labels
                        self._edge_weights = np.repeat(
                            plv_edge_weights, features.shape[0], axis=0
                        )

        if self.shuffle is True:
            (
                shuffled_features,
                shuffled_labels,
                shuffled_time_labels,
                shuffled_edge_weights,
            ) = sklearn.utils.shuffle(
                self._features, self._labels, self._time_labels, self._edge_weights
            )
            self._features = shuffled_features
            self._labels = shuffled_labels
            self._time_labels = shuffled_time_labels
            self._edge_weights = shuffled_edge_weights

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

        self._get_labels_features_edge_weights()
        self._get_edges()

        train_dataset = DynamicGraphTemporalSignal(
            self._edges,
            self._edge_weights,
            self._features,
            self._labels,
            time_labels=self._time_labels,
        )
        if self.loso_patient:
            val_dataset = DynamicGraphTemporalSignal(
                self._val_edges,
                self._val_edge_weights,
                self._val_features,
                self._val_labels,
                time_labels=self._val_time_labels,
            )
            return train_dataset, val_dataset

        return train_dataset


class RecurrentGCN(torch.nn.Module):
    def __init__(self, timestep, sfreq, n_nodes=18):
        super(RecurrentGCN, self).__init__()
        self.out_features = 128
        self.recurrent_1 = GCNConv(timestep * sfreq, 32, add_self_loops=False)
        self.recurrent_2 = GCNConv(32, 64, add_self_loops=False)
        self.recurrent_3 = GCNConv(64, 128, add_self_loops=False)
        self.fc1 = torch.nn.Linear(self.out_features * n_nodes, 128)
        self.fc2 = torch.nn.Linear(128, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        self.flatten = torch.nn.Flatten(start_dim=0)
        self.dropout = torch.nn.Dropout()

    def forward(self, x, edge_index, edge_weight):
        x = torch.squeeze(x)
        # x = F.normalize(x,dim=1)
        h = self.recurrent_1(x, edge_index, edge_weight)
        h = BatchNorm(32)(h)
        h = F.leaky_relu(h)
        h = self.recurrent_2(h, edge_index, edge_weight)
        h = BatchNorm(64)(h)
        h = F.leaky_relu(h)
        h = self.recurrent_3(h, edge_index, edge_weight)
        h = BatchNorm(128)(h)
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
        h = F.leaky_relu(h)
        h = self.dropout(h)
        h = self.fc4(h)
        return h


class EEGDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        npy_dataset_path: str = None,
        event_tables_path: str = None,
        plv_values_path: str = None,
        loso_patient: str = None,
        sfreq: int = 256,
        seizure_lookback: int = 600,
        sample_timestep: int = 10,
        inter_overlap: int = 0,
        ictal_overlap: int = 0,
        self_loops: bool = True,
    ) -> None:
        super(EEGDataLoader, self).__init__()
        self.dataloader = SeizureDataLoader(
            npy_dataset_path=npy_dataset_path,
            event_tables_path=event_tables_path,
            plv_values_path=plv_values_path,
            loso_patient=loso_patient,
            sampling_f=sfreq,
            seizure_lookback=seizure_lookback,
            sample_timestep=sample_timestep,
            inter_overlap=inter_overlap,
            ictal_overlap=ictal_overlap,
            self_loops=self_loops,
        )

    def setup(self, stage=None):
        self.train_loader, self.valid_loader = self.dataloader.get_dataset()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.valid_loader


class GraphConv(pl.LightningModule):
    def __init__(
        self,
        n_nodes: int = 18,
        timestep: int = 10,
        sfreq: int = 256,
        loss_weight: int = 2,
        binary_recall_threshold: float = 0.5,
    ) -> None:
        super(GraphConv, self).__init__()
        self.n_nodes = n_nodes
        self.timestep = timestep
        self.sfreq = sfreq
        self.model = RecurrentGCN(timestep, sfreq, n_nodes)
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.full([1], loss_weight))
        self.recall = BinaryRecall(threshold=binary_recall_threshold)

    def forward(self, x, edge_index, edge_attr):
        # print(x.shape)
        # print(edge_index.shape)
        # print(edge_attr.shape)
        return self.model(x, edge_index, edge_attr)

    def training_step(self, snapshot):
        # print(snapshot.x.shape)
        # print(snapshot.edge_index.shape)
        # print(snapshot.edge_attr.shape)
        logits = self(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        loss = self.loss(logits, snapshot.y)
        sensitivity = self.recall(logits, snapshot.y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_sensitivity", sensitivity, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, snapshot):
        logits = self(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        val_loss = self.loss(logits, snapshot.y)
        sensitivity = self.recall(logits, snapshot.y)
        self.log("train_loss", val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train_sensitivity", sensitivity, on_step=True, on_epoch=True, prog_bar=True
        )
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

print("Classes done")
pl_dataloader = EEGDataLoader(
    npy_dataset_path=Path("data/npy_data"),
    event_tables_path=Path("data/event_tables"),
    plv_values_path=Path("data/plv_arrays"),
    loso_patient="chb16",
    sfreq=SFREQ,
    seizure_lookback=600,
    sample_timestep=TIMESTEP,
    inter_overlap=INTER_OVERLAP,
    ictal_overlap=ICTAL_OVERLAP,
    self_loops=True,
)
print("Setup...")
pl_dataloader.setup()
print("Setup done")
pl_model = GraphConv(18, TIMESTEP, SFREQ)
print("Model done")
print("Trainer...")
trainer = pl.Trainer(
    accelerator="gpu", devices=2, num_nodes=1, max_epochs=10, num_sanity_val_steps=0
)
print("Training...")
trainer.fit(pl_model,pl_dataloader)