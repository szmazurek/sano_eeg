import pandas as pd
import numpy as np
from pathlib import Path
import os
from dataclasses import dataclass
import utils
from torch_geometric_temporal import  DynamicGraphTemporalSignal,StaticGraphTemporalSignal, temporal_signal_split, DynamicGraphTemporalSignalBatch
import networkx as nx
from torch_geometric.utils import from_networkx
import scipy
import sklearn
import random
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DynamicBatchSampler,DataLoader
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import sigmoid_focal_loss
from torch_geometric_temporal.nn.recurrent import DCRNN,  GConvGRU, A3TGCN, TGCN2, TGCN, A3TGCN2
from torch_geometric_temporal.nn.attention import STConv
from torchmetrics.classification import BinaryRecall,BinarySpecificity, AUROC, ROC
from torch_geometric.nn import global_mean_pool
import pytorch_lightning as pl
from torch_geometric.nn import GCNConv,BatchNorm,GATv2Conv
from sklearn.model_selection import KFold,StratifiedKFold, StratifiedShuffleSplit
from mne_features.univariate import compute_kurtosis, compute_hjorth_complexity, compute_hjorth_mobility
import torchaudio
import mne_features
import wandb
torch.manual_seed(42)
random.seed(42)

api_key = open("wandb_api_key.txt", "r")
key = api_key.read()
api_key.close()
os.environ["WANDB_API_KEY"] = key
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
torch.backends.cudnn.benchmark = False
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
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
    downsample: int = None
    buffer_time: int = 15
    batch_size : int = 32
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

    def _val_array_to_tensor(self):
        self._val_features = torch.tensor(self._val_features, dtype=torch.float32)
        self._val_labels = torch.tensor(self._val_labels)
        self._val_time_labels = torch.tensor(self._val_time_labels)
        self._val_edge_weights = torch.tensor(self._val_edge_weights)

    def _get_labels_count(self):
        labels, counts = np.unique(self._labels, return_counts=True)
        self._label_counts = {}
        for n, label in enumerate(labels):
            self._label_counts[int(label)] = counts[n]

    def _get_val_labels_count(self):
        labels, counts = np.unique(self._val_labels, return_counts=True)
        self._val_label_counts = {}
        for n, label in enumerate(labels):
            self._val_label_counts[int(label)] = counts[n]

    def _perform_features_train_fft(self):
        self._features = torch.fft.rfft(self._features)

    def _perform_features_val_fft(self):
        self._val_features = torch.fft.rfft(self._val_features)

    def _downsample_features_train(self):
        resampler = torchaudio.transforms.Resample(self.sampling_f, self.downsample)
        self._features = resampler(self._features)

    def _downsample_features_val(self):
        resampler = torchaudio.transforms.Resample(self.sampling_f, self.downsample)
        self._val_features = resampler(self._val_features)

    def _calculate_hjorth_features_train(self):

        new_features = [
            np.concatenate(
                [
                    compute_hjorth_mobility(feature),
                    compute_hjorth_complexity(feature),
                ],
                axis=1,
            )
            for feature in self._features
        ]
        self._features = np.array(new_features)

    def _calculate_hjorth_features_val(self):

        new_features = [
            np.concatenate(
                [
                    compute_hjorth_mobility(feature),
                    compute_hjorth_complexity(feature),
                ],
                axis=1,
            )
            for feature in self._val_features
        ]
        self._val_features = np.array(new_features)
    def _features_to_data_list(self,features,edges,edge_weights,labels, time_label):
        data_list = [
                    Data(
                        x=features[i],
                        edge_index=edges[i],
                        edge_attr=edge_weights[i],
                        y=labels[i],
                        time=time_label[i],
                    )
                    for i in range(len(features))
                ]
        return data_list
    def _split_data_list(self,data_list):
        class_labels = [data.y.item() for data in data_list]
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.train_test_split, random_state=42)
        train_indices, val_indices = next(splitter.split(data_list, class_labels))
        data_list_train = [data_list[i] for i in train_indices]
        dataset_list_val = [data_list[i] for i in val_indices]
        return data_list_train, dataset_list_val
    def _balance_classes(self):
        negative_label = self._label_counts[0]
        positive_label = self._label_counts[1]

        imbalance = negative_label - positive_label
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
    def _standardize_data(self,features,labels):
        indexes = np.where(labels == 0)[0]
        features_negative = features[indexes]
        channel_mean = features_negative.mean(axis=0).mean(1)
        channel_std = features_negative.std(axis=0).std(1)
        if 0 in channel_std:
            channel_mean = features_negative.mean(2)
            channel_std = features_negative.std(2)
        for i in range(features.shape[0]):
            for n in range(features.shape[1]):
                features[i,n,:] = (features[i,n,:] - channel_mean[n])/channel_std[n]
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
                    buffer_time=self.buffer_time,
                )

                if features is None:
                    continue
                features = features.squeeze()
                self._standardize_data(features,labels)
                time_labels = np.expand_dims(time_labels.astype(np.int32), 1)
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
        self.train_features_min_max = [self._features.min(), self._features.max()]
        if self.balance:
            self._get_labels_count()
            self._balance_classes()
        self._get_edges()
        self._get_labels_count()
        if self.hjorth:
            self._calculate_hjorth_features_train()
        self._array_to_tensor()
        if self.downsample:
            self._downsample_features_train()
        if self.fft:
            self._perform_features_train_fft()
        train_dataset = torch.utils.data.TensorDataset(
            self._features,
            self._edges,
            self._edge_weights,
            self._labels,
            self._time_labels,
        )
        if self.train_test_split is not None:
            if self.fft or self.hjorth:
                data_list = self._features_to_data_list(
                    self._features,self._edges,self._edge_weights, self._labels, self._time_labels
                )
                train_data_list, val_data_list = self._split_data_list(data_list)
                label_count= np.unique([data.y.item() for data in train_data_list],return_counts=True)[1]
                self.alpha = label_count[0]/label_count[1]
                loaders = [
                    DataLoader(train_data_list, batch_size=self.batch_size, shuffle=True,drop_last=False),
                    DataLoader(val_data_list, batch_size=len(val_data_list), shuffle=False,drop_last=False)
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
                    num_workers=2,
                    pin_memory=True,
                    prefetch_factor=4,
                    drop_last=False,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True,
                    prefetch_factor=4,
                    drop_last=False,
                )
                loaders = [train_dataloader, val_dataloader]
        else:
            if self.fft or self.hjorth:
                train_data_list = self._features_to_data_list(
                    self._features,self._edges,self._edge_weights, self._labels, self._time_labels
                )
                loaders = [DataLoader(train_data_list, batch_size=self.batch_size, shuffle=True,drop_last=False)]
            else:
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True,
                    prefetch_factor=4,
                    drop_last=False,
                )
                loaders = [train_dataloader]
        if self.loso_patient:
            self.val_features_min_max = [
                self._val_features.min(),
                self._val_features.max(),
            ]
            self._get_val_labels_count()
            if self.hjorth:
                self._calculate_hjorth_features_val()
            self._val_array_to_tensor()
            if self.downsample:
                self._downsample_features_val()
            if self.fft:
                self._perform_features_val_fft()
            if self.fft or self.hjorth:
                loso_data_list = self._features_to_data_list(
                    self._val_features,self._val_edges,self._val_edge_weights, self._val_labels, self._val_time_labels
                )
                return (*loaders, DataLoader(loso_data_list, batch_size=len(loso_data_list), shuffle=False,drop_last=False))
            loso_dataset = torch.utils.data.TensorDataset(
                self._val_features,
                self._val_edges,
                self._val_edge_weights,
                self._val_labels,
                self._val_time_labels,
            )
            loso_dataloader = torch.utils.data.DataLoader(
                loso_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
                prefetch_factor=4,
                drop_last=False,
            )
            return (*loaders, loso_dataloader)

        return (*loaders,)

class RecurrentGCN(torch.nn.Module):
    def __init__(self, timestep,sfreq, n_nodes=18,batch_size=32):
        super(RecurrentGCN, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        self.recurrent_1 = GCNConv(sfreq*timestep,32, add_self_loops=True,improved=False)
        self.recurrent_2 = GCNConv(32,64,add_self_loops=True,improved=False)
        self.recurrent_3 = GCNConv(64,128,add_self_loops=True,improved=False)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(32)
        self.batch_norm_2 = torch.nn.BatchNorm1d(64)
        self.batch_norm_3 = torch.nn.BatchNorm1d(128)
        self.flatten = torch.nn.Flatten(start_dim=0)
        self.dropout = torch.nn.Dropout()
    def forward(self, x, edge_index,edge_weight,batch):
        x = torch.squeeze(x)
        h = self.recurrent_1(x, edge_index=edge_index, edge_weight = edge_weight)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = self.recurrent_2(h, edge_index,edge_weight)
        h = self.batch_norm_2(h)
        h = F.leaky_relu(h)
        h = self.recurrent_3(h, edge_index,edge_weight)
        h = self.batch_norm_3(h)
        h = F.leaky_relu(h)
        h = global_mean_pool(h,batch)
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
    def __init__(self, timestep,sfreq, n_nodes=18,batch_size=32):
        super(AT3Batched, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        self.recurrent_1 = A3TGCN2(1,64, periods = timestep*sfreq,add_self_loops=True,improved=False,batch_size=batch_size)
        self.fc1 = torch.nn.Linear(64*n_nodes, 64)
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
    def __init__(self, timestep,sfreq, n_nodes=18,batch_size=32):
        super(GATv2, self).__init__()
        self.n_nodes = n_nodes
        self.out_features = 128
        self.recurrent_1 = GATv2Conv(int((sfreq*timestep/2)+1),32,heads=4,negative_slope=0.01,dropout=0.4, add_self_loops=True,improved=True,edge_dim=1)
        # for children in list(self.recurrent_1.children()):
        #     for param in list(children.named_parameters()):
        #         if param[0] == 'weight':
        #             nn.init.kaiming_uniform_(param[1], a=0.01)
        #nn.init.kaiming_uniform_(self.recurrent_1.att,a=0.01)
        self.fc1 = torch.nn.Linear(32*4, 64)
        nn.init.kaiming_uniform_(self.fc1.weight,a=0.01)
        self.fc2 = torch.nn.Linear(64, 32)
        nn.init.kaiming_uniform_(self.fc2.weight,a=0.01)
        self.fc3 = torch.nn.Linear(32, 1)
        nn.init.kaiming_uniform_(self.fc3.weight,a=0.01)
        self.fc4 = torch.nn.Linear(128, 1)
        self.batch_norm_1 = torch.nn.BatchNorm1d(32*4)
        self.batch_norm_2 = torch.nn.BatchNorm1d(32*8)
        self.dropout = torch.nn.Dropout()
        self.flatten = torch.nn.Flatten(start_dim=1)
    def forward(self, x, edge_index, edge_attr,batch):
        
        h = self.recurrent_1(x, edge_index=edge_index, edge_attr = edge_attr)
        h = self.batch_norm_1(h)
        h = F.leaky_relu(h)
        h = global_mean_pool(h,batch)
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
        return h.squeeze()

TIMESTEP = 3
INTER_OVERLAP = 0
ICTAL_OVERLAP = 2
SFREQ = 256
DOWNSAMPLING_F = 60
TRAIN_TEST_SPLIT = 0.1
SEIZURE_LOOKBACK = 600
BATCH_SIZE = 64
FFT = True
device = torch.device("cuda:0")
#for loso_patient in os.listdir('data/npy_data'):
dataloader = SeizureDataLoader(
    npy_dataset_path=Path('data/npy_data'),
    event_tables_path=Path('data/event_tables'),
    plv_values_path=Path('data/plv_arrays'),
    loso_patient='chb08',
    sampling_f=SFREQ,
    seizure_lookback=SEIZURE_LOOKBACK,
    sample_timestep= TIMESTEP,
    inter_overlap=INTER_OVERLAP,
    ictal_overlap=ICTAL_OVERLAP,
    self_loops=False,
    balance=False,
    train_test_split=TRAIN_TEST_SPLIT,
    fft=FFT,
    hjorth=False,
    downsample=DOWNSAMPLING_F,
    batch_size=BATCH_SIZE
    )
CONFIG = dict(
    timestep = TIMESTEP,
    inter_overlap = INTER_OVERLAP,
    ictal_overlap = ICTAL_OVERLAP,
    downsampling_f = DOWNSAMPLING_F,
    train_test_split = TRAIN_TEST_SPLIT,
    fft = FFT,
    seizure_lookback = SEIZURE_LOOKBACK,
    batch_size = BATCH_SIZE
)

wandb.init(
    project="sano_eeg",
    group="Gatv2",
    name='chb16',
    job_type="FFT_normalized",
    config=CONFIG,
)
train_loader,valid_loader,loso_loader=dataloader.get_dataset()
alpha = list(dataloader._label_counts.values())[0]/list(dataloader._label_counts.values())[1]
## normal loop


model = GATv2(TIMESTEP,60,batch_size=32).to(device)
#model = AT3Batched(TIMESTEP,60,batch_size=64).to(device)
loss_fn =  nn.BCEWithLogitsLoss(pos_weight=torch.full([1], alpha)).to(device)

#loss_fn = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
recall = BinaryRecall(threshold=0.5).to(device)
specificity = BinarySpecificity(threshold=0.5).to(device)
auroc = AUROC(task="binary").to(device)
roc = ROC('binary')

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2)

for epoch in tqdm(range(10)):

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
    for time, batch in enumerate(train_loader): ## TODO - this thing is still operating with no edge weights!!!
            ## find a way to compute plv per batch fast (is it even possible?)
            x = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.float().to(device)

            y = batch.y.to(device)
            batch_idx = batch.batch.to(device)
            time_to_seizure = batch.time.float().to(device)
            
            # x = x.squeeze()
            # x = (x-x.mean(dim=0))/x.std(dim=0)
    
            # x =torch.square(torch.abs(x))
            
            y_hat = model(x, edge_index,edge_attr,batch_idx)
            loss = loss_fn(y_hat,y)
            #loss = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
            epoch_loss += loss
            ## get preds & gorund truth
            try:
                preds = torch.cat([preds,y_hat.detach()],dim=0)
                ground_truth = torch.cat([ground_truth,y],dim=0)
        
            except:
                preds= y_hat.detach()
                ground_truth = y
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    ## calculate acc

    train_auroc = auroc(preds,ground_truth)
    train_sensitivity = recall(preds,ground_truth)
    train_specificity = specificity(preds,ground_truth)
    del preds, ground_truth
    wandb.log({"train_loss": epoch_loss.detach().cpu().numpy()/(time+1), 
                "epoch": epoch,
                "train_sensitivity": train_sensitivity,
                "train_specificity": train_specificity,
                "train_AUROC": train_auroc})
    print(f'Epoch: {epoch}', f'Epoch loss: {epoch_loss.detach().cpu().numpy()/(time+1)}')
    print(f'Epoch sensitivity: {train_sensitivity}')
    print(f'Epoch specificity: {train_specificity}')
    print(f'Epoch AUROC: {train_auroc} ')
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
                    
                    #time_to_seizure_val = batch_valid.time.float()
                    # x = batch_valid[0].to(device)
                    # edge_index = batch_valid[1].to(device)
                    # y_val = batch_valid[3].squeeze().to(device)
                    # x = x.squeeze()
                    x = (x-x.mean(dim=0))/x.std(dim=0)
                    x =  torch.square(torch.abs(x))
                    
                    
                    y_hat_val = model(x, edge_index,edge_attr,batch_idx)
                    loss_valid = loss_fn(y_hat_val,y_val)
                    #loss_valid = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
                    epoch_loss_valid += loss_valid
                    try:
                        preds_valid = torch.cat([preds_valid,y_hat_val],dim=0)
                        ground_truth_valid = torch.cat([ground_truth_valid,y_val],dim=0)
                    except:
                        preds_valid= y_hat_val
                        ground_truth_valid = y_val
    scheduler.step(epoch_loss_valid)
    val_auroc = auroc(preds_valid,ground_truth_valid)
    val_sensitivity = recall(preds_valid,ground_truth_valid)
    val_specificity = specificity(preds_valid,ground_truth_valid)
    del preds_valid, ground_truth_valid
    wandb.log({ "val_loss": epoch_loss_valid.cpu().numpy()/(time_valid+1), 
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_AUROC": val_auroc})
    print(f'Epoch val_loss: {epoch_loss_valid.cpu().numpy()/(time_valid+1)}')
    print(f'Epoch val_sensitivity: {val_sensitivity}')
    print(f'Epoch val specificity: {val_specificity}')
    print(f'Epoch val AUROC: {val_auroc} ')
    

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
            
            #time_to_seizure_val = batch_valid.time.float()
            #x = x.squeeze()
            # x = batch_valid[0].to(device)
            # edge_index = batch_valid[1].to(device)
            # y_val = batch_valid[3].squeeze().to(device)
            x = (x-x.mean(dim=0))/x.std(dim=0)
            x = torch.square(torch.abs(x))
            
            
            y_hat_loso = model(x, edge_index,edge_attr,batch_idx)
            loss_loso = loss_fn(y_hat_loso,y_loso)
            #loss_valid = torchvision.ops.sigmoid_focal_loss(y_hat,y,alpha=alpha*0.1,gamma=2,reduction='mean')
            epoch_loss_loso += loss_loso
            try:
                preds_loso = torch.cat([preds_loso,y_hat_loso],dim=0)
                ground_truth_loso= torch.cat([ground_truth_loso,y_loso],dim=0)
            except:
                preds_loso= y_hat_loso
                ground_truth_loso = y_loso
    loso_auroc = auroc(preds_loso,ground_truth_loso)
    loso_sensitivity = recall(preds_loso,ground_truth_loso)
    loso_specificity = specificity(preds_loso,ground_truth_loso)
    del preds_loso, ground_truth_loso
    wandb.log({" loso_loss": epoch_loss_loso.cpu().numpy()/(time_loso+1), 
                "loso_sensitivity": loso_sensitivity,
                "loso_specificity": loso_specificity,
                "loso_AUROC": loso_auroc})
    print(f'Loso_loss: {epoch_loss_loso.cpu().numpy()/(time_loso+1)}')
    print(f'Loso_sensitivity: {loso_sensitivity}')
    print(f'Loso_specificity: {loso_specificity}')
    print(f'Loso_AUROC: {loso_auroc} ')
    wandb.finish()