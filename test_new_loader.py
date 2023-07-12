import torch
import h5py
import os
import numpy as np
from pathlib import Path
import pandas as pd
from dataclasses import dataclass, field
import torch_geometric
from torch_geometric.data import Data, Dataset
from dataclasses import dataclass
import utils
from imblearn.over_sampling import SMOTE
from torch_geometric.utils import from_networkx
from scipy.signal import resample
import networkx as nx
from joblib import Parallel, delayed
from time import time


@dataclass
class HDFDataset_Writer:
    npy_dataset_path: str
    event_tables_path: str
    cache_folder: str
    seizure_lookback: int = 600
    sample_timestep: int = 5
    inter_overlap: int = 0
    preictal_overlap: int = 0
    ictal_overlap: int = 0
    downsample: int = None
    sampling_f: int = 256
    self_loops: bool = False
    balance: bool = False
    smote: bool = False
    buffer_time: int = 15
    used_classes_dict: dict[str] = field(
        default_factory=lambda: {"interictal": True, "preictal": True, "ictal": True}
    )

    def _get_event_tables(self, patient_name: str) -> tuple[dict, dict]:
        """Read events for given patient into start and stop times lists from .csv extracted files.
        Args:
            patient_name: (str) Name of the patient to get events for.
        Returns:
            start_events_dict: (dict) Dictionary with start events for given patient.
            stop_events_dict: (dict) Dictionary with stop events for given patient.
        """

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
        """Read seizure times into list from event_dict.
        Args:
            events_dict: (dict) Dictionary with events for given patient.
            recording: (str) Name of the recording to get events for.
        Returns:
            recording_events: (list) List of seizure event start and stop time for given recording.
        """
        recording_list = list(events_dict[recording + ".edf"].values())
        recording_events = [int(x) for x in recording_list if not np.isnan(x)]
        return recording_events

    def _create_edge_idx_and_attributes(
        self, connectivity_matrix: np.ndarray, threshold: int = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create adjacency matrix from connectivity matrix. Edges are created for values above threshold.
        If the edge is created, it has an attribute "weight" with the value of the connectivity measure associated.
        Args:
            connectivity_matrix: (np.ndarray) Array with connectivity values.
            threshold: (float) Threshold for creating edges. (default: 0.0)
        Returns:
            edge_index: (np.ndarray) Array with edge indices.
            edge_weights: (np.ndarray) Array with edge weights.
        """
        result_graph = nx.graph.Graph()
        n_nodes = connectivity_matrix.shape[0]
        result_graph.add_nodes_from(range(n_nodes))
        edge_tuples = [
            (i, j)
            for i in range(n_nodes)
            for j in range(n_nodes)
            if connectivity_matrix[i, j] > threshold
        ]
        result_graph.add_edges_from(edge_tuples)
        edge_index = nx.convert_matrix.to_numpy_array(result_graph)
        # connection_indices = np.where(edge_index==1)
        # edge_weights = connectivity_matrix[connection_indices] ## ??

        return edge_index

    def _features_to_data_list(self, features, edges, labels, edge_weights=None):
        """Converts features, edges and labels to list of torch_geometric.data.Data objects.
        Args:
            features: (np.ndarray) Array with features.
            edges: (np.ndarray) Array with edges.
            labels: (np.ndarray) Array with labels.
        Returns:
            data_list: (list) List of torch_geometric.data.Data objects.
        """
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

    def _apply_smote(self, features, labels):
        """Performs SMOTE oversampling on the dataset. Implemented for preictal vs ictal scenarion only.
        Args:
            features: (np.ndarray) Array with features.
            labels: (np.ndarray) Array with labels.
        Returns:
            x_train_smote: (np.ndarray) Array with SMOTE oversampled features.
            y_train_smote: (np.ndarray) Array with SMOTE oversampled labels.
        """
        dim_1, dim_2, dim_3 = features.shape

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

    def _initialize_dict(self):
        """Initializes the dictionary with features, labels and edge weights."""
        self.features_patient = {}
        self.labels_patient = {}
        self.edge_idx_patient = {}
        self.edge_weights_patient = {}

    def _get_labels_features_edge_weights_seizure(self, patient):
        """Method to extract features, labels and edge weights for seizure and interictal samples."""

        event_tables = self._get_event_tables(
            patient
        )  # extract start and stop of seizure for patient
        patient_path = os.path.join(self.npy_dataset_path, patient)
        recording_list = [
            recording
            for recording in os.listdir(patient_path)
            if "seizures" in recording
        ]
        for n, record in enumerate(
            recording_list
        ):  # iterate over recordings for a patient
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

            # plv_edge_weights = np.expand_dims(
            #     self._get_edge_weights_recording(
            #         np.load(os.path.join(self.plv_values_path, patient, record))
            #     ),
            #     axis=0,
            # )

            features, labels, time_labels = utils.extract_training_data_and_labels(
                data_array,
                start_event_tables,
                stop_event_tables,
                fs=self.sampling_f,
                seizure_lookback=self.seizure_lookback,
                sample_timestep=self.sample_timestep,
                preictal_overlap=self.preictal_overlap,
                ictal_overlap=self.ictal_overlap,
                buffer_time=self.buffer_time,
            )

            if features is None:
                print(
                    f"Skipping the recording {record} patients {patient} cuz features are none"
                )
                continue

            features = features.squeeze(2)
            conn_matrix_list = [
                utils.compute_spect_corr_matrix(feature, 256) for feature in features
            ]
            edge_idx = np.stack(
                [
                    self._create_edge_idx_and_attributes(
                        conn_matrix, threshold=np.mean(conn_matrix)
                    )
                    for conn_matrix in conn_matrix_list
                ]
            )
            # edge_idx, edge_weights = zip(*edges_and_weights)
            edge_idx = np.stack(edge_idx)
            # edge_weights = np.stack(edge_weights)
            if self.downsample:
                new_sample_count = int(self.downsample * self.sample_timestep)
                features = resample(features, new_sample_count, axis=2)
            if self.smote:
                features, labels = self._apply_smote(features, labels)
            labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)

            self.features_patient[patient] = (
                features
                if n == 0
                else np.concatenate([self.features_patient[patient], features])
            )
            self.labels_patient[patient] = (
                labels
                if n == 0
                else np.concatenate([self.labels_patient[patient], labels])
            )
            self.edge_idx_patient[patient] = (
                edge_idx
                if n == 0
                else np.concatenate([self.edge_idx_patient[patient], edge_idx])
            )

    #     edge_weights_patient = edge_weights if n == 0 else np.concatenate([edge_weights_patient, edge_weights])

    def get_dataset(self):
        config_name = f"lookback_{self.seizure_lookback}_timestep_{self.sample_timestep}_overlap_{self.inter_overlap}_{self.preictal_overlap}_{self.ictal_overlap}_downsample_{self.downsample}_smote_{self.smote}_self_loops_{self.self_loops}_balance_{self.balance}_buffer_{self.buffer_time}_connectivity_{self.connectivity_measure}"
        dataset_folder = os.path.join(self.cache_folder, config_name)
        dataset_path = os.path.join(dataset_folder, "dataset.hdf5")
        if os.path.exists(dataset_path):
            print(
                f"Folder {config_name} already exists in folder {dataset_folder}. Dataset not created."
            )
            return None
        os.makedirs(dataset_folder, exist_ok=True)

        patient_list = os.listdir(self.npy_dataset_path)
        self._initialize_dict()
        try:
            if self.smote:
                for patient in patient_list:
                    self._get_labels_features_edge_weights_seizure(patient)
            else:
                start = time()
                Parallel(n_jobs=6, backend="loky")(
                    delayed(self._get_labels_features_edge_weights_seizure)(patient)
                    for patient in patient_list
                )
                print(f"Time taken for parallel processing: {time()-start}")
            self.hdf5_file = h5py.File(dataset_path, "w")
            for patient in patient_list:
                self.hdf5_file.create_group(patient)
                self.hdf5_file[patient].create_dataset(
                    "features", data=self.features_patient[patient]
                )
                self.hdf5_file[patient].create_dataset(
                    "labels", data=self.labels_patient[patient]
                )
                self.hdf5_file[patient].create_dataset(
                    "edge_idx", data=self.edge_idx_patient[patient]
                )
            self.hdf5_file.close()
            print(f"Dataset created in folder {dataset_folder}.")
        except:
            self.hdf5_file.close()
            os.remove(dataset_path)
            raise Exception("Dataset creation failed. Dataset deleted.")


temp_names = ["cache", "cache1"]
for folder in temp_names:
    torch_geometric.seed_everything(42)
    hdf_writer = HDFDataset_Writer(
        npy_dataset_path="data/npy_data_full",
        event_tables_path="data/event_tables",
        cache_folder=folder,
        seizure_lookback=600,
        sample_timestep=9,
        downsample=60,
    )

    torch_geometric.seed_everything(42)
    hdf_writer.get_dataset()
