import fnmatch
import logging
import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Union

import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from imblearn.over_sampling import SMOTE
from mne_features.univariate import (compute_energy_freq_bands,
                                     compute_higuchi_fd,
                                     compute_hjorth_complexity,
                                     compute_hjorth_mobility, compute_katz_fd,
                                     compute_line_length, compute_variance)
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

import utils.utils as utils


@dataclass
class HDFDataset_Writer:
    seizure_lookback: int = 600
    sample_timestep: int = 5
    inter_overlap: int = 0
    preictal_overlap: int = 0
    ictal_overlap: int = 0
    downsample: Union[int, None] = None
    sampling_f: int = 256
    smote: bool = False
    buffer_time: int = 15
    connectivity_metric: str = "plv"
    npy_dataset_path: str = "npy_dataset"
    event_tables_path: str = "event_tables"
    cache_folder: str = "cache"

    """Class for creating hdf5 dataset from npy files.
    Args:
        seizure_lookback: (int) Time in seconds to look back from seizure onset. Default: 600.
        sample_timestep: (int) Time in seconds to sample the data. Default: 5.
        inter_overlap: (int) Time in seconds to overlap between interictal samples. Default: 0.
        preictal_overlap: (int) Time in seconds to overlap between preictal samples. Default: 0.
        icatal_overlap: (int) Time in seconds to overlap between ictal samples. Default: 0.
        downsample: (int) Downsampling factor. Default: None.
        sampling_f: (int) Sampling frequency of the input data (before downsampling). Default: 256.
        smote: (bool) Whether to use SMOTE oversampling. Default: False.
        buffer_time: (int) Time in seconds to add to the beggining and the end of the seizure. Default: 15.
        connectivity_metric: (str) Connectivity metric to use. Either 'plv' or 'spectral_corr'. Default: 'plv'.
        npy_dataset_path: (str) Path to the folder with npy recording files. Default: 'npy_dataset'.
        event_tables_path: (str) Path to the folder with event tables. Default: 'event_tables'.
        cache_folder: (str) Path to the folder to store the dataset. Default: 'cache'.
    """

    def __post_init__(self):
        self._initialize_logger()
        assert self.connectivity_metric in [
            "plv",
            "spectral_corr",
        ], "Connectivity metric must be either 'plv' or 'spectral_corr'"
        assert (
            self.downsample is None or self.downsample > 0
        ), "Downsample must be either None or positive integer"
        assert (
            self.sampling_f > 0
        ), "Sampling frequency must be positive integer"
        assert (
            self.sample_timestep > 0
        ), "Sample timestep must be positive integer"
        assert (
            self.seizure_lookback > 0
        ), "Seizure lookback must be positive integer"
        assert self.inter_overlap >= 0, "Inter overlap must be positive integer"
        assert (
            self.preictal_overlap >= 0
        ), "Preictal overlap must be positive integer"
        assert self.ictal_overlap >= 0, "Ictal overlap must be positive integer"
        assert self.buffer_time >= 0, "Buffer time must be positive integer"
        assert (
            self.inter_overlap < self.sample_timestep
        ), "Inter overlap must be smaller than sample timestep"
        assert (
            self.preictal_overlap < self.sample_timestep
        ), "Preictal overlap must be smaller than sample timestep"
        assert (
            self.ictal_overlap < self.sample_timestep
        ), "Ictal overlap must be smaller than sample timestep"

    def _initialize_logger(self):
        """Initializing logger."""
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=f"logs/hdf_dataset_writer_{start_time}.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        self.logger = logging.getLogger("hdf_dataset_writer")

    def _find_matching_configs(self, current_config):
        def find_yaml_files(directory):
            yaml_files = []
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if fnmatch.fnmatch(file, "*.yaml"):
                        yaml_files.append(os.path.join(root, file))
            return yaml_files

        config_files = find_yaml_files(self.cache_folder)
        for config_file in config_files:
            with open(config_file) as f:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
                if config_dict == current_config:
                    self.logger.info(
                        f"Found matching config file {config_file}"
                    )
                    print(f"Found matching config file {config_file}")
                    self.found_dataset_path = os.path.dirname(config_file)
                    return True
        return False

    def _create_config_dict(self):
        dataclass_keys = list(self.__dataclass_fields__.keys())
        dict_values = [self.__getattribute__(key) for key in dataclass_keys]
        initial_config_dict = dict(zip(dataclass_keys, dict_values))
        return initial_config_dict

    def _create_config_file(self, config_dict, dataset_folder_path):
        with open(os.path.join(dataset_folder_path, "config.yaml"), "w") as f:
            yaml.dump(config_dict, f)

    def _get_event_tables(self, patient_name: str) -> tuple[dict, dict]:
        """Read events for given patient into start and stop times lists
        from .csv extracted files.
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
        ]  # done terribly, but it has to be so for win/linux compat
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
        self, connectivity_matrix: np.ndarray, threshold: float = 0
    ) -> np.ndarray:
        """Create adjacency matrix from connectivity matrix. Edges are created for values above threshold.
        If the edge is created, it has an attribute "weight" with the value of the connectivity measure associated.
        Args:
            connectivity_matrix: (np.ndarray) Array with connectivity values.
            threshold: (float) Threshold for creating edges. (default: 0.0)
        Returns:
            edge_index: (np.ndarray) Array with edge indices.
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
        for label in labels:
            new_y_train.extend(label * dim_2)

        new_y_train = np.array(new_y_train)

        # transform the dataset
        oversample = SMOTE(random_state=42)
        x_train, y_train = oversample.fit_resample(new_x_train, new_y_train)
        x_train_smote = x_train.reshape(
            int(x_train.shape[0] / dim_2), dim_2, dim_3
        )
        y_train_smote = []
        for i in range(int(x_train.shape[0] / dim_2)):
            # print(i)
            value_list = list(
                y_train.reshape(int(x_train.shape[0] / dim_2), dim_2)[i]
            )
            # print(list(set(value_list)))
            y_train_smote.extend(list(set(value_list)))
            # Check: if there is any different value in a list
            if len(set(value_list)) != 1:
                print(
                    "\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n"
                )
        y_train_smote = np.array(y_train_smote)
        return x_train_smote, y_train_smote

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
        self.logger.info(
            f"Starting seizure period data loading for patient {patient}"
        )
        for record in recording_list:  # iterate over recordings for a patient
            recording_path = os.path.join(patient_path, record)
            record = record.replace(
                "seizures_", ""
            )  # some magic to get it properly working with event tables
            record_id = record.split(".npy")[0]  # get record id
            start_event_tables = self._get_recording_events(
                event_tables[0], record_id
            )  # get start events
            stop_event_tables = self._get_recording_events(
                event_tables[1], record_id
            )  # get stop events
            data_array = np.load(recording_path)  # load the recording

            (
                features,
                labels,
                time_labels,
            ) = utils.extract_training_data_and_labels(
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

            if features is None or features.shape[0] == 0:
                self.logger.info(
                    f"Skipping the recording {record} patients {patient} - features are none"
                )
                continue

            features = features.squeeze(2)
            sampling_f = (
                self.sampling_f if self.downsample is None else self.downsample
            )
            if self.downsample:
                new_sample_count = int(self.downsample * self.sample_timestep)
                features = resample(features, new_sample_count, axis=2)

            conn_matrix_list = [
                self.connectivity_function(feature, sampling_f)
                for feature in features
            ]

            edge_idx = np.stack(
                [
                    self._create_edge_idx_and_attributes(
                        conn_matrix, threshold=np.mean(conn_matrix)
                    )
                    for conn_matrix in conn_matrix_list
                ]
            )

            if self.smote:
                features, labels = self._apply_smote(features, labels)
            labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)
            try:
                features_patient = np.concatenate([features_patient, features])
                labels_patient = np.concatenate([labels_patient, labels])
                edge_idx_patient = np.concatenate([edge_idx_patient, edge_idx])
                # edge_weights_patient = np.concatenate([edge_weights_patient, edge_weights])
            except NameError:
                features_patient = features
                labels_patient = labels
                edge_idx_patient = edge_idx
                # edge_weights_patient = edge_weights

        try:
            sample_count = features_patient.shape[0]
            n_channels = features_patient.shape[1]
            n_features = features_patient.shape[2]

            while True:
                try:
                    with h5py.File(self.dataset_path, "a") as hdf5_file:
                        hdf5_file[patient].create_dataset(
                            "features",
                            data=features_patient,
                            maxshape=(None, n_channels, n_features),
                        )
                        hdf5_file[patient].create_dataset(
                            "labels", data=labels_patient, maxshape=(None, 1)
                        )
                        hdf5_file[patient].create_dataset(
                            "edge_idx",
                            data=edge_idx_patient,
                            maxshape=(None, n_channels, n_channels),
                        )

                        self.logger.info(
                            f"Dataset for patient {patient} created successfully!."
                        )
                    break
                except BlockingIOError:
                    self.logger.warning(
                        f"Waiting for dataset {patient} to be created."
                    )
                    continue

        except Exception as e:
            self.logger.error(e)
            traceback_str = traceback.format_exc()
            self.logger.error(traceback_str)
            self.logger.error("##############################################")
            self.logger.error(f"Cannot create dataset for patient {patient}!")
            self.logger.error("##############################################")
        preictal_samples = np.unique(labels_patient, return_counts=True)[1][0]
        return (patient, preictal_samples, sample_count)

    def _get_labels_features_edge_weights_interictal(
        self, patient, samples_patient: Union[int, None] = None
    ):
        """Method to extract features, labels and edge weights for interictal samples.
        Args:
            patient: (str) Name of the patient to extract the data for.
            samples_patient (optional): (int) Number of samples to extract for a patient.
        Samples are extracted from non-seizure recordings for a patient, starting from random time point.
        If not specified, the number of samples is calculated as the number of interictal samples for a patient
        divided by the number of recordings for a patient.

        """
        patient_path = os.path.join(self.npy_dataset_path, patient)
        # get all non-seizure recordings for a patient
        recording_list = [
            recording
            for recording in os.listdir(patient_path)
            if "seizures_" not in recording
        ]
        if samples_patient is None:
            # if not specified use the same number of samples for each recording as for preictal samples
            samples_per_recording = int(
                self.preictal_samples_dict[patient] / len(recording_list)
            )
        else:
            samples_per_recording = int(samples_patient / len(recording_list))

        for n, record in enumerate(recording_list):
            recording_path = os.path.join(patient_path, record)
            data_array = np.expand_dims(np.load(recording_path), 1)
            try:
                (
                    features,
                    labels,
                ) = utils.extract_training_data_and_labels_interictal(
                    input_array=data_array,
                    samples_per_recording=samples_per_recording,
                    fs=self.sampling_f,
                    timestep=self.sample_timestep,
                    overlap=self.inter_overlap,
                )
            except ValueError:
                self.logger.error(
                    f"Cannot extract demanded amount of interictal samples from recording {record} for patient {patient}"
                )
                continue

            if features is None:
                self.logger.info(
                    f"Skipping the recording {record} patients {patient} - features are none"
                )
                continue

            idx_to_delete = np.where(
                np.array(
                    [np.diff(feature, axis=-1).mean() for feature in features]
                )
                == 0
            )[0]
            if len(idx_to_delete) > 0:
                features = np.delete(features, obj=idx_to_delete, axis=0)
                labels = np.delete(labels, obj=idx_to_delete, axis=0)
            if features.shape[0] == 0:
                self.logger.info(
                    f"No samples left after removing bad ones for patient {patient} - recording {record}"
                )
                continue
            features = features.squeeze(2)
            sampling_f = (
                self.sampling_f if self.downsample is None else self.downsample
            )
            if self.downsample:
                new_sample_count = int(self.downsample * self.sample_timestep)
                features = resample(features, new_sample_count, axis=2)
            conn_matrix_list = [
                self.connectivity_function(feature, sampling_f)
                for feature in features
            ]
            try:
                edge_idx = np.stack(
                    [
                        self._create_edge_idx_and_attributes(
                            conn_matrix, threshold=float(np.mean(conn_matrix))
                        )
                        for conn_matrix in conn_matrix_list
                    ]
                )
            except Exception as e:
                self.logger.error(e)
                self.logger.error(
                    f"Cannot create edge index for patient {patient} \n recording {record}"
                )
                continue

            labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)

            try:
                features_patient = np.concatenate([features_patient, features])
                labels_patient = np.concatenate([labels_patient, labels])
                edge_idx_patient = np.concatenate([edge_idx_patient, edge_idx])
                # edge_weights_patient = np.concatenate([edge_weights_patient, edge_weights])
            except NameError:
                features_patient = features
                labels_patient = labels
                edge_idx_patient = edge_idx
                # edge_weights_patient = edge_weights

        while True:
            try:
                with h5py.File(self.dataset_path, "a") as hdf5_file:
                    current_patient_features = hdf5_file[patient][
                        "features"
                    ].shape[0]
                    current_patient_labels = hdf5_file[patient]["labels"].shape[
                        0
                    ]
                    current_patient_edge_idx = hdf5_file[patient][
                        "edge_idx"
                    ].shape[0]
                    hdf5_file[patient]["features"].resize(
                        (current_patient_features + features_patient.shape[0]),
                        axis=0,
                    )
                    hdf5_file[patient]["features"][
                        -features_patient.shape[0] :
                    ] = features_patient
                    hdf5_file[patient]["labels"].resize(
                        (current_patient_labels + labels_patient.shape[0]),
                        axis=0,
                    )
                    hdf5_file[patient]["labels"][
                        -labels_patient.shape[0] :
                    ] = labels_patient
                    hdf5_file[patient]["edge_idx"].resize(
                        (current_patient_edge_idx + edge_idx_patient.shape[0]),
                        axis=0,
                    )
                    hdf5_file[patient]["edge_idx"][
                        -edge_idx_patient.shape[0] :
                    ] = edge_idx_patient
                    self.logger.info(
                        f"Dataset for patient {patient} appended successfully!."
                    )
                break
            except BlockingIOError:
                self.logger.warning(f"Waiting for appending of {patient}.")
                continue
        return features_patient.shape[0]

    def _multiprocess_seizure_period_data_loading(self):
        num_processes = 24  # mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        self.logger.info(num_processes)

        result = pool.map(
            self._get_labels_features_edge_weights_seizure, self.patient_list
        )
        pool.close()
        pool.join()
        self.sample_count = 0
        for patient, preictal_samples, sample_count in result:
            self.preictal_samples_dict[patient] = preictal_samples
            self.sample_count += sample_count

    def _multiprocess_interictal_data_loading(self):
        num_processes = 24  # mp.cpu_count()
        self.logger.info(num_processes)
        pool = mp.Pool(processes=num_processes)
        
        result = pool.map(
            self._get_labels_features_edge_weights_interictal, self.patient_list
        )
        pool.close()
        pool.join()
        for sample_count in result:
            self.sample_count += sample_count

    def get_dataset(self):
        folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dataset_folder = os.path.join(self.cache_folder, folder_name)
        self.dataset_path = os.path.join(dataset_folder, "dataset.hdf5")
        current_config = self._create_config_dict()
        if self._find_matching_configs(current_config):
            print("Dataset already exists. Dataset not created.")
            return self.found_dataset_path

        os.makedirs(dataset_folder, exist_ok=True)
        self._create_config_file(current_config, dataset_folder)
        self.patient_list = os.listdir(self.npy_dataset_path)
        with h5py.File(self.dataset_path, "w") as hdf5_file:
            for patient in self.patient_list:
                hdf5_file.create_group(patient)
        self.connectivity_function = (
            utils.compute_plv_matrix
            if self.connectivity_metric == "plv"
            else utils.compute_spect_corr_matrix
        )
        self.preictal_samples_dict = {}
        try:
            if self.smote:
                for patient in self.patient_list:
                    self._get_labels_features_edge_weights_seizure(patient)
            else:
                self._multiprocess_seizure_period_data_loading()
                self.logger.info("Seizure period data loaded.")
                self._multiprocess_interictal_data_loading()
                self.logger.info("Interictal data loaded.")

            print(f"Dataset created in folder {dataset_folder}.")
            print(f"Dataset contains {self.sample_count} samples.")
        except Exception as exc:
            if os.path.exists(self.dataset_path):
                os.remove(self.dataset_path)
            if os.path.exists(os.path.join(dataset_folder, "config.yaml")):
                os.remove(os.path.join(dataset_folder, "config.yaml"))
            if os.path.exists(dataset_folder):
                os.rmdir(dataset_folder)
            self.logger.error("Dataset creation failed. Dataset deleted.")
            raise RuntimeError(
                "Dataset creation failed. Dataset deleted."
            ) from exc

        return dataset_folder


@dataclass
class HDFDatasetLoader:
    DEFAULT_CLASS_LABELS = {"preictal": 0, "ictal": 1, "interictal": 2}
    FREQ_BANDS = [0.5, 4, 8, 13, 29]
    BAND_COUNT = len(FREQ_BANDS) - 1
    """
    Class to load graph data from HDF5 file as lists of torch.geomtric.data.Data objects.
        Args:
            root: (str) Path to the HDF5 file.
            train_val_split_ratio: (float) Ratio of samples to be used for validation. Default: 0.0
            loso_subject: (str) Name of the patient to be used for leave-one-subject-out cross-validation. Default: None
            sampling_f: (int) Sampling frequency of the data. Default: 60
            extract_features: (bool) If True, the timeseries are transformed into a set of chosen EEG features. Default: False
            fft: (bool) If True, FFT is calculated on the timeseries. Default: False
            seed: (int) Random seed. Default: 42
            used_classes_dict: (dict[str]) Dictionary with periods to be used for training.
            Default: {"interictal": True, "preictal": True, "ictal": True}
            normalize_with : (str) Class to use for computing mean and standard deviation for normalization.
            Available options are 'interictal', 'preictal', 'ictal' or 'global'. Default: "interictal"
            kfold_cval_mode: (bool) If True, the get_dataset method returns a single list of data objects
            and labels for stratification. Default: False
    """

    root: str
    train_val_split_ratio: float = 0.0
    loso_subject: Union[str, None] = None
    sampling_f: int = 60
    extract_features: bool = False
    fft: bool = False
    seed: int = 42
    used_classes_dict: dict[str, bool] = field(
        default_factory=lambda: {
            "interictal": True,
            "preictal": True,
            "ictal": True,
        }
    )
    normalize_with: str = "interictal"
    kfold_cval_mode: bool = False

    def __post_init__(self):
        self._check_arguments()
        self._logger_init()
        self._determine_dataset_characteristics()
        self._show_used_classes()

    def _check_arguments(self):
        """Method to check if the arguments are valid."""
        assert (
            self.sampling_f > 0
        ), "Sampling frequency must be positive integer"
        assert not (
            self.extract_features and self.fft
        ), "Cannot extract both features and FFT"
        assert (
            self.train_val_split_ratio >= 0.0
            and self.train_val_split_ratio <= 1.0
        ), "Train val split ratio must be between 0.0 and 1.0"
        assert (
            sum(self.used_classes_dict.values()) > 1
        ), "At least two classes must be used for training!"
        assert (
            len(self.used_classes_dict) == 3
        ), "Used classes dict must contain 3 keys"
        assert (
            "interictal" in self.used_classes_dict.keys()
        ), "Used classes dict must contain 'interictal' key"
        assert (
            "preictal" in self.used_classes_dict.keys()
        ), "Used classes dict must contain 'preictal' key"
        assert (
            "ictal" in self.used_classes_dict.keys()
        ), "Used classes dict must contain 'ictal' key"
        assert self.normalize_with in [
            "interictal",
            "preictal",
            "ictal",
            "global",
        ], "Normalize with must be either 'interictal', 'preictal', 'ictal' or 'global'"

        if self.kfold_cval_mode:
            assert (
                self.train_val_split_ratio == 0.0
            ), "Train val split ratio must be 0.0 in kfold mode"
            assert (
                self.loso_subject is None
            ), "Loso subject must be None in kfold mode"

    def _show_used_classes(self):
        """Method to show which classes are used for training."""
        self.logger.info(f"Used classes: {self.used_classes_dict}")
        used_periods = [
            key for key, value in self.used_classes_dict.items() if value
        ]
        for period in used_periods:
            print(f"USING CLASS: {period}")

    def _determine_dataset_characteristics(self):
        """Method to determine dataset characteristics."""
        self.hdf_data_path = f"{self.root}/dataset.hdf5"
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            self.patient_list = list(hdf5_file.keys())
            self.n_channels, self.n_features = hdf5_file[self.patient_list[0]][
                "features"
            ].shape[1:]
        if self.loso_subject is not None:
            self.patient_list.remove(self.loso_subject)
        self._determine_sample_count()
        self._get_mean_std()

    def _logger_init(self):
        """Initializing logger."""
        if not os.path.exists("logs/"):
            os.makedirs("logs/")
        start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        logging.basicConfig(
            filename=f"logs/hdf_dataloader_{start_time}.log",
            level=logging.INFO,
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            force=True,
        )
        self.logger = logging.getLogger("hdf_dataloader")
        print(
            f"Logger initialized. Logs saved to logs/hdf_dataloader_{start_time}.log"
        )

    def _determine_sample_count(self):
        """Method to determine number of samples in the dataset. the values are later used to
        create placeholder arrays during mean and standard deviation calculation.
        """
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            total_samples = 0
            for patient in self.patient_list:
                total_samples += hdf5_file[patient]["features"].shape[0]

            if self.train_val_split_ratio > 0:
                self.train_samples = int(
                    total_samples * (1 - self.train_val_split_ratio)
                )
                self.val_samples = total_samples - self.train_samples
            else:
                self.train_samples = total_samples
                self.val_samples = 0
            if self.loso_subject is not None:
                self.loso_samples = hdf5_file[self.loso_subject][
                    "features"
                ].shape[0]
                total_samples += self.loso_samples
            else:
                self.loso_samples = 0

        self.logger.info(
            f"Counting samples successful. Dataset contains {total_samples} samples."
        )
        self.logger.info(
            f"Dataset contains {self.train_samples} training samples."
        )
        self.logger.info(
            f"Dataset contains {self.val_samples} validation samples."
        )
        self.logger.info(f"Dataset contains {self.loso_samples} loso samples.")

    def _get_mean_std(self):
        """Method to determine mean and standard deviation of interictal samples. Those values are used late to normalize all data."""

        current_sample = 0
        features_all = np.empty(
            (
                self.train_samples + self.val_samples,
                self.n_channels,
                self.n_features,
            )
        )  # for standarization only
        labels_all = np.empty((self.train_samples + self.val_samples, 1))
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            for patient in self.patient_list:
                features_all[
                    current_sample : current_sample
                    + hdf5_file[patient]["features"].shape[0]
                ] = hdf5_file[patient]["features"]
                labels_all[
                    current_sample : current_sample
                    + hdf5_file[patient]["features"].shape[0]
                ] = hdf5_file[patient]["labels"]
                current_sample += hdf5_file[patient]["features"].shape[0]
        if self.normalize_with != "global":
            idx = np.where(
                labels_all == self.DEFAULT_CLASS_LABELS[self.normalize_with]
            )[0]
            self.data_mean = np.mean(features_all[idx])
            self.data_std = np.std(features_all[idx])
        else:
            self.data_mean = np.mean(features_all)
            self.data_std = np.std(features_all)
        self.logger.info(
            "Mean and standard deviation calculated for interictal samples."
        )

    def _normalize_data(self, features: np.ndarray):
        """Method to normalize input features data using mean and standard deviation extracted previously.
        Args:
            features: (np.ndarray) Array with features to be normalized.
        Returns:
            normalized_features: (np.ndarray) Array with normalized features.
        """
        normalized_features = features.copy()
        for i in range(features.shape[0]):
            for n in range(features.shape[1]):
                normalized_features[i, n, :] = (
                    features[i, n, :] - self.data_mean
                ) / self.data_std
        return normalized_features

    def update_classes(
        self, features: np.ndarray, labels: np.ndarray, edge_idx: np.ndarray
    ):
        """Method to properly relabel classes for training accordint to used_classes_dict argument."""
        """Convention:
            Base case: 0 - preictal, 1 - ictal, 2 - interictal
            Case 1: 0 - preictal, 1 - ictal
            Case 2: 0 - interictal, 1 - ictal
            Case 3: 0 - preictal, 1 - interictal
        If present, ictal period always as class 1.
        Args:
            labels: (np.ndarray) Array with labels to be relabeled.
        Returns:
            new_features: (np.ndarray) Array with removed examples from unused classes.
            new_labels: (np.ndarray) Array with relabeled labels.
            new_edge_idx: (np.ndarray) Array with removed examples from unused classes.
        """
        if (
            self.used_classes_dict["preictal"]
            and self.used_classes_dict["ictal"]
        ):
            label_to_remove = 2
            new_labels = np.delete(
                labels, np.where(labels == label_to_remove)[0], axis=0
            )
            new_features = np.delete(
                features, np.where(labels == label_to_remove)[0], axis=0
            )
            new_idx = np.delete(
                edge_idx, np.where(labels == label_to_remove)[0], axis=0
            )
        elif (
            self.used_classes_dict["ictal"]
            and self.used_classes_dict["interictal"]
        ):
            label_to_remove = 0
            new_labels = np.delete(
                labels, np.where(labels == label_to_remove)[0], axis=0
            )
            new_features = np.delete(
                features, np.where(labels == label_to_remove)[0], axis=0
            )
            new_idx = np.delete(
                edge_idx, np.where(labels == label_to_remove)[0], axis=0
            )
            new_labels = np.where(new_labels == 2, 0, new_labels)
        elif (
            self.used_classes_dict["preictal"]
            and self.used_classes_dict["interictal"]
        ):
            label_to_remove = 1
            new_labels = np.delete(
                labels, np.where(labels == label_to_remove)[0], axis=0
            )
            new_features = np.delete(
                features, np.where(labels == label_to_remove)[0], axis=0
            )
            new_idx = np.delete(
                edge_idx, np.where(labels == label_to_remove)[0], axis=0
            )
            new_labels = np.where(new_labels == 2, 1, new_labels)

        return new_features, new_labels, new_idx

    def _calculate_timeseries_features(self, features: np.ndarray):
        """Converting features to EEG timeseries features.
        Args:
            features: (np.ndarray) Array with features to be converted.
        Returns:
            new_features: (np.ndarray) Array with Hjorth features.
        """

        new_features = np.array(
            [
                np.concatenate(
                    [
                        np.expand_dims(compute_variance(feature), 1),
                        np.expand_dims(compute_hjorth_mobility(feature), 1),
                        np.expand_dims(compute_hjorth_complexity(feature), 1),
                        np.expand_dims(compute_line_length(feature), 1),
                        np.expand_dims(compute_katz_fd(feature), 1),
                        np.expand_dims(compute_higuchi_fd(feature), 1),
                        compute_energy_freq_bands(
                            self.sampling_f, feature, self.FREQ_BANDS
                        ).reshape(self.n_channels, self.BAND_COUNT),
                    ],
                    axis=1,
                )
                for feature in features
            ]
        )
        return new_features

    def _get_train_val_indices(self, n_samples: np.ndarray, labels: np.ndarray):
        """Perfom train/validation split on provided indices with labels.
        Args:
            n_samples: (np.ndarray) Array with indices.
            labels: (np.ndarray) Array with labels.
        Returns:
            train_indices: (np.ndarray) Array with indices for training.
            val_indices: (np.ndarray) Array with indices for validation.
        """

        train_indices, val_indices = train_test_split(
            n_samples,
            test_size=self.train_val_split_ratio,
            shuffle=True,
            stratify=labels,
            random_state=self.seed,
        )
        train_indices = np.sort(train_indices)
        val_indices = np.sort(val_indices)
        return train_indices, val_indices

    def _transform_features(self, features: np.ndarray):
        """Performs normalization and  optionally transformation of features.
        Args:
            features: (np.ndarray) Array with features to be transformed.
        Returns:
            processed_features: (np.ndarray) Array with transformed features.
        """
        # processed_features = self._normalize_data(features)
        processed_features = (features - self.data_mean) / self.data_std

        if self.extract_features:
            processed_features = self._calculate_timeseries_features(
                processed_features
            )

        elif self.fft:
            processed_features = np.fft.rfft(processed_features)
            processed_features = torch.from_numpy(processed_features)
            return processed_features
        processed_features = torch.from_numpy(processed_features).float()
        return processed_features

    def _transform_edges(self, edge_idx: np.ndarray):
        """Converts the edges from numpy format to torch_geometric format.
        Args:
            edge_idx: (np.ndarray) Array with edges.
        Returns:
            edge_index: (torch.tensor) Tensor with edges.
        """
        edge_index = nx.convert_matrix.from_numpy_array(edge_idx)
        data_object = torch_geometric.utils.from_networkx(edge_index)
        edge_index = data_object.edge_index
        # edge_weight = data_object.weight
        return edge_index

    def _transform_labels(self, labels: np.ndarray):
        """Convert labels to torch tensor.
        Args:
            labels: (np.ndarray) Array with labels.
        Returns:
            labels_transformed: (torch.tensor) Tensor with labels.
        """
        labels_transformed = torch.from_numpy(labels).float()
        return labels_transformed

    def _features_to_data_list(
        self,
        features: np.ndarray,
        edge_idx: np.ndarray,
        labels: np.ndarray,
        patient_id: Union[int, None] = None,
    ) -> List[Data]:
        """Converts features, edges and labels to list of torch_geometric.data.Data objects.
        Before the conversion, features are normalized using mean and standard deviation of interictal samples
        and transformed if specified in the config.
        Args:
            features: (np.ndarray) Array with features.
            edges: (np.ndarray) Array with edges.
            labels: (np.ndarray) Array with labels.
            patient_id: (int) Patient id (used only to split data using kfold-cval). Default: None
        Returns:
            data_list: (list) List of torch_geometric.data.Data objects.
        """
        processed_features = self._transform_features(features)
        processed_labels = self._transform_labels(labels)
        if self.kfold_cval_mode:
            data_list = [
                Data(
                    x=processed_features[i],
                    edge_index=self._transform_edges(edge_idx[i]),
                    y=processed_labels[i],
                    patient_id=patient_id,
                )
                for i in range(len(processed_features))
            ]
            return data_list

        data_list = [
            Data(
                x=processed_features[i],
                edge_index=self._transform_edges(edge_idx[i]),
                y=processed_labels[i],
            )
            for i in range(len(processed_features))
        ]
        return data_list

    def _get_single_patient_data_train_val(
        self, patient: str
    ) -> Tuple[List[Data], List[Data]]:
        """Get patient data for train and validation sets.
        Args:
            patient: (str) Name of the patient to get the data for.
        Returns:
            train_data_list: (list) List of torch_geometric.data.Data objects for training.
            val_data_list: (list) List of torch_geometric.data.Data objects for validation.
        """
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            n_samples = np.arange(hdf5_file[patient]["features"].shape[0])
            labels = np.squeeze(hdf5_file[patient]["labels"])
            train_indices, val_indices = self._get_train_val_indices(
                n_samples, labels
            )

            features_train = hdf5_file[patient]["features"][train_indices]
            labels_train = hdf5_file[patient]["labels"][train_indices]
            edge_idx_train = hdf5_file[patient]["edge_idx"][train_indices]

            features_val = hdf5_file[patient]["features"][val_indices]
            labels_val = hdf5_file[patient]["labels"][val_indices]
            edge_idx_val = hdf5_file[patient]["edge_idx"][val_indices]
            if sum(self.used_classes_dict.values()) < 3:
                (
                    features_train,
                    labels_train,
                    edge_idx_train,
                ) = self.update_classes(
                    features_train, labels_train, edge_idx_train
                )
                features_val, labels_val, edge_idx_val = self.update_classes(
                    features_val, labels_val, edge_idx_val
                )

            data_list_train = self._features_to_data_list(
                features_train, edge_idx_train, labels_train
            )
            data_list_val = self._features_to_data_list(
                features_val, edge_idx_val, labels_val
            )
        self.logger.info(
            f"Processed patient {patient} for train and validation sets."
        )
        return (data_list_train, data_list_val)

    def _get_single_patient_data(self, patient: str) -> List[Data]:
        """Get patient data for training set.
        Args:
            patient: (str) Name of the patient to get the data for.
        Returns:
            train_data_list: (list) List of torch_geometric.data.Data objects for training.
        """
        patient_id = (
            int("".join(filter(str.isdigit, patient)))
            if self.kfold_cval_mode
            else None
        )
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            features = hdf5_file[patient]["features"][:]
            labels = hdf5_file[patient]["labels"][:]
            edge_idx = hdf5_file[patient]["edge_idx"][:]
            if sum(self.used_classes_dict.values()) < 3:
                features, labels, edge_idx = self.update_classes(
                    features, labels, edge_idx
                )
            data_list = self._features_to_data_list(
                features, edge_idx, labels, patient_id
            )
        self.logger.info(f"Processed patient {patient} set.")
        return data_list

    def get_datasets(
        self,
    ) -> tuple[List[Data], ...]:
        """Get data for training, validation and leave-one-subject-out cross-validation.
        Returns:
            data_lists: (list) List of lists of torch_geometric.data.Data objects loaded.
        """
        start = time.time()
        train_data_list: List[Data] = []
        if self.train_val_split_ratio > 0:
            val_data_list: List[Data] = []
        num_processes = mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        try:
            if self.train_val_split_ratio > 0:
                result_list = pool.map(
                    self._get_single_patient_data_train_val, self.patient_list
                )
                pool.close()
                pool.join()
                for train_data, val_data in result_list:
                    train_data_list = train_data_list + train_data
                    val_data_list = val_data_list + val_data
            else:
                result_list = pool.map(
                    self._get_single_patient_data, self.patient_list
                )
                pool.close()
                pool.join()
                for train_data in result_list:
                    train_data_list = train_data_list + train_data
        except KeyboardInterrupt:
            pool.terminate()
            print("Keyboard interrupt detected, terminating worker pool.")
            raise KeyboardInterrupt
        data_lists = (
            [train_data_list]
            if self.train_val_split_ratio == 0
            else [train_data_list, val_data_list]
        )
        if self.loso_subject is not None:
            loso_data_list = self._get_single_patient_data(self.loso_subject)
            data_lists.append(loso_data_list)
        self.logger.info(f"Data loading took {time.time() - start} seconds.")

        return tuple(data_lists)


# @dataclass
# class SeizureDataLoader:
#     npy_dataset_path: Path
#     event_tables_path: Path
#     plv_values_path: Path
#     loso_patient: str = None
#     sampling_f: int = 256
#     seizure_lookback: int = 600
#     sample_timestep: int = 5
#     inter_overlap: int = 0
#     preictal_overlap: int = 0
#     ictal_overlap: int = 0
#     self_loops: bool = True
#     balance: bool = True
#     train_test_split: float = None
#     fft: bool = False
#     hjorth: bool = False
#     downsample: int = None
#     buffer_time: int = 15
#     batch_size: int = 32
#     smote: bool = False
#     tsfresh: bool = False
#     rescale: bool = False
#     used_classes_dict: dict[str] = field(
#         default_factory=lambda: {
#             "interictal": True,
#             "preictal": True,
#             "ictal": True,
#         }
#     )
#     """Class to prepare dataloaders for eeg seizure perdiction from stored files.

#     Attributes:
#         npy_dataset_path: (Path) Path to directory with .npy files
#         event_tables_path: (Path) Path to directory with .csv files
#         plv_values_path: (Path) Path to directory with .npy files
#         loso_patient: (str) Patient name to be left out of training set.
#     If None, no patient is left out for testing. (default: None)
#         sampling_f: (int) Sampling frequency of the recordings. (default: 256)
#         seizure_lookback: (int) Time in seconds to look back from seizure onset. (default: 600)
#         sample_timestep: (int) Time in seconds between samples. (default: 5)
#         inter_overlap: (int) Time in seconds to overlap between interictal samples. (default: 0)
#         preictal_overlap: (int) Time in seconds to overlap between preictal samples. (default: 0)
#         ictal_overlap: (int) Time in seconds to overlap between ictal samples. (default: 0)
#         self_loops: (bool) Whether to add self loops to the graph. (default: True)
#         balance: (bool) Whether to balance the classes. (default: True)
#         train_test_split: (float) Percentage of data to be used for testing. (default: None)
#         fft: (bool) Whether to use fft features. (default: False)
#         hjorth: (bool) Whether to use hjorth features. (default: False)
#         downsample: (int) Factor by which to downsample the data. (default: None)
#         buffer_time: (int) Time in seconds to skip before and after every sample from seizure period.
#     (default: 15)
#         batch_size: (int) Batch size for dataloaders. (default: 32)
#         smote: (bool) Whether to use smote to balance the classes. (default: False)
#         used_classes_ditct: (dict) Dictionary with classes to be used.
#     (default: {'interictal': True, 'preictal': True, 'ictal': True})


#     """
#     # if used_classes_dict is None:
#     #     used_classes_dict = {"interictal": True, "preictal": True, "ictal": True}
#     assert (fft and hjorth) == False, "When fft is True, hjorth should be False"
#     assert (downsample is None) or (
#         downsample > 0
#     ), "Downsample should be None or positive integer"
#     assert (train_test_split is None) or (
#         train_test_split > 0 and train_test_split < 1
#     ), "Train test split should be None or float between 0 and 1"

#     assert not (
#         smote and balance
#     ), "Cannot use smote and balance at the same time"
#     assert not (
#         (fft or hjorth) and tsfresh
#     ), "Cannot use fft or hjorth and tsfresh at the same time"

#     def _get_event_tables(self, patient_name: str) -> tuple[dict, dict]:
#         """Read events for given patient into start and stop times lists from .csv extracted files.
#         Args:
#             patient_name: (str) Name of the patient to get events for.
#         Returns:
#             start_events_dict: (dict) Dictionary with start events for given patient.
#             stop_events_dict: (dict) Dictionary with stop events for given patient.
#         """

#         event_table_list = os.listdir(self.event_tables_path)
#         patient_event_tables = [
#             os.path.join(self.event_tables_path, ev_table)
#             for ev_table in event_table_list
#             if patient_name in ev_table
#         ]
#         patient_event_tables = sorted(patient_event_tables)
#         patient_start_table = patient_event_tables[
#             0
#         ]  ## done terribly, but it has to be so for win/linux compat
#         patient_stop_table = patient_event_tables[1]
#         start_events_dict = pd.read_csv(patient_start_table).to_dict("index")
#         stop_events_dict = pd.read_csv(patient_stop_table).to_dict("index")
#         return start_events_dict, stop_events_dict

#     def _get_recording_events(self, events_dict, recording) -> list[int]:
#         """Read seizure times into list from event_dict.
#         Args:
#             events_dict: (dict) Dictionary with events for given patient.
#             recording: (str) Name of the recording to get events for.
#         Returns:
#             recording_events: (list) List of seizure event start and stop time for given recording.
#         """
#         recording_list = list(events_dict[recording + ".edf"].values())
#         recording_events = [int(x) for x in recording_list if not np.isnan(x)]
#         return recording_events

#     def _get_graph(self, n_nodes: int) -> nx.Graph:
#         """Creates Networx complete graph with self loops
#         for given number of nodes.
#         Args:
#             n_nodes: (int) Number of nodes in the graph.
#         Returns:
#             graph: (nx.Graph) Fully connected graph with self loops.
#         """
#         graph = nx.complete_graph(n_nodes)
#         self_loops = [[node, node] for node in graph.nodes()]
#         graph.add_edges_from(self_loops)
#         return graph

#     def _get_edge_weights_recording(self, plv_values: np.ndarray) -> np.ndarray:
#         """Method for extracting PLV values associated with given edges for a recording.
#         The PLV was computed for the entire recroding for all channels when the recording was
#         processed.
#         Args:
#             plv_values: (np.ndarray) Array with PLV values for given recording.
#         Returns:
#             edge_weights: (np.ndarray) Array with PLV values for given edges.
#         """
#         graph = self._get_graph(plv_values.shape[0])
#         garph_dict = {}
#         for edge in graph.edges():
#             e_start, e_end = edge
#             garph_dict[edge] = {"plv": plv_values[e_start, e_end]}
#         nx.set_edge_attributes(graph, garph_dict)
#         edge_weights = from_networkx(graph).plv.numpy()
#         return edge_weights

#     def _get_edges(self):
#         """Method to assign edge attributes. Has to be called AFTER get_dataset() method."""
#         graph = self._get_graph(self._features.shape[1])
#         edges = np.expand_dims(from_networkx(graph).edge_index.numpy(), axis=0)
#         edges_per_sample_train = np.repeat(
#             edges, repeats=self._features.shape[0], axis=0
#         )
#         self._edges = torch.tensor(edges_per_sample_train)
#         if self.loso_patient is not None:
#             edges_per_sample_val = np.repeat(
#                 edges, repeats=self._loso_features.shape[0], axis=0
#             )
#             self._loso_edges = torch.tensor(edges_per_sample_val)

#     def _array_to_tensor(self):
#         """Method converting features, edges and weights to torch.tensors"""

#         self._features = torch.from_numpy(self._features)
#         self._labels = torch.from_numpy(self._labels)
#         # self._time_labels = torch.from_numpy(self._time_labels)
#         # self._edge_weights = torch.from_numpy(self._edge_weights)
#         if self.loso_patient is not None:
#             self._loso_features = torch.from_numpy(self._loso_features)
#             self._loso_labels = torch.from_numpy(self._loso_labels)
#             # self._loso_time_labels = torch.from_numpy(self._loso_time_labels)
#             # self._loso_edge_weights = torch.from_numpy(self._loso_edge_weights)

#     def _get_labels_count(self):
#         """Convenience method to get counts of labels in the dataset."""
#         labels, counts = np.unique(self._labels, return_counts=True)
#         self._label_counts = {}
#         for n, label in enumerate(labels):
#             self._label_counts[int(label)] = counts[n]
#         if self.loso_patient is not None:
#             labels, counts = np.unique(self._loso_labels, return_counts=True)
#             self._val_label_counts = {}
#             for n, label in enumerate(labels):
#                 self._val_label_counts[int(label)] = counts[n]

#     def _calculate_hjorth_features(self, features):
#         """Converting features to Hjorth features.
#         Args:
#             features: (np.ndarray) Array with features to be converted.
#         Returns:
#             new_features: (np.ndarray) Array with Hjorth features.
#         """
#         new_features = np.array(
#             [
#                 np.concatenate(
#                     [
#                         np.expand_dims(compute_variance(feature), 1),
#                         np.expand_dims(compute_hjorth_mobility(feature), 1),
#                         np.expand_dims(compute_hjorth_complexity(feature), 1),
#                         np.expand_dims(compute_line_length(feature), 1),
#                         np.expand_dims(compute_katz_fd(feature), 1),
#                         np.expand_dims(compute_higuchi_fd(feature), 1),
#                     ],
#                     axis=1,
#                 )
#                 for feature in features
#             ]
#         )
#         # new_mean = new_features.mean(axis=0)
#         # new_std = new_features.std(axis=0)
#         # new_features = (new_features - new_mean) / new_std
#         return new_features

#     def _features_to_data_list(self, features, edges, labels):
#         """Converts features, edges and labels to list of torch_geometric.data.Data objects.
#         Args:
#             features: (np.ndarray) Array with features.
#             edges: (np.ndarray) Array with edges.
#             labels: (np.ndarray) Array with labels.
#         Returns:
#             data_list: (list) List of torch_geometric.data.Data objects.
#         """
#         data_list = [
#             Data(
#                 x=features[i],
#                 edge_index=edges[i],
#                 # edge_attr=edge_weights[i],
#                 y=labels[i],
#                 # time=time_label[i],
#             )
#             for i in range(len(features))
#         ]
#         return data_list

#     def _split_data_list(self, data_list):
#         """Methods for splitting list of torch_geometric.data.Data objects into train and validation sets.
#         Uses StratifiedShuffleSplit to ensure that the classes are balanced in both sets.
#         Args:
#             data_list: (list) List of torch_geometric.data.Data objects.
#         Returns:
#             data_list_train: (list) List of torch_geometric.data.Data objects for training.
#             dataset_list_val: (list) List of torch_geometric.data.Data objects for validation.
#         """
#         class_labels = torch.tensor(
#             [data.y.item() for data in data_list], dtype=torch.float32
#         ).unsqueeze(1)
#         patient_labels = torch.tensor(
#             np.expand_dims(self._patient_number, 1), dtype=torch.float32
#         )
#         class_labels_patient_labels = torch.cat(
#             [class_labels, patient_labels], dim=1
#         )
#         splitter = StratifiedShuffleSplit(
#             n_splits=1, test_size=self.train_test_split, random_state=42
#         )
#         train_indices, val_indices = next(
#             splitter.split(data_list, class_labels_patient_labels)
#         )
#         self._indexes_to_later_delete = {
#             "train": train_indices,
#             "val": val_indices,
#         }
#         data_list_train = [data_list[i] for i in train_indices]
#         dataset_list_val = [data_list[i] for i in val_indices]
#         return data_list_train, dataset_list_val

#     def _initialize_dicts(self):
#         """Temporary method to initialize dictionaries for storing features, labels, etc.
#         Looks terrible, but convenient so far.
#         """
#         self._features_dict = {}
#         self._labels_dict = {}
#         self._time_labels_dict = {}
#         self._edge_weights_dict = {}
#         self._patient_number_dict = {}
#         if self.loso_patient:
#             self._loso_features_dict = {}
#             self._loso_labels_dict = {}
#             self._loso_time_labels_dict = {}
#             self._loso_edge_weights_dict = {}
#             self._loso_patient_number_dict = {}

#     def _convert_dict_to_array(self):
#         """A method to convert dictionaries to numpy arrays. This approach with dicts is redundant,
#         but allows for joblib parallelization for data loading by not using concatenation in the loading loop.
#         """
#         self._features = np.concatenate(
#             [self._features_dict[key] for key in self._features_dict.keys()]
#         )
#         del self._features_dict
#         self._labels = np.concatenate(
#             [self._labels_dict[key] for key in self._labels_dict.keys()]
#         )
#         del self._labels_dict
#         # self._time_labels = np.concatenate(
#         #     [self._time_labels_dict[key] for key in self._time_labels_dict.keys()]
#         # )
#         # del self._time_labels_dict
#         # self._edge_weights = np.concatenate(
#         #     [self._edge_weights_dict[key] for key in self._edge_weights_dict.keys()]
#         # )
#         # del self._edge_weights_dict
#         self._patient_number = np.concatenate(
#             [
#                 self._patient_number_dict[key]
#                 for key in self._patient_number_dict.keys()
#             ]
#         )
#         del self._patient_number_dict
#         if self.loso_patient:
#             self._loso_features = np.concatenate(
#                 [
#                     self._loso_features_dict[key]
#                     for key in self._loso_features_dict.keys()
#                 ]
#             )
#             del self._loso_features_dict
#             self._loso_labels = np.concatenate(
#                 [
#                     self._loso_labels_dict[key]
#                     for key in self._loso_labels_dict.keys()
#                 ]
#             )
#             del self._loso_labels_dict
#             # self._loso_time_labels = np.concatenate(
#             #     [
#             #         self._loso_time_labels_dict[key]
#             #         for key in self._loso_time_labels_dict.keys()
#             #     ]
#             # )
#             # del self._loso_time_labels_dict
#             # self._loso_edge_weights = np.concatenate(
#             #     [
#             #         self._loso_edge_weights_dict[key]
#             #         for key in self._loso_edge_weights_dict.keys()
#             #     ]
#             # )
#             # del self._loso_edge_weights_dict
#             self._loso_patient_number = np.concatenate(
#                 [
#                     self._loso_patient_number_dict[key]
#                     for key in self._loso_patient_number_dict.keys()
#                 ]
#             )
#             del self._loso_patient_number_dict

#     def _balance_classes(self) -> None:
#         """Method to balance classes in the dataset by removing samples from the majority class.
#         Currently works only for interictal and ictal classes."""
#         negative_label = self._label_counts[0]
#         positive_label = self._label_counts[1]

#         print(f"Number of negative samples pre removal {negative_label}")
#         print(f"Number of positive samples pre removal {positive_label}")
#         imbalance = negative_label - positive_label
#         print(f"imbalance {imbalance}")
#         negative_indices = np.where(self._labels == 0)[0]
#         indices_to_discard = np.random.choice(
#             negative_indices, size=imbalance, replace=False
#         )

#         self._features = np.delete(
#             self._features, obj=indices_to_discard, axis=0
#         )
#         self._labels = np.delete(self._labels, obj=indices_to_discard, axis=0)
#         self._time_labels = np.delete(
#             self._time_labels, obj=indices_to_discard, axis=0
#         )
#         self._edge_weights = np.delete(
#             self._edge_weights, obj=indices_to_discard, axis=0
#         )
#         self._patient_number = np.delete(
#             self._patient_number, obj=indices_to_discard, axis=0
#         )

#     def _standardize_data(self, features, labels, loso_features=None) -> None:
#         """Standardize features by subtracting mean and dividing by standard deviation.
#         The mean and std are computed from the interictal class. The same values are used for loso_features.
#         Args:
#             features: (np.ndarray) Array with features.
#             labels: (np.ndarray) Array with labels.
#             loso_features (optional): (np.ndarray) Array with features for LOSO patient.
#         """
#         indexes = np.where(labels == 0)[0]
#         features_negative = features[indexes]
#         channel_mean = features_negative.mean()
#         channel_std = features_negative.std()
#         for i in range(features.shape[0]):
#             for n in range(features.shape[1]):
#                 features[i, n, :] = (
#                     features[i, n, :] - channel_mean
#                 ) / channel_std
#         if (
#             loso_features is not None
#         ):  ## standardize loso features with the same values as for training data
#             for i in range(loso_features.shape[0]):
#                 for n in range(loso_features.shape[1]):
#                     loso_features[i, n, :] = (
#                         loso_features[i, n, :] - channel_mean
#                     ) / channel_std

#     def _min_max_scale(self, features, labels, loso_features=None) -> None:
#         """Min max scale features to range [0,1]. The min and max values are computed from the interictal class.
#         Args:
#             features: (np.ndarray) Array with features.
#             labels: (np.ndarray) Array with labels.
#         """
#         indexes = np.where(labels == 0)[0]
#         features_negative = features[indexes]

#         channel_min = features_negative.min()
#         channel_max = features_negative.max()
#         for i in range(features.shape[0]):
#             for n in range(features.shape[1]):
#                 features[i, n, :] = (features[i, n, :] - channel_min) / (
#                     channel_max - channel_min
#                 )
#         if loso_features is not None:
#             for i in range(loso_features.shape[0]):
#                 for n in range(loso_features.shape[1]):
#                     loso_features[i, n, :] = (
#                         loso_features[i, n, :] - channel_min
#                     ) / (channel_max - channel_min)

#     def _apply_smote(self, features, labels):
#         """Performs SMOTE oversampling on the dataset. Implemented for preictal vs ictal scenarion only.
#         Args:
#             features: (np.ndarray) Array with features.
#             labels: (np.ndarray) Array with labels.
#         Returns:
#             x_train_smote: (np.ndarray) Array with SMOTE oversampled features.
#             y_train_smote: (np.ndarray) Array with SMOTE oversampled labels.
#         """
#         dim_1, dim_2, dim_3 = features.shape

#         new_dim = dim_1 * dim_2
#         new_x_train = features.reshape(new_dim, dim_3)
#         new_y_train = []
#         for i in range(len(labels)):
#             new_y_train.extend([labels[i]] * dim_2)

#         new_y_train = np.array(new_y_train)

#         # transform the dataset
#         oversample = SMOTE(random_state=42)
#         x_train, y_train = oversample.fit_resample(new_x_train, new_y_train)
#         x_train_smote = x_train.reshape(
#             int(x_train.shape[0] / dim_2), dim_2, dim_3
#         )
#         y_train_smote = []
#         for i in range(int(x_train.shape[0] / dim_2)):
#             # print(i)
#             value_list = list(
#                 y_train.reshape(int(x_train.shape[0] / dim_2), dim_2)[i]
#             )
#             # print(list(set(value_list)))
#             y_train_smote.extend(list(set(value_list)))
#             ## Check: if there is any different value in a list
#             if len(set(value_list)) != 1:
#                 print(
#                     "\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n"
#                 )
#         y_train_smote = np.array(y_train_smote)
#         # print(np.unique(y_train_smote,return_counts=True))
#         return x_train_smote, y_train_smote

#     def _compute_tsfresh_features(self, features):
#         """Compute tsfresh features for given features. The features are computed for each channel in every sample.
#         Currently the extracted features are hardcoded into the method.
#         Args:
#             features: (np.ndarray) Array with features.
#         Returns:
#             new_features: (np.ndarray) Array with tsfresh features.
#         """
#         fc_parameters = {
#             "abs_energy": None,
#             "absolute_sum_of_changes": None,
#             "agg_autocorrelation": [{"f_agg": "mean", "maxlag": 10}],
#             "autocorrelation": [{"lag": 10}],
#             "number_peaks": [{"n": 5}],
#             "c3": [{"lag": 10}],
#             "cid_ce": [{"normalize": True}],
#             "longest_strike_below_mean": None,
#             "longest_strike_above_mean": None,
#             "fourier_entropy": [{"bins": 10}],
#             "mean_change": None,
#             "number_crossing_m": [{"m": 0}],
#             "sample_entropy": None,
#             "variance": None,
#             "variation_coefficient": None,
#         }
#         n_variables = len(fc_parameters.keys())
#         n_nodes = features[0].shape[0]
#         for n, case in enumerate(features):
#             new_dataframe = pd.DataFrame(
#                 columns=["id", "time", "channel", "value"]
#             )
#             new_dataframe["id"] = np.repeat(n, case.shape[0] * case.shape[1])
#             new_dataframe["time"] = np.tile(
#                 np.arange(case.shape[1]), case.shape[0]
#             )
#             new_dataframe["channel"] = np.stack(
#                 [torch.full([case.shape[1]], i) for i in range(case.shape[0])]
#             ).flatten()
#             new_dataframe["value"] = abs(case.flatten())
#             extracted_features = extract_features(
#                 new_dataframe,
#                 column_id="id",
#                 column_sort="time",
#                 column_kind="channel",
#                 column_value="value",
#                 default_fc_parameters=fc_parameters,
#                 n_jobs=6,
#             )

#             try:
#                 final_df = pd.concat([final_df, extracted_features], axis=0)
#             except:
#                 final_df = extracted_features
#         new_features = final_df.to_numpy().reshape(-1, n_nodes, n_variables)
#         return new_features

#     def _extend_data(
#         self,
#         patient,
#         patient_number,
#         features,
#         labels,
#         time_labels=None,
#         plv_edge_weights=None,
#     ):
#         """Convenience method to extend the dictionaries with features, labels, time labels and edge weights.
#         Args:
#             patient: (str) Name of the patient to extend the dictionaries for.
#             patient_number: (int) Patient number to extend the dictionaries for.
#             features: (np.ndarray) Array with features.
#             labels: (np.ndarray) Array with labels.
#             time_labels (optional): (np.ndarray) Array with time labels.
#             plv_edge_weights (optional): (np.ndarray) Array with edge weights.
#         """
#         if patient == self.loso_patient:
#             # logging.info(f"Adding recording {record} of patient {patient}")
#             try:
#                 self._loso_features_dict[patient] = np.concatenate(
#                     (self._loso_features_dict[patient], features), axis=0
#                 )
#                 self._loso_labels_dict[patient] = np.concatenate(
#                     (self._loso_labels_dict[patient], labels), axis=0
#                 )
#                 # self._loso_time_labels_dict[patient] = np.concatenate(
#                 #     (self._loso_time_labels_dict[patient], time_labels), axis=0
#                 # )
#                 # self._loso_edge_weights_dict[patient] = np.concatenate(
#                 #     (
#                 #         self._loso_edge_weights_dict[patient],
#                 #         np.repeat(plv_edge_weights, features.shape[0], axis=0),
#                 #     )
#                 # )

#                 self._loso_patient_number_dict[patient] = np.concatenate(
#                     (self._loso_patient_number_dict[patient], patient_number)
#                 )
#             except:
#                 self._loso_features_dict[patient] = features
#                 self._loso_labels_dict[patient] = labels
#                 # self._loso_time_labels_dict[patient] = time_labels
#                 # self._loso_edge_weights_dict[patient] = np.repeat(
#                 #     plv_edge_weights, features.shape[0], axis=0
#                 # )
#                 self._loso_patient_number_dict[patient] = patient_number

#         else:
#             try:
#                 self._features_dict[patient] = np.concatenate(
#                     (self._features_dict[patient], features), axis=0
#                 )
#                 self._labels_dict[patient] = np.concatenate(
#                     (self._labels_dict[patient], labels), axis=0
#                 )
#                 # self._time_labels_dict[patient] = np.concatenate(
#                 #     (self._time_labels_dict[patient], time_labels), axis=0
#                 # )
#                 # self._edge_weights_dict[patient] = np.concatenate(
#                 #     (
#                 #         self._edge_weights_dict[patient],
#                 #         np.repeat(plv_edge_weights, features.shape[0], axis=0),
#                 #     )
#                 # )

#                 self._patient_number_dict[patient] = np.concatenate(
#                     (self._patient_number_dict[patient], patient_number)
#                 )
#             except:
#                 self._features_dict[patient] = features
#                 self._labels_dict[patient] = labels
#                 # self._time_labels_dict[patient] = time_labels
#                 # self._edge_weights_dict[patient] = np.repeat(
#                 #     plv_edge_weights, features.shape[0], axis=0
#                 # )
#                 self._patient_number_dict[patient] = patient_number

#     def _get_labels_features_edge_weights_seizure(self, patient):
#         """Method to extract features, labels and edge weights for seizure and interictal samples."""

#         event_tables = self._get_event_tables(
#             patient
#         )  # extract start and stop of seizure for patient
#         patient_path = os.path.join(self.npy_dataset_path, patient)
#         recording_list = [
#             recording
#             for recording in os.listdir(patient_path)
#             if "seizures" in recording
#         ]
#         for record in recording_list:  # iterate over recordings for a patient
#             recording_path = os.path.join(patient_path, record)
#             record = record.replace(
#                 "seizures_", ""
#             )  ## some magic to get it properly working with event tables
#             record_id = record.split(".npy")[0]  #  get record id
#             start_event_tables = self._get_recording_events(
#                 event_tables[0], record_id
#             )  # get start events
#             stop_event_tables = self._get_recording_events(
#                 event_tables[1], record_id
#             )  # get stop events
#             data_array = np.load(recording_path)  # load the recording

#             # plv_edge_weights = np.expand_dims(
#             #     self._get_edge_weights_recording(
#             #         np.load(os.path.join(self.plv_values_path, patient, record))
#             #     ),
#             #     axis=0,
#             # )

#             (
#                 features,
#                 labels,
#                 time_labels,
#             ) = utils.extract_training_data_and_labels(
#                 data_array,
#                 start_event_tables,
#                 stop_event_tables,
#                 fs=self.sampling_f,
#                 seizure_lookback=self.seizure_lookback,
#                 sample_timestep=self.sample_timestep,
#                 preictal_overlap=self.preictal_overlap,
#                 ictal_overlap=self.ictal_overlap,
#                 buffer_time=self.buffer_time,
#             )

#             if features is None:
#                 print(
#                     f"Skipping the recording {record} patients {patient} cuz features are none"
#                 )
#                 continue

#             features = features.squeeze(2)

#             if self.downsample:
#                 new_sample_count = int(self.downsample * self.sample_timestep)
#                 features = scipy.signal.resample(
#                     features, new_sample_count, axis=2
#                 )
#             if self.fft:
#                 features = np.fft.rfft(features, axis=2)
#             if self.smote:
#                 features, labels = self._apply_smote(features, labels)
#             time_labels = np.expand_dims(time_labels.astype(np.int32), 1)
#             labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)
#             patient_number = torch.full(
#                 [labels.shape[0]],
#                 int("".join(x for x in patient if x.isdigit())),
#                 dtype=torch.float32,
#             )

#             self._extend_data(patient, patient_number, features, labels)

#     def _get_labels_features_edge_weights_interictal(
#         self, patient, samples_patient: int = None
#     ):
#         """Method to extract features, labels and edge weights for interictal samples.
#         Args:
#             patient: (str) Name of the patient to extract the data for.
#             samples_patient (optional): (int) Number of samples to extract for a patient.
#         Samples are extracted from non-seizure recordings for a patient, starting from random time point.
#         If not specified, the number of samples is calculated as the number of interictal samples for a patient
#         divided by the number of recordings for a patient.

#         """
#         patient_path = os.path.join(self.npy_dataset_path, patient)
#         ## get all non-seizure recordings for a patient
#         recording_list = [
#             recording
#             for recording in os.listdir(patient_path)
#             if not "seizures_" in recording
#         ]
#         if samples_patient is None:
#             patient_num = int("".join(filter(str.isdigit, patient)))
#             if patient == self.loso_patient:
#                 patient_negatives = np.unique(
#                     self._loso_labels_dict[patient], return_counts=True
#                 )[1][0]
#                 samples_per_recording = int(
#                     patient_negatives / len(recording_list)
#                 )
#             else:
#                 patient_negatives = np.unique(
#                     self._labels_dict[patient], return_counts=True
#                 )[1][0]
#                 samples_per_recording = int(
#                     patient_negatives / len(recording_list)
#                 )
#         else:
#             samples_per_recording = int(samples_patient / len(recording_list))
#         for recording in recording_list:
#             recording_path = os.path.join(patient_path, recording)
#             data_array = np.expand_dims(np.load(recording_path), 1)
#             try:
#                 (
#                     features,
#                     labels,
#                 ) = utils.extract_training_data_and_labels_interictal(
#                     input_array=data_array,
#                     samples_per_recording=samples_per_recording,
#                     fs=self.sampling_f,
#                     timestep=self.sample_timestep,
#                     overlap=self.inter_overlap,
#                 )
#             except:
#                 print(
#                     f"Skipping recording {recording} for patient due to the error"
#                 )
#                 continue
#             idx_to_delete = np.where(
#                 np.array(
#                     [np.diff(feature, axis=-1).mean() for feature in features]
#                 )
#                 == 0
#             )[0]
#             if len(idx_to_delete) > 0:
#                 features = np.delete(features, obj=idx_to_delete, axis=0)
#                 labels = np.delete(labels, obj=idx_to_delete, axis=0)
#                 print(
#                     f"Deleted {len(idx_to_delete)} samples from patient {patient} \n recording {recording} due to zero variance"
#                 )
#             patient_number = torch.full(
#                 [labels.shape[0]],
#                 patient_num,
#                 dtype=torch.float32,
#             )
#             features = features.squeeze(2)
#             if self.downsample:
#                 new_sample_count = int(self.downsample * self.sample_timestep)
#                 features = scipy.signal.resample(
#                     features, new_sample_count, axis=2
#                 )
#             if self.fft:
#                 features = np.fft.rfft(features, axis=2)
#             labels = labels.reshape((labels.shape[0], 1)).astype(np.float32)
#             self._extend_data(patient, patient_number, features, labels)

#     def _update_classes(self):
#         """Method to remove samples of period that we do not want to load, as specified in used_classes_dict.
#         If it is possible, the method aims set the interictal period as class 0 to be used for extracting normalization parameters.
#         If it is not possible, preictal period remains chosen as class 0.
#         """
#         if (
#             not self.used_classes_dict["ictal"]
#             or not self.used_classes_dict["preictal"]
#         ):
#             label_to_delete = 0 if self.used_classes_dict["ictal"] else 1
#             idx_to_delete = np.where(self._labels == label_to_delete)[0]
#             self._features = np.delete(
#                 self._features, obj=idx_to_delete, axis=0
#             )
#             self._labels = np.delete(self._labels, obj=idx_to_delete, axis=0)
#             self._patient_number = np.delete(
#                 self._patient_number, obj=idx_to_delete, axis=0
#             )
#             ## change labels of remaining classes
#             if label_to_delete == 0:
#                 self._labels[self._labels == 2] = 0
#                 print(
#                     "Deleted preictal samples, changed interictal label to 0,  ictal remains 1 "
#                 )
#             else:
#                 self._labels[self._labels == 0] = 1
#                 self._labels[self._labels == 2] = 0
#                 print(
#                     "Deleted ictal samples, changed interictal label to 0, preictal to 1"
#                 )
#             if self.loso_patient is not None:
#                 idx_to_delete = np.where(self._loso_labels == label_to_delete)[
#                     0
#                 ]
#                 self._loso_features = np.delete(
#                     self._loso_features, obj=idx_to_delete, axis=0
#                 )
#                 self._loso_labels = np.delete(
#                     self._loso_labels, obj=idx_to_delete, axis=0
#                 )
#                 self._loso_patient_number = np.delete(
#                     self._loso_patient_number, obj=idx_to_delete, axis=0
#                 )
#                 if label_to_delete == 0:
#                     self._loso_labels[self._loso_labels == 2] = 0
#                     print(
#                         "Deleted preictal samples from LOSO patient, changed interictal label to 0, ictal remains 1 "
#                     )
#                 else:
#                     self._loso_labels[self._loso_labels == 0] = 1
#                     self._loso_labels[self._loso_labels == 2] = 0
#                     print(
#                         "Deleted ictal from LOSO patient, changed interictal label to 0, preictal to 1"
#                     )
#         elif (
#             sum(self.used_classes_dict.values()) == 3
#         ):  ## case when all three classes are used - just flipping labels
#             self._labels[
#                 self._labels == 2
#             ] = 4  ## change interictal to 4 from 2 temporarily
#             self._labels[self._labels == 0] = 2  ## change preictal to 2 from 0
#             self._labels[
#                 self._labels == 4
#             ] = 0  ## change interictal to 0 from 4
#             if self.loso_patient is not None:
#                 self._loso_labels[self._loso_labels == 2] = 4
#                 self._loso_labels[self._loso_labels == 0] = 2
#                 self._loso_labels[self._loso_labels == 4] = 0

#     # TODO define a method to create edges and calculate plv to get weights
#     def get_dataset(self):
#         """Creating graph data iterators. The iterator yelds dynamic, weighted and undirected graphs
#         containing self loops. Every node represents one electrode in EEG. The graph is fully connected,
#         edge weights are calculated for every EEG recording as PLV between channels (edge weight describes
#         the "strength" of connectivity between two channels in a recording). Node features are values of
#         channel voltages in time. Features are of shape [nodes,features,timesteps].

#         Returns:
#             train_dataset {DynamicGraphTemporalSignal} -- Training data iterator.
#             valid_dataset {DynamicGraphTemporalSignal} -- Validation data iterator (only if loso_patient is
#             specified in class constructor).
#         """
#         ### TODO rozkminić o co chodzi z tym całym time labels - na razie wartość liczbowa która tam wchodzi
#         ### to shape atrybutu time_labels
#         assert (
#             "interictal" in self.used_classes_dict.keys()
#         ), "Please define the behavior for interictal class in used_classes_dict"
#         assert (
#             "preictal" in self.used_classes_dict.keys()
#         ), "Please define the behavior for preictal class in used_classes_dict"
#         assert (
#             "ictal" in self.used_classes_dict.keys()
#         ), "Please define the behavior for ictal class in used_classes_dict"

#         assert (
#             sum(self.used_classes_dict.values()) > 1
#         ), "Please define at least two classes to use in used_classes_dict"

#         self._initialize_dicts()
#         patient_list = os.listdir(self.npy_dataset_path)
#         start_time = time.time()
#         if self.smote:
#             for patient in patient_list:
#                 self._get_labels_features_edge_weights_seizure(patient)
#         else:
#             Parallel(n_jobs=6, require="sharedmem")(
#                 delayed(self._get_labels_features_edge_weights_seizure)(patient)
#                 for patient in patient_list
#             )
#         print(
#             f"Finished reading in {time.time() - start_time} seconds for seizure data"
#         )
#         if self.used_classes_dict["interictal"]:
#             Parallel(n_jobs=6, require="sharedmem")(
#                 delayed(self._get_labels_features_edge_weights_interictal)(
#                     patient
#                 )
#                 for patient in patient_list
#             )

#         self._convert_dict_to_array()
#         self._update_classes()

#         self._get_labels_count()

#         if self.balance:
#             self._balance_classes()

#         print(
#             f"Finished reading in {time.time() - start_time} seconds for non seizure data"
#         )
#         start_time_preprocessing = time.time()
#         if self.rescale:
#             self._features *= (
#                 self._features * 1e6
#             )  ## rescale to back to volts, numeric stability problems
#             if self.loso_patient is not None:
#                 self._loso_features *= self._loso_features * 1e6
#         self._standardize_data(
#             self._features, self._labels, self._loso_features
#         )
#         if self.tsfresh:
#             self._features = self._compute_tsfresh_features(self._features)
#             if self.loso_patient is not None:
#                 self._loso_features = self._compute_tsfresh_features(
#                     self._loso_features
#                 )

#         if self.hjorth:
#             self._features = self._calculate_hjorth_features(self._features)
#             if self.loso_patient is not None:
#                 self._loso_features = self._calculate_hjorth_features(
#                     self._loso_features
#                 )
#         self._get_edges()
#         self._array_to_tensor()

#         if self.train_test_split is not None:
#             if self.fft or self.hjorth:
#                 data_list = self._features_to_data_list(
#                     self._features,
#                     self._edges,
#                     # self._edge_weights,
#                     self._labels,
#                     # self._time_labels,
#                 )
#                 train_data_list, val_data_list = self._split_data_list(
#                     data_list
#                 )
#                 label_count = np.unique(
#                     [data.y.item() for data in train_data_list],
#                     return_counts=True,
#                 )[1]
#                 self.alpha = label_count[0] / label_count[1]
#                 loaders = [
#                     DataLoader(
#                         train_data_list,
#                         batch_size=self.batch_size,
#                         shuffle=True,
#                         drop_last=False,
#                     ),
#                     DataLoader(
#                         val_data_list,
#                         batch_size=len(val_data_list),
#                         shuffle=False,
#                         drop_last=False,
#                     ),
#                 ]

#             else:
#                 train_dataset = torch.utils.data.TensorDataset(
#                     self._features,
#                     self._edges,
#                     # self._edge_weights,
#                     self._labels,
#                     # self._time_labels,
#                 )

#                 train_dataset, val_dataset = torch.utils.data.random_split(
#                     train_dataset,
#                     [1 - self.train_test_split, self.train_test_split],
#                     generator=torch.Generator().manual_seed(42),
#                 )

#                 train_dataloader = torch.utils.data.DataLoader(
#                     train_dataset,
#                     batch_size=self.batch_size,
#                     shuffle=True,
#                     drop_last=False,
#                 )

#                 val_dataloader = torch.utils.data.DataLoader(
#                     val_dataset,
#                     batch_size=self.batch_size,
#                     shuffle=False,
#                     drop_last=False,
#                 )
#                 loaders = [train_dataloader, val_dataloader]
#         else:
#             if self.fft or self.hjorth:
#                 train_data_list = self._features_to_data_list(
#                     self._features,
#                     self._edges,
#                     # self._edge_weights,
#                     self._labels,
#                     # self._time_labels,
#                 )
#                 loaders = [
#                     DataLoader(
#                         train_data_list,
#                         batch_size=self.batch_size,
#                         shuffle=True,
#                         drop_last=False,
#                     )
#                 ]
#             else:
#                 train_dataset = torch.utils.data.TensorDataset(
#                     self._features,
#                     self._edges,
#                     # self._edge_weights,
#                     self._labels,
#                     # self._time_labels,
#                 )
#                 train_dataloader = torch.utils.data.DataLoader(
#                     train_dataset,
#                     batch_size=self.batch_size,
#                     shuffle=True,
#                     drop_last=False,
#                 )
#                 loaders = [train_dataloader]
#         if self.loso_patient:
#             if self.fft or self.hjorth:
#                 loso_data_list = self._features_to_data_list(
#                     self._loso_features,
#                     self._loso_edges,
#                     # self._loso_edge_weights,
#                     self._loso_labels,
#                     # self._loso_time_labels,
#                 )
#                 print(
#                     "Preprocessing time: ",
#                     time.time() - start_time_preprocessing,
#                 )
#                 return (
#                     *loaders,
#                     DataLoader(
#                         loso_data_list,
#                         batch_size=len(loso_data_list),
#                         shuffle=False,
#                         drop_last=False,
#                     ),
#                 )
#             loso_dataset = torch.utils.data.TensorDataset(
#                 self._loso_features,
#                 self._loso_edges,
#                 # self._loso_edge_weights,
#                 self._loso_labels,
#                 #  self._loso_time_labels,
#             )
#             loso_dataloader = torch.utils.data.DataLoader(
#                 loso_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=False,
#                 drop_last=False,
#             )

#             return (*loaders, loso_dataloader)

#         return (*loaders,)