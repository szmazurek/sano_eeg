import multiprocessing as mp
from torch_geometric.data.separate import separate
from torch_geometric.data import Data
import fnmatch
import logging
import shutil
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Tuple, Union
import psutil
import h5py
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch_geometric
import yaml
from imblearn.over_sampling import SMOTE
from mne_features.univariate import (
    compute_energy_freq_bands,
    compute_higuchi_fd,
    compute_hjorth_complexity,
    compute_hjorth_mobility,
    compute_katz_fd,
    compute_line_length,
    compute_variance,
)
from scipy.signal import resample
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
import gc
import utils
from collections import defaultdict
from torch_geometric.data import Dataset, InMemoryDataset

CPUS_PER_TASK = 4  # int(os.environ["SLURM_CPUS_PER_TASK"])
CTX = mp.get_context("fork")


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

    def __post_init__(self) -> None:
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
        return None

    def _initialize_logger(self) -> None:
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
        return None

    def _find_matching_configs(self, current_config: dict) -> bool:
        """Utility to find configs with the same parameters in the cache folder.
        Args:
            current_config: (dict) Dictionary with the current config.
        Returns:
            bool: True if the config is found, False otherwise.
        """

        def find_yaml_files(directory: str) -> List[str]:
            """Utility to get the yaml files from directory
            Args:
                directory: (str) Path to the directory to search in.
            Returns:
                yaml_files (list[str]): List of yaml files in the directory.
            """
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

    def _create_config_dict(self) -> dict:
        """Method to create config dict with the dataset parameters.
        Returns:
            initial_config_dict: (dict) Dictionary with the dataset parameters.
        """
        dataclass_keys = list(self.__dataclass_fields__.keys())
        dict_values = [self.__getattribute__(key) for key in dataclass_keys]
        initial_config_dict = dict(zip(dataclass_keys, dict_values))
        return initial_config_dict

    def _create_config_file(
        self, config_dict: dict, dataset_folder_path: str
    ) -> None:
        """Method to save config dict to yaml file.
        Args:
            config_dict: (dict) Dictionary with the dataset parameters.
            dataset_folder_path: (str) Path to the folder to save the config file to.
        """
        with open(os.path.join(dataset_folder_path, "config.yaml"), "w") as f:
            yaml.dump(config_dict, f)
        return None

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
            # here lies the error
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
            except UnboundLocalError:
                self.logger.warning(
                    f"Cannot append {patient}, record {record}."
                )
                return 0

        return features_patient.shape[0]

    def _multiprocess_seizure_period_data_loading(self):
        num_processes = CPUS_PER_TASK  # mp.cpu_count()
        pool = mp.Pool(processes=num_processes)
        self.logger.info(num_processes)

        result = pool.map(
            self._get_labels_features_edge_weights_seizure, self.patient_list
        )
        pool.close()
        pool.join()
        if pool:
            pool.terminate()
            pool = None  # workaround for mp bug
        self.sample_count = 0
        for patient, preictal_samples, sample_count in result:
            self.preictal_samples_dict[patient] = preictal_samples
            self.sample_count += sample_count

    def _multiprocess_interictal_data_loading(self):
        num_processes = CPUS_PER_TASK  # mp.cpu_count()
        self.logger.info(num_processes)
        pool = mp.Pool(processes=num_processes)

        result = pool.map(
            self._get_labels_features_edge_weights_interictal, self.patient_list
        )
        pool.close()
        pool.join()
        if pool:
            pool.terminate()
            pool = None  # workaround for mp bug
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
                self.logger.info(self.preictal_samples_dict)
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
class HDFDatasetLoader(InMemoryDataset):
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

    def _create_paths(self) -> List[str]:
        main_root_dir = os.path.join(self.root, "processed_cache")
        try:
            if not len(os.listdir(main_root_dir)) == 0:
                shutil.rmtree(main_root_dir)
        except FileNotFoundError:
            self.logger.info("No processed cache found.")
        if not os.path.exists(main_root_dir):
            os.makedirs(main_root_dir)
            self.logger.info("Created processed cache folder.")
        if self.train_val_split_ratio > 0:
            path_list = [
                f"{main_root_dir}/train/",
                f"{main_root_dir}/val/",
            ]
        else:
            path_list = [f"{main_root_dir}/train/"]
        if self.loso_subject is not None:
            path_list.append(f"{main_root_dir}/test/")

        for path in path_list:
            os.makedirs(path, exist_ok=True)
        self.logger.info("Created processed cache subfolders.")
        return path_list

    # def _create_paths(self) -> None:

    def __post_init__(self) -> None:
        #  super().__init__()
        self._check_arguments()
        self._logger_init()
        self._determine_dataset_characteristics()
        self._show_used_classes()

        self.processed_file_paths = self._create_paths()

    def _check_arguments(self) -> None:
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
        return None

    def _show_used_classes(self) -> None:
        """Method to show which classes are used for training."""
        self.logger.info(f"Used classes: {self.used_classes_dict}")
        used_periods = [
            key for key, value in self.used_classes_dict.items() if value
        ]
        for period in used_periods:
            print(f"USING CLASS: {period}")
        return None

    def _determine_dataset_characteristics(self) -> None:
        """Method to determine dataset characteristics."""
        self.hdf_data_path = f"{self.root}/dataset.hdf5"
        print(self.root)
        with h5py.File(self.hdf_data_path, "r") as hdf5_file:
            self.patient_list = list(hdf5_file.keys())
            self.n_channels, self.n_features = hdf5_file[self.patient_list[0]][
                "features"
            ].shape[1:]
        if self.loso_subject is not None:
            self.patient_list.remove(self.loso_subject)
        self._determine_sample_count()
        self._get_mean_std()
        return None

    def _logger_init(self) -> None:
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
        return None

    def _determine_sample_count(self) -> None:
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
        return None

    def _load_data_for_std_mean(
        self, patient: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            try:
                with h5py.File(self.hdf_data_path, "r") as hdf5_file:
                    return (
                        hdf5_file[patient]["features"][:],
                        hdf5_file[patient]["labels"][:],
                    )
            except BlockingIOError:
                self.logger.warning(
                    f"Waiting for dataset {patient} to be loaded."
                )
                continue

    def _get_mean_std(self) -> None:
        """Method to determine mean and standard deviation of interictal samples. Those values are used late to normalize all data."""

        start_time = time.time()
        pool = CTX.Pool(processes=CPUS_PER_TASK)
        results = pool.map(self._load_data_for_std_mean, self.patient_list)
        pool.close()
        pool.join()
        if pool:
            pool.terminate()
            pool = None
        features_all, labels_all = zip(*results)
        features_all = np.concatenate(features_all)
        labels_all = np.concatenate(labels_all)
        if self.normalize_with != "global":
            idx = np.where(
                labels_all == self.DEFAULT_CLASS_LABELS[self.normalize_with]
            )[0]
            self.median = np.median(features_all[idx])
            self.scale = iqr(features_all[idx])
            # self.data_mean = np.mean(features_all[idx])
            # self.data_std = np.std(features_all[idx])
            # self.logger.info(f"Mean {self.data_mean}")
            # self.logger.info(f"Std {self.data_std}")
        else:
            self.data_mean = np.mean(features_all)
            self.data_std = np.std(features_all)

        del features_all, labels_all
        gc.collect()
        self.logger.info(
            f"Mean and standard deviation calculated for interictal samples in {time.time()-start_time}."
        )
        return None

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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method to properly relabel classes for training accordint to used_classes_dict argument."""
        """Convention:
            Base case: 0 - preictal, 1 - ictal, 2 - interictal
            Case 1: 0 - preictal, 1 - ictal
            Case 2: 0 - interictal, 1 - ictal
            Case 3: 0 - preictal, 1 - interictal
        If present, ictal period always as class 1.
        Args:
            features: (np.ndarray) Array with features to be relabeled.
            labels: (np.ndarray) Array with labels to be relabeled.
            edge_idx: (np.ndarray) Array with edge indices to be relabeled.
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

    def _calculate_timeseries_features(
        self, features: np.ndarray
    ) -> np.ndarray:
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

    def _get_train_val_indices(
        self, n_samples: np.ndarray, labels: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    def _transform_features(self, features: np.ndarray) -> torch.tensor:
        """Performs normalization and  optionally transformation of features.
        Args:
            features: (np.ndarray) Array with features to be transformed.
        Returns:
            processed_features: (torch.tensor Array with transformed features.
        """

        # processed_features = self._normalize_data(features)
        processed_features = (features - self.median) / self.scale
        if self.extract_features:
            processed_features = self._calculate_timeseries_features(
                processed_features
            )
        elif self.fft:
            processed_features = np.fft.rfft(
                processed_features, axis=2, norm="forward"
            )
            processed_features = torch.from_numpy(processed_features)
            processed_features = torch.square(
                torch.abs(processed_features)
            ).float()
            return processed_features
        processed_features = torch.from_numpy(processed_features).float()
        return processed_features

    @staticmethod
    def _transform_edges(edge_idx: np.ndarray) -> torch.tensor:
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

    @staticmethod
    def _transform_labels(labels: np.ndarray) -> torch.tensor:
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
        del processed_features, processed_labels
        gc.collect()
        return data_list

    def _get_mem_info(self, pid: int) -> dict[str, int]:
        res = defaultdict(int)
        for mmap in psutil.Process(pid).memory_maps():
            res["rss"] += mmap.rss
            res["pss"] += mmap.pss
            res["uss"] += mmap.private_clean + mmap.private_dirty
            res["shared"] += mmap.shared_clean + mmap.shared_dirty
            if mmap.path.startswith("/"):  # looks like a file path
                res["shared_file"] += mmap.shared_clean + mmap.shared_dirty
        return res

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
        pid = os.getpid()
        # self.logger.info(f"Initial logging of memory: {self.get_mem_info(pid)} for {patient}")
        # nofile = len(os.listdir("/proc/" + str(os.getpid()) + "/fd/"))
        # self.logger.info(f"Nofile start {nofile}")
        self.logger.info("Alive")
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
        # nofile = len(os.listdir("/proc/" + str(os.getpid()) + "/fd/"))
        # self.logger.info(f"Nofile after hdf {nofile}")
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

        data_list_train, slices_train = self.collate(data_list_train)
        data_list_val, slices_val = self.collate(data_list_val)
        savepath_train = os.path.join(
            self.processed_file_paths[0], f"{patient}.pt"
        )
        savepath_val = os.path.join(
            self.processed_file_paths[1], f"{patient}.pt"
        )
        torch.save((data_list_train, slices_train), savepath_train)
        torch.save((data_list_val, slices_val), savepath_val)
        del (
            features_train,
            labels_train,
            edge_idx_train,
            labels,
            train_indices,
            val_indices,
        )
        del features_val, labels_val, edge_idx_val
        del data_list_train, data_list_val, slices_train, slices_val
        gc.collect()
        meminfo_table = self._get_mem_info(pid)
        self.logger.info(
            f"Processed patient {patient} for train and validation sets."
        )
        # self.logger.info(
        #     f"Reading patient {patient}, Virtual memory {psutil.virtual_memory()}"
        # )
        self.logger.info(
            f"End logging of memory: {meminfo_table} for {patient}"
        )
        # self.logger.info(f"Resource limit {resource.getrlimit(resource.RLIMIT_NOFILE)}")
        # nofile = len(os.listdir("/proc/" + str(os.getpid()) + "/fd/"))
        # self.logger.info(f"Nofile exit {nofile}")

        return None

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
        data, slices = self.collate(data_list)
        if patient == self.loso_subject:
            savepath = os.path.join(
                self.processed_file_paths[-1], f"{patient}.pt"
            )
        else:
            savepath = os.path.join(
                self.processed_file_paths[0], f"{patient}.pt"
            )
        torch.save((data, slices), savepath)

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
        num_processes = CPUS_PER_TASK
        pool = CTX.Pool(processes=num_processes)
        try:
            if self.train_val_split_ratio > 0:
                pool.map_async(
                    self._get_single_patient_data_train_val,
                    self.patient_list,
                )
                pool.close()
                pool.join()
                if pool:
                    pool.terminate()
                    pool = None  # workaround for mp bugs
                self.logger.info("Train and validation data loaded.")

            else:
                pool.map_async(self._get_single_patient_data, self.patient_list)
                pool.close()
                pool.join()
                if pool:
                    pool.terminate()
                    pool = None  # workaround for mp bugs
        except KeyboardInterrupt:
            pool.terminate()
            print("Keyboard interrupt detected, terminating worker pool.")
            raise KeyboardInterrupt

        if self.loso_subject is not None:
            self._get_single_patient_data(self.loso_subject)

        self.logger.info(f"Data loading took {time.time() - start} seconds.")

        return self.processed_file_paths


class GraphDataset:
    def __init__(self, root):
        super().__init__()
        self.object_paths = [
            os.path.join(root, path) for path in os.listdir(root)
        ]
        self.root = root
        self.load()

    def __len__(self) -> int:
        return len(self._data_list) if self._data_list is not None else 0

    def load(self) -> None:

        data, slices = zip(*[torch.load(path) for path in self.object_paths])
        data_list = []
        for n in range(len(data)):
            data_temp = data[n]
            slices_temp = slices[n]
            data_list += [
                separate(Data, data_temp, index, slices_temp, decrement=False)
                for index in range(len(data_temp.y))
            ]
        self._data_list = data_list
        print(len(self._data_list))

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]

