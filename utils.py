import mne
import numpy as np
from sklearn.decomposition import PCA
from mne_icalabel import label_components
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from pyprep.prep_pipeline import PrepPipeline
from scipy import signal
from mne.preprocessing import ICA
import re
import os

ch_demanded_order = [
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "P7",
    "P3",
    "P4",
    "P8",
    "O1",
    "O2",
]

current_order = [
    "Fp1",
    "F7",
    "T7",
    "P7",
    "F3",
    "C3",
    "P3",
    "O1",
    "Fz",
    "Cz",
    "Fp2",
    "F4",
    "C4",
    "P4",
    "O2",
    "F8",
    "P8",
    "T8",
]


standard_channel_order = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FZ-CZ",
    "CZ-PZ",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8-0",
    "P8-O2",
]


def remove_channels_dummy(ch_name):
    if re.findall("--", ch_name):
        print(f"Removing channel {ch_name}")
        return True
    return False


def remove_channels_duplicate(ch_name):
    ch_decomposed = ch_name.split(sep="-")
    if len(ch_decomposed) > 2:
        if int(ch_decomposed[-1]) != 0:
            print(f"Removing channel {ch_name}")
            return True
    return False


def remove_repeating_pairs(cleared_ch_names):
    repeating_chs = []
    for pair in cleared_ch_names:
        temp_ch_pairs = cleared_ch_names.copy()
        temp_ch_pairs.remove(pair)
        try:
            ch_1, ch_2 = pair.split(sep="-")
        except:
            ch_1, ch_2, ch_3 = pair.split(sep="-")
        for pair_2 in temp_ch_pairs:
            num_of_same_chs = pair_2.count(ch_1) + pair_2.count(ch_2)
            if num_of_same_chs > 1:
                repeating_chs.append(pair_2)

    return repeating_chs[0]  ## not so good function!


def reorder_channels_chbmit(raw):
    """Reorders the channels of chbmit dataset patients to the one we need."""
    ch_map = {}
    for n, old_name in enumerate(raw.ch_names):
        ch_map[old_name] = current_order[n]
    raw.rename_channels(ch_map)
    raw.reorder_channels(ch_demanded_order)
    montage = mne.channels.read_custom_montage(
        r"A:\Program Files\MATLAB\R2020a\eeglab2022.1\my_locs\chb_mit_ch_locs.loc"
    )
    raw.set_montage(montage)


def run_preprocessing(
    raw,
    n_ica_components=18,
    freq_l=1.0,
    freq_h=30.0,
    avg_ref=True,
    apply_pca=True,
    apply_ica=False,
    informax=False,
):
    """Runs preprocessing on given mne.raw instance."""
    if apply_ica and apply_pca:
        raise ValueError(
            "Values of apply_pca and apply_ica cannot both be True! Choose only one method of artifact removal."
        )

    raw.load_data()
    raw.filter(l_freq=freq_l, h_freq=freq_h, h_trans_bandwidth=1)
    if avg_ref:
        raw.set_eeg_reference()
    if apply_ica:
        if informax:
            ica = ICA(
                n_components=n_ica_components,
                max_iter="auto",
                random_state=97,
                method="infomax",
                fit_params=dict(extended=True),
            )
        else:
            ica = ICA(n_components=n_ica_components)
        ica.fit(raw)
        ic_labels = label_components(raw, ica, method="iclabel")
        labels = np.array(ic_labels["labels"])
        ica.exclude = np.where((labels != "other") & (labels != "brain"))[0].tolist()
        ica.apply(raw)

    elif apply_pca:
        pca = PCA()
        pca.fit(raw._data)
        components = pca.components_
        mu = pca.mean_
        transform_raw = pca.transform(raw._data)
        raw._data = np.squeeze(mu + transform_raw[:, 2:] @ components[2:])

    return raw


def load_and_dump_channels(filepath):
    """Reads CHB-MIT file and resets the order of channels to the demanded one"""
    data_raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    try:
        ch_names = data_raw.ch_names
        ch_to_remove = list(filter(remove_channels_dummy, ch_names))
        data_raw.drop_channels(ch_to_remove)
    except:
        print("No dummy channels to remove")
    try:
        ch_names = data_raw.ch_names
        ch_to_remove = list(filter(remove_channels_duplicate, ch_names))
        data_raw.drop_channels(ch_to_remove)
    except:
        print("No duplicate channels to remove")
    try:
        ch_names = data_raw.ch_names
        data_raw.drop_channels(remove_repeating_pairs(ch_names))
    except:
        print("No repeating pairs to remove")
    current_channels = data_raw.ch_names
    if current_channels != standard_channel_order:
        data_raw.reorder_channels(standard_channel_order)
    if len(current_channels) > 18:
        n_chs_to_drop = len(current_channels) - 18
        channel_ordering = current_channels[:-n_chs_to_drop]
        data_raw.reorder_channels(channel_ordering)
    if len(current_channels) < 18:
        print(f"Too few channels to processd, found {len(current_channels)}. Skipping.")
        return None
    return data_raw


def preprocess_dataset(
    subjects_with_seizures_path, dataset_dirpath, preprocessed_dirpath
):
    """Runs full preprocessing on the dataset cointained in the folder.
    Args:  
        subject_with_seizures_path: path to a text file containing recordings with seizures.
    The names have to be a row vector, with every row named patient_folder/recording_name.edf.

        dataset_path: path to a folder containing all patient folders.

        preprocessed_dirpath: path to folder in which preprocessed files will be saved.

    """
    subjects_with_seizures = [
        subject[:-1] for subject in open(subjects_with_seizures_path, "r").readlines()
    ]
    for subject in subjects_with_seizures:
        try:
            subject_path = os.path.join(dataset_dirpath, subject)
            raw_file = load_and_dump_channels(subject_path)
            if raw_file is None:
                continue
        except:
            print(f"Subject {subject} not found.")
            continue

        reorder_channels_chbmit(raw_file)

        raw_instance = run_preprocessing(raw_file, apply_pca=True, freq_l=2)
        save_path = os.path.join(preprocessed_dirpath, subject)
        if not os.path.exists(os.path.split(save_path)[0]):
            os.mkdir(os.path.split(save_path)[0])
        mne.export.export_raw(save_path, raw_instance, fmt="edf")
        print(f"Finished preprocessing subject {subject}.")


def prepare_timestep_array(array, timestep, overlap):
    """Preprocess input array of shape [n_nodes,feature_per_node,samples]
    into [samples_count,n_nodes,feature_per_node,timestep]."""
    features = [
        array[:, :, i : i + timestep]
        for i in range(0, array.shape[2] - timestep + 1, timestep - overlap)
    ]
    return np.array(features)


def prepare_timestep_label(array, timestep, overlap):
    """Preprocess input array of shape [n_nodes,feature_per_node,samples]
    into [samples_count,n_nodes,feature_per_node,timestep]."""
    time_to_seizure = array.shape[2]
    seconds = [
        (time_to_seizure - i) / 256
        for i in range(timestep, array.shape[2], timestep - overlap)
    ]
    return np.array(seconds)


def extract_training_data_and_labels(
    raw,
    start_ev_array,
    stop_ev_array,
    savepath=None,
    seizure_lookback: int = 600,
    sample_timestep: int = 10,
    overlap: int = 9,
):
    ## TODO - skończyć to i przetestować bo źle to wygląda, dorobić zapis
    """Function to extract seizure periods and preictal perdiods into samples ready to be put into graph neural network."""
    for n, start_ev in enumerate(start_ev_array):
        fs = int(raw.info["sfreq"])
        seizure_lookback = seizure_lookback
        prev_event_time = start_ev - stop_ev_array[n - 1] if n > 0 else start_ev

        if prev_event_time > seizure_lookback:
            interictal_period = raw.get_data(
                start=(start_ev - seizure_lookback) * fs, stop=start_ev * fs
            )
        else:
            interictal_period = raw.get_data(
                start=(start_ev - prev_event_time) * fs, stop=start_ev * fs
            )

        interictal_period = (
            np.expand_dims(interictal_period.transpose(), axis=2)
            .swapaxes(0, 2)
            .swapaxes(0, 1)
        )  ##reshape for preprocessing
        interictal_features = prepare_timestep_array(
            array=interictal_period, timestep=sample_timestep * fs, overlap=overlap * fs
        )
        interictal_event_labels = np.zeros(
            interictal_features.shape[0]
        )  ## assign label 0 to every interictal period sample
        interictal_event_time_labels = prepare_timestep_label(
            interictal_period, sample_timestep * fs, overlap * fs
        )  ## assign time to seizure for every sample [s]
        seizure_period = raw.get_data(start=start_ev * fs, stop=stop_ev_array[n] * fs)
        print(seizure_period.shape)
        seizure_period = (
            np.expand_dims(seizure_period.transpose(), axis=2)
            .swapaxes(0, 2)
            .swapaxes(0, 1)
        )
        print(seizure_period.shape)
        seizure_features = prepare_timestep_array(
            array=seizure_period, timestep=sample_timestep * fs, overlap=overlap * fs
        )
        seizure_event_labels = np.ones(seizure_features.shape[0])
        if n == 0:
            full_interictal_features = interictal_features
            full_interictal_event_labels = interictal_event_labels
            full_interictal_event_time_labels = interictal_event_time_labels
            full_seizure_features = seizure_features
            full_seizure_event_labels = seizure_event_labels
        else:
            full_interictal_features = np.concatenate(
                (full_interictal_features, interictal_features)
            )
            full_interictal_event_labels = np.concatenate(
                (full_interictal_event_labels, interictal_event_labels)
            )
            full_interictal_event_time_labels = np.concatenate(
                (full_interictal_event_time_labels, interictal_event_time_labels)
            )
            full_seizure_features = np.concatenate(
                (full_seizure_features, seizure_features)
            )
            full_seizure_event_labels = np.concatenate(
                (full_seizure_event_labels, seizure_event_labels)
            )
        
        ## assign label 1 to every seizure period sample
        # savepath_interictal_data = f"{savepath}_interictal_{n}"
        # savepath_interictal_labels_e = f"{savepath}_interictal_labels_e_{n}"
        # savepath_interictal_labels_t = f"{savepath}_interictal_labels_t_{n}"
        # savepath_ictal_data = f"{savepath}_ictal_{n}"
        # savepath_ictal_labels_e = f"{savepath}_ictal_labels_e_{n}"
        # np.save()
    return (
        full_interictal_features,
        full_interictal_event_labels,
        full_interictal_event_time_labels,
        full_seizure_features,
        full_seizure_event_labels,
    )


def run_prep(raw, line_freq, ransac=False, channel_wise=False):
    sfreq = raw.info["sfreq"]
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(line_freq, sfreq / 2, line_freq),
    }
    raw.load_data()
    montage = raw.get_montage()
    prep = PrepPipeline(
        raw, prep_params, montage, ransac=ransac, channel_wise=channel_wise
    )
    prep.fit()
    return prep
