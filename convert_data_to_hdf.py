"""Script to convert data to hdf5 format to reduce storage and amount of I/O operations."""

import numpy as np
import h5py
import os
import argparse
import multiprocessing as mp
from typing import List, Tuple, Dict, Union, Any, Optional
import dataclasses
import pandas as pd
import logging
from argparse import ArgumentParser
from datetime import datetime


def get_recording_events(events_dict, recording: str) -> list[int]:
    """Read seizure times into list from event_dict.
    Args:
        events_dict: (dict) Dictionary with events for given patient.
        recording: (str) Name of the recording to get events for.
    Returns:
        recording_events: (list) List of seizure event start and stop time for given recording.
    """
    try:
        recording_list = list(events_dict[recording].values())
        recording_events = [int(x) for x in recording_list if not np.isnan(x)]
    except KeyError as e:
        LOGGER.info("No events found for recording: " + recording)
        raise e

    return recording_events


def get_event_tables(
    patient_name: str, event_tables_path: str
) -> tuple[dict, dict]:
    """Read events for given patient into start and stop times lists
    from .csv extracted files.
    Args:
        patient_name: (str) Name of the patient to get events for.
    Returns:
        events: (tuple) Tuple of start and stop event dicts for given patient.
    """
    try:
        patient_start_table = os.path.join(
            event_tables_path, patient_name + "_start.csv"
        )
        patient_stop_table = os.path.join(
            event_tables_path, patient_name + "_stop.csv"
        )
        start_events_dict = pd.read_csv(patient_start_table).to_dict("index")
        stop_events_dict = pd.read_csv(patient_stop_table).to_dict("index")

    except FileNotFoundError as e:
        LOGGER.info("No event tables found for patient: " + patient_name)
        raise e
    events = (start_events_dict, stop_events_dict)
    return events


def get_patinet_recording_data(patient: str) -> None:
    """Reads patient data from .npy files and saves them into hdf5 format."""
    NPY_DATA_PATH = args.npy_data_path
    EVENT_TABLES_PATH = args.event_tables_path
    HDF_FILE_PATH = args.hdf_file_path
    patient_recordings_path = os.path.join(NPY_DATA_PATH, patient)
    patient_recordings = [
        os.path.join(patient_recordings_path, x)
        for x in os.listdir(patient_recordings_path)
    ]
    patient_events = get_event_tables(patient, EVENT_TABLES_PATH)
    patient_events_start = patient_events[0]
    patient_events_stop = patient_events[1]
    recordings_with_events = patient_events_start.keys()
    while True:
        try:
            with h5py.File(HDF_FILE_PATH, "a") as f:
                f.create_group(patient)
                break
        except BlockingIOError:
            continue
    LOGGER.info(f"Started converting data for patient: {patient}")
    for recording in patient_recordings:
        recording_name = os.path.basename(recording)
        print(recording_name)
        if "seizures" in recording_name:
            recording_name = recording_name.removeprefix("seizures_")
            recording_name_temp = recording_name.replace(".npy", ".edf")
            if recording_name_temp in recordings_with_events:
                recording_events_start = get_recording_events(
                    patient_events_start, recording_name_temp
                )
                recording_events_stop = get_recording_events(
                    patient_events_stop, recording_name_temp
                )
        recording_name = recording_name.removesuffix(".npy")
        while True:
            try:
                with h5py.File(HDF_FILE_PATH, "a") as f:
                    f[patient].create_dataset(
                        recording_name,
                        data=np.load(recording),
                        compression="gzip",
                    )
                    try:
                        f[patient][recording_name].attrs[
                            "events_start"
                        ] = recording_events_start
                        f[patient][recording_name].attrs[
                            "events_stop"
                        ] = recording_events_stop
                        del (
                            recording_events_start,
                            recording_events_stop,
                            recording_name_temp,
                        )
                    except UnboundLocalError:
                        LOGGER.info(
                            f"No events found for recording: {recording}"
                        )
                    break
            except BlockingIOError:
                # LOGGER.info("Waiting for file to be available.")
                continue
        LOGGER.info(
            f"Finished converting data for recording: {recording_name} for patient: {patient}"
        )

    LOGGER.info(f"Finished converting data for patient: {patient}")
    return None


def main(args) -> None:
    if not os.path.exists(os.path.dirname(HDF_FILE_PATH)):
        os.makedirs(os.path.dirname(HDF_FILE_PATH))

    patient_list = os.listdir(NPY_DATA_PATH)
    pool = mp.Pool(mp.cpu_count())
    pool.map(get_patinet_recording_data, patient_list)
    pool.close()
    pool.join()
    # for patient in patient_list:
    #     get_patinet_recording_data(patient)
    print("Done")
    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--npy_data_path", type=str, required=True)
    parser.add_argument("--event_tables_path", type=str, required=True)
    parser.add_argument("--hdf_file_path", type=str, required=True)
    args = parser.parse_args()
    NPY_DATA_PATH = args.npy_data_path
    EVENT_TABLES_PATH = args.event_tables_path
    HDF_FILE_PATH = args.hdf_file_path
    start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.basicConfig(
        filename=f"logs/npy_to_hdf_{start_time}.log",
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    LOGGER = logging.getLogger("hdf_dataset_writer")
    LOGGER.info("Started converting data to hdf5 format.")
    try:
        main(args)
    except (Exception, KeyboardInterrupt, SystemExit) as e:
        LOGGER.error(e)
        os.remove(HDF_FILE_PATH)
        print("Error occured. HDF file removed.")
        raise e
