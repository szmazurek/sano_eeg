import utils
from pathlib import Path
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_dataset_path", type=str)
    parser.add_argument("--preprocessed_data_save_path", type=str)
    parser.add_argument("--recordings_with_seizures_file_path", type=str)
    args = parser.parse_args()
    adult_data = args.raw_dataset_path
    peprocessed_data = args.preprocessed_data_save_path
    subject_seizures = Path("data/raw_dataset/RECORDS-WITH-SEIZURES")

    utils.preprocess_dataset_seizures(subject_seizures, adult_data, peprocessed_data)
