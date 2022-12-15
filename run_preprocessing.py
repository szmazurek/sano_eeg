import utils
from pathlib import Path

adult_data =Path('data/raw_dataset')
peprocessed_data = Path('data/preprocessed_data')
subject_seizures = Path('data/raw_dataset/RECORDS-WITH-SEIZURES')

utils.preprocess_dataset(subject_seizures,adult_data,peprocessed_data)