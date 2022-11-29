import utils

adult_data =r'dult_data'
peprocessed_data = r'preprocessed_adult'
subject_seizures = r'adult_data\RECORDS-WITH-SEIZURES'

utils.preprocess_dataset(subject_seizures,adult_data,peprocessed_data)