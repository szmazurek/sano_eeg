from utils import save_timeseries_array
from pathlib import Path

preprocessed_ds_path = Path('data/seizure_fiels_only')
npy_ds_path = Path('data/npy_data_no_preprocessing')

save_timeseries_array(preprocessed_ds_path,npy_ds_path)