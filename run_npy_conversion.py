from utils import save_timeseries_array
from pathlib import Path

preprocessed_ds_path = Path('data/preprocessed_data')
npy_ds_path = Path('data/npy_data')

save_timeseries_array(preprocessed_ds_path,npy_ds_path)