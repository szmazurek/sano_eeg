# A repository for system to predict seizure occurence in epileptic patients from EEG recordings
Using chb-mit dataset from https://physionet.org/content/chbmit/1.0.0/
Warning! In the file RECORDS-WITH-SEIZURES in line 35 (chb07/chb07_18.edf) should be changed into chb07/chb07_19.edf
How to use this repo:
1. Download chb-mit dataset using link from above into repo folder, pull the directory containing data as a main
directory and rename it to raw_dataset
2. Create python enviornment using either enviornment_linux.yml for linux or environment.yml for windows. 
Probably conflicts will emerge, they have to be dealt with manually. 
3. Create preprocessed_data_folder and run.
4. Activate new enviornment
5. python run_preprocessing.py
6. python run_event_extraction.py
7. python run_npy_conversion.py
8. Use dataloader defined in seizure_table_reading.ipynb file and run the model defined in the same file.