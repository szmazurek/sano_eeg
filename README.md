# A repository for system to predict seizure occurence in epileptic patients from EEG recordings
Using chb-mit dataset from https://physionet.org/content/chbmit/1.0.0/
Warning! In the file RECORDS-WITH-SEIZURES in line 35 (chb07/chb07_18.edf) should be changed into chb07/chb07_19.edf
How to use this repo:
1. Download chb-mit dataset using link from above into repo folder, pull the directory containing data as a main
directory and rename it to raw_dataset
2. Create python environment and install requirements.txt.
3. To run preprocessing please execute the scrip preprocessing/run_preprocessing.py
Note that preprocessing parameters are hardcoded in utils/utils.py file <br>
to change them they need to be configured manually.
4. To run training please execute the script train.py. By default wandb <br>
is enabled, to run it as is one needs to create wandb_api_key.txt file in <br>
src folder with wandb API key.
5. If one wants to replicate the wandb sweep for architecture search,
please refer to the instructions on https://wandb.ai/. Sweep file is in <br>
sweep_config.yaml. To run the sweep please modify the parameters, such as <br>
entity or model configuration.
6. To run explainability notebook, please follow the instructions in the notebook <br>
named explainability.ipynb.