# Repo description
Branches:

1. master - contains all the code used for the project, not updated for quite some time
2. pipeline_tests - contains the code used for local experiments and testing
3. dev - branch with code used on athena HPC for longer experiments (code succesfully tested in pipeline_tests is added to this branch)

## How to use this repo:

1. Run data extraction as described in README.md in raw_dataset folder
2. In pipeline_tests - the notebook named main_experiments.ipynb contains the code for ongoing experiemnts. \
It is subject to changes and ongoing work is happening here. Main parts are SeizureDataLoader class and training loop along with models.
The models and datloader are not in the seperate files as debugging is easier this way in the Jupyter. Dataloader is terribly bloated \
and needs to be refactored (although some things are left on purpose as they may be needed in further develeopment stages).
3. In dev - most changes happen in the train_hpc.py script. It also needs refactoring, but is subject to many changes (it is also hard to tell what will be needed in the future).