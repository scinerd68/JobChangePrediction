import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


job_change = pd.read_csv('data/aug_train.csv')

split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)
for train_index, test_index in split.split(job_change, job_change["relevent_experience"]):
    strat_train_set = job_change.reindex(index=train_index)
    strat_test_set = job_change.reindex(index=test_index)

strat_train_set.to_csv("Data/train.csv")
strat_test_set.to_csv("Data/test.csv")