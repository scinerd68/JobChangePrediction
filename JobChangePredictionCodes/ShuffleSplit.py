import pandas as pd
from sklearn.model_selection import ShuffleSplit

job_change = pd.read_csv('Data/aug_train.csv')
split = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_cv_index, test_index in split.split(job_change):
    train_cv_set = job_change.iloc[train_cv_index]
    test_set = job_change.iloc[test_index]

cv_split = ShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, cv_index in cv_split.split(train_cv_set):
    train_set = train_cv_set.iloc[train_index]
    cv_set = train_cv_set.iloc[cv_index]

train_set.to_csv('Data/train.csv', index=False)
test_set.to_csv('Data/test.csv', index=False)
cv_set.to_csv('Data/cv.csv', index=False)