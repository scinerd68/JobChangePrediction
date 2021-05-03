import numpy as np
import pandas as pd

data = pd.read_csv('Data/aug_train.csv')
data["city_development_index_cat"] = pd.cut(data["city_development_index"], bins = [0,0.5,0.6,0.7,0.8,0.9,0.95,1.0], labels = [1,2,3,4,5,6,7])
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits =1, test_size =0.2, random_state =2)
for train_index, test_index in split.split(data, data["city_development_index_cat"]):
  strat_train_set = data.loc[train_index]
  strat_test_set = data.loc[test_index]

split2 = StratifiedShuffleSplit(n_splits =1, test_size =0.25, random_state =2)
for train_index, cv_index in split.split(strat_train_set, strat_train_set["city_development_index_cat"]):
  strat_train_set = data.loc[train_index]
  strat_cv_set = data.loc[cv_index]

strat_train_set.to_csv('Data/strat_train.csv', index=False)
strat_test_set.to_csv('Data/strat_test.csv', index=False)
strat_cv_set.to_csv('Data/strat_cv.csv', index=False)

for set_ in (strat_train_set, strat_test_set, strat_cv_set):
  set_.drop("city_development_index_cat", axis =1, inplace =True)

training = strat_train_set.drop(["enrollee_id", "target"], axis=1)
training_labels = strat_train_set["target"].copy()

cv = strat_cv_set.drop(["enrollee_id", "target"], axis=1)
cv_labels = strat_cv_set["target"].copy()

testing = strat_test_set.drop(["enrollee_id", "target"], axis=1)
testing_labels = strat_test_set["target"].copy()

from  sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer

ord_attribs = ["enrolled_university", "relevent_experience",    "education_level",  "experience",   "company_size", "last_new_job"]
num_attribs = ["city_development_index", "training_hours"]
cat_attribs = ["city",  "gender",   "major_discipline", "company_type"]
all_attribs = ["enrolled_university", "relevent_experience",    "education_level",  "experience",   "company_size", "last_new_job", "city_development_index", "training_hours", "city", "gender",   "major_discipline", "company_type"]


class OrdTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, **trans_params):
        self.X = pd.DataFrame(X, columns=ord_attribs)
        self.X['relevent_experience'] = self.X['relevent_experience'].apply(self.relevent_experience_to_numeric)
        self.X['enrolled_university'] = self.X['enrolled_university'].apply(self.enrolled_university_to_numeric)
        self.X['education_level'] = self.X['education_level'].apply(self.education_level_to_numeric)
        self.X['experience'] = self.X['experience'].apply(self.experience_to_numeric)
        self.X['company_size'] = self.X['company_size'].apply(self.company_size_to_numeric)
        self.X['last_new_job'] = self.X['last_new_job'].apply(self.last_new_job_to_numeric)
        return self.X

    def relevent_experience_to_numeric(self, x):
        if x == "Has relevent experience":
            return 1
        if x == "No relevent experience":
            return 0
        return None

    def enrolled_university_to_numeric(self, x):
        if x == 'no_enrollment':
            return 0
        if x == 'Part time course':
            return 1
        if x == 'Full time course':
            return 2
        return None

    def education_level_to_numeric(self, x):
        if x == "Primary School":
            return 1
        if x == "High School":
            return 2
        if x == "Graduate":
            return 3
        if x == "Masters":
            return 4
        if x == "Phd":
            return 5
        return None

    def company_size_to_numeric(self, x):
        from math import log
        if x == '<10':
            return log(5)
        if x == '10/49':
            return log(25)
        if x == '50-99':
            return log(75)
        if x == '100-500':
            return log(300)
        if x == '500-999':
            return log(750)
        if x == '1000-4999':
            return log(3000)
        if x == '5000-9999':
            return log(7500)
        if x == '10000+':
            return log(10000)
        else:
            return None

    def experience_to_numeric(self, x):
        if x == '>20':
            return 21
        elif x == '<1':
            return 0
        elif x != x:
            return None
        else:
            return int(x)

    def last_new_job_to_numeric(self, x):
        if x == '>4':
            return 6
        elif x == 'never':
            return 1
        elif x != x:
            return None
        else:
            return int(x) + 1

class SparseToDense(TransformerMixin):
  def fit(self,X,y=None):
    return self
  def transform(self,X):
    X = X.toarray()
    return X


ord_pipeline = Pipeline([
    ('to_numeric', OrdTransformer()),
    ('imputer', KNNImputer()),
    ('std_scaler', StandardScaler())
])

num_pipeline = Pipeline([
    ('imputer', KNNImputer()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore')),
    ('to_dense', SparseToDense()),
    ('std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("ord", ord_pipeline, ord_attribs),
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

training_prepared = full_pipeline.fit_transform(training)
cv_prepared = full_pipeline.transform(cv)
testing_prepared = full_pipeline.transform(testing)

np.savetxt("Data/X_train_strat.csv", training_prepared, delimiter=',')
np.savetxt("Data/y_train_strat.csv", training_labels.to_numpy(), delimiter=',')

np.savetxt("Data/X_cv_strat.csv", cv_prepared, delimiter=',')
np.savetxt("Data/y_cv_strat.csv", cv_labels.to_numpy(), delimiter=',')

np.savetxt("Data/X_test_strat.csv", testing_prepared, delimiter=',')
np.savetxt("Data/y_test_strat.csv", testing_labels.to_numpy(), delimiter=',')

