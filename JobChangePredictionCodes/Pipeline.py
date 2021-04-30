from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

train_set = pd.read_csv("Data/train.csv")
cv_set = pd.read_csv("Data/cv.csv")
test_set = pd.read_csv("Data/test.csv")

# train_set = pd.read_csv("Data/strat_train.csv")
# cv_set = pd.read_csv("Data/strat_cv.csv")
# test_set = pd.read_csv("Data/strat_test.csv")

job_change = train_set.drop("target", axis=1)
job_change_labels = train_set["target"].copy()

num_attribs = ['city_development_index', 'training_hours']
ord_cat_attribs = ['relevent_experience', 'enrolled_university', 'education_level', 'experience', 'company_size', 'last_new_job']
nom_cat_attribs = ['city', 'gender', 'major_discipline', 'company_type']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy= "median")),
    ('std_scaler', StandardScaler())
])

class OrdToNumeric(TransformerMixin):
    def fit(self,X,y=None):
        return self

    def transform(self,X,y=None):
        X['relevent_experience'] = X['relevent_experience'].apply(self.relevent_experience_to_numeric)
        X['enrolled_university'] = X['enrolled_university'].apply(self.enrolled_university_to_numeric)
        X['education_level'] = X['education_level'].apply(self.education_level_to_numeric)
        X['experience'] = X['experience'].apply(self.experience_to_numeric)
        X['company_size'] = X['company_size'].apply(self.company_size_to_numeric)
        X['last_new_job'] = X['last_new_job'].apply(self.last_new_job_to_numeric)
        return X

    def relevent_experience_to_numeric(self, experience):
        if experience != experience:
            return None
        if experience == "Has relevent experience":
            return 1
        return 0

    def enrolled_university_to_numeric(self, x):
        if x != x:
            return None
        if x == 'no_enrollment':
            return 0
        if x == 'Part time course':
            return 1
        if x == 'Full time course':
            return 2

    def education_level_to_numeric(self, x):
        if x != x:
            return None
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

    def experience_to_numeric(self, x):
        if x != x:
            return None
        if x == '>20':
            return 21
        elif x == '<1':
            return 0
        else:
            return int(x)

    def company_size_to_numeric(self, x):
        if x != x:
            return None
        if x == '<10' :
            return 0
        if x == '10/49' :
            return 1
        if x == '50-99' :
            return 2
        if x == '100-500' :
            return 3
        if x == '500-999' :
            return 4
        if x == '1000-4999' :
            return 5
        if x == '5000-9999':
            return 6
        if x == '10000+':
            return 7

    def last_new_job_to_numeric(self, x):
        if x != x:
            return None
        if x == '>4' :
            return 5
        if x == 'never' :
            return 0
        else:
            return int(x)

ord_pipeline = Pipeline([
    ('ord_to_numeric', OrdToNumeric()),
    ('ord_imputer', SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
    ('ord_std_scaler', StandardScaler())
])

class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()

nom_pipeline = Pipeline([
    ('nom_imputer', SimpleImputer(strategy="constant")),
    ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore')),
    ('to_array', DenseTransformer()),
    ('nom_std_scaler', StandardScaler())
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("ord", ord_pipeline, ord_cat_attribs),
    ("nom", nom_pipeline, nom_cat_attribs)
])

job_change_prepared = full_pipeline.fit_transform(job_change)

np.savetxt("Data/X_train.csv", job_change_prepared, delimiter=',')
np.savetxt("Data/y_train.csv", job_change_labels.to_numpy(), delimiter=',')

from imblearn.under_sampling import RandomUnderSampler
undersample = RandomUnderSampler(sampling_strategy="majority", random_state=42)
# fit and apply the transform
X_under, y_under = undersample.fit_resample(job_change_prepared, job_change_labels.to_numpy())

np.savetxt("Data/X_train_under.csv", X_under, delimiter=',')
np.savetxt("Data/y_train_under.csv", y_under, delimiter=',')


job_change_test = test_set.drop("target", axis=1)
job_change_test_labels = test_set["target"].copy()

job_change_test_prepared = full_pipeline.transform(job_change_test)

np.savetxt("Data/X_test.csv", job_change_test_prepared, delimiter=',')
np.savetxt("Data/y_test.csv", job_change_test_labels.to_numpy(), delimiter=',')


job_change_cv = cv_set.drop("target", axis=1)
job_change_cv_labels = cv_set["target"].copy()

job_change_cv_prepared = full_pipeline.transform(job_change_cv)

np.savetxt("Data/X_cv.csv", job_change_cv_prepared, delimiter=',')
np.savetxt("Data/y_cv.csv", job_change_cv_labels.to_numpy(), delimiter=',')
