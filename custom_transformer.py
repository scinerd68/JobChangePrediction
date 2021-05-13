import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


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


class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = dict()

    def fit(self, X, y=None):
        for col in X.columns:
            le = LabelEncoder()
            le.fit(X.loc[X[col].notna(), col])
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))

            # Set unknown to new value so transform on test set handles unknown values
            max_value = max(le_dict.values())
            le_dict['_unk'] = max_value + 1

            self.encoders[col] = le_dict
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            le_dict = self.encoders[col]
            X.loc[X[col].notna(), col] = X.loc[X[col].notna(), col].apply(
                lambda x: le_dict.get(x, le_dict['_unk'])).values
        return X

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y)
        return self.transform(X, y)


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.toarray()