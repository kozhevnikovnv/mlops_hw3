import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


class Model:
    def __init__(self, model_type):
        self.model_name = model_type
        if model_type == 'LGBM':
            self.model = LGBMClassifier()
        elif model_type == 'Logreg':
            self.model = LogisticRegression()
        else:
            raise ValueError('unsupported model')
        self.feature_names = ['variance', 'skewness', 'curtois', 'enthropy']
        self.target_name = 'target'
        self.is_fitted = False
        self.score_train = None
        self.score_test = None
        self.hyperparams = None

    def preprocessing_data(self, data_path):
        data = pd.read_csv(data_path)
        X = data.drop([self.target_name], axis=1)
        y = data[self.target_name]
        return X, y

    def fit(self, data):
        X, y = self.preprocessing_data(data)
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, data_to_predict):
        return self.model.predict(data_to_predict)
