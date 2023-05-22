import unittest
import tsaugmentation as tsag
import pandas as pd
import numpy as np
from tsaugmentation.feature_engineering.static_features import (
    create_static_features,
    scale_static_features,
)
from tsaugmentation.feature_engineering.dynamic_features import create_dynamic_features
from tsaugmentation.feature_engineering.feature_transformations import (
    temporalize,
    detemporalize,
    combine_inputs_to_model,
)
from sklearn.preprocessing import MinMaxScaler


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.window_size = 10
        self.latent_dim = 2
        self.dataset = tsag.preprocessing.PreprocessDatasets(
            "tourism", freq='M'
        ).apply_preprocess()
        data = self.dataset["predict"]["data_matrix"]
        self.n = data.shape[0]
        self.s = data.shape[1]
        self.n_features = self.s
        self.n_train = self.n - self.window_size
        self.groups = list(self.dataset["train"]["groups_names"].keys())
        self.df = pd.DataFrame(data)
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.dataset["dates"], columns=["Date"])], axis=1
        )[: data.shape[0]]
        self.df = self.df.set_index("Date")

        self.X_train_raw = self.df.astype(np.float32).to_numpy()

        scaler_target = MinMaxScaler().fit(self.X_train_raw)
        self.X_train_raw_scaled = scaler_target.transform(self.X_train_raw)

    def test_feature_transformations_temporalize(self):
        X_train = temporalize(self.X_train_raw_scaled, self.window_size)

        self.assertTrue(X_train.shape == (219, 10, 304))

    def test_feature_transformations_detemporalize(self):
        X_train = temporalize(self.X_train_raw_scaled, self.window_size)
        X_train_orig = detemporalize(X_train, self.window_size)

        self.assertTrue(X_train_orig.shape == (228, 304))
