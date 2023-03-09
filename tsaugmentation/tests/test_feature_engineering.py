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
            "tourism"
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

        self.static_features_res = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                    0.16666667,
                ],
            ]
        )

        self.dynamic_features_res = pd.DataFrame(
            {
                "year_cos": {
                    10: 1.0,
                    11: 1.0,
                    12: 0.9396926164627075,
                    13: 0.9396926164627075,
                    14: 0.9396926164627075,
                    15: 0.9396926164627075,
                    16: 0.9396926164627075,
                    17: 0.9396926164627075,
                    18: 0.9396926164627075,
                    19: 0.9396926164627075,
                },
                "year_sin": {
                    10: 0.0,
                    11: 0.0,
                    12: 0.3420201539993286,
                    13: 0.3420201539993286,
                    14: 0.3420201539993286,
                    15: 0.3420201539993286,
                    16: 0.3420201539993286,
                    17: 0.3420201539993286,
                    18: 0.3420201539993286,
                    19: 0.3420201539993286,
                },
                "month_cos": {
                    10: 0.8412535190582275,
                    11: 1.0,
                    12: 1.0,
                    13: 0.8412535190582275,
                    14: 0.4154150187969208,
                    15: -0.1423148363828659,
                    16: -0.6548607349395752,
                    17: -0.9594929814338684,
                    18: -0.9594929814338684,
                    19: -0.6548607349395752,
                },
                "month_sin": {
                    10: -0.5406408309936523,
                    11: -1.1331077372775996e-15,
                    12: 0.0,
                    13: 0.5406408309936523,
                    14: 0.9096319675445557,
                    15: 0.9898214340209961,
                    16: 0.7557495832443237,
                    17: 0.28173255920410156,
                    18: -0.28173255920410156,
                    19: -0.7557495832443237,
                },
            }
        )

    def test_creating_static_features(self):
        static_features = create_static_features(
            self.window_size, self.groups, self.dataset, self.n
        )
        static_features_scaled = scale_static_features(static_features)

        np.testing.assert_array_almost_equal(
            static_features_scaled["state"][100:105, 50:70],
            self.static_features_res,
        )

    def test_creating_dynamic_features(self):
        dynamic_features = create_dynamic_features(self.df, "MS")

        pd.testing.assert_frame_equal(
            dynamic_features.iloc[10:20],
            self.dynamic_features_res.astype(np.float32),
        )

    def test_feature_transformations_temporalize(self):
        X_train, y_train = temporalize(self.X_train_raw_scaled, self.window_size)

        self.assertTrue(X_train.shape == (218, 10, 304))

    def test_feature_transformations_detemporalize(self):
        X_train, y_train = temporalize(self.X_train_raw_scaled, self.window_size)
        X_train_orig = detemporalize(X_train)

        self.assertTrue(X_train_orig.shape == (218, 304))

    def combine_preprocessed_inputs(self):
        dynamic_features_inp, X_inp, static_features_inp = combine_inputs_to_model(
            self.X_train,
            self.dynamic_features,
            self.static_features_scaled,
            self.window_size,
        )

        self.assertTrue(len(dynamic_features_inp + X_inp + static_features_inp) == 9)
