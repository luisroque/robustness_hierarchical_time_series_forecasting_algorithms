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
    combine_inputs_to_model,
)
from tsaugmentation.visualization.model_visualization import plot_loss
from sklearn.preprocessing import MinMaxScaler
from tsaugmentation.model.models import VAE, get_mv_model
from tensorflow import keras


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.window_size = 10
        self.latent_dim = 2
        self.dataset_name = 'tourism'
        dataset = tsag.preprocessing.PreprocessDatasets(
            self.dataset_name
        ).apply_preprocess()
        data = dataset["predict"]["data_matrix"]
        n = data.shape[0]
        s = data.shape[1]
        self.n_features = s
        n_train = n - self.window_size
        groups = list(dataset["train"]["groups_names"].keys())
        df = pd.DataFrame(data)
        df = pd.concat(
            [df, pd.DataFrame(dataset["dates"], columns=["Date"])], axis=1
        )[: data.shape[0]]
        df = df.set_index("Date")

        X_train_raw = df.astype(np.float32).to_numpy()

        scaler_target = MinMaxScaler().fit(X_train_raw)
        X_train_raw_scaled = scaler_target.transform(X_train_raw)

        static_features = create_static_features(
            self.window_size, groups, dataset, n
        )

        self.static_features_scaled = scale_static_features(static_features)
        self.dynamic_features = create_dynamic_features(df, 'MS')
        X_train, y_train = temporalize(X_train_raw_scaled, self.window_size)

        self.n_features_concat = X_train.shape[1] + self.dynamic_features.shape[1]

        self.features_inp = combine_inputs_to_model(
            X_train,
            self.dynamic_features,
            self.static_features_scaled,
            self.window_size,
        )

    def test_model_mv_normal(self):
        encoder, decoder = get_mv_model(
            mv_normal_dim=self.n_features,
            static_features=self.static_features_scaled,
            dynamic_features_df=self.dynamic_features,
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=self.n_features_concat,
            latent_dim=self.latent_dim
        )
        vae_full_mv = VAE(encoder, decoder, self.window_size)
        vae_full_mv.compile(optimizer=keras.optimizers.Adam())

        history = vae_full_mv.fit(
            x=self.features_inp,
            epochs=5,
            batch_size=5,
            shuffle=False,
        )

        self.assertTrue(history.history['loss'][-1] < 0.5)

    def test_model_univariate_normal(self):
        encoder, decoder = get_mv_model(
            mv_normal_dim=1,
            static_features=self.static_features_scaled,
            dynamic_features_df=self.dynamic_features,
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=self.n_features_concat,
            latent_dim=self.latent_dim
        )
        vae_uni_mv = VAE(encoder, decoder, self.window_size)
        vae_uni_mv.compile(optimizer=keras.optimizers.Adam())

        history = vae_uni_mv.fit(
            x=self.features_inp,
            epochs=5,
            batch_size=5,
            shuffle=False,
        )

        plot_loss(history, 0, self.dataset_name)

        self.assertTrue(history.history['loss'][-1] < 0.5)
