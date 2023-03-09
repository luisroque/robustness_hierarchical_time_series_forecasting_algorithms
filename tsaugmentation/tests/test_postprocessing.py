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
    detemporalize,
)
from tsaugmentation.postprocessing.generative_helper import generate_new_time_series
from tsaugmentation.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)
from tsaugmentation.visualization.model_visualization import plot_generated_vs_original
from sklearn.preprocessing import MinMaxScaler
from tsaugmentation.model.models import VAE, get_mv_model
from tensorflow import keras


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.window_size = 10
        latent_dim = 2
        dataset = tsag.preprocessing.PreprocessDatasets("tourism").apply_preprocess()
        data = dataset["predict"]["data_matrix"]
        self.n = data.shape[0]
        self.n_train = self.n - self.window_size
        s = data.shape[1]
        self.n_features = s
        groups = list(dataset["train"]["groups_names"].keys())
        df = pd.DataFrame(data)
        df = pd.concat([df, pd.DataFrame(dataset["dates"], columns=["Date"])], axis=1)[
            : data.shape[0]
        ]
        df = df.set_index("Date")

        self.X_train_raw = df.astype(np.float32).to_numpy()

        self.scaler_target = MinMaxScaler().fit(self.X_train_raw)
        X_train_raw_scaled = self.scaler_target.transform(self.X_train_raw)

        static_features = create_static_features(
            self.window_size, groups, dataset, self.n
        )

        static_features_scaled = scale_static_features(static_features)
        dynamic_features = create_dynamic_features(df, "MS")
        X_train, y_train = temporalize(X_train_raw_scaled, self.window_size)

        n_features_concat = X_train.shape[1] + dynamic_features.shape[1]

        (
            self.dynamic_features_inp,
            X_inp,
            self.static_features_inp,
        ) = combine_inputs_to_model(
            X_train,
            dynamic_features,
            static_features_scaled,
            self.window_size,
        )
        static_features_inp_window = [
            np.squeeze(arr[-218:]) for arr in self.static_features_inp
        ]

        encoder, decoder = get_mv_model(
            mv_normal_dim=self.n_features,
            static_features=static_features_scaled,
            dynamic_features_df=dynamic_features,
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=n_features_concat,
            latent_dim=latent_dim,
        )
        self.vae_full_mv = VAE(encoder, decoder, self.window_size)
        self.vae_full_mv.compile(optimizer=keras.optimizers.Adam())

        self.vae_full_mv.fit(
            x=(self.dynamic_features_inp, X_inp, static_features_inp_window),
            epochs=5,
            batch_size=5,
            shuffle=False,
        )

        _, _, self.z = self.vae_full_mv.encoder.predict(
            self.dynamic_features_inp + [X_inp] + static_features_inp_window
        )
        preds = self.vae_full_mv.decoder.predict(
            [self.z] + self.dynamic_features_inp + static_features_inp_window
        )
        preds = detemporalize(preds)
        self.X_hat = self.scaler_target.inverse_transform(preds)

    def test_generate_new_time_series(self):

        dec_pred_hat = generate_new_time_series(
            self.vae_full_mv,
            [0.5, 0.5],
            self.z,
            self.window_size,
            self.dynamic_features_inp,
            self.static_features_inp,
            self.scaler_target,
            self.n_features,
            self.n_train,
        )

        plot_generated_vs_original(
            dec_pred_hat=dec_pred_hat,
            X_train_raw=self.X_train_raw[10:],
            param_vae=10,
            dataset_name="tourism",
        )

        self.assertTrue(dec_pred_hat.shape == (218, 304))

    def test_compute_similarity(self):
        dec_pred_hat = generate_new_time_series(
            self.vae_full_mv,
            [0.5, 0.5],
            self.z,
            self.window_size,
            self.dynamic_features_inp,
            self.static_features_inp,
            self.scaler_target,
            self.n_features,
            self.n_train,
        )

        self.assertTrue(
            compute_similarity_transformed_vs_original(dec_pred_hat, self.X_train_raw)[
                0
            ]
            < 20
        )
