import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List

from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from tsaugmentation.model.models import CVAE, get_CVAE, get_flatten_size_encoder
from tsaugmentation.feature_engineering.static_features import (
    create_static_features,
)
from tsaugmentation.feature_engineering.dynamic_features import create_dynamic_features
from tsaugmentation.feature_engineering.feature_transformations import (
    temporalize,
    combine_inputs_to_model,
    detemporalize,
)
from tsaugmentation.postprocessing.generative_helper import generate_new_time_series
from tsaugmentation.visualization.model_visualization import plot_generated_vs_original

from tsaugmentation.preprocessing.pre_processing_datasets import (
    PreprocessDatasets as ppc,
)

from tsaugmentation import __version__


class InvalidFrequencyError(Exception):
    pass


class CreateTransformedVersionsCVAE:
    """
    Class for creating transformed versions of the dataset using a Conditional Variational Autoencoder (CVAE).

    This class contains several methods to preprocess data, fit a CVAE, generate new time series, and
    save transformed versions of the dataset. It's designed to be used with time-series data.

    The class follows the Singleton design pattern ensuring that only one instance can exist.

    Args:
        dataset_name: Name of the dataset.
        freq: Frequency of the time series data.
        input_dir: Directory where the input data is located. Defaults to "./".
        transf_data: Type of transformation applied to the data. Defaults to "whole".
        top: Number of top series to select. Defaults to None.
        window_size: Window size for the sliding window. Defaults to 10.
        weekly_m5: If True, use the M5 competition's weekly grouping. Defaults to True.
        test_size: Size of the test set. If None, the size is determined automatically. Defaults to None.
        dynamic_feat_trig: If True, apply dynamic feature transformation. Defaults to True.

        Below are parameters for the synthetic data creation:
            num_base_series_time_points: Number of base time points in the series. Defaults to 100.
            num_latent_dim: Dimension of the latent space. Defaults to 3.
            num_variants: Number of variants for the transformation. Defaults to 20.
            noise_scale: Scale of the Gaussian noise. Defaults to 0.1.
            amplitude: Amplitude of the time series data. Defaults to 1.0.
    """

    _instance = None

    def __new__(cls, *args, **kwargs) -> 'CreateTransformedVersionsCVAE':
        """
        Override the __new__ method to implement the Singleton design pattern.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        dataset_name: str,
        freq: str,
        input_dir: str = "./",
        transf_data: str = "whole",
        top: int = None,
        window_size: int = 10,
        weekly_m5: bool = True,
        test_size: int = None,
        dynamic_feat_trig: bool = True,
        num_base_series_time_points: int = 100,
        num_latent_dim: int = 3,
        num_variants: int = 20,
        noise_scale: float = 0.1,
        amplitude: float = 1.0,
    ):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.transf_data = transf_data
        self.freq = freq
        self.top = top
        self.test_size = test_size
        self.dynamic_feat_trig = dynamic_feat_trig
        self.weekly_m5 = weekly_m5
        self.num_base_series_time_points = num_base_series_time_points
        self.num_latent_dim = num_latent_dim
        self.num_variants = num_variants
        self.noise_scale = noise_scale
        self.amplitude = amplitude
        self.dataset = self._get_dataset()
        if window_size:
            self.window_size = window_size
        data = self.dataset["predict"]["data_matrix"]
        self.y = data
        self.n = data.shape[0]
        self.s = data.shape[1]
        self.n_features = self.s
        self.n_train = self.n - self.window_size + 1
        self.groups = list(self.dataset["train"]["groups_names"].keys())
        self.df = pd.DataFrame(data)
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.dataset["dates"], columns=["Date"])], axis=1
        )[: self.n]
        self.df = self.df.set_index("Date")
        self.df.asfreq(self.freq)
        self.preprocess_freq()

        self.features_input = (None, None, None)
        self._create_directories()
        self._save_original_file()

    def preprocess_freq(self):
        end_date = None

        # Create dataset with window_size more dates in the future to be used
        if self.freq in ["Q", "QS"]:
            if self.freq == "Q":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size * 3)
        elif self.freq in ["M", "MS"]:
            if self.freq == "M":
                self.freq += "S"
            end_date = self.df.index[-1] + pd.DateOffset(months=self.window_size)
        elif self.freq == "W":
            end_date = self.df.index[-1] + pd.DateOffset(weeks=self.window_size)
        elif self.freq == "D":
            end_date = self.df.index[-1] + pd.DateOffset(days=self.window_size)
        else:
            raise InvalidFrequencyError(
                f"Invalid frequency - {self.freq}. Please use one of the defined frequencies: Q, QS, M, MS, W, or D."
            )

        ix = pd.date_range(
            start=self.df.index[0],
            end=end_date,
            freq=self.freq,
        )
        self.df_generate = self.df.copy()
        self.df_generate = self.df_generate.reindex(ix)

    def _get_dataset(self):
        """
        Get dataset and apply preprocessing
        """
        ppc_args = {
            "dataset": self.dataset_name,
            "freq": self.freq,
            "input_dir": self.input_dir,
            "top": self.top,
            "test_size": self.test_size,
            "weekly_m5": self.weekly_m5,
            "num_base_series_time_points": self.num_base_series_time_points,
            "num_latent_dim": self.num_latent_dim,
            "num_variants": self.num_variants,
            "noise_scale": self.noise_scale,
            "amplitude": self.amplitude,
        }

        dataset = ppc(**ppc_args).apply_preprocess()

        return dataset

    def _create_directories(self):
        """
        Create dynamically the directories to store the data if they don't exist
        """
        # Create directory to store transformed datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}data/transformed_datasets").mkdir(
            parents=True, exist_ok=True
        )

    def _save_original_file(self):
        """
        Store original dataset
        """
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_original.npy",
            "wb",
        ) as f:
            np.save(f, self.y)

    def _save_version_file(
        self,
        y_new: np.ndarray,
        version: int,
        sample: int,
        transformation: str,
        method: str = "single_transf",
    ) -> None:
        """
        Store the transformed dataset

        :param y_new: transformed data
        :param version: version of the transformation
        :param sample: sample of the transformation
        :param transformation: name of the transformation applied
        """
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_version_{version}_{sample}samples_{method}_{transformation}_{self.transf_data}.npy",
            "wb",
        ) as f:
            np.save(f, y_new)

    def _generate_static_features(self, n: int) -> None:
        """Helper method to create the static feature and scale them

        Args:
            n: number of samples
        """
        self.static_features = create_static_features(self.groups, self.dataset)

    def _feature_engineering(
        self, n: int
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Create static and dynamic features as well as apply preprocess to raw time series

        Args:
            n: number of samples
        """
        self.X_train_raw = self.df.astype(np.float32).to_numpy()

        self.scaler_target = MinMaxScaler().fit(self.X_train_raw)
        X_train_raw_scaled = self.scaler_target.transform(self.X_train_raw)

        self._generate_static_features(n)

        if n == self.n:
            # if we want to generate new time series with the same size
            # as the original ones
            self.dynamic_features = create_dynamic_features(
                self.df_generate, self.freq, trigonometric=self.dynamic_feat_trig
            )
        else:
            self.dynamic_features = create_dynamic_features(
                self.df, self.freq, trigonometric=self.dynamic_feat_trig
            )

        X_train = temporalize(X_train_raw_scaled, self.window_size)

        self.n_features_concat = X_train.shape[1] + self.dynamic_features.shape[1]

        (
            self.dynamic_features_inp,
            X_inp,
            self.static_features_inp,
        ) = combine_inputs_to_model(
            X_train,
            self.dynamic_features,
            self.static_features,
            self.window_size,
        )

        return self.dynamic_features_inp, X_inp, self.static_features_inp

    def get_flatten_size_encoder(self):
        _ = self._feature_engineering(self.n_train)
        flatten_size = get_flatten_size_encoder(
            static_features=self.static_features,
            dynamic_features_df=self.dynamic_features,
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=self.n_features_concat,
        )

        return flatten_size

    def fit(
        self,
        epochs: int = 750,
        batch_size: int = 5,
        patience: int = 30,
        latent_dim: int = 2,
        learning_rate: float = 0.001,
        hyper_tuning: bool = False,
        load_weights: bool = True,
    ) -> tuple[CVAE, dict, EarlyStopping]:
        """
        Training our CVAE on the dataset supplied

        :param epochs: number of epochs to train the model
        :param batch_size: batch size to train the model
        :param patience: parameter for early stopping
        :param latent_dim: dimensionality of the normal dist
                -> if = 1 univariate; if = n_features full multivariate

        :return: model trained
        """
        self.features_input = self._feature_engineering(self.n_train)

        encoder, decoder = get_CVAE(
            static_features=self.features_input[2],
            dynamic_features=self.features_input[0],
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=self.n_features_concat,
            latent_dim=latent_dim,
            embedding_dim=8,
        )

        cvae = CVAE(encoder, decoder, self.window_size)
        cvae.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )

        weights_folder = "model_weights"
        os.makedirs(weights_folder, exist_ok=True)

        weights_file = os.path.join(
            weights_folder, f"{self.dataset_name}_vae_weights.h5"
        )
        history = None

        if os.path.exists(weights_file) and not hyper_tuning and load_weights:
            _ = cvae(self.features_input)
            print("Loading existing weights...")
            cvae.load_weights(weights_file)
        else:
            mc = ModelCheckpoint(
                weights_file,
                save_best_only=True,
                save_weights_only=True,
                monitor="loss",
                mode="auto",
                verbose=1,
            )

            history = cvae.fit(
                x=self.features_input,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=False,
                callbacks=[es, mc],
            )

        return cvae, history, es

    def predict(
        self, cvae: CVAE, similar_static_features: int = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict original time series using CVAE

        Args:
            cvae: cvae model
            similar_static_features: int to indicate if we use the original static feature of a
                    specific series in the dataset

        Returns:
            Tuple containing the predictions and also the z value computed by calling predict on
            the encoder

        """
        dynamic_feat, X_inp, static_feat = self.features_input

        if similar_static_features:
            sim_static_features = []
            for i in range(len(static_feat)):
                sim_static_features.append(
                    np.zeros((self.n_train, self.n_features, 1))
                    + static_feat[i][:, similar_static_features]
                )

            z_mean, z_log_var, z = cvae.encoder.predict(
                dynamic_feat + X_inp + sim_static_features
            )
        else:
            z_mean, z_log_var, z = cvae.encoder.predict(
                dynamic_feat + X_inp + static_feat
            )

        preds = cvae.decoder.predict([z] + dynamic_feat + static_feat)
        preds = detemporalize(preds, self.window_size)
        X_hat = self.scaler_target.inverse_transform(preds)

        # To train the CVAE the first points (equal to the window size) of the dataset were not predicted
        X_hat_complete = np.concatenate(
            (self.X_train_raw[: self.window_size], X_hat), axis=0
        )

        return X_hat_complete, z, z_mean, z_log_var

    def generate_transformed_time_series(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[str] = None,
        transf_param: float = 0.5,
        plot_predictions: bool = True,
        n_series_plot: int = 8,
    ) -> np.ndarray:
        """
        Generate new time series by sampling from the latent space of a Conditional Variational Autoencoder (CVAE).

        Args:
            cvae: A trained Conditional Variational Autoencoder (CVAE) model.
            z_mean: Mean parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            z_log_var: Log variance parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            transformation: Transformation to apply to the data, if any.
            transf_param: Parameter for the transformation.
            plot_predictions: If True, plots examples of generated series versus original and stores in a PDF.
            n_series_plot: Number of series to plot.

        Returns:
            A new generated dataset (time series).
        """
        self.features_input = self._feature_engineering(self.n)
        dynamic_feat, X_inp, static_feat = self.features_input

        dec_pred_hat = generate_new_time_series(
            cvae=cvae,
            z_mean=z_mean,
            z_log_var=z_log_var,
            window_size=self.window_size,
            dynamic_features_inp=dynamic_feat,
            static_features_inp=static_feat,
            scaler_target=self.scaler_target,
            n_features=self.n_features,
            n=self.n,
            transformation=transformation,
            transf_param=transf_param,
        )

        if plot_predictions:
            plot_generated_vs_original(
                dec_pred_hat=dec_pred_hat,
                X_train_raw=self.X_train_raw,
                transformation=transformation,
                transf_param=transf_param,
                dataset_name=self.dataset_name,
                n_series=n_series_plot,
                model_version=__version__,
            )
        return dec_pred_hat

    def generate_new_datasets(
        self,
        cvae: CVAE,
        z_mean: np.ndarray,
        z_log_var: np.ndarray,
        transformation: Optional[str] = None,
        transf_param: List[float] = None,
        n_versions: int = 6,
        n_samples: int = 10,
        save: bool = True,
    ) -> np.ndarray:
        """
        Generate new datasets using the CVAE trained model and different samples from its latent space.

        Args:
            cvae: A trained Conditional Variational Autoencoder (CVAE) model.
            z_mean: Mean parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            z_log_var: Log variance parameters of the latent space distribution (Gaussian). Shape: [num_samples, window_size].
            transformation: Transformation to apply to the data, if any.
            transf_param: Parameter for the transformation.
            n_versions: Number of versions of the dataset to create.
            n_samples: Number of samples of the dataset to create.
            save: If True, the generated datasets are stored locally.

        Returns:
            An array containing the new generated datasets.
        """
        if transf_param is None:
            transf_param = [0.5, 2, 4, 10, 20, 50]
        y_new = np.zeros((n_versions, n_samples, self.n, self.s))
        s = 0
        for v in range(1, n_versions + 1):
            for s in range(1, n_samples + 1):
                y_new[v - 1, s - 1] = self.generate_transformed_time_series(
                    cvae=cvae,
                    z_mean=z_mean,
                    z_log_var=z_log_var,
                    transformation=transformation,
                    transf_param=transf_param[v - 1],
                )
            if save:
                self._save_version_file(y_new[v - 1], v, s, "vae")
        return y_new
