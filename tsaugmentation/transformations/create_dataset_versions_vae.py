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
from sklearn.preprocessing import MinMaxScaler
from tsaugmentation.model.models import VAE, get_mv_model
from ..preprocessing.pre_processing_datasets import PreprocessDatasets as ppc
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.callbacks import EarlyStopping
from tsaugmentation.visualization.model_visualization import plot_generated_vs_original


class CreateTransformedVersionsVAE:
    """
    A class used to create new datasets from an original one using a VAE

    Attributes
    ----------
    dataset : the original dataset to consider
    freq: frequency of the dataset
    rel_dir : relative directory where to store the downloaded files (e.g. './' current dir, '../' parent dir)
    transf_data: what data to transform: only training data 'train' or the whole dataset 'whole'
    """

    def __init__(
        self,
        dataset_name: str,
        freq: str,
        input_dir: str = "./",
        transf_data: str = "whole",
    ):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.transf_data = transf_data
        self.freq = freq
        self.dataset = self._get_dataset()
        self.window_size = 10
        self.latent_dim = 2
        data = self.dataset["predict"]["data_matrix"]
        self.y = data
        self.n = data.shape[0]
        self.s = data.shape[1]
        self.n_features = self.s
        self.n_train = self.n - self.window_size
        self.groups = list(self.dataset["train"]["groups_names"].keys())
        self.df = pd.DataFrame(data)
        self.df = pd.concat(
            [self.df, pd.DataFrame(self.dataset["dates"], columns=["Date"])], axis=1
        )[: self.n]
        self.df = self.df.set_index("Date")
        self.df.asfreq(self.freq)

        # Create dataset with window_size more dates in the past to be used
        if self.freq == 'Q':
            start_date = self.df.index[0] - pd.DateOffset(months=10*3)
        elif self.freq == 'M':
            start_date = self.df.index[0] - pd.DateOffset(months=10)
        elif self.freq == 'W':
            start_date = self.df.index[0] - pd.DateOffset(weeks=10)
        elif self.freq == 'D':
            start_date = self.df.index[0] - pd.DateOffset(days=10)
        ix = pd.date_range(
            start=start_date,
            end=self.df.index[-1],
            freq=self.freq + "S",  # S ensures that we are at the start of the period
        )
        self.df_generate = self.df.copy()
        self.df_generate = self.df_generate.reindex(ix)

        self.features_input = (None, None, None)
        self._create_directories()
        self._save_original_file()

    def _get_dataset(self):
        """
        Get dataset and apply preprocessing
        """
        return ppc(self.dataset_name).apply_preprocess()

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
        static_features = create_static_features(
            self.window_size, self.groups, self.dataset, n
        )
        self.static_features_scaled = scale_static_features(static_features)

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
            self.dynamic_features = create_dynamic_features(self.df_generate)
        else:
            self.dynamic_features = create_dynamic_features(self.df)

        X_train, y_train = temporalize(X_train_raw_scaled, self.window_size)

        self.n_features_concat = X_train.shape[1] + self.dynamic_features.shape[1]

        (
            self.dynamic_features_inp,
            X_inp,
            self.static_features_inp,
        ) = combine_inputs_to_model(
            X_train,
            self.dynamic_features,
            self.static_features_scaled,
            self.window_size,
        )

        return self.dynamic_features_inp, X_inp, self.static_features_inp

    def fit(
        self,
        epochs: int = 750,
        batch_size: int = 5,
        patience: int = 30,
        mv_normal_dim: int = None,
    ) -> tuple[VAE, dict]:
        """
        Training our VAE on the dataset supplied

        :param epochs: number of epochs to train the model
        :param batch_size: batch size to train the model
        :param patience: parameter for early stopping
        :param mv_normal_dim: dimensionality of the normal dist
                -> if = 1 univariate; if = n_features full multivariate

        :return: model trained
        """
        if not mv_normal_dim:
            mv_normal_dim = self.n_features
        self.features_input = self._feature_engineering(self.n_train)

        encoder, decoder = get_mv_model(
            mv_normal_dim=mv_normal_dim,
            static_features=self.static_features_scaled,
            dynamic_features_df=self.dynamic_features,
            window_size=self.window_size,
            n_features=self.n_features,
            n_features_concat=self.n_features_concat,
            latent_dim=self.latent_dim,
        )

        vae = VAE(encoder, decoder, self.window_size)
        vae.compile(optimizer=keras.optimizers.Adam())
        es = EarlyStopping(
            patience=patience,
            verbose=1,
            monitor="loss",
            mode="auto",
            restore_best_weights=True,
        )

        history = vae.fit(
            x=self.features_input,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            callbacks=[es],
        )

        return vae, history

    def predict(
        self, vae: VAE, similar_static_features: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict original time series using VAE, note that by default we will use the middle value
         (0.5 since we have a MinMax scaling) for the static features so that we are sampling using
         the most similar structure possible and so reduce the distance between the series

        Args:
            vae: vae model
            similar_static_features: bool to indicate if we use the original static feature of a
                    middle value to increase the similarity between the series generated

        Returns:
            Tuple containing the predictions and also the z value computed by calling predict on
            the encoder

        """
        dynamic_feat, X_inp, static_feat = self.features_input

        if similar_static_features:
            sim_static_features = []
            for i in range(len(static_feat)):
                sim_static_features.append(
                    0.5 * np.ones((self.n_train, self.n_features, 1))
                )

            _, _, z = vae.encoder.predict(dynamic_feat + X_inp + sim_static_features)
        else:
            _, _, z = vae.encoder.predict(dynamic_feat + X_inp + static_feat)

        preds = vae.decoder.predict([z] + dynamic_feat + static_feat)
        preds = detemporalize(preds)
        X_hat = self.scaler_target.inverse_transform(preds)

        # To train the VAE the first points (equal to the window size) of the dataset were not predicted
        X_hat_complete = np.concatenate((self.X_train_raw[:10], X_hat), axis=0)

        return X_hat_complete, z

    def generate_transformed_time_series(
        self,
        vae: VAE,
        z: np.ndarray,
        std_latent_space: float,
        plot_predictions: bool = True,
        n_series_plot: int = 8,
    ) -> np.ndarray:
        """
        Generate new time series by sampling from the latent space

        Args:
            vae: trained model
            z: parameters of the latent space distribution (gaussian) of shape
                        [num_samples, window_size, param] where param is 0 (mean) or 1 (std)
            std_latent_space: standard deviation to use when sampling from the learned distributions
                        e.g. x_mean_sample = np.random.normal(z[id_seq, :, 0], std_latent_space[0])
            plot_predictions: plot some examples of generated series vs original and store in pdf
            n_series_plot: number of series to plot

        Returns:
            new generated dataset
        """
        self.features_input = self._feature_engineering(self.n)
        dynamic_feat, X_inp, static_feat = self.features_input

        dec_pred_hat = generate_new_time_series(
            vae,
            [std_latent_space, std_latent_space],
            z,
            self.window_size,
            dynamic_feat,
            static_feat,
            self.scaler_target,
            self.n_features,
            self.n,
        )

        if plot_predictions:
            plot_generated_vs_original(
                dec_pred_hat,
                self.X_train_raw,
                std_latent_space,
                self.dataset_name,
                n_series_plot,
            )
        return dec_pred_hat

    def generate_new_datasets(
        self,
        vae: VAE,
        z: np.ndarray,
        std_latent_space: list[float],
        n_versions: int = 6,
        n_samples: int = 10,
        save: str = True,
    ) -> np.ndarray:
        """
        Generate new datasets using the VAE trained model and different samples from its latent space

        :param vae: model
        :param z: parameters of the latent space distribution (gaussian) of shape
                    [num_samples, window_size, param] where param is 0 (mean) or 1 (std)
        :param std_latent_space: list of standard deviations to use when sampling from the learned distributions
        :param n_versions: number of versions of the dataset to create
        :param n_samples: number of samples of the dataset to create
        :param save: if true the datasets are stored locally
        """
        y_new = np.zeros((n_versions, n_samples, self.n, self.s))
        s = 0
        for v in range(1, n_versions + 1):
            for s in range(1, n_samples + 1):
                y_new[v - 1, s - 1] = self.generate_transformed_time_series(
                    vae, z, std_latent_space[v - 1]
                )
            if save:
                self._save_version_file(y_new[v - 1], v, s, "vae")
        return y_new
