from typing import List, Optional
from tensorflow import keras
import numpy as np
from tsaugmentation.feature_engineering.feature_transformations import detemporalize
from tsaugmentation.transformations.manipulate_data import ManipulateData


def generate_new_time_series(
    cvae: keras.Model,
    z_mean: np.ndarray,
    z_log_var: np.ndarray,
    window_size: int,
    dynamic_features_inp: List[np.ndarray],
    static_features_inp: List[np.ndarray],
    scaler_target: object,
    n_features: int,
    n: int,
    transformation: Optional[str] = None,
    transf_param: float = 0.5,
) -> np.ndarray:
    """
    Generate a new time series using a trained Conditional Variational Autoencoder (CVAE) model.

    This function starts by sampling the first point of the series from the latent space. Then,
    it generates the rest of the series using the trained CVAE model. If a transformation is specified,
    it is applied to the standard deviation of the latent space before sampling.

    Args:
        cvae (keras.Model): A trained CVAE model to use for series generation.
        z_mean (np.ndarray): Mean of the latent space Gaussian distribution for each series.
        z_log_var (np.ndarray): Logarithm of the variance of the latent space Gaussian distribution for each series.
        window_size (int): Size of the rolling window used in the series generation.
        dynamic_features_inp (List[np.ndarray]): List of dynamic features to be inputted to the model.
        static_features_inp (List[np.ndarray]): List of static features to be inputted to the model.
        scaler_target (object): A scaler object, trained on the training data, used for inverse transformations.
        n_features (int): Number of input features.
        n (int): Number of time points in the series to be generated.
        transformation (str, optional): Name of the transformation to apply to the standard deviation of the latent space.
                                        If None, no transformation is applied.
        transf_param (float, optional): Parameter for the transformation. Default is 0.5.

    Returns:
        np.ndarray: Generated series of shape (n - window_size, n_features).
    """
    latent_dim = z_mean.shape[-1]  # Get the latent dimension from the z_mean shape
    z_std = np.exp(z_log_var * 0.5)

    if transformation is not None:
        z_std = np.clip(
            ManipulateData(
                x=z_std, transformation=transformation, parameters=[transf_param]
            ).apply_transf(),
            0,
            None,
        )

    dec_pred = []

    for id_seq in range(n - window_size + 1):
        # Sample from a multivariate normal distribution for each series
        z_sample = np.random.normal(z_mean[id_seq], z_std[id_seq], size=(latent_dim,))

        d_feat = [dy[id_seq, :].reshape(1, window_size) for dy in dynamic_features_inp]
        s_feat = [st[id_seq, :].reshape(1, n_features, 1) for st in static_features_inp]
        dec_pred.append(
            cvae.decoder.predict([z_sample.reshape(1, latent_dim)] + d_feat + s_feat)
        )

    dec_pred_hat = detemporalize(np.squeeze(np.array(dec_pred)), window_size)
    dec_pred_hat = scaler_target.inverse_transform(dec_pred_hat)

    return dec_pred_hat
