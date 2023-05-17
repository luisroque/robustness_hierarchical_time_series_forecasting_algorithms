from tensorflow import keras
import numpy as np
from tsaugmentation.feature_engineering.feature_transformations import detemporalize


def generate_new_time_series(
    cvae: keras.Model,
    z_mean: np.ndarray,
    z_log_var: np.ndarray,
    window_size: int,
    dynamic_features_inp: list[np.ndarray],
    static_features_inp: list[np.ndarray],
    scaler_target: object,
    n_features: int,
    n: int,
    init_samples_std: float = None,
) -> np.ndarray:
    """
    This function generates a new time series using a trained Variational Autoencoder (VAE) model.
    It samples the first point of the series from the latent space and generates the rest of the series
    using the learned decoder.

    Args:
        cvae (keras.Model): Trained VAE model to use for generation.
        z_mean (np.ndarray): Mean of the latent space multivariate normal distribution for each series.
        z_log_var (np.ndarray): Logarithm of the variance of the latent space multivariate normal distribution for each series.
        window_size (int): Size of the rolling window.
        dynamic_features_inp (list[np.ndarray]): Dynamic features in the format to be inputted to the RNN.
        static_features_inp (list[np.ndarray]): Static features in the format to be inputted to the RNN.
        scaler_target (object): Scaler trained on training data to be able to perform inverse transformations.
        n_features (int): Number of input features.
        n (int): Number of time points in the series to be generated.
        init_samples_std (float): Standard deviation for the initial samples.

    Returns:
        np.ndarray: Predicted series in the shape (n - window_size, n_features).
    """
    latent_dim = z_mean.shape[-1]  # Get the latent dimension from the z_mean shape

    if init_samples_std is not None:
        z_std = np.tile(np.array(init_samples_std), z_log_var.shape)
    else:
        z_std = np.exp(z_log_var * 0.5)

    dec_pred = []

    for id_seq in range(n - window_size + 1):
        # Sample from a multivariate normal distribution for each series
        z_sample = np.random.normal(z_mean[id_seq], z_std[id_seq], size=(latent_dim, ))

        d_feat = [dy[id_seq, :].reshape(1, window_size) for dy in dynamic_features_inp]
        s_feat = [st[id_seq, :].reshape(1, n_features, 1) for st in static_features_inp]
        dec_pred.append(
            cvae.decoder.predict(
                [z_sample.reshape(1, latent_dim)]
                + d_feat
                + s_feat
            )
        )

    dec_pred_hat = detemporalize(np.squeeze(np.array(dec_pred)), window_size)
    dec_pred_hat = scaler_target.inverse_transform(dec_pred_hat)

    return dec_pred_hat

