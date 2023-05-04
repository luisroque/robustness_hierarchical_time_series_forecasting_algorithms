from tensorflow import keras
import numpy as np
from tsaugmentation.feature_engineering.feature_transformations import detemporalize


def generate_new_time_series(
    vae: keras.Model,
    init_samples_std: list[float],
    z: np.ndarray,
    window_size: int,
    dynamic_features_inp: list[np.ndarray],
    static_features_inp: list[np.ndarray],
    scaler_target: object,
    n_features: int,
    n: int,
) -> np.ndarray:
    """
    Sample first point of the series from the latent space and
    generate rest of the series using the learned decoder

    :param vae: model to use
    :param init_samples_std: standard deviation for the initial samples
    :param z: latent space mv normal dist (mean and std for each series)
    :param window_size: rolling window
    :param dynamic_features_inp: dynamic features in the format to be inputed to the RNN
    :param static_features_inp: static features in the format to be inputed to the RNN
    :param scaler_target: scaler learning on training data to be able to inverse transform
    :param n_features: number of input features
    :param n: number of time points

    :return: predictions in the shape (n - window_size, s)

    """
    # generate based on the first mean and std value for the first point of the series
    id_seq = 0

    x_mean_sample = np.random.normal(z[id_seq, :, 0], init_samples_std[0])
    x_std_sample = np.random.normal(z[id_seq, :, 1], init_samples_std[1])

    dec_pred = []

    for id_seq in range(n - window_size + 1):
        d_feat = [dy[id_seq, :].reshape(1, window_size) for dy in dynamic_features_inp]
        s_feat = [st[id_seq, :].reshape(1, n_features, 1) for st in static_features_inp]
        dec_pred.append(
            vae.decoder.predict(
                [np.asarray([[x_mean_sample, x_std_sample]]).transpose(0, 2, 1)]
                + d_feat
                + s_feat
            )
        )

    dec_pred_hat = detemporalize(np.squeeze(np.array(dec_pred)), window_size)
    dec_pred_hat = scaler_target.inverse_transform(dec_pred_hat)

    return dec_pred_hat
