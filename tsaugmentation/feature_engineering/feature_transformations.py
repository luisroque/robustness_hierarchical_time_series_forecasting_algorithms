import numpy as np
import pandas as pd


def temporalize(data: np.ndarray, window_size: int) -> np.ndarray:
    """
    Transforming the data to the following shape using a rolling window:
    from (n, s) to (n-window_size+1, window_size, s)

    :param data: input data to transform
    :param window_size: input window to consider on the transformation

    :return X: ndarray of the transformed features
    :return Y: ndarray of the transformed labels
    """

    X = []
    for i in range(len(data) - window_size + 1):
        row = [r for r in data[i : i + window_size]]
        X.append(row)
    return np.array(X)


def detemporalize(data: np.ndarray) -> np.ndarray:
    """
    Transform the data back to the original shape

    :param data: data to back transform
    :param window_size: rolling window

    :return: data in the original shape
    """
    return np.array([x[0] for x in data])


def combine_inputs_to_model(
    X_train: np.ndarray,
    dynamic_features: pd.DataFrame,
    static_features_scaled: dict,
    window_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Combining the input features to the model: dynamic features, raw time series data and static features

    :param X_train: raw time series data
    :param dynamic_features: dynamic features already processed
    :param static_features: static features already processed
    :param window_size: rolling window

    :return: dynamic features ready to be inputed by the model
    :return: raw time series features ready to be inputed by the model
    :return: static features ready to be inputed by the model

    """

    X_dyn = temporalize(dynamic_features.to_numpy(), window_size)

    dynamic_features_inp, X_inp, static_features_inp = (
        [X_dyn[:, :, i] for i in range(len(dynamic_features.columns))],
        [X_train],
        [
            np.expand_dims(group_array, axis=2)
            for group, group_array in static_features_scaled.items()
        ],
    )

    return dynamic_features_inp, X_inp, static_features_inp
