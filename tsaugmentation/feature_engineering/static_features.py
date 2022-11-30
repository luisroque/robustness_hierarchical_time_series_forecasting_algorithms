import numpy as np
from sklearn.preprocessing import MinMaxScaler


def get_group_features(feature: str, dataset: dict) -> np.ndarray:
    """
    Get an array for a group feature of shape (n, s). Note that the static features are defined by serie(s)
    so they are repeated for dim n

    :param feature: feature to select from the group static information
    :param dataset: dataset full info

    :return: np.darray with shape (n,s)
    """

    return dataset['train']['groups_idx'][feature]


def create_static_features(window_size: int, groups: list, dataset: dict, n: int) -> dict:
    """
    Create an object with key group and value (n,s) ndarray encoding the static features.
    Note that the features are only different for each series (dimension s) and are
    then tiled for the other dimension (samples).

    Note that it returns shorter time series, a number of samples equal to the window size
    is removed from all the time series to ensure that the shapes match.

    :param window_size: size of the rolling window used to transform the dataset to be
        fit the requirements of a RNN
    :param groups: groups in the data
    :param dataset: dataset full info
    :param n: number of samples

    :return: object with key group and value ndarray (n,s) encoding the static features
    """
    groups_arrays = {}

    for group in groups:
        groups_arrays[group] = {}
        groups_arrays[group] = get_group_features(group, dataset)
        groups_arrays[group] = np.tile(groups_arrays[group].reshape(-1, 1), n).T

    return groups_arrays


def scale_static_features(groups_arrays: dict) -> dict:
    """
    Scale the static features using MinMax

    :param groups_arrays: object with key group and value ndarray (n,s) encoding the static features

    :return: object with key group and value ndarray (n,s) encoding the scaled static features
    """
    for group, arr in groups_arrays.items():
        scaler = MinMaxScaler().fit(arr.T)
        scaled_data = scaler.transform(arr.T)
        groups_arrays[group] = scaled_data.T

    return groups_arrays
