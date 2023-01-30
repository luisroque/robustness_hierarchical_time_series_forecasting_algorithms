from typing import List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_generated_vs_original(
    dec_pred_hat: np.ndarray, X_train_raw: np.ndarray
) -> None:
    _, ax = plt.subplots(4, 2, figsize=(18, 10))
    ax = ax.ravel()
    for i in range(8):
        ax[i].plot(dec_pred_hat[:, i], label="new sample")
        ax[i].plot(X_train_raw[:, i], label="orig")
    plt.legend()
    plt.show()


class DataTransform:
    def __init__(self, groups):
        self.g = groups
        self.mu_data = np.mean(self.g["train"]["data"], axis=0)
        self.std_data = np.std(self.g["train"]["data"], axis=0)

    def std_transf_train(self):
        self.g["train"]["data"] = (
            self.g["train"]["data"] - self.mu_data
        ) / self.std_data
        return self.g

    def inv_transf_train(self):
        self.g["train"]["data"] = (
            self.g["train"]["data"] * self.std_data
        ) + self.mu_data
        return self.g

    def inv_transf_train_general(self, pred):
        pred_samples = pred.shape[0]
        pred = (
            (pred.reshape(-1, self.g["train"]["s"]) * self.std_data) + self.mu_data
        ).reshape(pred_samples, self.g["train"]["n"], self.g["train"]["s"])
        return pred

    def inv_transf_predict_general(self, pred):
        pred_samples = pred.shape[0]
        pred = (
            (pred.reshape(-1, self.g["predict"]["s"]) * self.std_data) + self.mu_data
        ).reshape(pred_samples, self.g["predict"]["n"], self.g["predict"]["s"])
        return pred


def sample_data(y: pd.DataFrame, dates: List, sample_perc: float, h: int):
    """
    Samples data from the original dataframe and dates

    Args:
        y: dataframe with the original dataset
        dates: original dates
        sample_perc: percentage of the dataset to sample from the original
    """
    y_reindex = y.reset_index()
    # we want to keep the first and last point of the train data
    # we also only want to sample the training and not the test data
    y_reindex_copy = y_reindex.copy()
    y_reindex_copy = y_reindex_copy[~y_reindex_copy.index.isin(y_reindex.iloc[-h - 1:].index)]
    y_reindex_copy = y_reindex_copy[~y_reindex_copy.index.isin(y_reindex.iloc[0].index)]
    sampled_df = y_reindex_copy.sample(frac=sample_perc)
    sampled_with_ends = pd.concat([y_reindex[:1], y_reindex.iloc[-h-1:], sampled_df], ignore_index=True)

    sampled_with_ends_date = sampled_with_ends.set_index("Date")
    sample_index = sampled_with_ends_date.index
    y_reindex_date = y_reindex.set_index('Date')
    y_sample = y_reindex_date[y_reindex_date.index.isin(sample_index)]
    original_index = y_reindex_date.index
    sample_index = y_sample.index
    difference_index = original_index.intersection(sample_index)

    filtered_dates = []
    x_values = []
    for i in range(len(y_reindex)):
        if y_reindex_date.index[i] in difference_index:
            filtered_dates.append(dates[i])
            x_values.append(i)

    return y_sample, filtered_dates, x_values


def generate_groups_data_flat(y, dates, groups_input, seasonality, h, sample_perc=0.5):
    """
    It works for two kinds of structures:
        1) The name of the columns have specific length for specific groups
        2) There is a multiIndex column structure for each group
    """
    if sample_perc:
        y, dates, x_values = sample_data(y, dates, sample_perc=sample_perc, h=h)

    groups = {}

    for i in ["train", "predict"]:
        groups[i] = {}
        if i == "train":
            y_ = y.iloc[:-h, :]
            if sample_perc:
                groups[i]["x_values"] = x_values[:-h]
            else:
                groups[i]["x_values"] = list(np.arange(y_.shape[0]))
        else:
            y_ = y
            if sample_perc:
                groups[i]["x_values"] = x_values
            else:
                groups[i]["x_values"] = list(np.arange(y_.shape[0]))
        groups[i]["groups_idx"] = {}
        groups[i]["groups_n"] = {}
        groups[i]["groups_names"] = {}

        groups[i]["n"] = y_.shape[0]
        groups[i]["s"] = y_.shape[1]

        # Test if we are receiving format 1) or 2)
        if len(next(iter(groups_input.values()))) == 1:
            for g in groups_input:
                group_idx = pd.get_dummies(
                    [i[groups_input[g][0]] for i in y_]
                ).values.argmax(1)
                groups[i]["groups_idx"][g] = np.tile(
                    group_idx, (groups[i]["n"], 1)
                ).flatten("F")
                groups[i]["groups_n"][g] = np.unique(group_idx).shape[0]
                group_names = [i[groups_input[g][0]] for i in y_]
                groups[i]["groups_names"][g] = np.unique(group_names)
        else:
            for g in groups_input:
                group_idx = pd.get_dummies(
                    [i[groups_input[g][0] : groups_input[g][1]] for i in y_]
                ).values.argmax(1)
                groups[i]["groups_idx"][g] = np.tile(
                    group_idx, (groups[i]["n"], 1)
                ).flatten("F")
                groups[i]["groups_n"][g] = np.unique(group_idx).shape[0]
                group_names = [i[groups_input[g][0] : groups_input[g][1]] for i in y_]
                groups[i]["groups_names"][g] = np.unique(group_names)

        groups[i]["n_series_idx"] = np.tile(
            np.arange(groups[i]["s"]), (groups[i]["n"], 1)
        ).flatten("F")
        groups[i]["n_series"] = np.arange(groups[i]["s"])

        groups[i]["g_number"] = len(groups_input)

        groups[i]["data"] = y_.values.T.ravel()

    groups["seasonality"] = seasonality
    groups["h"] = h
    groups["dates"] = dates

    print("Number of groups: " + str(len(groups["train"]["groups_names"])))
    for name, group in groups["train"]["groups_names"].items():
        print("\t" + str(name) + ": " + str(len(group)))
    print("Total number of series: " + str(groups["train"]["s"]))
    print("Number of points per series for train: " + str(groups["train"]["n"]))
    print("Total number of points: " + str(groups["predict"]["n"]))
    print("Seasonality: " + str(seasonality))
    print("Forecast horizon: " + str(h))

    return groups


def generate_groups_data_matrix(groups):

    for group in groups["train"]["groups_idx"].keys():
        groups["train"]["groups_idx"][group] = (
            groups["train"]["groups_idx"][group]
            .reshape(groups["train"]["s"], groups["train"]["n"])
            .T[0, :]
        )
        groups["predict"]["groups_idx"][group] = (
            groups["predict"]["groups_idx"][group]
            .reshape(groups["predict"]["s"], groups["predict"]["n"])
            .T[0, :]
        )

    groups["train"]["full_data"] = (
        groups["train"]["data"].reshape(groups["train"]["s"], groups["train"]["n"]).T
    )
    groups["train"]["data"] = (
        groups["train"]["data"].reshape(groups["train"]["s"], groups["train"]["n"]).T
    )

    groups["train"]["n_series_idx_full"] = (
        groups["train"]["n_series_idx"]
        .reshape(groups["train"]["s"], groups["train"]["n"])
        .T[0, :]
    )
    groups["train"]["n_series_idx"] = (
        groups["train"]["n_series_idx"]
        .reshape(groups["train"]["s"], groups["train"]["n"])
        .T[0, :]
    )

    groups["predict"]["n_series_idx"] = (
        groups["predict"]["n_series_idx"]
        .reshape(groups["predict"]["s"], groups["predict"]["n"])
        .T[0, :]
    )
    groups["predict"]["data_matrix"] = (
        groups["predict"]["data"]
        .reshape(groups["predict"]["s"], groups["predict"]["n"])
        .T
    )

    return groups
