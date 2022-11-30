import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def create_dynamic_features(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Create a dataframe with the dynamic features computed using trigonometric functions -> cos(year), sin(year), cos(month), sin(month).
    The features are dynamic since they change over time (over the number of samples n)

    :param df: dataframe with the raw data
    :param freq: frequency of the dataset

    :return: new dataframe with the dynamic features
    :return: dataframe converted to np.ndarray
    """

    train_df_input = pd.DataFrame()

    if (df.index.year.max() - df.index.year.min()) > 0:
        train_df_input['year_cos'] = np.cos(
            2 * np.pi * (df.index.year - df.index.year.min()) / (df.index.year.max() - df.index.year.min()))
        train_df_input['year_sin'] = np.sin(
            2 * np.pi * (df.index.year - df.index.year.min()) / (df.index.year.max() - df.index.year.min()))
    if (df.index.month.max() - df.index.month.min()) > 0:
        train_df_input['month_cos'] = np.cos(
            2 * np.pi * (df.index.month - df.index.month.min()) / (df.index.month.max() - df.index.month.min()))
        train_df_input['month_sin'] = np.sin(
            2 * np.pi * (df.index.month - df.index.month.min()) / (df.index.month.max() - df.index.month.min()))

    # Only create the features below when the frequency is high enough (e.g. weekly or daily)
    if (df.index.week.max() - df.index.week.min()) > 0 and freq not in ('QS', 'MS'):
        train_df_input['week_cos'] = np.cos(
            2 * np.pi * (df.index.week - df.index.week.min()) / (df.index.week.max() - df.index.week.min()))
        train_df_input['week_sin'] = np.sin(
            2 * np.pi * (df.index.week - df.index.week.min()) / (df.index.week.max() - df.index.week.min()))
    if (df.index.day.max() - df.index.day.min()) > 0 and freq not in ('QS', 'MS', 'W'):
        train_df_input['day_cos'] = np.cos(
            2 * np.pi * (df.index.day - df.index.day.min()) / (df.index.day.max() - df.index.day.min()))
        train_df_input['day_sin'] = np.sin(
            2 * np.pi * (df.index.day - df.index.day.min()) / (df.index.day.max() - df.index.day.min()))

    return train_df_input.astype(np.float32)


def plot_dynamic_features(df_dynamic: pd.DataFrame) -> None:
    """
    Help function to plot the dynamic features

    :param df_dynamic: dataframe containing the dynamic features
    """
    _, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax = ax.ravel()

    ax[0].plot(df_dynamic.index, df_dynamic.iloc[:, :2])
    ax[1].plot(df_dynamic.index, df_dynamic.iloc[:, 2:])
    plt.show()
