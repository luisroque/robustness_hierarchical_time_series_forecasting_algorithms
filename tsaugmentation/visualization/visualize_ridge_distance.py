import numpy as np
import pandas as pd
from joypy import joyplot
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def build_df_ridge(
    d_transf: np.ndarray,
    d_orig: np.ndarray,
    n_d: int,
    transformations: list,
    versions: int,
) -> pd.DataFrame:
    """Build dataframe for the ridge plot

    Args:
        d_transf: distances between series for the transformed versions
        d_orig: distances between series for the original version
        n_d: number of combinations of time series in the dataset
        transformations: set of transformations appled to this dataset
        versions: number of versions created of the dataset

    Returns:
        Dataframe with the structure necessary for the ridge plot API

    """

    df = pd.DataFrame(
        {
            "transformation": np.repeat(
                np.arange(len(transformations)), versions * n_d
            ),
            "version": np.tile(
                np.repeat(np.arange(versions, dtype=np.int32), n_d),
                len(transformations),
            ),
            "dist": np.squeeze(d_transf.reshape(-1).astype("float64")),
        }
    )

    df = df.sort_values(by=["version", "transformation"]).reset_index(drop=True)

    df_dummy = pd.DataFrame(
        {"dummy": np.tile(np.arange(n_d * len(transformations)), versions)}
    )

    df = pd.concat((df, df_dummy), axis=1)

    df["index"] = df["transformation"].astype("str") + df["dummy"].astype("str")
    df["index"] = df["index"].astype("int")

    df = df.pivot_table(
        index="index", columns="version", values="dist", aggfunc=np.mean
    )
    df = df.reset_index(drop=True)

    df_dummy = pd.DataFrame({"transformation": np.repeat(transformations, n_d)})
    df = pd.concat((df, df_dummy), axis=1)

    # add original series
    df["orig"] = np.tile(d_orig, len(transformations))

    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df[cols]

    return df


def store_df_distances(
    df: pd.DataFrame, dataset_name: str, directory: str = "."
) -> None:
    df.to_pickle(f"{directory}/{dataset_name}_distances_transformed.pkl")


def load_df_distances(dataset_name: str, directory: str = ".") -> pd.DataFrame:
    df = pd.read_pickle(f"{directory}/{dataset_name}_distances_transformed.pkl")
    return df


def load_distances(dataset_name: str, directory: str = "."):
    with open(f"{directory}/{dataset_name}_distances_transformed.npy", "rb") as f:
        d_transf = np.load(f, allow_pickle=True)
    with open(f"{directory}/{dataset_name}_distances_original.npy", "rb") as f:
        d_orig = np.load(f, allow_pickle=True)
    return d_transf, d_orig


def plot_distances(
    dataset_name: str,
    df_rige: pd.DataFrame,
    versions: int,
    x_range: list[int] = None,
) -> None:
    """Plot the distances for a specific dataset

    Args:
        dataset_name: name of the dataset
        df_rige: df with data in the correct shape for ridge plots
        versions: number of created versions of the dataset
        x_range: range to plot in the x_axis

    Returns:
        Dataframe containing the distances for each version and transformation
    """
    if x_range is None:
        x_range = [6, 18]
    plt.figure()
    # map versions to a 0-1 interval to sample different blue colors
    m = interp1d([0, versions], [0, 0.8])

    ax, fig = joyplot(
        data=df_rige,
        by="transformation",
        legend=True,
        alpha=1,
        figsize=(7, 7),
        x_range=x_range,
        fill=False,
        overlap=5,
        color=["darkorange"] + [cm.Blues_r(m(i)) for i in range(versions)],
    )
    plt.title(
        f"Ridgeline Plot for {dataset_name} of the DTW distance per version and transformation",
        fontsize=20,
    )
    plt.show()
