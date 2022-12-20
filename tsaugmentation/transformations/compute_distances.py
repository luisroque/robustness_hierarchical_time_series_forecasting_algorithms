import numpy as np
from dtaidistance import dtw
from sklearn.preprocessing import StandardScaler


def compute_distances(
    orig_scaled: np.ndarray,
    transf: np.ndarray,
    s: int,
    transformations: list,
    versions: int,
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute distances between each series in a dataset

    Args:
        orig_scaled: original data scaled
        transf: array with original and transformed data
        s: number of time series
        transformations: set of transformations applied to this dataset
        versions: number of different versions stored of this dataset
        window: Only allow for shifts up to this amount away from the two diagonals

    Returns:
        Array with the distances between series
    """

    d_orig = np.array(
        dtw.distance_matrix_fast(
            orig_scaled.T, compact=True, only_triu=True, window=window
        )
    )

    # Scaling the data and compute the DTW distance
    d_transf = np.zeros((len(transformations), versions, int(s * (s - 1) / 2)))

    for t in range(len(transformations)):
        for v in range(versions):
            scaler = StandardScaler()
            scaler.fit(transf[t][v][0])
            transf_scaled = scaler.transform(transf[t][v][0])

            d_transf[t][v] = np.array(
                dtw.distance_matrix_fast(
                    transf_scaled.T, compact=True, only_triu=True, window=window
                )
            )

    return d_orig, d_transf


def compute_store_distances(
    dataset_name: str,
    orig: np.ndarray,
    transf: np.ndarray,
    transformations: list[str],
    versions: int,
    directory: str = ".",
    window: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute DTW for each time series in the dataset

    Args:
        dataset_name: name of the dataset
        orig: array with original data
        transf: array with transformed data
        transformations: list of transformations
        versions: number of versions that exist of the dataset
        directory: local directory to store the file
        window: Only allow for shifts up to this amount away from the two diagonals

    Returns:
        Arrays containing respectively the original distances and the transformed ones
    """
    # Scaling the data
    scaler = StandardScaler()
    scaler.fit(orig)
    orig_scaled = scaler.transform(orig)

    # Getting parameters n_samples and n_series
    s = transf.shape[4]

    d_orig, d_transf = compute_distances(
        orig_scaled, transf, s, transformations, versions, window
    )

    with open(f"{directory}/{dataset_name}_distances_transformed.npy", "wb") as f:
        np.save(f, d_transf, allow_pickle=True)
    with open(f"{directory}/{dataset_name}_distances_original.npy", "wb") as f:
        np.save(f, d_orig, allow_pickle=True)

    return d_orig, d_transf
