import numpy as np


def get_data(
    local_dir_orig: str,
    local_dir_transf: str,
    dataset_name: str,
    transformations: list,
    versions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Loads the original dataset and the transformed versions from files

    Args:
        local_dir_orig: local dir where the original data is stored
        local_dir_transf: local dir where the transformed data is stored
        dataset_name: name of the dataset to load
        transformations: set of transformations applied to this dataset
        versions: number of versions

    Returns:
        Tuple with original data and an array of shape (number_transformations + 1, n_versions, n_samples_versions, n_samples, n_series)
        with the original + transformed data

    """
    orig = np.load(f"{local_dir_orig}/{dataset_name}_original.npy")
    transf = []
    for t in transformations:
        transf_ = []
        for v in range(1, versions + 1):
            transf_.append(
                np.load(
                    f"{local_dir_transf}/{dataset_name}_version_{v}_10samples_single_transf_{t}_whole.npy"
                )
            )
        transf.append(transf_)
    return orig, np.array(transf)
