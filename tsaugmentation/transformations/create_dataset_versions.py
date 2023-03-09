from ..preprocessing.pre_processing_datasets import PreprocessDatasets as ppc
from tsaugmentation.transformations.manipulate_data import ManipulateData
import numpy as np
from tsaugmentation.visualization.visualize_transformed_datasets import Visualizer
from .compute_similarities_summary_metrics import ComputeSimilaritiesSummaryMetrics
from pathlib import Path


class CreateTransformedVersions:
    """
    A class used to create new datasets from an original one using time series augmentation techniques

    Attributes
    ----------
    dataset : str
        the original dataset to consider
    rel_dir : str
        relative directory where to store the downloaded files (e.g. './' current dir, '../' parent dir)
    transf_data: str
        what data to transform: only training data 'train' or the whole dataset 'whole'
    """

    def __init__(self, dataset_name, input_dir="./", transf_data="whole"):
        self.dataset_name = dataset_name
        self.input_dir = input_dir
        self.transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
        self.transformations_w_random = self.transformations.copy()
        self.transformations_w_random.insert(0, "random")
        self.parameters = {
            "jitter": 0.5,
            "scaling": 0.02,
            "magnitude_warp": 0.02,
            "time_warp": 0.02,
        }
        self.data = self._get_dataset()
        self.s = self.data["train"]["s"]  # number of series in dataset
        self.n_samples = 10

        self.transf_data = transf_data
        if transf_data == "train":
            self.y = self.data["train"]["data"]
            self.n = self.data["train"][
                "n"
            ]  # number of points in each series for the training dataset
        else:
            self.y = self.data["predict"]["data_matrix"]
            self.n = self.data["predict"][
                "n"
            ]  # number of points in each series for the whole dataset

        self.groups_idx = self.data["train"]["groups_idx"]
        self._create_directories()
        self._save_original_file()
        self.n_versions = 6
        self.visualizer = Visualizer(
            dataset=self.dataset_name,
            n_versions=self.n_versions,
            n_series=6,
            transf_data=self.transf_data,
        )
        self.y_new_all = np.zeros(
            (len(self.transformations), self.n_versions, self.n_samples, self.n, self.s)
        )
        self.transfs = np.tile(
            np.array(self.transformations).reshape(-1, 1),
            (self.n_versions, self.n_samples, 1, self.s),
        ).transpose(2, 0, 1, 3)

        self.y_loaded_original = None
        self.y_loaded_transformed = None

    def _create_directories(self):
        # Create directory to store transformed datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}data/transformed_datasets").mkdir(
            parents=True, exist_ok=True
        )

    def _save_original_file(self):
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_original.npy",
            "wb",
        ) as f:
            np.save(f, self.y)

    def _save_version_file(self, y_new, version, sample, transformation, method):
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_version_{version}_{sample}samples_{method}_{transformation}_{self.transf_data}.npy",
            "wb",
        ) as f:
            np.save(f, y_new)

    def _get_dataset(self):
        return ppc(self.dataset_name).apply_preprocess()

    def _visualize_transf_series(self, transf, method):
        self.visualizer.visualize_series_transf(transf=transf, method=method)

    def _get_parameters_map(self, transfs):
        params = np.vectorize(self.parameters.get, otypes=["float"])(transfs)
        return params

    def _create_new_version(self, method, n_versions=None, save=True):
        if not n_versions:
            n_versions = self.n_versions
        y_new = np.zeros(
            (len(self.transformations), self.n_versions, self.n_samples, self.n, self.s)
        )
        params = self._get_parameters_map(self.transformations)
        j = 0
        for k in range(len(self.transformations)):
            # Create versions and samples per transformation
            for i in range(1, n_versions + 1):
                # Create 6 different versions of a dataset
                for j in range(1, self.n_samples + 1):
                    # Create 10 samples of each version
                    y_new[k, i - 1, j - 1] = ManipulateData(
                        self.y, self.transformations[k], parameters=params * i
                    ).apply_transf()
                if save:
                    self._save_version_file(
                        y_new[k, i - 1], i, j, self.transformations[k], method
                    )
        return y_new

    def read_groups_transformed(self, method):
        with open(
            f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_original.npy",
            "rb",
        ) as f:
            self.y_loaded_original = np.load(f)

        y_new = []
        for version in range(1, self.n_versions + 1):
            with open(
                f"{self.input_dir}data/transformed_datasets/{self.dataset_name}_version_{version}_10samples_single_transf_{method}_{self.transf_data}.npy",
                "rb",
            ) as f_new:
                y_ver = np.load(f_new)
                y_new.append(y_ver)
        self.y_loaded_transformed = np.array(y_new)

    def create_new_version_single_transf(self):
        self.y_new_all = self._create_new_version(method="single_transf")
        for i in range(len(self.transformations)):
            self._visualize_transf_series(
                transf=self.transfs[i, :, 0, :],
                method=f"single_transf_{self.transformations[i]}",
            )
        print(
            f"\nSUCCESS: Stored {self.transfs.shape[0]*self.transfs.shape[1]*self.transfs.shape[2]} transformed datasets"
        )

    def _plot_distances(self, dict_transf_ver, title):
        self.visualizer.visualize_avg_distance_by_version(dict_transf_ver, title)

    def compute_distances_transformed_datasets(self):
        res_dict = {}
        for i in range(len(self.transformations_w_random)):
            res_dict[self.transformations_w_random[i]] = {}
            for j in range(self.n_versions):
                compute_similarities = ComputeSimilaritiesSummaryMetrics(
                    self.y_new_all[i, j], self.groups_idx
                )
                res_dict[self.transformations_w_random[i]][
                    f"v{j+1}"
                ] = compute_similarities.compute_avg_similarities()["all"]
        title = (
            "Average distance within each group of the transformed dataset\n"
            "using DTW (by transformation and version)"
        )
        self._plot_distances(res_dict, title)

    def compute_distances_transf_vs_original_by_transf_and_ver(self):
        res_dict = {}
        for i in range(len(self.transformations)):
            res_dict[self.transformations[i]] = {}
            for j in range(self.n_versions):
                compute_similarities = ComputeSimilaritiesSummaryMetrics(
                    dataset=self.y,
                    group_dict_idxs=self.groups_idx,
                    transformed_dataset=self.y_new_all[i, j],
                )
                res_dict[self.transformations[i]][
                    f"v{j+1}"
                ] = (
                    compute_similarities.compute_avg_similarities_transf_dataset_vs_original()
                )
        title = (
            "Average distance of the transformed time series and the original dataset\n"
            "using DTW (by transformation and version)"
        )
        self._plot_distances(res_dict, title)

    def _compute_distances_transf_vs_original(self, n_versions, y_new_all):
        res_dict = {}
        for j in range(n_versions):
            compute_similarities = ComputeSimilaritiesSummaryMetrics(
                dataset=self.y,
                group_dict_idxs=self.groups_idx,
                transformed_dataset=y_new_all[j],
            )
            res_dict[
                f"v{j + 1}"
            ] = (
                compute_similarities.compute_avg_similarities_transf_dataset_vs_original()
            )
        return res_dict

    def _plot_transformations_by_version(self, dist_dict, title):
        self.visualizer.visualize_transformations_by_version(dist_dict, title)
