from ..preprocessing.pre_processing_datasets import PreprocessDatasets as ppc
from tsaugmentation.transformations.manipulate_data import ManipulateData
import numpy as np
from tsaugmentation.visualization.visualize_transformed_datasets import Visualizer
from .compute_similarities_summary_metrics import ComputeSimilaritiesSummaryMetrics
from pathlib import Path


class CreateTransformedVersions:

    def __init__(self, dataset):
        self.dataset = dataset
        self.transformations = ['jitter', 'scaling', 'magnitude_warp', 'time_warp']
        self.transformations_w_random = self.transformations.copy()
        self.transformations_w_random.insert(0, 'random')
        self.parameters = {"jitter": 0.5,
                           "scaling": 0.02,
                           "magnitude_warp": 0.02,
                           "time_warp": 0.02}
        self.data = self.get_dataset()
        self.n = self.data['train']['n'] # number of points in each series
        self.s = self.data['train']['s'] # number of series in dataset
        self.n_samples = 10
        self.y = self.data['train']['data']
        self.groups_idx = self.data['train']['groups_idx']
        self._save_original_file()
        self.n_versions = 6
        self.visualizer = Visualizer(dataset=self.dataset, n_versions=self.n_versions, n_series=6)
        self.y_new_all = np.zeros((len(self.transformations), self.n_versions, self.n_samples, self.n, self.s))
        self._create_directories()
        self.transfs = np.tile(np.array(self.transformations).reshape(-1, 1),
                          (self.n_versions, self.n_samples, 1, self.s)).transpose(2, 0, 1, 3)

    @staticmethod
    def _create_directories():
        # Create directory to store transformed datasets if does not exist
        Path("./transformed_datasets").mkdir(parents=True, exist_ok=True)

    def _save_original_file(self):
        with open(f'./transformed_datasets/{self.dataset}_original.npy', 'wb') as f:
            np.save(f, self.y)

    def _save_version_file(self, y_new, version, sample, transformation, method):
        with open(f'./transformed_datasets/{self.dataset}_version_{version}_{sample}samples_{method}_{transformation}.npy', 'wb') as f:
            np.save(f, y_new)

    def get_dataset(self):
        return ppc(self.dataset).apply_preprocess()

    def _visualize_transf_series(self, transf, method):
        self.visualizer.visualize_series_transf(transf=transf, method=method)

    def _get_parameters_map(self, transfs):
        params = np.vectorize(self.parameters.get, otypes=['float'])(transfs)
        return params

    def _create_new_version(self, method, n_versions=None, save=True):
        if not n_versions:
            n_versions = self.n_versions
        y_new = np.zeros((len(self.transformations), self.n_versions, self.n_samples, self.n, self.s))
        params = self._get_parameters_map(self.transformations)
        for k in range(len(self.transformations)):
            # Create versions and samples per transformation
            for i in range(1, n_versions+1):
                # Create 6 different versions of a dataset
                for j in range(1, self.n_samples+1):
                    # Create 10 samples of each version
                    y_new[k, i-1, j-1] = ManipulateData(self.y,
                                                        self.transformations[k],
                                                        parameters=params*i).apply_transf()
                if save:
                    self._save_version_file(y_new[k, i-1], i, j, self.transformations[k], method)
        return y_new

    def create_new_version_single_transf(self):
        self.y_new_all = self._create_new_version(method='single_transf')
        for i in range(len(self.transformations)):
            self._visualize_transf_series(transf=self.transfs[i, :, 0, :], method=f'single_transf_{self.transformations[i]}')
        print(f'\nSUCCESS: Stored {self.transfs.shape[0]*self.transfs.shape[1]*self.transfs.shape[2]} transformed datasets')

    def _plot_distances(self, dict_transf_ver, title):
        self.visualizer.visualize_avg_distance_by_version(dict_transf_ver, title)

    def compute_distances_transformed_datasets(self):
        res_dict = {}
        for i in range(len(self.transformations_w_random)):
            res_dict[self.transformations_w_random[i]] = {}
            for j in range(self.n_versions):
                compute_similarities = ComputeSimilaritiesSummaryMetrics(self.y_new_all[i, j], self.groups_idx)
                res_dict[self.transformations_w_random[i]][f'v{j+1}'] = compute_similarities.compute_avg_similarities()['all']
        title = 'Average distance within each group of the transformed dataset\n' \
                'using DTW (by transformation and version)'
        self._plot_distances(res_dict, title)

    def compute_distances_transf_vs_original_by_transf_and_ver(self):
        res_dict = {}
        for i in range(len(self.transformations)):
            res_dict[self.transformations[i]] = {}
            for j in range(self.n_versions):
                compute_similarities = ComputeSimilaritiesSummaryMetrics(dataset=self.y,
                                                                         group_dict_idxs=self.groups_idx,
                                                                         transformed_dataset=self.y_new_all[i, j])
                res_dict[self.transformations[i]][f'v{j+1}'] = compute_similarities\
                    .compute_avg_similarities_transf_dataset_vs_original()
        title = 'Average distance of the transformed time series and the original dataset\n' \
                'using DTW (by transformation and version)'
        self._plot_distances(res_dict, title)

    def _compute_distances_transf_vs_original(self, n_versions, y_new_all):
        res_dict = {}
        for j in range(n_versions):
            compute_similarities = ComputeSimilaritiesSummaryMetrics(dataset=self.y,
                                                                     group_dict_idxs=self.groups_idx,
                                                                     transformed_dataset=y_new_all[j])
            res_dict[f'v{j + 1}'] = compute_similarities.compute_avg_similarities_transf_dataset_vs_original()
        return res_dict

    def _plot_transformations_by_version(self, dist_dict, title):
        self.visualizer.visualize_transformations_by_version(dist_dict, title)