from ..preprocessing.pre_processing_datasets import PreprocessDatasets as ppc
from .apply_transformations_dataset import ApplyTransformationsDataset as cnvd
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
        self.n_s = 10 # number of samples per version
        self.n = self.data['train']['n'] # number of points in each series
        self.s = self.data['train']['s'] # number of series in dataset
        self.y = np.tile(np.expand_dims(self.data['train']['data'], axis=0), (self.n_s, 1, 1))
        self.groups_idx = self.data['train']['groups_idx']
        self._save_original_file()
        self.n_versions = 6
        self.n_samples = 10
        self.visualizer = Visualizer(dataset=self.dataset, n_versions=self.n_versions, n_series=6)
        self.y_new_all = np.zeros((len(self.transformations_w_random), self.n_versions, self.n_samples, self.n, self.s))
        self._create_directories()

    @staticmethod
    def _create_directories():
        # Create directory to store transformed datasets if does not exist
        Path("./transformed_datasets").mkdir(parents=True, exist_ok=True)

    def _save_original_file(self):
        with open(f'./transformed_datasets/{self.dataset}_original.npy', 'wb') as f:
            np.save(f, self.y)

    def _save_version_file(self, y_new, version, sample, method):
        with open(f'./transformed_datasets/{self.dataset}_version_{version}_{sample}samples_{method}.npy', 'wb') as f:
            np.save(f, y_new)

    def get_dataset(self):
        return ppc(self.dataset).apply_preprocess()

    def _visualize_transf_series(self, version_to_plot, transf, method):
        # self.visualizer.visualize_ver_transf(version=version_to_plot, transf=transf, method=method)
        self.visualizer.visualize_series_transf(transf=transf, method=method)

    def _get_parameters_map(self, transfs):
        params = np.vectorize(self.parameters.get)(transfs)
        return params

    def _create_new_version(self, transfs, parameters, method, n_versions=None, save=True):
        if not n_versions:
            n_versions = self.n_versions
        y_new = np.zeros((self.n_s, self.n, self.s))
        y_new_all = np.zeros((n_versions, self.n_s, self.n, self.s))
        for i in range(1, n_versions+1):
            # Create 6 different versions of a dataset
            for j in range(1, self.n_samples+1):
                # Create 10 samples of each version
                y_new[j-1] = cnvd(self.y[j-1],
                                  transfs[i-1, j-1],
                                  version=i,
                                  sample=j,
                                  parameters=parameters[i-1, j-1]).apply_transformations()
            if save:
                self._save_version_file(y_new, i, j, method)
            y_new_all[i-1] = y_new
        return y_new_all

    def _create_new_version_random(self):
        transfs = np.tile(np.random.choice(self.transformations, size=(6, 1, 32)), (1, 10, 1)) # (n_versions, n_samples, n_series)
        params = self._get_parameters_map(transfs)
        method = 'random'
        self.y_new_all[0] = self._create_new_version(transfs, params, method)
        print(f'\nSUCCESS: Stored {transfs.shape[0]*transfs.shape[1]} transformed datasets using Random transformations')
        self._visualize_transf_series(version_to_plot=1, transf=transfs[:, 0, :], method=method)

    def create_new_version_single_transf(self):
        self._create_new_version_random()
        transfs = np.tile(np.array(self.transformations).reshape(-1, 1), (6, 10, 1, 32)).transpose(2, 0, 1, 3)
        params = self._get_parameters_map(transfs)
        params = np.arange(1, 7).reshape((1, -1, 1, 1)) * params
        for i in range(len(self.transformations)):
            self.y_new_all[i+1] = self._create_new_version(transfs[i], params[i], f'single_transf_{self.transformations[i]}')
            self._visualize_transf_series(version_to_plot=1, transf=transfs[i, :, 0, :], method=f'single_transf_{self.transformations[i]}')
        print(f'\nSUCCESS: Stored {transfs.shape[0]*transfs.shape[1]*transfs.shape[2]} transformed datasets')

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
        for i in range(len(self.transformations_w_random)):
            res_dict[self.transformations_w_random[i]] = {}
            for j in range(self.n_versions):
                compute_similarities = ComputeSimilaritiesSummaryMetrics(dataset=self.y,
                                                                         group_dict_idxs=self.groups_idx,
                                                                         transformed_dataset=self.y_new_all[i, j])
                res_dict[self.transformations_w_random[i]][f'v{j+1}'] = compute_similarities\
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

    def create_new_version_using_n_random_transformations(self, n_versions):
        """
        Method to create new versions of the dataset by applying cumulatively transformations to the data
        without storing them to a file (only stored in memory)
        """
        transfs = np.tile(np.random.choice(self.transformations, size=(n_versions, 1, 32)), (1, 10, 1)) # (n_versions, n_samples, n_series)
        params = self._get_parameters_map(transfs)
        method = 'random'
        y_new_all = self._create_new_version(transfs=transfs,
                                             parameters=params,
                                             method=method,
                                             n_versions=n_versions,
                                             save=False)
        dist_dict = self._compute_distances_transf_vs_original(n_versions, y_new_all)
        title = 'Distance between transformed dataset and the original dataset\n' \
                'by version'
        self._plot_transformations_by_version(dist_dict, title)
