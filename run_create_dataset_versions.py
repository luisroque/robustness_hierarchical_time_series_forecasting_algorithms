from datasets.pre_processing_datasets import preprocess_datasets as ppc
from create_new_dataset_versions import CreateNewVersionDataset as cnvd
import numpy as np
from visualize_transformed_datasets import Visualizer


class CreateNewVersion:

    def __init__(self, dataset):
        self.dataset = dataset
        self.transformations = ['jitter', 'scaling', 'magnitude_warp', 'time_warp']
        self.parameters = {"jitter": 0.5,
                           "scaling": 0.02,
                           "magnitude_warp": 0.02,
                           "time_warp": 0.02}
        self.data = self.get_dataset()
        self.n_s = 10 # number of samples per version
        self.n = self.data['train']['n'] # number of points in each series
        self.s = self.data['train']['s'] # number of series in dataset
        self.y = np.tile(np.expand_dims(self.data['train']['data'], axis=0), (10, 1, 1))
        self._save_original_file()
        self.n_versions = 6
        self.n_samples = 10
        self.visualizer = Visualizer(dataset=self.dataset, n_versions=self.n_versions, n_series=6)

    def _save_original_file(self):
        with open(f'./datasets/transformed_datasets/{self.dataset}_original.npy', 'wb') as f:
            np.save(f, self.y)

    def _save_version_file(self, y_new, version, sample, method):
        with open(f'./datasets/transformed_datasets/{self.dataset}_version_{version}_{sample}samples_{method}.npy', 'wb') as f:
            np.save(f, y_new)

    def get_dataset(self):
        return ppc(self.dataset).apply_preprocess()

    def _visualize_transf_series(self, version_to_plot, transf, method):
        # self.visualizer.visualize_ver_transf(version=version_to_plot, transf=transf, method=method)
        self.visualizer.visualize_series_transf(transf=transf, method=method)

    def _get_parameters_map(self, transfs):
        params = np.vectorize(self.parameters.get)(transfs)
        return params

    def _create_new_version(self, transfs, parameters, method):
        y_new = np.zeros((self.n_s, self.n, self.s))
        y_new_all = []
        for i in range(1, self.n_versions+1):
            # Create 6 different versions of a dataset
            for j in range(1, self.n_samples+1):
                # Create 10 samples of each version
                y_new[j-1] = cnvd(self.y[j-1],
                                  transfs[i-1, j-1],
                                  version=i,
                                  sample=j,
                                  parameters=parameters[i-1, j-1]).apply_transformations()
            y_new_all.append(y_new)
            self._save_version_file(y_new, i, j, method)

    def create_new_version_random(self):
        transfs = np.tile(np.random.choice(self.transformations, size=(6, 1, 32)), (1, 10, 1)) # (n_versions, n_samples, n_series)
        params = self._get_parameters_map(transfs)
        method = 'random'
        self._create_new_version(transfs, params, method)
        print(f'\nSUCCESS: Stored {transfs.shape[0]*transfs.shape[1]} transformed datasets')
        self._visualize_transf_series(version_to_plot=1, transf=transfs[:, 0, :], method=method)

    def create_new_version_single_transf(self):
        transfs = np.tile(np.array(self.transformations).reshape(-1, 1), (6, 10, 1, 32)).transpose(2, 0, 1, 3)
        params = self._get_parameters_map(transfs)
        params = np.arange(1, 7).reshape((1, -1, 1, 1)) * params
        for i in range(len(self.transformations)):
            self._create_new_version(transfs[i], params[i], f'single_transf_{self.transformations[i]}')
            self._visualize_transf_series(version_to_plot=1, transf=transfs[i, :, 0, :], method=f'single_transf_{self.transformations[i]}')
        print(f'\nSUCCESS: Stored {transfs.shape[0]*transfs.shape[1]*transfs.shape[2]} transformed datasets')
