from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


class Visualizer:
    """
    A class used to visualize transformations

    ...

    Attributes
    ----------
    dataset : str
        the original dataset to consider
    rel_dir : str
        relative directory where to store the downloaded files (e.g. './' current dir, '../' parent dir)
    n_versions: int
        number of versions created by applying the transformations
    n_series: int
        number of series to plot
    """

    def __init__(self, dataset, n_versions=6, n_series=6, input_dir='./', transf_data='whole'):
        self.n_series = n_series
        self.dataset = dataset
        self.n_versions = n_versions
        self.input_dir = input_dir
        self.transf_data = transf_data

    def _read_files(self, method):
        with open(f'{self.input_dir}data/transformed_datasets/{self.dataset}_original.npy', 'rb') as f:
            self.y = np.load(f)

        y_new = []
        for version in range(1, self.n_versions + 1):
            with open(f'{self.input_dir}data/transformed_datasets/{self.dataset}_version_{version}_10samples_{method}_{self.transf_data}.npy',
                      'rb') as f_new:
                y_ver = np.load(f_new)
                y_new.append(y_ver)
        self.y_new = np.array(y_new)

    def visualize_ver_transf(self, version, transf, method):
        self._read_files(method=method)
        fig, ax = plt.subplots(2, int(np.floor(self.n_series / 2)), sharex='all')
        ax = ax.ravel()

        for i in range(self.y_new.shape[1]):
            for j in range(self.n_series):
                if i % 9 == 0 and not i == 0:
                    ax[j].plot(self.y[:, j], label='original', color='darkorange')
                    ax[j].set_title(f'{transf[version, j]}, s={j}')
                ax[j].plot(self.y_new[version, i, :, j], color='darkblue', alpha=0.3)

        plt.legend()
        fig.suptitle(f'10 samples of {self.n_series} of the transformed series')
        plt.show()

    def visualize_series_transf(self, transf, method):
        self._read_files(method=method)
        _, ax = plt.subplots(2, int(np.floor(self.n_series / 2)), sharex=True)
        ax = ax.ravel()

        colors = plt.cm.Blues_r(np.linspace(0, 0.65, self.n_versions))[::-1]

        for i in range(self.y_new.shape[0]):
            for j in range(self.n_series):
                if (i + 1) % 6 == 0:
                    ax[j].plot(self.y[:, j], label='original', color='darkorange')
                ax[j].plot(self.y_new[i, 0, :, j], label=f'version {i}', color=colors[i])
                ax[j].set_title(f'Series {j}, [{"".join(list(transf[:, j].astype("<U1")))}]')
        plt.legend()
        plt.show()

    @staticmethod
    def visualize_avg_distance_by_version(dict_transf_version_dist, title):
        df = pd.DataFrame.from_dict({i: dict_transf_version_dist[i]
                                    for i in dict_transf_version_dist.keys()},
                                    orient='index')
        df.plot.bar(colormap='RdBu', rot=0)
        plt.title(title)
        plt.show()

    @staticmethod
    def visualize_transformations_by_version(dict_transf_version_dist, title):
        df = pd.DataFrame.from_dict(dict_transf_version_dist, orient='index')
        df.plot.line(colormap='RdBu', rot=0, legend=False)
        plt.title(title)
        plt.show()
