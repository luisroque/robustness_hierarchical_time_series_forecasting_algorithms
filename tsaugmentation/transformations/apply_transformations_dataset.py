from .manipulate_data import ManipulateData
import numpy as np


class ApplyTransformationsDataset:
    """
    Create a new version of a dataset. It works by performing one
    one transformation randomly chosen from a list of possible
    transformations and apply it to a random time series. This process
    is done for every time series in the dataset.

    Each time series is modified by one transformation only,
    no cumulative transformations are applied

    inputs:
        y - data
        transformations - list of transformations
        version - version of the dataset created (1 to n, 1 has only one random
            transformation per series and n has n random transformations per series
        sample - sample of the dataset created, for dataset version 1, for instance, we create 10 different samples
    """

    def __init__(self, y, transformations, version, sample, parameters):
        self.y = y
        self.transformations = transformations
        self.version = version
        self.sample = sample
        self.parameters = parameters

    def apply_transformations(self):
        y_new = np.zeros((self.y.shape[0], self.y.shape[1]))
        for i in range(self.y.shape[1]):
            y_new[:, i] = ManipulateData(self.y[:, i].reshape(-1, 1),
                                         self.transformations[i],
                                         self.parameters[i]).apply_transf()
        
        return y_new
