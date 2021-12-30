from .similarity_model import SimilarityModel
import numpy as np


class ComputeSimilarities:

    def __init__(self, dataset, transf_series=None, transf_dataset=None):
        self.dataset = dataset
        self.transf_series = transf_series
        self.transf_dataset = transf_dataset

    def compute_similarity_all_pairs(self):
        similarity_model = SimilarityModel()
        res = []
        for i in range(self.dataset.shape[1]):
            for j in range(i+1, self.dataset.shape[1]):
                res.append(similarity_model.estimate_similarity_two_series(self.dataset[:, i], self.dataset[:, j]))
        return np.array(res)

    def compute_similarity_single_series_and_dataset(self):
        similarity_model = SimilarityModel()
        res = []
        for i in range(self.dataset.shape[1]):
            res.append(similarity_model.estimate_similarity_two_series(self.dataset[:, i], self.transf_series))
        return np.mean(res)

    def compute_mean_similarity_elementwise(self):
        similarity_model = SimilarityModel()
        res = []
        for i in range(self.dataset.shape[1]):
            res.append(similarity_model.estimate_similarity_two_series(self.dataset[:, i], self.transf_dataset[:, i]))
        return np.mean(res)



