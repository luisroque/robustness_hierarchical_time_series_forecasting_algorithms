from similarity_model import SimilarityModel
import numpy as np


class ComputeSimilarities:

    def __init__(self, dataset):
        self.dataset = dataset

    def compute_similarity_all_pairs(self):
        similarity_model = SimilarityModel()
        res = []
        for i in range(self.dataset.shape[1]):
            for j in range(i+1, self.dataset.shape[1]):
                res.append(similarity_model.estimate_similarity_two_series(self.dataset[:, i], self.dataset[:, j]))
        return np.array(res)




