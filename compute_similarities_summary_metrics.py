from compute_similarities import ComputeSimilarities
import numpy as np


class ComputeSimilaritiesSummaryMetrics:

    def __init__(self, dataset, group_dict_idxs):
        self.dataset = dataset
        self.group_dict_idxs = group_dict_idxs

    def compute_similarities_within_group(self, group, group_val):
        similarities = ComputeSimilarities(
            self.dataset[:, np.where(self.group_dict_idxs[f'{group}'] == group_val)][:, 0, :]
        ).compute_similarity_all_pairs()
        return similarities

    def compute_avg_similarities_within_group(self, group, group_val):
        return np.mean(self.compute_similarities_within_group(group, group_val))

    def compute_similarities_for_every_group_element(self):
        res_dict = {}
        for k, val in self.group_dict_idxs.items():
            list_val = []
            val = set(val)
            for v in val:
                list_val.append(self.compute_avg_similarities_within_group(k, v))
            res_dict[k] = list_val
        return res_dict

    def compute_similarities_for_every_element(self):
        res_dict = {}
        for k, val in self.group_dict_idxs.items():
            list_val = []
            val = set(val)
            for v in val:
                list_val.append(self.compute_avg_similarities_within_group(k, v))
            res_dict[k] = list_val
        return res_dict


