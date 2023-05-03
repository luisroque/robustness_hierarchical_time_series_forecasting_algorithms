from .compute_similarities import ComputeSimilarities
import numpy as np
from sklearn.preprocessing import StandardScaler
from dtaidistance import dtw


def compute_similarity_transformed_vs_original(
    dec_pred_hat: np.ndarray, orig: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute DTW between all series in the original dataset and between all series in the transformed dataset
    The DTW is computed after scaling the data

    :param dec_pred_hat: transformed dataset
    :orig_data: original data

    :return: mean of the distance - transformed dataset
    :return: mean of the distance - original dataset
    """
    scaler = StandardScaler()
    scaler.fit(orig)
    orig_scaled = scaler.transform(orig)

    orig_scaled = orig_scaled.astype("float64")

    d_orig = np.array(dtw.distance_matrix_fast(orig_scaled.T[:, 10:], compact=True))

    dec_pred_hat = dec_pred_hat.astype("float64")

    scaler = StandardScaler()
    scaler.fit(dec_pred_hat)
    dec_pred_hat_scaled = scaler.transform(dec_pred_hat)

    d_generated = np.array(
        dtw.distance_matrix_fast(dec_pred_hat_scaled.T, compact=True)
    )

    return np.mean(d_generated), np.mean(d_orig)


class ComputeSimilaritiesSummaryMetrics:
    def __init__(self, dataset, group_dict_idxs, transformed_dataset=None):
        self.dataset = dataset
        self.group_dict_idxs = group_dict_idxs
        self.transformed_dataset = transformed_dataset

    def compute_similarities_within_group(self, group, group_val):
        similarities = ComputeSimilarities(
            self.dataset[:, np.where(self.group_dict_idxs[f"{group}"] == group_val)][
                :, 0, :
            ]
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

    def compute_avg_similarities_for_every_group(self):
        res_dict = self.compute_similarities_for_every_group_element()
        res = {}
        for k, val in res_dict.items():
            res[k] = np.mean(val)
        return res

    def compute_avg_similarities(self):
        res_dict = self.compute_avg_similarities_for_every_group()
        res_dict["all"] = np.mean(list(res_dict.values()))
        return res_dict

    def compute_avg_similarities_transf_dataset_vs_original(self):
        avg_sim = []
        for i in range(self.transformed_dataset.shape[1]):
            avg_sim.append(
                ComputeSimilarities(
                    dataset=self.dataset, transf_series=self.transformed_dataset[:, i]
                ).compute_similarity_single_series_and_dataset()
            )
        return np.mean(avg_sim)
