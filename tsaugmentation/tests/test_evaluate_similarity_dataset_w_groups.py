import unittest
import numpy as np
from tsaugmentation.transformations.compute_similarities_summary_metrics import ComputeSimilaritiesSummaryMetrics


class TestEvaluateSimilarityWithGroups(unittest.TestCase):

    def setUp(self):
        np.random.seed(43)
        self.n_points = 10
        self.n_series = 20
        self.base_dataset = np.random.normal(0, 1, size=(self.n_points, self.n_series))
        self.transf_dataset = np.random.normal(0, 2, size=(self.n_points, self.n_series))
        self.groups = 3
        self.groups_idx = {'group1': np.arange(self.n_series) % 3,
                           'group2': np.arange(self.n_series) % 2,
                           'group3': np.arange(self.n_series)[::-1] % 2}

    def test_evaluate_similarity_within_specific_group(self):
        group = 'group1'
        group_val = 1
        similarities = [
            9.906357, 14.4180557, 10.7834545, 9.8581009, 13.2951942, 8.7437967, 11.319695, 12.8673413, 10.984207,
            13.9656696, 10.4894326, 12.0199686, 10.8596356, 13.1942304, 10.1134963, 9.9802701, 10.4758453, 11.0350871,
            14.602276,  8.563134, 11.873626
        ]
        compute_similarities = ComputeSimilaritiesSummaryMetrics(self.base_dataset, self.groups_idx) \
            .compute_similarities_within_group(group, group_val)
        self.assertIsNone(np.testing.assert_almost_equal(similarities, compute_similarities))

    def test_evaluate_similarity_for_every_specific_group_element(self):
        similarities = {'group1': [10.624993776985308, 11.397565415605204, 10.451836642651857],
                        'group2': [11.932008691345734, 9.162135946742984],
                        'group3': [9.162135946742984, 11.932008691345734]}
        compute_similarities = ComputeSimilaritiesSummaryMetrics(self.base_dataset, self.groups_idx) \
            .compute_similarities_for_every_group_element()
        self.assertDictEqual(similarities, compute_similarities)

    def test_evaluate_avg_similarity_for_every_specific_group_element(self):
        similarities = {'group1': 10.824798611747456,
                        'group2': 10.547072319044359,
                        'group3': 10.547072319044359}
        compute_similarities = ComputeSimilaritiesSummaryMetrics(self.base_dataset, self.groups_idx) \
            .compute_avg_similarities_for_every_group()
        self.assertDictEqual(similarities, compute_similarities)

    def test_evaluate_avg_similarity_for_every_specific_group_element_and_all(self):
        similarities = {'group1': 10.824798611747456,
                        'group2': 10.547072319044359,
                        'group3': 10.547072319044359,
                        'all': 10.639647749945391}
        compute_similarities = ComputeSimilaritiesSummaryMetrics(self.base_dataset, self.groups_idx) \
            .compute_avg_similarities()
        self.assertEqual(similarities, compute_similarities)

    def test_calculate_similarity_between_transformed_series_and_base_dataset(self):
        similarities = 17.07120129561533
        compute_similarities = ComputeSimilaritiesSummaryMetrics(dataset=self.base_dataset,
                                                                 group_dict_idxs=self.groups_idx,
                                                                 transformed_dataset=self.transf_dataset) \
            .compute_avg_similarities_transf_dataset_vs_original()
        self.assertEqual(similarities, compute_similarities)


