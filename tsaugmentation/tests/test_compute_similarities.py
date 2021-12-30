import unittest
import numpy as np
from tsaugmentation.transformations.compute_similarities import ComputeSimilarities
from tsaugmentation.transformations.similarity_model import SimilarityModel


class TestComputeSimilarities(unittest.TestCase):

    def setUp(self):
        self.n_points = 10
        self.n_series = 100
        self.base_dataset = np.tile(np.arange(10).reshape(1, -1), (100, 1)).T
        self.transformed_series = np.arange(10, 20)

    def test_compute_similarities_between_EQUAL_pairs_of_ts_of_dataset(self):
        sim = ComputeSimilarities(self.base_dataset).compute_similarity_all_pairs()
        # Number of combinations of two series, without repeating: m*(m-1)/2
        expected_output = np.zeros((int(self.n_series*(self.n_series-1)/2),))
        self.assertIsNone(np.testing.assert_almost_equal(sim, expected_output))

    def test_compute_similarities_between_pairs_of_ts_of_dataset_with_one_series_different(self):
        k = 0
        selected_series = 4
        expected_output = np.zeros((int(self.n_series*(self.n_series-1)/2),))
        similarity_model = SimilarityModel()
        sim_diff_series_value = similarity_model.estimate_similarity_two_series(np.arange(self.n_points), np.cos(np.arange(self.n_points)))
        for i in range(self.base_dataset.shape[1]):
            for j in range(i+1, self.base_dataset.shape[1]):
                if i == selected_series or j == selected_series:
                    expected_output[k] = sim_diff_series_value
                k += 1
        dataset = np.array(self.base_dataset, dtype='float64')
        dataset[:, selected_series] = np.cos(np.arange(self.n_points))
        sim = ComputeSimilarities(dataset).compute_similarity_all_pairs()
        self.assertIsNone(np.testing.assert_almost_equal(sim, expected_output))

    def test_compute_avg_similarity_between_single_series_and_all_series_in_a_dataset(self):
        avg_sim = ComputeSimilarities(self.base_dataset, self.transformed_series)\
            .compute_similarity_single_series_and_dataset()
        similarity = 109.0
        self.assertEqual(similarity, avg_sim)

    def test_mean_similarity_elementwise_with_same_dataset(self):
        avg_sim = ComputeSimilarities(dataset=self.base_dataset, transf_dataset=self.base_dataset) \
            .compute_mean_similarity_elementwise()
        self.assertEqual(0, avg_sim)

