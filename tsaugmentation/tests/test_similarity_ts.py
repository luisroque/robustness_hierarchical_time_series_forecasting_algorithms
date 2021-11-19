import unittest
import numpy as np
from tsaugmentation.transformations.similarity_model import SimilarityModel


class TestSimilarityMeasure(unittest.TestCase):

    def test_calculate_similarity_between_two_identical_series(self):
        similarity_model = SimilarityModel()
        similarity = similarity_model.estimate_similarity_two_series(np.arange(10), np.arange(10))
        expected_result = 0
        self.assertEqual(similarity, expected_result)

    def test_calculate_similarity_between_two_NON_identical_series(self):
        similarity_model = SimilarityModel()
        idx = np.linspace(0, 50, 50)
        similarity = np.round(similarity_model.estimate_similarity_two_series(np.sin(idx), np.cos(idx)), 3)
        expected_result = 27.069
        self.assertEqual(similarity, expected_result)
