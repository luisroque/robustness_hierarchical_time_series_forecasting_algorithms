import unittest
from tsaugmentation.transformations.manipulate_data import ManipulateData
import numpy as np
from scipy.interpolate import CubicSpline


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = 'tourism'
        self.parameters = {"jitter": 0.5,
                            "scaling": 0.2,
                            "magnitude_warp": 0.2,
                            "time_warp": 0.2}
        self.params = np.vectorize(self.parameters.get)(list(self.parameters.keys()))
        np.random.seed(0)
        self.data = np.random.randn(200, 100)

    def test_magnitude_warping_individual_series(self):
        res = ManipulateData(self.data, 'magnitude_warp', self.params).apply_transf()

        random_warp_1 = np.array([0.999904, 0.76587626, 0.92712345, 0.69992099, 0.87777622, 1.12951142])

        warp_steps = np.array([0., 39.8, 79.6, 119.4, 159.2, 199.])
        ret_1 = CubicSpline(warp_steps, random_warp_1)(np.arange(self.data.shape[0]))
        res_1 = self.data[:, 1]*ret_1

        self.assertIsNone(np.testing.assert_almost_equal(res_1, res[:, 1]))

    def test_time_warping_individual_series(self):
        res = ManipulateData(self.data, 'time_warp', self.params).apply_transf()

        random_warp_1 = np.array([0.999904, 0.76587626, 0.92712345, 0.69992099, 0.87777622, 1.12951142])
        warp_steps = np.array([0., 39.8, 79.6, 119.4, 159.2, 199.])
        time_warp_1 = CubicSpline(warp_steps, warp_steps*random_warp_1)(np.arange(self.data.shape[0]))
        res_1 = np.interp(np.arange(self.data.shape[0]), time_warp_1, self.data[:, 1])

        self.assertIsNone(np.testing.assert_almost_equal(res_1, res[:, 1], decimal=4))




