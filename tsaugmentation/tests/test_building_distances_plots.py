import unittest
import numpy as np

from tsaugmentation.transformations.compute_distances import compute_store_distances
from tsaugmentation.visualization.visualize_ridge_distance import (
    build_df_ridge,
    plot_distances,
    load_distances,
    store_df_distances,
    load_df_distances
)
from tsaugmentation.feature_engineering.get_data_distance import get_data
import pathlib


class TestBuildingDistancePlots(unittest.TestCase):
    def setUp(self):
        base_path = pathlib.Path(__file__).parent.resolve()
        self.dataset = "prison"
        self.versions = 6
        self.transformations = ["jitter", "scaling", "magnitude_warp", "time_warp"]
        self.data_orig, self.data_transf = get_data(
            f"{base_path}/data/transformed_datasets",
            f"{base_path}/data/transformed_datasets",
            self.dataset,
            transformations=self.transformations,
            versions=self.versions,
        )
        self.s = self.data_transf.shape[4]
        self.d_orig, self.d_transf = compute_store_distances(
            self.dataset,
            self.data_orig,
            self.data_transf,
            self.transformations,
            self.versions,
        )
        self.n_d = self.d_transf.shape[2]

    def test_compute_distances_shape(self):
        self.assertTrue(
            self.n_d
            == np.math.factorial(self.s)
            / (np.math.factorial(self.s - 2) * np.math.factorial(2))
        )

    def test_build_df_distances(self):
        df_ridge = build_df_ridge(
            self.d_transf, self.d_orig, self.n_d, self.transformations, self.versions
        )
        self.assertTrue(df_ridge.shape == (1984, 8))

    def test_store_load_data(self):
        df_transf_load, d_orig_load = load_distances(self.dataset)
        self.assertTrue(df_transf_load.shape == (4, 6, 496))

    def test_store_distances_df(self):
        df_ridge = build_df_ridge(
            self.d_transf, self.d_orig, self.n_d, self.transformations, self.versions
        )
        store_df_distances(df_ridge, self.dataset)

    def test_plot_distances(self):
        df_ridge = load_df_distances(self.dataset)
        plot_distances(self.dataset, df_ridge, self.versions, x_range=[0, 10])
