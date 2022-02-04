import unittest
from tsaugmentation.transformations.create_dataset_versions import CreateTransformedVersions
import shutil
import os
import numpy as np


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = 'police'
        self.transformed_datasets = CreateTransformedVersions(self.dataset)
        self.transformed_datasets.parameters = {"jitter": 0.5,
                                           "scaling": 0.1,
                                           "magnitude_warp": 0.05,
                                           "time_warp": 0.05}
        self.transformed_datasets.create_new_version_single_transf()
        np.random.seed(0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./data/original_datasets")
        shutil.rmtree("./data/transformed_datasets")

    def test_create_correct_number_transformed_datasets_single_transf(self):
        # shape (n_transformations + random_transf , n_versions, n_samples, n_points_train, n_series)
        self.assertTrue(self.transformed_datasets.y_new_all.shape == (4, 6, 10, 300, 500))

    def test_create_correct_number_transformed_datasets_FILES_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset)
        transformed_datasets.create_new_version_single_transf()
        file_count = len([name for name in os.listdir('./data/transformed_datasets/')])
        self.assertEqual(file_count, 25)
