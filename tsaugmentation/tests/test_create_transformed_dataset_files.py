import unittest
from tsaugmentation.transformations.create_dataset_versions import CreateTransformedVersions
import shutil
import os


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = 'prison'

    def test_create_correct_number_transformed_datasets_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset)
        transformed_datasets.create_new_version_single_transf()
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        # shape (n_transformations + random_transf , n_versions, n_samples, n_points_train, n_series)
        self.assertTrue(transformed_datasets.y_new_all.shape == (5, 6, 10, 40, 32))

    def test_create_correct_number_transformed_datasets_FILES_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset)
        transformed_datasets.create_new_version_single_transf()
        file_count = len([name for name in os.listdir('./transformed_datasets/')])
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        self.assertEqual(file_count, 31)

