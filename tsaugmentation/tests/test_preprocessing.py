import unittest
from tsaugmentation.preprocessing.pre_processing_datasets import PreprocessDatasets
import shutil


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.wrong_api = "http://test.com"
        self.dataset = 'prison'

    def test_get_dataset_with_wrong_api_or_server_down(self):
        preprocess_data = PreprocessDatasets(self.dataset)
        preprocess_data.api = self.wrong_api
        data = preprocess_data.apply_preprocess()
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        self.assertFalse(data)

    def test_get_dataset_prison(self):
        preprocess_data = PreprocessDatasets(self.dataset)
        data = preprocess_data.apply_preprocess()
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")
        self.assertTrue(data)
