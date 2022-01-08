import unittest
from tsaugmentation.visualization.visualize_transformed_datasets import Visualizer
from tsaugmentation.transformations.create_dataset_versions import CreateTransformedVersions
import shutil


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset = 'prison'

    def test_read_files(self):
        transformed_datasets = CreateTransformedVersions(self.dataset)
        transformed_datasets.create_new_version_single_transf()
        vi = Visualizer(self.dataset)
        vi._read_files(method='single_transf_jitter')

        shutil.rmtree("./data/original_datasets")
        shutil.rmtree("./data/transformed_datasets")
        self.assertTrue(vi.y_new.shape == (6, 10, 40, 32))
        self.assertTrue(vi.y.shape == (40, 32))
