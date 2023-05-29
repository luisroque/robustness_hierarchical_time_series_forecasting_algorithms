import unittest
import pandas as pd
from tsaugmentation.preprocessing import PreprocessDatasets


class PreprocessDatasetsTestCase(unittest.TestCase):
    def setUp(self):
        self.preprocess_datasets = PreprocessDatasets(dataset="synthetic", freq="D")

    def test_add_synthetic_dataset(self):
        self.preprocess_datasets._synthetic()

        groups = self.preprocess_datasets._load_pickle_file(
            self.preprocess_datasets.pickle_path
        )
        self.assertIsNotNone(groups)
        self.assertIn("train", groups)
        self.assertIn("predict", groups)
        self.assertIn("groups_idx", groups["train"])
        self.assertIn("groups_n", groups["train"])
        self.assertIn("groups_names", groups["train"])
        self.assertIn("s", groups["train"])
        self.assertIn("n", groups["train"])
        self.assertTrue((100, 3), groups["base_series"].shape)
