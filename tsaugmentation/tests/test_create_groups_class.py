import unittest

from tsaugmentation.preprocessing.subsample_dataset import CreateGroups


class TestCreateGroups(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "prison"
        self.sample_perc = 0.9
        self.create_group = CreateGroups(self.dataset_name, self.sample_perc)

    def test_create_subsampled_dataset(self):
        groups = self.create_group.create_subsampled_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_subsampled_dataset(self):
        groups = self.create_group.read_subsampled_groups()
        self.assertTrue(groups["h"] == 8)

    def test_create_original_dataset(self):
        groups = self.create_group.create_original_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_original_dataset(self):
        groups = self.create_group.read_original_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_groups_transformed(self):
        y_new = self.create_group.read_groups_transformed("jitter")
        self.assertTrue(y_new.shape == (6, 10, 48, 32))
