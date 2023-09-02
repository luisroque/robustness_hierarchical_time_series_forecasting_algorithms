import unittest

from tsaugmentation.preprocessing.subsample_dataset import CreateGroups
from tsaugmentation.visualization.visualize_subsampled_dataset import plot_series


class TestCreateGroups(unittest.TestCase):
    def setUp(self):
        self.dataset_name = "prison"
        self.sample_perc = 0.9
        self.create_group = CreateGroups(self.dataset_name, "Q", self.sample_perc)
        self.dataset_name_t = "tourism"
        self.sample_perc = 0.60
        self.create_group_t = CreateGroups(self.dataset_name_t, "M", self.sample_perc)
        self.create_group_t_original = CreateGroups(self.dataset_name_t, "M")

    def test_create_subsampled_dataset(self):
        groups = self.create_group.create_subsampled_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_subsampled_dataset(self):
        groups = self.create_group.read_subsampled_groups()
        plot_series(groups)
        self.assertTrue(groups["h"] == 8)

    def test_read_subsampled_dataset_tourism(self):
        groups_missing = self.create_group_t.read_subsampled_groups()
        groups = self.create_group_t_original.read_original_groups()

        plot_series(groups_missing, groups)
        self.assertTrue(groups["h"] == 24)

    def test_create_original_dataset(self):
        groups = self.create_group.create_original_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_original_dataset(self):
        groups = self.create_group.read_original_groups()
        self.assertTrue(groups["h"] == 8)

    def test_read_groups_transformed(self):
        y_new = self.create_group.read_groups_transformed("jitter")
        self.assertTrue(y_new.shape == (6, 10, 48, 32))
