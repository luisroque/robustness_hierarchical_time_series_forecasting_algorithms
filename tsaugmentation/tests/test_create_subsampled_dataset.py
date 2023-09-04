import unittest
import os
import math

from tsaugmentation.preprocessing.subsample_dataset import CreateGroups


class TestCreateGroups(unittest.TestCase):
    def setUp(self):
        self.dataset_name_t = "tourism"
        self.sample_perc = 0.50

    def test_create_subsample_dataset(self):
        expected_filename = f"{self.dataset_name_t}_{int(self.sample_perc * 100)}.pkl"
        expected_filepath = os.path.join(
            "./data/subsampled_datasets", expected_filename
        )

        # Delete the file if it already exists to ensure the test is valid
        if os.path.exists(expected_filepath):
            os.remove(expected_filepath)

        groups = CreateGroups(
            dataset_name=self.dataset_name_t, sample_perc=self.sample_perc, freq="M"
        ).create_subsampled_groups()

        groups = CreateGroups(
            dataset_name=self.dataset_name_t, sample_perc=self.sample_perc, freq="M"
        ).read_subsampled_groups()

        groups_orig = CreateGroups(
            dataset_name=self.dataset_name_t, freq="M"
        ).create_original_groups()

        groups_orig = CreateGroups(
            dataset_name=self.dataset_name_t, freq="M"
        ).read_original_groups()

        self.assertTrue(
            os.path.exists(expected_filepath),
            f"File {expected_filepath} does not exist",
        )

        # Verify that the length of the x_values in the subsampled group is approximately x%
        # the length of the x_values in the original group
        len_subsample = len(groups["train"]["x_values"])
        len_orig = len(groups_orig["train"]["x_values"])
        expected_len_subsample = math.ceil(len_orig * self.sample_perc)

        # We can allow a small deviation
        deviation = math.ceil(len_orig * 0.05)
        self.assertTrue(
            expected_len_subsample - deviation
            <= len_subsample
            <= expected_len_subsample + deviation,
            f"The length of the subsampled x_values is not approximately {self.sample_perc*100}% of the original length.",
        )

        self.assertTrue(
            groups["predict"]["data"].shape != groups_orig["predict"]["data"].shape
        )

        self.assertTrue(
            groups["predict"]["original_data"].shape == groups_orig["predict"]["data"].shape
        )

        self.assertTrue("original_data" in groups_orig["predict"])
