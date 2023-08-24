import unittest
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from tsaugmentation.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    def test_test_size_prison(self):
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="prison", freq="Q", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (40, 2)
        )

    def test_test_size_tourism(self):
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="tourism", freq="M", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (204, 2)
        )

    def test_test_size_m5(self):
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="m5", freq="W", test_size=2, weekly_m5=True
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (261, 2)
        )

    def test_test_size_m5_daily(self):
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="m5", freq="D", test_size=2, weekly_m5=False
        )

        print(self.create_dataset_vae.dataset["train"]["data"].shape)

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (1869, 2)
        )

    def test_test_size_police(self):
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="police", freq="D", test_size=2
        )

        self.assertTrue(
            self.create_dataset_vae.dataset["train"]["data"].shape == (304, 2)
        )
