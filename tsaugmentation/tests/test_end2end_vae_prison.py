import unittest
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsVAE,
)
from tsaugmentation.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.create_dataset_vae = CreateTransformedVersionsVAE(
            dataset_name="prison", freq="Q"
        )

        self.model, _, _ = self.create_dataset_vae.fit(epochs=5)
        self.preds, self.z = self.create_dataset_vae.predict(self.model)

    def test_compute_similarity(self):
        dec_pred_hat = self.create_dataset_vae.generate_transformed_time_series(
            self.model, self.z, 0.5
        )

        self.assertTrue(
            compute_similarity_transformed_vs_original(
                dec_pred_hat, self.create_dataset_vae.X_train_raw
            )[0]
            < 20
        )

    def test_create_correct_number_transformed_datasets(self):
        new_datasets = self.create_dataset_vae.generate_new_datasets(
            self.model, self.z, [0.5, 1, 2, 5, 10, 15], 2, 3
        )
        # shape (n_versions, n_samples_generated, n_samples_dataset, n_series)
        self.assertTrue(new_datasets.shape == (2, 3, 48, 32))

    def test_dynamic_features_creation(self):
        dynamic_feat, _, _ = self.create_dataset_vae._feature_engineering(48)
        self.assertTrue(len(dynamic_feat) == 4)
