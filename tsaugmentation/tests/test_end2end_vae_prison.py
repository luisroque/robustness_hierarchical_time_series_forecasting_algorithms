import unittest
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from tsaugmentation.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name="prison", freq="Q", dynamic_feat_trig=False
        )

        self.model, _, _ = self.create_dataset_vae.fit(epochs=5, load_weights=False)

    def test_compute_similarity(self):
        (
            self.preds,
            self.z,
            self.z_mean,
            self.z_log_var,
        ) = self.create_dataset_vae.predict(self.model)

        dec_pred_hat = self.create_dataset_vae.generate_transformed_time_series(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transf_param=0.5,
        )

        self.assertTrue(
            compute_similarity_transformed_vs_original(
                dec_pred_hat, self.create_dataset_vae.X_train_raw
            )[0]
            < 20
        )

    def test_create_correct_number_transformed_datasets_similiar_static_features(self):
        (
            self.preds,
            self.z,
            self.z_mean,
            self.z_log_var,
        ) = self.create_dataset_vae.predict(self.model, similar_static_features=0)

        new_datasets = self.create_dataset_vae.generate_new_datasets(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transf_param=[0.5, 1, 2, 5, 10, 15],
            n_versions=2,
            n_samples=3,
        )
        # shape (n_versions, n_samples_generated, n_samples_dataset, n_series)
        self.assertTrue(new_datasets.shape == (2, 3, 48, 32))

    def test_dynamic_features_creation(self):
        dynamic_feat, _, _ = self.create_dataset_vae._feature_engineering(48)
        self.assertTrue(len(dynamic_feat) == 2)
