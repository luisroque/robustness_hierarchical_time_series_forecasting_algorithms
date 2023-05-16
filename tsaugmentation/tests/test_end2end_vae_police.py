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
            dataset_name="police", freq="D", top=10
        )

        self.model, _, _ = self.create_dataset_vae.fit(epochs=5)
        self.model, _, _ = self.create_dataset_vae.fit(epochs=5)
        (
            self.preds,
            self.z,
            self.z_mean,
            self.z_log_var,
        ) = self.create_dataset_vae.predict(self.model)

    def test_compute_similarity(self):
        dec_pred_hat = self.create_dataset_vae.generate_transformed_time_series(
            vae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            std_latent_space=0.5,
        )

        self.assertTrue(
            compute_similarity_transformed_vs_original(
                dec_pred_hat, self.create_dataset_vae.X_train_raw
            )[0]
            < 20
        )

