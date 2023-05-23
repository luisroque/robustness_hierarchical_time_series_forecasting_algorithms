import unittest
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)
from tsaugmentation.transformations.compute_similarities_summary_metrics import (
    compute_similarity_transformed_vs_original,
)


class TestModel(unittest.TestCase):
    """
    Class for testing CVAE model's ability to generate new time series
    and to compute similarity between the generated time series and the original.
    """

    def setUp(self) -> None:
        """
        Set up a test instance of CreateTransformedVersionsCVAE and train the model.
        """
        self.dataset_name = "prison"
        self.freq = "Q"
        self.dynamic_feat_trig = False
        self.create_dataset_vae = CreateTransformedVersionsCVAE(
            dataset_name=self.dataset_name,
            freq=self.freq,
            dynamic_feat_trig=self.dynamic_feat_trig,
        )
        self.epochs = 5
        self.load_weights = True
        self.model, _, _ = self.create_dataset_vae.fit(
            epochs=self.epochs, load_weights=self.load_weights
        )
        self.std_latent_space = 0.5
        self.similarity_threshold = 20

        (
            self.preds,
            self.z,
            self.z_mean,
            self.z_log_var,
        ) = self.create_dataset_vae.predict(self.model)

    def test_compute_similarity(self):
        """
        Test if the similarity between the generated time series and original
        is below a certain threshold.
        """
        dec_pred_hat = self.create_dataset_vae.generate_transformed_time_series(
            cvae=self.model,
            z_mean=self.z_mean,
            z_log_var=self.z_log_var,
            transformation="magnitude_warp",
            transf_param=5,
        )

        similarity_score = compute_similarity_transformed_vs_original(
            dec_pred_hat, self.create_dataset_vae.X_train_raw
        )[0]

        self.assertTrue(
            similarity_score < self.similarity_threshold,
            f"Similarity score {similarity_score} is not less than the threshold {self.similarity_threshold}",
        )
