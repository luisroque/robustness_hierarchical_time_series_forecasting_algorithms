import unittest
from tsaugmentation.preprocessing import PreprocessDatasets
from tsaugmentation.model.create_dataset_versions_vae import (
    CreateTransformedVersionsCVAE,
)


class PreprocessDatasetsTestCase(unittest.TestCase):
    def setUp(self):
        self.dataset = "synthetic"
        self.freq = "D"

    def test_add_synthetic_dataset(self):
        self.preprocess_datasets = PreprocessDatasets(dataset=self.dataset, freq=self.freq)
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

    def test_synthetic_dataset_CVAE(self):
        create_dataset_cvae = CreateTransformedVersionsCVAE(
            dataset_name=self.dataset,
            freq=self.freq,
            dynamic_feat_trig=False,
        )

        model, _, _ = create_dataset_cvae.fit(latent_dim=3)
        (
            preds,
            z,
            z_mean,
            z_log_var,
        ) = create_dataset_cvae.predict(model)

        dec_pred_hat = create_dataset_cvae.generate_transformed_time_series(
            cvae=model,
            z_mean=z_mean,
            z_log_var=z_log_var,
            transformation="jitter",
        )

        self.assertTrue(dec_pred_hat.shape, (100, 60))

