import unittest
from tsaugmentation.transformations.create_dataset_versions import CreateTransformedVersions
import shutil
import os
import matplotlib.pyplot as plt
from tsaugmentation.transformations.compute_similarities_summary_metrics import ComputeSimilarities
import numpy as np
from tsaugmentation.visualization.visualize_transformed_datasets import Visualizer


class TestCreateTransformedDatasets(unittest.TestCase):

    def setUp(self):
        self.dataset1 = 'prison'
        self.dataset2 = 'tourism'
        self.transformed_datasets = CreateTransformedVersions(self.dataset2)
        self.transformed_datasets.parameters = {"jitter": 0.5,
                                           "scaling": 0.1,
                                           "magnitude_warp": 0.05,
                                           "time_warp": 0.05}
        self.transformed_datasets.create_new_version_single_transf()
        np.random.seed(0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree("./original_datasets")
        shutil.rmtree("./transformed_datasets")

    def test_create_new_version(self):
        td = CreateTransformedVersions(self.dataset1)
        y_new = td._create_new_version('test', save=False)
        self.assertTrue(y_new.shape == (4, 6, 10, 40, 32))

    def test_create_correct_number_transformed_datasets_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset1)
        transformed_datasets.create_new_version_single_transf()
        # shape (n_transformations + random_transf , n_versions, n_samples, n_points_train, n_series)
        self.assertTrue(transformed_datasets.y_new_all.shape == (4, 6, 10, 40, 32))

    def test_create_correct_number_transformed_datasets_FILES_single_transf(self):
        transformed_datasets = CreateTransformedVersions(self.dataset1)
        transformed_datasets.create_new_version_single_transf()
        file_count = len([name for name in os.listdir('./transformed_datasets/')])
        self.assertEqual(file_count, 25)

    def test_create_transformations_with_tourism_dataset(self):
        for i in range(4):
            plt.plot(self.transformed_datasets.y_new_all[i, 0, 9][:, 0],
                       label=f'{self.transformed_datasets.transformations[i]}_v1')
            plt.plot(self.transformed_datasets.y_new_all[i, 5, 9][:, 0],
                       label=f'{self.transformed_datasets.transformations[i]}_v6')
            plt.plot(self.transformed_datasets.y[:, 0], label='original')
            plt.title(f'{self.transformed_datasets.transformations[i]}')
            plt.legend()
            plt.show()

        mean_sim_time_warp_version_1 = ComputeSimilarities(dataset=self.transformed_datasets.y,
                                                           transf_dataset=self.transformed_datasets.y_new_all[3, 0, 9]) \
            .compute_mean_similarity_elementwise()

        mean_sim_time_warp_version_6 = ComputeSimilarities(dataset=self.transformed_datasets.y,
                                                           transf_dataset=self.transformed_datasets.y_new_all[3, 5, 9]) \
            .compute_mean_similarity_elementwise()

        mean_sim_magnitude_warp_version_1 = ComputeSimilarities(dataset=self.transformed_datasets.y,
                                                                transf_dataset=self.transformed_datasets.y_new_all[2, 0, 9]) \
            .compute_mean_similarity_elementwise()

        mean_sim_magnitude_warp_version_6 = ComputeSimilarities(dataset=self.transformed_datasets.y,
                                                                transf_dataset=self.transformed_datasets.y_new_all[2, 5, 9]) \
            .compute_mean_similarity_elementwise()

        self.assertEqual(mean_sim_magnitude_warp_version_6, 5570.868881506015)
        self.assertEqual(mean_sim_magnitude_warp_version_1, 1135.2260361015149)

        self.assertEqual(mean_sim_time_warp_version_6, 6863.478158992539)
        self.assertEqual(mean_sim_time_warp_version_1, 5181.410707701956)

    def test_create_transformations_with_tourism_dataset_and_compare_with_files(self):
        vi = Visualizer(self.dataset2)
        vi._read_files(method='single_transf_time_warp')

        for i in range(4):
            vi._read_files(method=f'single_transf_{self.transformed_datasets.transformations[i]}')
            plt.plot(self.transformed_datasets.y_new_all[i, 5, 9][:, 0],
                       label=f'{self.transformed_datasets.transformations[i]}_v6')
            plt.plot(vi.y_new[5, 9][:, 0],
                     label=f'{self.transformed_datasets.transformations[i]}_v6_from_file')
            plt.title(f'{self.transformed_datasets.transformations[i]}')
            plt.legend()
            plt.show()

        mean_sim_transf_and_file_single_series = ComputeSimilarities(dataset=self.transformed_datasets.y_new_all[i, 5, 9][:, 10].reshape(-1, 1),
                                                                     transf_dataset=vi.y_new[5, 9][:, 10].reshape(-1, 1)) \
            .compute_mean_similarity_elementwise()
        self.assertEqual(mean_sim_transf_and_file_single_series, 0)


