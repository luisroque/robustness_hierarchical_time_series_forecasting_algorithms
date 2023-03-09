import unittest
import numpy as np
import tsaugmentation as tsag


class TestModel(unittest.TestCase):

    def test_import_prison(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison', test_size=48*2).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (40, 2))

    def test_import_tourism(self):
        self.data = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228*50).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (204, 50))

    def test_import_m5(self):
        self.data = tsag.preprocessing.PreprocessDatasets('m5', test_size=2).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (263, 2))

    def test_import_police(self):
        self.data = tsag.preprocessing.PreprocessDatasets('police', top=2).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (304, 2))

    def test_import_prison_50perc_data(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison', test_size=48 * 2, sample_perc=0.5).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (int((48-self.data['h'])/2)+1, 2))

    def test_import_prison_50perc_data_x_values(self):
        self.data = tsag.preprocessing.PreprocessDatasets('prison', test_size=48 * 2, sample_perc=0.5).apply_preprocess()
        self.assertListEqual(self.data['predict']['x_values'][-self.data['h']:], list(np.arange(40, 48)))

    def test_import_tourism_50perc_data(self):
        self.data = tsag.preprocessing.PreprocessDatasets('tourism', test_size=228 * 50, sample_perc=0.5).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (int((228-self.data['h'])/2)+1, 50))

    def test_import_m5_50perc_data(self):
        self.data = tsag.preprocessing.PreprocessDatasets('m5', test_size=2, sample_perc=0.5).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (int((275-self.data['h'])/2)+1, 2))

    def test_import_police_50perc_data(self):
        self.data = tsag.preprocessing.PreprocessDatasets('police', top=2, sample_perc=0.5).apply_preprocess()
        self.assertTrue(self.data['train']['data'].shape == (int((334-self.data['h'])/2)+1, 2))
