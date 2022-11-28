import unittest
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
