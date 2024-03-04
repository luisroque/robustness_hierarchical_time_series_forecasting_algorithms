import unittest
from tsaugmentation.preprocessing import PreprocessDatasets


class TestLoadCompletem5(unittest.TestCase):

    def test_load_complete_m5(self):
        m5 = PreprocessDatasets(dataset='m5', freq='D', weekly_m5=False, top=None).apply_preprocess()
        self.assertEqual(m5['train']['s'], 30490)
