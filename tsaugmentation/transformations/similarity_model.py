from .utils import block_print, enable_print

block_print()
from dtw import dtw
enable_print()


class SimilarityModel:

    def __init__(self):
        self.model = dtw

    def estimate_similarity_two_series(self, series1, series2):
        sim = self.model(series1, series2, keep_internals=True).distance
        return sim

