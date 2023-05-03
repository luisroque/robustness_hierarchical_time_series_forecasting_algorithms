import unittest
from tsaugmentation.model.hyperparameter_tuning import optimize_hyperparameters


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        result = optimize_hyperparameters(dataset_name='prison', freq='Q', n_calls=5)
        print(result)

    def test_hyperparameter_tuning(self):
        pass
