import os
import pickle
import numpy as np
from tsaugmentation.preprocessing.pre_processing_datasets import PreprocessDatasets
from tsaugmentation.transformations.create_dataset_versions import CreateTransformedVersions


class CreateGroups:
    def __init__(self, dataset_name: str, sample_perc: float = None):
        """Creates a new instance of CreateGroups."""
        self.dataset_name = dataset_name
        self.sample_perc = sample_perc
        self.sample_int_perc = int(sample_perc * 100) if sample_perc is not None else None

    def create_subsampled_groups(self) -> dict:
        """Creates subsampled groups from the dataset and stores them in a pickle file."""
        groups = self._preprocess_dataset(self.sample_perc)
        file_path = f"./data/subsampled_datasets/{self.dataset_name}_{self.sample_int_perc}.pkl"
        self._store_groups(groups, file_path)
        return groups

    def create_original_groups(self) -> dict:
        """Creates original groups from the dataset and stores them in a pickle file."""
        groups = self._preprocess_dataset()
        file_path = f"./data/original_datasets/{self.dataset_name}.pkl"
        self._store_groups(groups, file_path)
        return groups

    def read_subsampled_groups(self) -> dict:
        """Loads subsampled groups from a pickle file."""
        file_path = f"./data/subsampled_datasets/{self.dataset_name}_{self.sample_int_perc}.pkl"
        return self._read_groups(file_path)

    def read_original_groups(self) -> dict:
        """Loads original groups from a pickle file."""
        file_path = f"./data/original_datasets/{self.dataset_name}.pkl"
        return self._read_groups(file_path)

    def _preprocess_dataset(self, sample_perc: float = None) -> dict:
        """Preprocesses the dataset with the given sample percentage."""
        dataset = PreprocessDatasets(self.dataset_name, sample_perc=sample_perc)
        return dataset.apply_preprocess()

    @staticmethod
    def _store_groups(data_dict: dict, file_path: str) -> None:
        """Stores a dictionary as a pickle file at the specified file path."""
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data_dict, f)

    @staticmethod
    def _read_groups(file_path: str) -> dict:
        """Loads a pickle file at the specified file path and returns its contents as a dictionary."""
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def read_groups_transformed(self, method: str) -> np.ndarray:
        transformer = CreateTransformedVersions(self.dataset_name)
        transformer.read_groups_transformed(method)
        return transformer.y_loaded_transformed
