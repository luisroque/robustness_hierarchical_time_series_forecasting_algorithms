import pandas as pd
from .utils import generate_groups_data_flat, generate_groups_data_matrix
from urllib import request
from pathlib import Path
import os


class PreprocessDatasets:

    def __init__(self, dataset):
        self.dataset = dataset
        self.api = 'http://www.machinelearningtimeseries.com/apidownload/'
        self._create_directories()

    @staticmethod
    def _create_directories():
        # Create directory to store original datasets if does not exist
        Path("./original_datasets").mkdir(parents=True, exist_ok=True)
        # Create directory to store transformed datasets if does not exist
        Path("./transformed_datasets").mkdir(parents=True, exist_ok=True)

    def _get_dataset(self):
        path = f'./original_datasets/{self.dataset}.csv'
        # Download the original file if it does not exist
        if not os.path.isfile(path):
            try:
                request.urlretrieve(f'{self.api}{self.dataset}', path)
                return pd.read_csv(path, sep=",")
            except:
                print('It is not possible to download the dataset at this time!')
                return pd.DataFrame()
        else:
            return pd.read_csv(path, sep=",")

    def _prison(self):
        prison = self._get_dataset()
        if prison.empty:
            return {}
        prison = prison.drop('Unnamed: 0', axis =1)
        prison['t'] = prison['t'].astype('datetime64[ns]')
        prison_pivot = prison.pivot(index='t',columns=['state', 'gender', 'legal'], values='count')

        groups_input = {
        'state': [0],
        'gender': [1],
        'legal': [2]
        }

        groups = generate_groups_data_flat(y = prison_pivot, 
                                   groups_input = groups_input, 
                                   seasonality=4, 
                                   h=8)
        groups = generate_groups_data_matrix(groups)
        return groups

    def _tourism(self):
        data = self._get_dataset()
        if data.empty:
            return {}

        data['t'] = data['Date'].astype('datetime64[ns]')
        data = data.drop('Date', axis=1)
        data_pivot = data.pivot(index='t', columns=['state', 'zone', 'region', 'purpose'], values='Count')

        groups_input = {
            'state': [0],
            'zone': [1],
            'region': [2],
            'purpose': [3]
        }

        groups = generate_groups_data_flat(y=data_pivot,
                                           groups_input=groups_input,
                                           seasonality=12,
                                           h=24)
        groups = generate_groups_data_matrix(groups)
        return groups

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, '_' + self.dataset)(self)
        return dataset_new
