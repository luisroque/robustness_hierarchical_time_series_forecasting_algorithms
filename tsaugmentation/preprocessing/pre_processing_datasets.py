import pandas as pd
from .utils import generate_groups_data_flat, generate_groups_data_matrix
import calendar
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
        data['Year'] = data['Year'].fillna(method='ffill')

        d = dict((v,k) for k,v in enumerate(calendar.month_name))
        data.Month = data.Month.map(d)
        data = data.assign(t=pd.to_datetime(data[['Year', 'Month']].assign(day=1))).set_index('t')
        data = data.drop(['Year', 'Month'], axis=1)

        groups_input = {
        'State': [0,1],
        'Zone': [0,2],
        'Region': [0,3],
        'Purpose': [3,6]
        }

        groups = generate_groups_data_flat(data, groups_input, seasonality=12, h=12)
        groups = generate_groups_data_matrix(groups)
        return groups

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, '_' + self.dataset)(self)
        return dataset_new
