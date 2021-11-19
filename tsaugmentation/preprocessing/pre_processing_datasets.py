import pandas as pd
from .utils import generate_groups_data_flat, generate_groups_data_matrix
import calendar
from pathlib import Path


class PreprocessDatasets:

    def __init__(self, dataset):
        self.dataset = dataset

    @staticmethod
    def _prison():
        # prison dataset
        path = Path(__file__).parent / './original_datasets/prisonLF.csv'
        prison = pd.read_csv(path, sep=",")
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

    @staticmethod
    def _tourism():
        # tourism dataset
        path = Path(__file__).parent / './original_datasets/TourismData_v3.csv'
        data = pd.read_csv(path)
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
        dataset_new = getattr(PreprocessDatasets, '_' + self.dataset)()
        return dataset_new
