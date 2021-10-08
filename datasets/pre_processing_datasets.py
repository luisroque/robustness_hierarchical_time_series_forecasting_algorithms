import pandas as pd
from ..pre_processing import generate_groups_data_flat

class preprocess_datasets():

    def __init__(self, dataset):
        self.dataset = dataset

    def _prison():
        # prison dataset
        prison = pd.read_csv('./original_datasets/prisonLF.csv', sep=",")
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
        return groups

    def _tourism():
        # tourism dataset
        data = pd.read_csv('./original_datasets/TourismData_v3.csv')
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

        return groups

    def apply_preprocess(self):
        dataset_new = getattr(preprocess_datasets, '_' + self.dataset)()
        return dataset_new
