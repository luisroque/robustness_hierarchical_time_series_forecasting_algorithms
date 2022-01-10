import pandas as pd
from .utils import generate_groups_data_flat, generate_groups_data_matrix
from urllib import request
from pathlib import Path
import os
import zipfile
import numpy as np
import datetime


class PreprocessDatasets:
    """
    A class used to preprocess datasets

    ...

    Attributes
    ----------
    dataset : str
        the dataset to download and preprocess
    rel_dir : str
        relative directory where to store the downloaded files (e.g. './' current dir, '../' parent dir)
    """

    def __init__(self, dataset, input_dir='./'):
        if dataset == 'm5':
            dataset = dataset.capitalize()
        self.dataset = dataset
        self.input_dir = input_dir
        self.api = 'http://www.machinelearningtimeseries.com/apidownload/'
        self._create_directories()

    def _create_directories(self):
        # Create directory to store original datasets if does not exist
        Path(f'{self.input_dir}data').mkdir(parents=True, exist_ok=True)
        Path(f'{self.input_dir}data/original_datasets').mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _floor(x, freq):
        offset = x[0].ceil(freq) - x[0] + datetime.timedelta(days=-1)
        return (x + offset).floor(freq) - offset

    def _get_dataset(self, file_type='csv'):
        path = f'{self.input_dir}data/original_datasets/{self.dataset}.{file_type}'
        # Download the original file if it does not exist
        if not os.path.isfile(path):
            try:
                request.urlretrieve(f'{self.api}{self.dataset}', path)
                return path
            except:
                print('It is not possible to download the dataset at this time!')
        else:
            return path

    def _prison(self):
        path = self._get_dataset()
        if not path:
            return {}
        prison = pd.read_csv(path, sep=",")

        prison = prison.drop('Unnamed: 0', axis =1)
        prison['t'] = prison['t'].astype('datetime64[ns]')
        prison_pivot = prison.pivot(index='t', columns=['state', 'gender', 'legal'], values='count')

        groups_input = {
        'state': [0],
        'gender': [1],
        'legal': [2]
        }

        groups = generate_groups_data_flat(y=prison_pivot,
                                   dates=list(prison_pivot.index),
                                   groups_input=groups_input,
                                   seasonality=4,
                                   h=8)
        groups = generate_groups_data_matrix(groups)
        return groups

    def _tourism(self):
        path = self._get_dataset()
        if not path:
            return {}
        tourism = pd.read_csv(path, sep=",")

        tourism['t'] = tourism['Date'].astype('datetime64[ns]')
        tourism = tourism.drop('Date', axis=1)
        tourism_pivot = tourism.pivot(index='t', columns=['state', 'zone', 'region', 'purpose'], values='Count')

        groups_input = {
            'state': [0],
            'zone': [1],
            'region': [2],
            'purpose': [3]
        }
        groups = generate_groups_data_flat(y=tourism_pivot,
                                           dates=list(tourism_pivot.index),
                                           groups_input=groups_input,
                                           seasonality=12,
                                           h=24)
        groups = generate_groups_data_matrix(groups)
        return groups

    def _m5(self):
        path = self._get_dataset(file_type='zip')
        if not path:
            return {}
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(f'{self.input_dir}data/original_datasets/')

        INPUT_DIR = f'{self.input_dir}data/original_datasets/m5-data'
        cal = pd.read_csv(f'{INPUT_DIR}/calendar.csv')
        stv = pd.read_csv(f'{INPUT_DIR}/sales_train_validation.csv')

        # Transform column wide days to single column
        stv = stv.melt(list(stv.columns[:6]), var_name='day', value_vars=list(stv.columns[6:]), ignore_index=True)

        # Group by the groups to consider (item_id have 3049 unique)
        stv = stv.groupby(['dept_id', 'cat_id', 'store_id', 'state_id', 'item_id', 'day']).sum('value').reset_index()
        days_calendar = np.concatenate((stv['day'].unique().reshape(-1, 1), cal['date'][:-56].unique().reshape(-1, 1)),
                                       axis=1)
        df_caldays = pd.DataFrame(days_calendar, columns=['day', 'Date'])

        # Add calendar days
        stv = stv.merge(df_caldays, how='left', on='day')
        stv['Date'] = stv['Date'].astype('datetime64[ns]')
        stv = stv.set_index('Date')

        # Transform in weekly data
        rule = '7D'
        f = self._floor(stv.index, rule)
        stv_weekly = stv.groupby(['dept_id', 'cat_id', 'store_id', 'state_id', 'item_id', f]).sum()

        stv_pivot = stv_weekly.reset_index().pivot(index='Date',
                                                   columns=['dept_id', 'cat_id', 'store_id', 'state_id', 'item_id'],
                                                   values='value')
        stv_pivot = stv_pivot.fillna(0)

        groups_input = {
            'Department': [0],
            'Category': [1],
            'Store': [2],
            'State': [3]
        }

        groups = generate_groups_data_flat(y=stv_pivot,
                                           dates=list(stv_pivot.index),
                                           groups_input=groups_input,
                                           seasonality=52,
                                           h=12)
        groups = generate_groups_data_matrix(groups)
        return groups

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, '_' + self.dataset.lower())(self)
        return dataset_new
