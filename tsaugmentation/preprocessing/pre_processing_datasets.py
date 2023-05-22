import pandas as pd
import pickle
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

    def __init__(
        self,
        dataset,
        freq,
        input_dir="./",
        top=500,
        test_size=None,
        sample_perc=None,
        weekly_m5=True,
    ):
        self.weekly = weekly_m5
        if dataset == "m5":
            dataset = dataset.capitalize()
        self.dataset = dataset
        self.freq = freq
        self.input_dir = input_dir
        self.api = "http://94.60.148.158:8086/apidownload/"
        self.top = top
        self.test_size = test_size
        self.sample_perc = sample_perc
        if self.sample_perc is not None and self.sample_perc > 1:
            raise ValueError("sample_perc must be between 0 and 1")
        elif self.sample_perc:
            self.sample_perc_int = int(self.sample_perc * 100)
        else:
            self.sample_perc_int = ""
        self._create_directories()
        self.n = {"prison": 48, "tourism": 228, "m5": 275, "police": 334}

        self.pickle_path = (
            f"{self.input_dir}data/original_datasets/"
            f"{self.dataset}_groups_{self.freq}_{self.sample_perc_int}_"
            f"{self.test_size}.pickle"
        )

    def _create_directories(self):
        # Create directory to store original datasets if does not exist
        Path(f"{self.input_dir}data").mkdir(parents=True, exist_ok=True)
        Path(f"{self.input_dir}data/original_datasets").mkdir(
            parents=True, exist_ok=True
        )

    @staticmethod
    def _floor(x, freq):
        offset = x[0].ceil(freq) - x[0] + datetime.timedelta(days=-1)
        return (x + offset).floor(freq) - offset

    def _get_dataset_path(self, file_type="csv"):
        path = f"{self.input_dir}data/original_datasets/{self.dataset}.{file_type}"
        if not os.path.isfile(path):
            try:
                request.urlretrieve(f"{self.api}{self.dataset}", path)
            except request.URLError as e:
                print(f"Failed to download the dataset. Error: {e}")
        return path

    @staticmethod
    def _load_pickle_file(file_path):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as handle:
                return pickle.load(handle)

    @staticmethod
    def _transform_and_group_stv(stv):
        """Transforms and groups the stv DataFrame."""
        stv = stv.melt(
            list(stv.columns[:6]),
            var_name="day",
            value_vars=list(stv.columns[6:]),
            ignore_index=True,
        )

        stv = (
            stv.groupby(["dept_id", "cat_id", "store_id", "state_id", "item_id", "day"])
            .sum("value")
            .reset_index()
        )

        return stv

    @staticmethod
    def _generate_calendar_days(stv, cal):
        """Generates a DataFrame of calendar days."""
        days_calendar = np.concatenate(
            (
                stv["day"].unique().reshape(-1, 1),
                cal["date"][:-56].unique().reshape(-1, 1),
            ),
            axis=1,
        )
        df_caldays = pd.DataFrame(days_calendar, columns=["day", "Date"])

        return df_caldays

    def _convert_to_weekly_data(self, stv, cols):
        """Converts the stv DataFrame to weekly data."""
        rule = "7D"
        f = self._floor(stv.index, rule)

        cols_group = cols.copy()
        cols_group.append(f)

        stv_weekly = stv.groupby(cols_group).sum()

        return stv_weekly

    def _filter_top_series(self, df, group_columns):
        """Filters the top series from the df DataFrame."""
        sort_column = (
            "value"
            if "value" in df.columns
            else "Count"
            if "Count" in df.columns
            else None
        )
        if sort_column is None:
            raise ValueError("Neither 'value' nor 'Count' column found in DataFrame")

        df_top = (
            df.groupby(group_columns)
            .sum()
            .reset_index()
            .sort_values(by=sort_column, ascending=False)
            .head(self.top)
            .drop(sort_column, axis=1)
        ).reset_index(drop=True)

        df_f = pd.merge(
            df_top,
            df.reset_index(drop=True),
            on=group_columns,
            how="left",
        )

        df_f.drop_duplicates(inplace=True)

        return df_f

    def _generate_groups(self, df_pivot, groups_input, seasonality, h):
        """Generates groups from the pivoted DataFrame."""

        groups = generate_groups_data_flat(
            y=df_pivot,
            dates=list(df_pivot.index),
            groups_input=groups_input,
            seasonality=seasonality,
            h=h,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)

        with open(self.pickle_path, "wb") as handle:
            pickle.dump(groups, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return groups

    def _load_and_preprocess_data(self, path, date_column, drop_columns=None):
        """Loads and preprocesses data from the specified path."""
        if not path:
            return None
        df = pd.read_csv(path, sep=",")
        if self.test_size:
            test_size = self.test_size * self.n[self.dataset]
            df = df[:test_size]
        if drop_columns:
            df = df.drop(drop_columns, axis=1)
        df["Date"] = df[date_column].astype("datetime64[ns]")
        return df

    @staticmethod
    def _pivot_data(df, index, columns, values):
        """Pivots the specified DataFrame."""
        df_pivot = df.pivot(index=index, columns=columns, values=values)
        return df_pivot

    def _prison(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

        path = self._get_dataset_path()
        prison = self._load_and_preprocess_data(path, "t", ["Unnamed: 0"])
        if prison is None:
            return {}

        prison["Date"] = pd.PeriodIndex(prison["Date"], freq="Q").to_timestamp()
        prison_pivot = self._pivot_data(
            prison, "Date", ["state", "gender", "legal"], "count"
        )

        groups_input = {"state": [0], "gender": [1], "legal": [2]}
        groups = self._generate_groups(prison_pivot, groups_input, 4, 8)
        return groups

    def _tourism(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

        path = self._get_dataset_path()
        tourism = self._load_and_preprocess_data(path, "Date")
        if tourism is None:
            return {}

        tourism_pivot = self._pivot_data(
            tourism, "Date", ["state", "zone", "region", "purpose"], "Count"
        )
        tourism_pivot = tourism_pivot.reindex(sorted(tourism_pivot.columns), axis=1)

        groups_input = {"state": [0], "zone": [1], "region": [2], "purpose": [3]}
        groups = self._generate_groups(tourism_pivot, groups_input, 12, 24)
        return groups

    def _m5(self):
        """Preprocess the M5 dataset."""
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

        path = self._get_dataset_path(file_type="zip")
        if not path:
            return {}

        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(f"{self.input_dir}data/original_datasets/")

        input_dir = f"{self.input_dir}data/original_datasets/m5-data"
        cal = pd.read_csv(f"{input_dir}/calendar.csv")
        stv = pd.read_csv(f"{input_dir}/sales_train_validation.csv")

        if self.test_size:
            stv = stv[: self.test_size]

        stv = self._transform_and_group_stv(stv)
        df_caldays = self._generate_calendar_days(stv, cal)
        stv = stv.merge(df_caldays, how="left", on="day")
        stv["Date"] = stv["Date"].astype("datetime64[ns]")
        stv = stv.set_index("Date")
        cols = ["dept_id", "cat_id", "store_id", "state_id", "item_id"]

        if self.weekly:
            stv = self._convert_to_weekly_data(stv, cols)

        stv = stv.reset_index()

        if self.top:
            stv = self._filter_top_series(stv, cols)

        stv_pivot = self._pivot_data(
            stv.reset_index(),
            "Date",
            cols,
            "value",
        )
        stv_pivot = stv_pivot.fillna(0)

        groups_input = {
            "Department": [0],
            "Category": [1],
            "Store": [2],
            "State": [3],
            "Item": [4],
        }

        seasonality, h = (52, 12) if self.weekly else (365, 30)
        groups = self._generate_groups(stv_pivot, groups_input, seasonality, h)
        return groups

    def _police(self):
        data = self._load_pickle_file(self.pickle_path)
        if data is not None:
            return data

        path = self._get_dataset_path(file_type="xlsx")
        police = pd.read_excel(path)
        cols = ["Crime", "Beat", "Street", "ZIP"]
        cols_date = cols.copy()
        cols_date.append("Date")

        # Drop unwanted columns
        police = police.drop(
            [
                "RMSOccurrenceHour",
                "StreetName",
                "Suffix",
                "NIBRSDescription",
                "Premise",
            ],
            axis=1,
        )
        police.columns = [
            "Id",
            "Date",
            "Crime",
            "Count",
            "Beat",
            "Block",
            "Street",
            "City",
            "ZIP",
        ]
        police = police.drop(["Id"], axis=1)
        police = police.astype({"ZIP": "string"})

        police = self._filter_top_series(police, ["Crime", "Beat", "Street", "ZIP"])

        police = self._fill_missing_dates(police, cols_date)

        police_pivot = self._pivot_data(police.reset_index(), "Date", cols, "Count")
        police_pivot = police_pivot.fillna(0)

        if self.test_size:
            police_pivot = police_pivot.iloc[:, : self.test_size]

        groups_input = {"Crime": [0], "Beat": [1], "Street": [2], "ZIP": [3]}

        groups = self._generate_groups(police_pivot, groups_input, 365, 30)
        return groups

    @staticmethod
    def _fill_missing_dates(df, cols):
        """
        Fill missing dates in DataFrame.
        """
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)

        if "Date" in cols:
            cols.remove("Date")

        df = df.groupby(cols).resample("D").sum().reset_index().sort_values(by="Date")

        df["Count"].fillna(0, inplace=True)

        return df

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, "_" + self.dataset.lower())(self)
        return dataset_new
