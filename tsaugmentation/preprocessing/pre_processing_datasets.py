import pandas as pd
from .utils import generate_groups_data_flat, generate_groups_data_matrix
from urllib import request
from pathlib import Path
import os
import zipfile
import numpy as np
import datetime
from itertools import product


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
        self, dataset, input_dir="./", top=500, test_size=None, sample_perc=None
    ):
        if dataset == "m5":
            dataset = dataset.capitalize()
        self.dataset = dataset
        self.input_dir = input_dir
        self.api = "http://94.60.148.158:8086/apidownload/"
        self.top = top
        self.test_size = test_size
        self.sample_perc = sample_perc
        if self.sample_perc is not None and self.sample_perc > 1:
            raise ValueError("sample_perc must be between 0 and 1")
        self._create_directories()

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

    def _get_dataset(self, file_type="csv"):
        path = f"{self.input_dir}data/original_datasets/{self.dataset}.{file_type}"
        # Download the original file if it does not exist
        if not os.path.isfile(path):
            try:
                request.urlretrieve(f"{self.api}{self.dataset}", path)
                return path
            except:
                print("It is not possible to download the dataset at this time!")
        else:
            return path

    def _prison(self):
        path = self._get_dataset()
        if not path:
            return {}
        prison = pd.read_csv(path, sep=",")
        if self.test_size:
            prison = prison[: self.test_size]

        prison = prison.drop("Unnamed: 0", axis=1)
        prison["Date"] = prison["t"].astype("datetime64[ns]")
        prison.drop("t", axis=1)
        prison["Date"] = pd.PeriodIndex(prison["Date"], freq="Q").to_timestamp()
        prison_pivot = prison.pivot(
            index="Date", columns=["state", "gender", "legal"], values="count"
        )

        groups_input = {"state": [0], "gender": [1], "legal": [2]}

        groups = generate_groups_data_flat(
            y=prison_pivot,
            dates=list(prison_pivot.index),
            groups_input=groups_input,
            seasonality=4,
            h=8,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)
        return groups

    def _tourism(self):
        path = self._get_dataset()
        if not path:
            return {}
        tourism = pd.read_csv(path, sep=",")
        if self.test_size:
            tourism = tourism[: self.test_size]

        tourism["Date"] = tourism["Date"].astype("datetime64[ns]")
        tourism_pivot = tourism.pivot(
            index="Date", columns=["state", "zone", "region", "purpose"], values="Count"
        )
        tourism_pivot = tourism_pivot.reindex(sorted(tourism_pivot.columns), axis=1)

        groups_input = {"state": [0], "zone": [1], "region": [2], "purpose": [3]}
        groups = generate_groups_data_flat(
            y=tourism_pivot,
            dates=list(tourism_pivot.index),
            groups_input=groups_input,
            seasonality=12,
            h=24,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)
        return groups

    def _m5(self):
        path = self._get_dataset(file_type="zip")
        if not path:
            return {}
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(f"{self.input_dir}data/original_datasets/")

        INPUT_DIR = f"{self.input_dir}data/original_datasets/m5-data"
        cal = pd.read_csv(f"{INPUT_DIR}/calendar.csv")
        stv = pd.read_csv(f"{INPUT_DIR}/sales_train_validation.csv")

        # M5 is too big to fit into memory, using test_size for testing purposes
        if self.test_size:
            stv = stv[: self.test_size]

        # Transform column wide days to single column
        stv = stv.melt(
            list(stv.columns[:6]),
            var_name="day",
            value_vars=list(stv.columns[6:]),
            ignore_index=True,
        )

        # Group by the groups to consider (item_id have 3049 unique)
        # item_id could be added here
        stv = (
            stv.groupby(["dept_id", "cat_id", "store_id", "state_id", "item_id", "day"])
            .sum("value")
            .reset_index()
        )
        days_calendar = np.concatenate(
            (
                stv["day"].unique().reshape(-1, 1),
                cal["date"][:-56].unique().reshape(-1, 1),
            ),
            axis=1,
        )
        df_caldays = pd.DataFrame(days_calendar, columns=["day", "Date"])

        # Add calendar days
        stv = stv.merge(df_caldays, how="left", on="day")
        stv["Date"] = stv["Date"].astype("datetime64[ns]")
        stv = stv.set_index("Date")

        # Transform in weekly data
        rule = "7D"
        f = self._floor(stv.index, rule)

        # item_id could be added here
        stv_weekly = stv.groupby(
            ["dept_id", "cat_id", "store_id", "state_id", "item_id", f]
        ).sum()

        # Filter top 1000 series
        stv_weekly_top = (
            stv_weekly.groupby(["dept_id", "cat_id", "store_id", "state_id", "item_id"])
            .sum()
            .sort_values(by="value", ascending=False)
            .head(self.top)
            .drop("value", axis=1)
        )

        # create a column marking df2 values
        stv_weekly_top["marker"] = 1

        # join the two, keeping all of df1's indices
        joined = pd.merge(
            stv_weekly.reset_index(),
            stv_weekly_top,
            on=["dept_id", "cat_id", "store_id", "state_id", "item_id"],
            how="left",
        )
        stv_weekly_f = joined[joined["marker"] == 1][stv_weekly.reset_index().columns]

        # item_id could be added here
        stv_pivot = stv_weekly_f.reset_index().pivot(
            index="Date",
            columns=["dept_id", "cat_id", "store_id", "state_id", "item_id"],
            values="value",
        )
        stv_pivot = stv_pivot.fillna(0)

        # item_id could be added here
        groups_input = {
            "Department": [0],
            "Category": [1],
            "Store": [2],
            "State": [3],
            "Item": [4],
        }

        groups = generate_groups_data_flat(
            y=stv_pivot,
            dates=list(stv_pivot.index),
            groups_input=groups_input,
            seasonality=52,
            h=12,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)
        return groups

    def _police(self, start_date="2021-01-01", end_date="2021-11-30"):
        path = self._get_dataset(file_type="xlsx")
        if not path:
            return {}
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

        # Filter top 1000 series
        police_top = (
            police.groupby(cols)
            .sum()
            .sort_values(by="Count", ascending=False)
            .head(self.top)
            .drop("Count", axis=1)
            .reset_index()
        )
        police = police.groupby(cols_date).sum().reset_index()

        # create a column marking df2 values
        police_top["marker"] = 1

        # join the two, keeping all of df1's indices
        joined = pd.merge(police, police_top, on=cols, how="left")
        police_f = joined[joined["marker"] == 1][police.columns]
        police_f = police_f.reset_index().drop("index", axis=1)
        police_f = police_f.groupby(cols_date).sum().reset_index().set_index("Date")

        # build a reference dataframe with all the dates to be merges, as the original does not have data for all days
        police_to_merge = (
            police_f.reset_index().drop(["Date", "Count"], axis=1).drop_duplicates()
        )
        idx = pd.date_range(start_date, end_date)
        lens = len(idx)
        rest_to_concat = pd.DataFrame(
            np.array(
                [
                    np.repeat(police_to_merge.iloc[:, i].values, lens)
                    for i in range(len(cols) - 1)
                ]
            ).T
        )
        complete_data = pd.DataFrame(product(list(police_to_merge.iloc[:, -1]), idx))
        frames = [rest_to_concat, complete_data]
        police_base = pd.concat(frames, axis=1)
        police_base.columns = cols_date

        police_f = police_f.reset_index()
        police = police_base.merge(police_f, how="left", on=cols_date)

        police_pivot = police.reset_index().pivot(
            index="Date", columns=cols, values="Count"
        )
        police_pivot = police_pivot.fillna(0)

        groups_input = {"Crime": [0], "Beat": [1], "Street": [2], "ZIP": [3]}

        groups = generate_groups_data_flat(
            y=police_pivot,
            dates=list(police_pivot.index),
            groups_input=groups_input,
            seasonality=7,
            h=30,
            sample_perc=self.sample_perc,
        )
        groups = generate_groups_data_matrix(groups)

        return groups

    def apply_preprocess(self):
        dataset_new = getattr(PreprocessDatasets, "_" + self.dataset.lower())(self)
        return dataset_new
