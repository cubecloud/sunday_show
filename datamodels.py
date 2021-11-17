import os
import sys
import time
import copy
import pytz
import numpy as np
import datetime
import pandas as pd
# from pandas import DataFrame
from tsdataparams import TSDataParams, DatasetParams
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple

from tsdataparams import DatasetParams
import tensorflow as tf

__version__ = 0.0016

DEV: str
DEV_DATA: str


def get_local_timezone_name():
    if time.daylight:
        offset_hour = time.altzone / 3600
    else:
        offset_hour = time.timezone / 3600

    offset_hour_msg = f"{offset_hour:.0f}"
    if offset_hour > 0:
        offset_hour_msg = f"+{offset_hour:.0f}"
    return f'Etc/GMT{offset_hour_msg}'


# noinspection PyStatementEffect,PyUnresolvedReferences
def prepare_dev_stationary():
    import os
    import sys
    global DEV
    global DEV_DATA
    try:
        test_ipython = str(get_ipython())
    except NameError:
        print('Running on local environment')
        DEV = os.getcwd()
    else:
        if 'google.colab' in test_ipython:
            print('Running on CoLab')
            from google.colab import drive
            drive.mount('/content/drive')
            cmd = "pip install -U keras-tuner"
            os.system(cmd)
            DEV = '/content/drive/MyDrive/Python/sunday'
        elif 'ipykernel' in test_ipython:
            print('Running on Jupyter Notebook')
            DEV = os.getcwd()
    sys.path.append(DEV)
    path_head = os.path.split(DEV)[0]
    DEV_DATA = os.path.join(path_head, 'sunday_show/')
    pass


class TSModel:
    def __init__(self,
                 dataset_period: Tuple[str, str],
                 data_path: str) -> None:
        """
        Returns:
            None
        """
        self.data_storage: dict = {}
        # self.data_df: pd.DataFrame = pd.DataFrame()
        self.pair_symbol: str
        self.dataset_period: Tuple = dataset_period
        self.pairs_data_path = os.path.join(data_path, "pairs_data")
        self.default_columns = ("timestamp",
                                "open",
                                "close",
                                "high",
                                "low",
                                "volume",
                                "trades")
        self.history: dict = {}
        self.tsdp = TSDataParams
        self.use_pairs: Tuple = ()
        self.use_intervals: Tuple = ()
        self.use_trend_weights: Tuple = ()
        self.use_steps_in_the_last: Tuple = ()
        self.dataset_model_compiled = False
        pass

    def get_symbol_interval_data(self,
                                 pair_symbol,
                                 time_interval="1h",
                                 usecols=("timestamp",
                                          "open",
                                          "close",
                                          "high",
                                          "low",
                                          "volume",
                                          "trades"),
                                 tz=None,
                                 ):

        """
        Loading currency trading data from file
        """
        path_filename = os.path.join(self.pairs_data_path, f"{pair_symbol}-{time_interval}-data.csv")
        if not "timestamp" in usecols:
            usecols.insert(0, "timestamp")

        data_df = pd.read_csv(path_filename, usecols=usecols)
        data_df.rename(columns={'timestamp': 'datetime'}, inplace=True)
        mask = (data_df['datetime'] >= self.dataset_period[0]) & (data_df['datetime'] <= self.dataset_period[1])
        data_df = data_df[mask]

        if tz is not None:
            temp_index = pd.to_datetime(data_df['datetime'])
            data_df.index, data_df.index.name = temp_index, "datetimeindex"
            """
            Warning! Time converted to local time
            """
            data_df.index = data_df.index.tz_localize(tz="UTC")
            data_df.index = data_df.index.tz_convert(tz=tz)
            """
            Warning! Time converted to local time
            """
        else:
            temp_index = pd.to_datetime(data_df['datetime'])
            data_df.index, data_df.index.name = temp_index, "datetimeindex"
        data_df.drop(columns=['datetime'], inplace=True)
        return data_df

    @staticmethod
    def split_datetime_data(datetime_index: pd.DatetimeIndex,
                            cols_defaults: list
                            ) -> pd.DataFrame:
        """
        Args:

            datetime_index (pd.DatetimeIndex):      datetimeindex object
            cols_defaults (list):                   list of names for dividing datetime for encoding

        Returns:
            object:     pd.Dataframe with columns for year, quarter, month, weeknum, weekday, hour(*), minute(*)
                        * only if datetimeindex have data about
        """
        temp_df: pd.DataFrame = pd.DataFrame()
        temp_df.index, temp_df.index.name = datetime_index, "datetimeindex"

        datetime_funct: dict = {'year': temp_df.index.year,
                                'quarter': temp_df.index.quarter,
                                'month': temp_df.index.month,
                                'weeknum': temp_df.index.isocalendar().week,
                                'weekday': temp_df.index.day_of_week,
                                'hour': temp_df.index.hour,
                                'minute': temp_df.index.minute,
                                }
        for col_name in cols_defaults:
            if col_name == 'weeknum':
                temp_df[col_name] = temp_df.index.isocalendar().week
            else:
                if datetime_funct[col_name].nunique() != 1:
                    temp_df[col_name] = datetime_funct[col_name]
        return temp_df

    # noinspection PyTypeChecker
    def prepare_datetime_4embedding(self,
                                    cols_defaults=('year',
                                                   'quarter',
                                                   'month',
                                                   'weeknum',
                                                   'weekday',
                                                   'hour',
                                                   'minute'
                                                   ),
                                    ) -> pd.DataFrame:
        """
        Args:
            cols_defaults (list):       default columns names for encoding

        Returns:
            object:     pd.Dataframe with dummy encoded datetimeindex columns with prefix 'de_'
        """
        for pair_symbol in self.use_pairs:
            for interval in self.use_intervals:
                raw_df = self.data_storage[pair_symbol][interval]['raw_df']

                de_df = TSModel.split_datetime_data(raw_df.index,
                                                    cols_defaults,
                                                    )

                cols_names = de_df.columns
                de_df = pd.get_dummies(de_df, columns=cols_names, drop_first=False)
                for col in de_df.columns:
                    de_df.rename(columns={col: f'de_{col}'}, inplace=True)
                self.data_storage[pair_symbol][interval]['de_df'] = copy.deepcopy(de_df)
        pass

    @staticmethod
    def calculate_pct_change(input_df: pd.DataFrame,
                             use_col: str or list,
                             steps_in_the_last,
                             drop_input: bool = True,
                             drop_nan: bool = False,
                             ) -> pd.DataFrame:

        assert type(use_col) is str or list, f"Error: {use_col}  type is {type(use_col)} not str or list"
        df_pct_change = pd.DataFrame()
        df_pct_change.index = input_df.index
        try:
            for col_name in use_col:
                df_pct_change[col_name] = input_df[col_name]
        except KeyError:
            df_pct_change[use_col] = input_df[use_col]


        for col_name in df_pct_change.columns:
            for idx in range(1, steps_in_the_last + 1):
                indicator_name = f"{col_name}_pct_chng_{idx}"
                df_pct_change[indicator_name] = df_pct_change[col_name].pct_change(idx)
        if drop_nan:
            df_pct_change = df_pct_change.dropna()
        if drop_input:
            df_pct_change = df_pct_change.drop(columns=use_col)
        return df_pct_change

    @staticmethod
    def calculate_trend(input_df: pd.DataFrame,
                        W: float = 0.15
                        ) -> pd.DataFrame:
        """
        Args:
            input_df (pd.DataFrame):    input DataFrame with index and OHCLV data
            W (float):                  weight percentage for calc trend default 0.15
        Returns:
           trend (pd.DataFrame):        output DataFrame
        """

        # volatility = calc_volatility(df)
        # max_vol = volatility.max()
        # min_vol = volatility.min()
        # std_vol = volatility.std()
        # vol_scaling = max_vol*W/min_vol
        # print(f"{min_vol}/{max_vol}= {min_vol/max_vol}, {std_vol} scaling={vol_scaling}")

        # TODO: create autoweight calculation based on current volatility (check hypothesis)
        def weighted_W(idx, W: float = 0.15):
            """
            Args:
                idx (int):  index of the row from pd.Dataframe
                W (float): increase or decrease weight (percentage) of the 'close' bar

            Returns:

            """
            weighted = W
            # if volatility[idx]< 0.9:
            #     weighted = 0.13
            # else:
            #     weighted = W - (np.log(volatility[idx])*0.8)
            #     # print(weighted)
            return weighted

        """
        Setup. Getting first data and initialize variables
        """
        x_bar_0 = [input_df.index[0],  # [0] - datetime
                   input_df["open"][0],  # [1] - open
                   input_df["high"][0],  # [2] - high
                   input_df["low"][0],  # [3] - low
                   input_df["close"][0],  # [4] - CLOSE
                   input_df["volume"][0],
                   input_df["trades"][0]
                   ]
        FP_first_price = x_bar_0[4]
        xH_highest_price = x_bar_0[2]
        HT_highest_price_timemark = 0
        xL_lowest_price = x_bar_0[3]
        LT_lowest_price_timemark = 0
        Cid = 0
        FPN_first_price_idx = 0
        Cid_array = np.zeros(input_df.shape[0])
        """
        Setup. Getting first data and initialize variables
        """

        for idx in range(input_df.shape[0] - 1):
            x_bar = [input_df.index[idx],
                     input_df["open"][idx],
                     input_df["high"][idx],
                     input_df["low"][idx],
                     input_df["close"][idx],
                     input_df["volume"][idx],
                     input_df["trades"][idx]
                     ]
            # print(x_bar)
            # print(x_bar[4])
            W = weighted_W(idx, W)
            if x_bar[2] > (FP_first_price + x_bar_0[4] * W):
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = idx
                FPN_first_price_idx = idx
                Cid = 1
                Cid_array[idx] = 1
                Cid_array[0] = 1
                break
            if x_bar[3] < (FP_first_price - x_bar_0[4] * W):
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = idx
                FPN_first_price_idx = idx
                Cid = -1
                Cid_array[idx] = -1
                Cid_array[0] = -1
                break

        for ix in range(FPN_first_price_idx + 1, input_df.shape[0] - 2):
            x_bar = [input_df.index[ix],
                     input_df["open"][ix],
                     input_df["high"][ix],
                     input_df["low"][ix],
                     input_df["close"][ix],
                     input_df["volume"][ix],
                     input_df["trades"][ix]
                     ]
            W = weighted_W(ix, W)
            if Cid > 0:
                if x_bar[2] > xH_highest_price:
                    xH_highest_price = x_bar[2]
                    HT_highest_price_timemark = ix
                if x_bar[2] < (
                        xH_highest_price - xH_highest_price * W) and LT_lowest_price_timemark <= HT_highest_price_timemark:
                    for j in range(1, input_df.shape[0] - 1):
                        if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                            Cid_array[j] = 1
                    xL_lowest_price = x_bar[2]
                    LT_lowest_price_timemark = ix
                    Cid = -1

            if Cid < 0:
                if x_bar[3] < xL_lowest_price:
                    xL_lowest_price = x_bar[3]
                    LT_lowest_price_timemark = ix
                if x_bar[3] > (
                        xL_lowest_price + xL_lowest_price * W) and HT_highest_price_timemark <= LT_lowest_price_timemark:
                    for j in range(1, input_df.shape[0] - 1):
                        if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                            Cid_array[j] = -1
                    xH_highest_price = x_bar[2]
                    HT_highest_price_timemark = ix
                    Cid = 1

        # TODO: rewrite this block in intelligent way !!! Now is working but code is ugly
        """ Checking last bar in input_df """
        ix = input_df.shape[0] - 1
        x_bar = [input_df.index[ix],
                 input_df["open"][ix],
                 input_df["high"][ix],
                 input_df["low"][ix],
                 input_df["close"][ix],
                 input_df["volume"][ix],
                 input_df["trades"][ix]
                 ]
        if Cid > 0:
            if x_bar[2] > xH_highest_price:
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
            if x_bar[2] <= xH_highest_price:
                for j in range(1, input_df.shape[0]):
                    if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                        Cid_array[j] = 1
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
                Cid = -1
        if Cid < 0:
            if x_bar[3] < xL_lowest_price:
                xL_lowest_price = x_bar[3]
                LT_lowest_price_timemark = ix
                # print(True)
            if x_bar[3] >= xL_lowest_price:
                for j in range(1, input_df.shape[0]):
                    if HT_highest_price_timemark < j <= LT_lowest_price_timemark:
                        Cid_array[j] = -1
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
                Cid = 1
        if Cid > 0:
            if x_bar[2] > xH_highest_price:
                xH_highest_price = x_bar[2]
                HT_highest_price_timemark = ix
            if x_bar[2] <= xH_highest_price:
                for j in range(1, input_df.shape[0]):
                    if LT_lowest_price_timemark < j <= HT_highest_price_timemark:
                        Cid_array[j] = 1
                # xL_lowest_price = x_bar[3]
                # LT_lowest_price_timemark = ix
                # Cid = -1
        trend = pd.DataFrame(data=Cid_array,
                             index=input_df.index,
                             columns=["trend"])
        return trend

    def prepare_trend_data(self,
                           use_trend_weights: Tuple = (0.15, 0.055),
                           ) -> None:
        for pair_symbol in self.use_pairs:
            for interval in self.use_intervals:
                raw_df = self.data_storage[pair_symbol][interval]['raw_df']
                for weight in use_trend_weights:
                    trend_df = self.calculate_trend(raw_df, W=weight)
                    self.data_storage[pair_symbol][interval]['trend_df'][weight] = copy.deepcopy(trend_df)
        pass

    def prepare_pct_change_data(self,
                                steps_in_the_last,
                                ohlc_data: bool = True,
                                ) -> None:
        """
        Args:

            steps_in_the_last (int):
            ohlc_data (bool):           if True using 'open', 'high, 'low', 'close' columns
                                        if False using only 'close' column
        """
        use_col = "close"
        if ohlc_data:
            use_col = ['open', 'high', 'low', 'close',]
        for pair_symbol in self.use_pairs:
            for interval in self.use_intervals:
                raw_df = self.data_storage[pair_symbol][interval]['raw_df']
                pct_change_df = self.calculate_pct_change(raw_df,
                                                          use_col,
                                                          steps_in_the_last=steps_in_the_last)
                if ohlc_data:
                    self.data_storage[pair_symbol][interval]['pct_df']['OHLC'] = copy.deepcopy(pct_change_df)
                else:
                    self.data_storage[pair_symbol][interval]['pct_df']['close'] = copy.deepcopy(pct_change_df)
        pass


    @staticmethod
    def show_trend_buy_sell_points(input_df: pd.DataFrame,
                                   use_col: str = "trend",
                                   show: bool = False) -> pd.DataFrame:
        """
        Args:
            input_df (pd.DataFrame):    pd.DataFrame with column with trend_data [-1,1]
            use_col (str):              name of column. default "trend"
            show (bool):                show figure with trend, dataframe must have OHLCV columns

        Returns:
            pd.DataFrame:  pd.Dataframe with "buy" & "sell" column
        """
        col_list = input_df.columns.to_list()
        try:
            trend_col_num = col_list.index(use_col)
        except ValueError:
            msg = f"Error: {use_col} not found in pd.DataFrame only {col_list}"
            sys.exit(msg)

        if show:
            try:
                col_list.index("close")
            except ValueError:
                msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
                print(msg)
                show = False

        col_list_len = len(col_list)
        current_trend = 0
        buy_sell_df = copy.deepcopy(input_df)
        buy_sell_df["buy"] = np.nan
        buy_sell_df["sell"] = np.nan
        idx = 0
        while idx < buy_sell_df.shape[0] - 1:
            previous_trend = current_trend
            idx += 1
            current_trend = buy_sell_df.iloc[idx, trend_col_num]
            if current_trend == -1 and previous_trend == 1:
                # mark previous position for selling
                buy_sell_df.iloc[idx - 1, col_list_len + 1] = -1
            elif current_trend == 1 and previous_trend == -1:
                # mark previous position for buying
                buy_sell_df.iloc[idx - 1, col_list_len] = 1

        if show:
            fig = plt.figure(figsize=(45, 18))
            ax1 = fig.add_subplot(3, 1, 1)
            # Don't allow the axis to be on top of your data
            ax1.set_axisbelow(True)
            # Turn on the minor TICKS, which are required for the minor GRID
            ax1.minorticks_on()
            # Customize the major grid
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            # Customize the minor grid
            ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            ax1.plot(buy_sell_df["close"], 'b-')
            ax2 = ax1.twinx()
            ax2.plot(buy_sell_df["trend"], 'r-')
            ax2.set_ylabel('y2', color='r')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')

            ax6 = fig.add_subplot(3, 1, 2)
            # Don't allow the axis to be on top of your data
            ax6.set_axisbelow(True)
            # Turn on the minor TICKS, which are required for the minor GRID
            ax6.minorticks_on()
            # Customize the major grid
            ax6.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            # Customize the minor grid
            ax6.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

            ax6.plot(buy_sell_df["close"], 'b-')
            ax7 = ax6.twinx()
            ax7.plot(buy_sell_df['trend'], '-c')

            ax8 = ax6.twinx()
            buy_points = np.asarray(buy_sell_df['buy'] > 0)
            ax8.plot(buy_sell_df['trend'], 'gD', markevery=buy_points)

            ax9 = ax6.twinx()
            sell_points = np.asarray(buy_sell_df['sell'] < 0)
            ax9.plot(buy_sell_df['trend'], 'rD', markevery=sell_points)

            ax7.set_ylabel('y2', color='r')
            for tl in ax7.get_yticklabels():
                tl.set_color('r')

            plt.show()
        return buy_sell_df

    def check_trends_weights(self,
                             pair_symbol: str,
                             time_interval: str,
                             use_col: str = "trend"
                             ) -> None:
        """
        Args:
            pair_symbol (str):      name of pair_symbol
            use_col (str):          name of column. default "trend"

        Returns:
            None:
        """
        assert pair_symbol in self.use_pairs, f"Pair symbol {pair_symbol} is not in compiled pair symbols"
        data_df = self.data_storage[pair_symbol][time_interval]['raw_df']

        for weight in self.use_trend_weights:
            trend_df = copy.deepcopy(self.data_storage[pair_symbol][time_interval]['trend_df'][weight])

            # for visualization we use scaling of trend = 1 to data_df["close"].max()
            max_close = data_df["close"].max()
            min_close = data_df["close"].min()
            mean_close = data_df["close"].mean()
            trend_df.loc[(trend_df["trend"] == 1), "trend"] = max_close
            trend_df.loc[(trend_df["trend"] == -1), "trend"] = min_close
            trend_df.loc[(trend_df["trend"] == 0), "trend"] = mean_close
            data_df[f"trend_{weight}"] = trend_df[use_col]

        col_list = data_df.columns.to_list()

        try:
            col_list.index("close")
        except ValueError:
            msg = f"Error: 'close' column not found in pd.DataFrame only {col_list}. Can't show figure"
            sys.exit(msg)

        weights_list_len = len(self.use_trend_weights)
        fig = plt.figure(figsize=(45, 6 * weights_list_len))

        for i, weight in enumerate(self.use_trend_weights):
            ax1 = fig.add_subplot(weights_list_len, 1, i+1)
            # Don't allow the axis to be on top of your data
            ax1.set_axisbelow(True)
            # Turn on the minor TICKS, which are required for the minor GRID
            ax1.minorticks_on()
            # Customize the major grid
            ax1.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
            # Customize the minor grid
            ax1.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
            # ax1.plot(data_df.index, data_df["close"],  'b-')
            # ax2 = ax1.twinx()
            ax1.plot(data_df.index, data_df[f"trend_{weight}"], data_df.index, data_df["close"])
            ax1.set_ylabel(f'weight = {weight}', color='r')
            plt.title(f"Trend with weight: {weight}")
        plt.show()
        pass

    def init_data_storage(self) -> None:
        timezone = get_local_timezone_name()

        # setting the storage_dictionary
        use_intervals = self.tsdp.intervals_bars
        for pair_symbol in self.use_pairs:
            self.data_storage.update({pair_symbol: {}})
            intervals_dict = [{interval: {'bars': bars,
                                          'raw_df': None,
                                          'trend_df': {'weights': {}},
                                          'de_df': None,
                                          "pct_df": {'close': None,
                                                     'OHLC': None},
                                          }
                               } for interval, bars in use_intervals.items()]
            for x in intervals_dict:
                self.data_storage[pair_symbol].update(x)

        # loading raw_data to data_storage
        for pair_symbol in self.use_pairs:
            for interval in self.use_intervals:
                raw_df = self.get_symbol_interval_data(pair_symbol=pair_symbol,
                                                       time_interval=interval,
                                                       tz=timezone)
                interval_dict = self.data_storage[pair_symbol].get(interval)
                interval_dict['raw_df'] = copy.deepcopy(raw_df)
                self.data_storage[pair_symbol].update({interval: interval_dict})
        pass

    def compile(self,
                use_pairs=("BTCUSDT",
                           "ETHUSDT"),
                use_intervals=("15m",
                               "1h"),
                use_trend_weights=(0.15,
                                   0.75,
                                   0.055,
                                   0.0275,
                                   ),
                use_steps_in_the_last=(20,
                                       16,
                                       5)
                ):
        self.use_pairs = use_pairs
        self.use_intervals = use_intervals
        self.use_trend_weights = use_trend_weights
        self.use_steps_in_the_last = use_steps_in_the_last
        self.init_data_storage()
        self.prepare_trend_data(use_trend_weights)
        self.prepare_datetime_4embedding()
        self.prepare_pct_change_data(steps_in_the_last=self.use_steps_in_the_last[0])
        self.dataset_model_compiled = True
        pass

    def fit(self) -> dict:
        # TODO: fit method
        return self.history


if __name__ == '__main__':
    prepare_dev_stationary()
    pd.set_option('display.max_colwidth', None)
    symbol = "ETHUSDT"

    # noinspection PyUnboundLocalVariable
    ds = TSModel(dataset_period=('2019-12-31 20:59:59.000',
                                 '2021-10-31 20:59:59.000'),
                 data_path=DEV_DATA,
                 )
    ds.compile(use_intervals=("15m", "1h"),
               use_trend_weights=(0.15, 0.075, 0.055, 0.0275),
               )
    ds.check_trends_weights(pair_symbol=symbol,
                            time_interval="1h",
                            )
