import datetime

__version__ = 0.0003

class TSDataParams:
    intervals_bars: dict = {'1m': 1,
                            '3m': 3,
                            '5m': 5,
                            '15m': 15,
                            '30m': 30,
                            '1h': 60,
                            '2h': 120,
                            '4h': 240,
                            '6h': 360,
                            '8h': 480,
                            '12h': 720,
                            '1D': 1440,
                            '3D': 4320,
                            '1W': 10080,
                            '1M': 43200,
                            }

    intervals_dict = {"1m": {"bars": 1,
                             "val_len": 43200,
                             "test": 720,
                             "steps_in_the_last": 5,  # 5
                             "n_steps": 19,
                             "forward_lag": 1,
                             "plato_lvl": 0.0005,
                             "period": ('2020-01-01 10:00:00.000',
                                        '2021-12-31 23:59:00.000'),
                             "trend_neutral": 3,
                             "trend_up": 1,
                             "trend_down": 3,
                             },
                      "30m": {"bars": 30,
                              "val_len": 1440,
                              "test": 600,
                              "steps_in_the_last": 15,
                              "n_steps": 19,
                              "forward_lag": 1,
                              "plato_lvl": 0.0013,
                              "period": ('2020-01-01 10:00:00.000',
                                         '2021-12-31 23:59:00.000'),
                              "trend_neutral": 1,
                              "trend_up": 3,
                              "trend_down": 4,
                              },
                      "1h": {"bars": 60,
                             "val_len": 720,
                             "test": 600,
                             "steps_in_the_last": 5,
                             "n_steps": 19,
                             "forward_lag": 1,
                             "plato_lvl": 0.0027,
                             "period": ('2020-01-01 10:00:00.000',
                                        '2021-12-31 23:59:00.000'),
                             "trend_neutral": 1,
                             "trend_up": 3,
                             "trend_down": 2,
                             },
                      }

    datetime_format = "%Y-%m-%d %H:%M:%S"
    pass


class DatasetParams:
    def __init__(self,
                 interval):
        self.intervals_dict = TSDataParams.intervals_dict
        self.datetime_format = TSDataParams.datetime_format
        self.get_interval_params(interval)
        pass

    @staticmethod
    def get_one_month_ago() -> datetime.datetime:
        today = datetime.datetime.today()
        if today.month == 1:
            one_month_ago = today.replace(year=today.year - 1, month=12)
        else:
            extra_days = 0
            while True:
                try:
                    one_month_ago = today.replace(month=today.month - 1, day=today.day - extra_days)
                    break
                except ValueError:
                    extra_days += 1
        return one_month_ago

    @staticmethod
    def get_one_year_ago() -> datetime.datetime:
        today = datetime.datetime.today()
        extra_days = 0
        while True:
            try:
                one_year_ago = today.replace(year=today.year - 1, month=today.month, day=today.day - extra_days)
                break
            except ValueError:
                extra_days += 1
        return one_year_ago

    def get_interval_params(self, time_interval):
        params_dict = self.intervals_dict.get(time_interval)
        if params_dict is None:
            params_dict = self.intervals_dict.get("1h")
            print("Warning: interval dictionary is not defined, getting '1h' time interval params")
        print(f"Params for interval {time_interval}")
        for name, param in params_dict.items():
            setattr(self, name, param)
            if name == "period":
                if getattr(self, name)[0] is None:
                    setattr(self, name, self.get_one_year_ago().strftime(self.datetime_format))
                elif getattr(self, name)[1] is None:
                    setattr(self, name, datetime.datetime.today().strftime(self.datetime_format))
            print(f"{name} = {param}")
        pass

    def del_interval_params(self, time_interval):
        params_dict = self.intervals_dict.get(time_interval)
        for name, param in params_dict.items():
            delattr(self, name)
        pass