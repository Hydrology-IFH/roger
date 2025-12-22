from pathlib import Path
import os
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def read_meteo(path_to_dir: Path):
    """Reads the meteorological data

    Data is imported from .txt files and stored in dataframes. Format of NA/NaN
    values is -9999.

    Args
    ----------
    path_to_dir : Path
        path to directions which contains input data

    Returns
    ----------
    prec_10mins : pd.DataFrame
        precipitation (in mm/10 mins)

    df_ta : pd.DataFrame
        air temperature (in °C)

    df_pet : pd.DataFrame
        potential evapotranspiration (in mm/day)
    """
    if not os.path.isdir(path_to_dir):
        raise ValueError(path_to_dir, "does not exist")

    Ta_path = path_to_dir / "TA.txt"
    PREC_path = path_to_dir / "PREC.txt"
    PET_path = path_to_dir / "PET.txt"

    df_PREC = pd.read_csv(
        PREC_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_PREC.index = pd.to_datetime(dict(year=df_PREC.YYYY, month=df_PREC.MM, day=df_PREC.DD, hour=df_PREC.hh, minute=df_PREC.mm))
    df_PREC = df_PREC.loc[:, ["PREC"]]
    df_PREC.index = df_PREC.index.rename("Index")

    if os.path.exists(PET_path):
        df_pet = pd.read_csv(
            PET_path,
            sep=r"\s+",
            skiprows=0,
            header=0,
            na_values=-9999,
        )
        df_pet.index = pd.to_datetime(dict(year=df_pet.YYYY, month=df_pet.MM, day=df_pet.DD, hour=df_pet.hh, minute=df_pet.mm))
        df_pet = df_pet.loc[:, ["PET"]]
        df_pet.index = df_pet.index.rename("Index")

    else:
        df_pet = None

    df_ta = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_ta.index = pd.to_datetime(dict(year=df_ta.YYYY, month=df_ta.MM, day=df_ta.DD, hour=df_ta.hh, minute=df_ta.mm))
    df_ta = df_ta.loc[:, "TA":]
    df_ta = df_ta.interpolate(method='linear', limit_direction='forward', axis=0)
    df_ta.index = df_ta.index.rename("Index")

    # reset index of precipitation time series
    # time series starts on first day at 00:00 and ends on last day at 23:50
    prec_ind = df_PREC.index
    new_prec_ind = pd.date_range(
        start=datetime(prec_ind[0].year, prec_ind[0].month, prec_ind[0].day, 0, 0),
        end=datetime(prec_ind[-1].year, prec_ind[-1].month, prec_ind[-1].day, 23, 50),
        freq="10min",
    )
    prec_10mins = pd.DataFrame(index=new_prec_ind)
    prec_10mins["PREC"] = 0.
    prec_10mins.loc[df_PREC.index, "PREC"] = df_PREC["PREC"].values.astype(float)

    return prec_10mins, df_pet, df_ta


base_path = Path(__file__).parent  # current directory; change if files are elsewhere
meteo_path = base_path / "input" / "2013-2023"

# load stress magnitude tables
file = base_path / "input" / "stress_tests_meteo" / "ta_spring_stress_magnitude.csv"
df_ta_spring_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)
file = base_path / "input" / "stress_tests_meteo" / "ta_summer_stress_magnitude.csv"
df_ta_summer_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)
file = base_path / "input" / "stress_tests_meteo" / "prec_spring_stress_magnitude.csv"
df_prec_spring_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)
file = base_path / "input" / "stress_tests_meteo" / "prec_summer_stress_magnitude.csv"
df_prec_summer_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)
file = base_path / "input" / "stress_tests_meteo" / "pet_spring_stress_magnitude.csv"
df_pet_spring_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)
file = base_path / "input" / "stress_tests_meteo" / "pet_summer_stress_magnitude.csv"
df_pet_summer_stress_magnitude = pd.read_csv(file, sep=";", skiprows=1)


meteo_stations = [1443, 684, 1346, 2388, 259, 757, 1224]
durations = [0, 2, 3]
magnitudes = [0, 1, 2]

for station in meteo_stations:
    prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
    # select summer period of 2018
    prec_10mins_summer_2018 = prec_10mins.loc["2018-06-01":"2018-08-31"]
    df_ta_summer_2018 = df_ta.loc["2018-06-01":"2018-08-31"]
    df_pet_summer_2018 = df_pet.loc["2018-06-01":"2018-08-31"]
    # select summer period of 2017
    prec_10mins_summer_2017 = prec_10mins.loc["2017-06-01":"2017-08-31"]
    df_ta_summer_2017 = df_ta.loc["2017-06-01":"2017-08-31"]
    df_pet_summer_2017 = df_pet.loc["2017-06-01":"2017-08-31"]
    # select spring period of 2020
    prec_10mins_spring_2020 = prec_10mins.loc["2020-03-01":"2020-05-31"]
    df_ta_spring_2020 = df_ta.loc["2020-03-01":"2020-05-31"]
    df_pet_spring_2020 = df_pet.loc["2020-03-01":"2020-05-31"]
    # select spring period of 2015
    prec_10mins_spring_2015 = prec_10mins.loc["2015-03-01":"2015-05-31"]
    df_ta_spring_2015 = df_ta.loc["2015-03-01":"2015-05-31"]
    df_pet_spring_2015 = df_pet.loc["2015-03-01":"2015-05-31"]
    # select summer and spring period of 2021
    prec_10mins_spring_summer_2021 = prec_10mins.loc["2021-03-01":"2021-08-31"]
    df_ta_spring_summer_2021 = df_ta.loc["2021-03-01":"2021-08-31"]
    df_pet_spring_summer_2021 = df_pet.loc["2021-03-01":"2021-08-31"]
    for duration in durations:
        for magnitude in magnitudes:
            if magnitude == 0 and duration == 3:
                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert spring period of 2020 in spring period of 2019
                prec_10mins.loc["2019-03-01":"2019-05-31"] = prec_10mins_spring_2020.values
                df_ta.loc["2019-03-01":"2019-05-31"] = df_ta_spring_2020.values
                df_pet.loc["2019-03-01":"2019-05-31"] = df_pet_spring_2020.values
                # insert spring period of 2020 in spring period of 2018
                prec_10mins.loc["2018-03-01":"2018-05-31"] = prec_10mins_spring_2020.values
                df_ta.loc["2018-03-01":"2018-05-31"] = df_ta_spring_2020.values
                df_pet.loc["2018-03-01":"2018-05-31"] = df_pet_spring_2020.values

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31"] = prec_10mins_summer_2018.values
                df_ta.loc["2017-06-01":"2017-08-31"] = df_ta_summer_2018.values
                df_pet.loc["2017-06-01":"2017-08-31"] = df_pet_summer_2018.values
                # insert summer period of 2018 in summer period of 2016
                prec_10mins.loc["2016-06-01":"2016-08-31"] = prec_10mins_summer_2018.values
                df_ta.loc["2016-06-01":"2016-08-31"] = df_ta_summer_2018.values
                df_pet.loc["2016-06-01":"2016-08-31"] = df_pet_summer_2018.values

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert spring period of 2020 in spring period of 2017
                prec_10mins.loc["2017-03-01":"2017-05-31"] = prec_10mins_spring_2020.values
                df_ta.loc["2017-03-01":"2017-05-31"] = df_ta_spring_2020.values
                df_pet.loc["2017-03-01":"2017-05-31"] = df_pet_spring_2020.values
                # insert spring period of 2020 in spring period of 2016
                prec_10mins.loc["2016-03-01":"2016-05-31"] = prec_10mins_spring_2020.values
                df_ta.loc["2016-03-01":"2016-05-31"] = df_ta_spring_2020.values
                df_pet.loc["2016-03-01":"2016-05-31"] = df_pet_spring_2020.values
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31"] = prec_10mins_summer_2018.values
                df_ta.loc["2017-06-01":"2017-08-31"] = df_ta_summer_2018.values
                df_pet.loc["2017-06-01":"2017-08-31"] = df_pet_summer_2018.values
                # insert summer period of 2018 in summer period of 2016
                prec_10mins.loc["2016-06-01":"2016-08-31"] = prec_10mins_summer_2018.values
                df_ta.loc["2016-06-01":"2016-08-31"] = df_ta_summer_2018.values
                df_pet.loc["2016-06-01":"2016-08-31"] = df_pet_summer_2018.values

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-summer-wet" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert spring period of 2015 in spring period of 2020
                prec_10mins.loc["2020-03-01":"2020-05-31"] = prec_10mins_spring_2015.values
                df_ta.loc["2020-03-01":"2020-05-31"] = df_ta_spring_2015.values
                df_pet.loc["2020-03-01":"2020-05-31"] = df_pet_spring_2015.values
                # insert summer period of 2017 in summer period of 2018
                prec_10mins.loc["2018-06-01":"2018-08-31"] = prec_10mins_summer_2017.values
                df_ta.loc["2018-06-01":"2018-08-31"] = df_ta_summer_2017.values
                df_pet.loc["2018-06-01":"2018-08-31"] = df_pet_summer_2017.values
                # insert summer and spring period of 2021 in spring and summer period of 2020
                prec_10mins.loc["2020-03-01":"2020-08-31"] = prec_10mins_spring_summer_2021.values
                df_ta.loc["2020-03-01":"2020-08-31"] = df_ta_spring_summer_2021.values
                df_pet.loc["2020-03-01":"2020-08-31"] = df_pet_spring_summer_2021.values
                # insert summer and spring period of 2021 in spring and summer period of 2019
                prec_10mins.loc["2019-03-01":"2019-08-31"] = prec_10mins_spring_summer_2021.values
                df_ta.loc["2019-03-01":"2019-08-31"] = df_ta_spring_summer_2021.values
                df_pet.loc["2019-03-01":"2019-08-31"] = df_pet_spring_summer_2021.values

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

            elif magnitude == 1 and duration == 2:
                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                prec_magnitude_spring = df_prec_spring_stress_magnitude.loc[df_prec_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_spring = df_ta_spring_stress_magnitude.loc[df_ta_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_spring = df_pet_spring_stress_magnitude.loc[df_pet_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                prec_magnitude_summer = df_prec_summer_stress_magnitude.loc[df_prec_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_summer = df_ta_summer_stress_magnitude.loc[df_ta_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_summer = df_pet_summer_stress_magnitude.loc[df_pet_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                # insert spring period of 2020 in spring period of 2019
                prec_10mins.loc["2019-03-01":"2019-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2019-03-01":"2019-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2019-03-01":"2019-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2017-06-01":"2017-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2017-06-01":"2017-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert spring period of 2020 in spring period of 2017
                prec_10mins.loc["2017-03-01":"2017-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2017-03-01":"2017-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2017-03-01":"2017-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2017-06-01":"2017-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2017-06-01":"2017-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

            elif magnitude == 2 and duration == 3:
                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                prec_magnitude_spring = df_prec_spring_stress_magnitude.loc[df_prec_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_spring = df_ta_spring_stress_magnitude.loc[df_ta_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_spring = df_pet_spring_stress_magnitude.loc[df_pet_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                prec_magnitude_summer = df_prec_summer_stress_magnitude.loc[df_prec_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_summer = df_ta_summer_stress_magnitude.loc[df_ta_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_summer = df_pet_summer_stress_magnitude.loc[df_pet_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                # insert spring period of 2020 in spring period of 2019
                prec_10mins.loc["2019-03-01":"2019-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2019-03-01":"2019-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2019-03-01":"2019-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))
                # insert spring period of 2020 in spring period of 2018
                prec_10mins.loc["2018-03-01":"2018-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2018-03-01":"2018-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2018-03-01":"2018-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2017-06-01":"2017-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2017-06-01":"2017-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))
                # insert summer period of 2018 in summer period of 2016
                prec_10mins.loc["2016-06-01":"2016-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2016-06-01":"2016-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2016-06-01":"2016-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                # insert spring period of 2020 in spring period of 2017
                prec_10mins.loc["2017-03-01":"2017-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2017-03-01":"2017-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2017-03-01":"2017-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))
                # insert spring period of 2020 in spring period of 2016
                prec_10mins.loc["2016-03-01":"2016-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2016-03-01":"2016-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2016-03-01":"2016-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))
                # insert summer period of 2018 in summer period of 2017
                prec_10mins.loc["2017-06-01":"2017-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2017-06-01":"2017-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2017-06-01":"2017-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))
                # insert summer period of 2018 in summer period of 2016
                prec_10mins.loc["2016-06-01":"2016-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2016-06-01":"2016-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2016-06-01":"2016-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

            elif magnitude == 2 and duration == 0:
                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                prec_magnitude_spring = df_prec_spring_stress_magnitude.loc[df_prec_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_spring = df_ta_spring_stress_magnitude.loc[df_ta_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_spring = df_pet_spring_stress_magnitude.loc[df_pet_spring_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                prec_magnitude_summer = df_prec_summer_stress_magnitude.loc[df_prec_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                ta_magnitude_summer = df_ta_summer_stress_magnitude.loc[df_ta_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                pet_magnitude_summer = df_pet_summer_stress_magnitude.loc[df_pet_summer_stress_magnitude["ID"] == station, f"magnitude{magnitude}"].values[0]
                prec_10mins.loc["2020-03-01":"2020-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2020-03-01":"2020-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2020-03-01":"2020-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                prec_10mins.loc["2018-06-01":"2018-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2018-06-01":"2018-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2018-06-01":"2018-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")

                path_to_dir = base_path  / "input" / "stress_tests_meteo" / "spring-summer-drought" / f"duration{duration}_magnitude{magnitude}" / f"{station}" 
                if not os.path.exists(path_to_dir):
                    os.makedirs(path_to_dir)  

                prec_10mins, df_pet, df_ta = read_meteo(meteo_path / str(station))
                prec_10mins.loc["2020-03-01":"2020-05-31", "PREC"] = prec_10mins_spring_2020.loc[:, "PREC"].values * (1 + (prec_magnitude_spring / 100))
                df_ta.loc["2020-03-01":"2020-05-31", "TA":] = df_ta_spring_2020.loc[:, "TA":].values + ta_magnitude_spring
                df_pet.loc["2020-03-01":"2020-05-31", "PET"] = df_pet_spring_2020.loc[:, "PET"].values * (1 + (pet_magnitude_spring / 100))
                prec_10mins.loc["2018-06-01":"2018-08-31", "PREC"] = prec_10mins_summer_2018.loc[:, "PREC"].values * (1 + (prec_magnitude_summer / 100))
                df_ta.loc["2018-06-01":"2018-08-31", "TA":] = df_ta_summer_2018.loc[:, "TA":].values + ta_magnitude_summer
                df_pet.loc["2018-06-01":"2018-08-31", "PET"] = df_pet_summer_2018.loc[:, "PET"].values * (1 + (pet_magnitude_summer / 100))

                PREC_path = path_to_dir / "PREC.txt"
                Ta_path = path_to_dir / "TA.txt"
                PET_path = path_to_dir / "PET.txt"

                # export precipitation to .txt
                prec_10mins["YYYY"] = prec_10mins.index.year.values
                prec_10mins["MM"] = prec_10mins.index.month.values
                prec_10mins["DD"] = prec_10mins.index.day.values
                prec_10mins["hh"] = prec_10mins.index.hour.values
                prec_10mins["mm"] = prec_10mins.index.minute.values
                prec_10mins = prec_10mins.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PREC"]]
                prec_10mins.to_csv(PREC_path, header=True, index=False, sep="\t")

                # export air temperature to .txt
                df_ta["YYYY"] = df_ta.index.year.values
                df_ta["MM"] = df_ta.index.month.values
                df_ta["DD"] = df_ta.index.day.values
                df_ta["hh"] = df_ta.index.hour.values
                df_ta["mm"] = 0
                df_ta = df_ta.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "TA", "TA_min", "TA_max"]]
                df_ta.to_csv(Ta_path, header=True, index=False, sep="\t")

                # export potential evapotranspiration to .txt
                df_pet["YYYY"] = df_pet.index.year.values
                df_pet["MM"] = df_pet.index.month.values
                df_pet["DD"] = df_pet.index.day.values
                df_pet["hh"] = df_pet.index.hour.values
                df_pet["mm"] = 0
                df_pet = df_pet.loc[:, ["YYYY", "MM", "DD", "hh", "mm", "PET"]]
                df_pet.to_csv(PET_path, header=True, index=False, sep="\t")





