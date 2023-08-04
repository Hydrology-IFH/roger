from pathlib import Path
import os
import glob
import pandas as pd
from datetime import datetime


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
    RS_path = path_to_dir / "RS.txt"

    df_PREC = pd.read_csv(
        PREC_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        parse_dates=[[0, 1, 2, 3, 4]],
        index_col=0,
        na_values=-9999,
    )
    df_PREC.index = pd.to_datetime(df_PREC.index, format="%Y %m %d %H %M")
    df_PREC.index = df_PREC.index.rename("Index")

    if os.path.exists(PET_path):
        df_pet = pd.read_csv(
            PET_path,
            sep=r"\s+",
            skiprows=0,
            header=0,
            parse_dates=[[0, 1, 2, 3, 4]],
            index_col=0,
            na_values=-9999,
        )
        df_pet.index = pd.to_datetime(df_pet.index, format="%Y %m %d %H %M")
        df_pet.index = df_pet.index.rename("Index")
    else:
        df_pet = None

    if os.path.exists(RS_path):
        df_rs = pd.read_csv(
            RS_path,
            sep=r"\s+",
            skiprows=0,
            header=0,
            parse_dates=[[0, 1, 2, 3, 4]],
            index_col=0,
            na_values=-9999,
        )
        df_rs.index = pd.to_datetime(df_rs.index, format="%Y %m %d %H %M")
        df_rs.index = df_rs.index.rename("Index")
    else:
        df_rs = None

    df_ta = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        parse_dates=[[0, 1, 2, 3, 4]],
        index_col=0,
        na_values=-9999,
    )
    df_ta.index = pd.to_datetime(df_ta.index, format="%Y %m %d %H %M")
    df_ta.index = df_ta.index.rename("Index")

    # reset index of precipitation time series
    # time series starts on first day at 00:00 and ends on last day at 23:50
    prec_ind = df_PREC.index
    new_prec_ind = pd.date_range(
        start=datetime(prec_ind[0].year, prec_ind[0].month, prec_ind[0].day, 0, 0),
        end=datetime(prec_ind[-1].year, prec_ind[-1].month, prec_ind[-1].day, 23, 50),
        freq="10T",
    )
    prec_10mins = pd.DataFrame(index=new_prec_ind)
    prec_10mins["PREC"] = 0
    prec_10mins.loc[df_PREC.index, "PREC"] = df_PREC["PREC"].values

    return prec_10mins, df_pet, df_ta, df_rs


def write_meteo_csv_from_dwd(path_to_dir: Path):
    """Writes the meteorological data downloaded from WeatherDB
    (https://weather.hydro.intra.uni-freiburg.de/)

    Data is imported from .txt files and stored in dataframes. Format of NA/NaN
    values is -9999.

    Args
    ----------
    path_to_dir : Path
        path to directions which contains input data
    """
    if not os.path.isdir(path_to_dir):
        raise ValueError(path_to_dir, "does not exist")

    Ta_path = glob.glob(str(path_to_dir / "T_*.txt"))[0]
    PREC_path = glob.glob(str(path_to_dir / "N_*.txt"))[0]
    PET_path = glob.glob(str(path_to_dir / "ET_*.txt"))[0]

    df_prec = pd.read_csv(
        PREC_path,
        sep=r"\s+",
        skiprows=2,
        header=0,
        parse_dates=[[0, 1, 2, 3, 4]],
        index_col=0,
        na_values=-9999,
        date_format="%Y %m %d %H %M",
    )
    df_prec.index = pd.to_datetime(df_prec.index, format="%Y %m %d %H %M")

    df_pet = pd.read_csv(
        PET_path,
        sep=r"\s+",
        skiprows=2,
        header=0,
        parse_dates=[[0, 1, 2]],
        index_col=0,
        na_values=-9999,
        date_format="%Y %m %d %H %M",
    )
    df_pet.index = pd.to_datetime(df_pet.index, format="%Y %m %d")

    df_ta = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=2,
        header=0,
        parse_dates=[[0, 1, 2]],
        index_col=0,
        na_values=-9999,
        date_format="%Y %m %d %H %M",
    )
    df_ta.index = pd.to_datetime(df_ta.index, format="%Y %m %d")

    # reset index of precipitation time series
    # time series starts on first day at 00:00 and ends on last day at 23:50
    prec_ind = df_prec.index
    new_prec_ind = pd.date_range(
        start=datetime(prec_ind[0].year, prec_ind[0].month, prec_ind[0].day, 0, 0),
        end=datetime(prec_ind[-1].year, prec_ind[-1].month, prec_ind[-1].day, 23, 50),
        freq="10T",
    )
    prec_10mins = pd.DataFrame(index=new_prec_ind)
    prec_10mins["PREC"] = 0
    prec_10mins.loc[df_prec.index, "PREC"] = df_prec["N"].values

    Ta_path = path_to_dir / "TA.txt"
    PREC_path = path_to_dir / "PREC.txt"
    PET_path = path_to_dir / "PET.txt"

    # export precipitation to .txt
    df_PREC = pd.DataFrame(index=prec_10mins.index, columns=["YYYY", "MM", "DD", "hh", "mm", "PREC"])
    df_PREC["YYYY"] = df_PREC.index.year.values
    df_PREC["MM"] = df_PREC.index.month.values
    df_PREC["DD"] = df_PREC.index.day.values
    df_PREC["hh"] = df_PREC.index.hour.values
    df_PREC["mm"] = df_PREC.index.minute.values
    df_PREC["PREC"] = prec_10mins["PREC"].values
    df_PREC.to_csv(PREC_path, header=True, index=False, sep="\t")

    # export air temperature to .txt
    df_TA = pd.DataFrame(index=df_ta.index, columns=["YYYY", "MM", "DD", "hh", "mm", "TA"])
    df_TA["YYYY"] = df_TA.index.year.values
    df_TA["MM"] = df_TA.index.month.values
    df_TA["DD"] = df_TA.index.day.values
    df_TA["hh"] = df_TA.index.hour.values
    df_TA["mm"] = 0
    df_TA["TA"] = df_ta.loc[:, "T"].values
    df_TA.to_csv(Ta_path, header=True, index=False, sep="\t")

    # export potential evapotranspiration to .txt
    df_PET = pd.DataFrame(index=df_pet.index, columns=["YYYY", "MM", "DD", "hh", "mm", "PET", "R_R0"])
    df_PET["YYYY"] = df_PET.index.year.values
    df_PET["MM"] = df_PET.index.month.values
    df_PET["DD"] = df_PET.index.day.values
    df_PET["hh"] = df_PET.index.hour.values
    df_PET["mm"] = 0
    df_PET["PET"] = df_pet.loc[:, "ET"].values
    df_PET["R_R0"] = 0
    df_PET.to_csv(PET_path, header=True, index=False, sep="\t")
