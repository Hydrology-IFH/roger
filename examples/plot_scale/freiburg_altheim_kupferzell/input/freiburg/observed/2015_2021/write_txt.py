from pathlib import Path
import glob
import os
import pandas as pd
from datetime import datetime

base_path = Path(__file__).parent
meteo_station_id = 1443

def write_meteo_txt_from_WeatherDB(input_dir: Path, output_dir: Path):
    """Writes the meteorological data downloaded from WeatherDB
    (https://weather.hydro.intra.uni-freiburg.de/)

    Args
    ----------
    input_dir : Path
        path to folder which contains input data

    output_dir : Path
        path to folder which contains input data
    """
    if not os.path.isdir(input_dir):
        raise ValueError(input_dir, 'does not exist')

    Ta_path = glob.glob(str(input_dir / "T_*.txt"))[0]
    PREC_path = glob.glob(str(input_dir / "N_*.txt"))[0]
    PET_path = glob.glob(str(input_dir / "ET_*.txt"))[0]

    df_prec = pd.read_csv(PREC_path, sep=r"\s+", skiprows=2, header=0, parse_dates=[[0, 1, 2, 3, 4]],
                          index_col=0, na_values=-9999)
    df_prec.index = pd.to_datetime(df_prec.index, format='%Y %m %d %H %M')

    df_pet = pd.read_csv(PET_path, sep=r"\s+", skiprows=2, header=0, parse_dates=[[0, 1, 2]],
                         index_col=0, na_values=-9999)
    df_pet.index = pd.to_datetime(df_pet.index, format='%Y %m %d')

    df_ta = pd.read_csv(Ta_path, sep=r"\s+", skiprows=2, header=0, parse_dates=[[0, 1, 2]],
                        index_col=0, na_values=-9999)
    df_ta.index = pd.to_datetime(df_ta.index, format='%Y %m %d')

    # reset index of precipitation time series
    # time series starts on first day at 00:00 and ends on last day at 23:50
    prec_ind = df_prec.index
    new_prec_ind = pd.date_range(start=datetime(prec_ind[0].year, prec_ind[0].month, prec_ind[0].day, 0, 0),
                              end=datetime(prec_ind[-1].year, prec_ind[-1].month, prec_ind[-1].day, 23, 50),
                              freq='10T')
    prec_10mins = pd.DataFrame(index=new_prec_ind)
    prec_10mins['PREC'] = 0
    prec_10mins.loc[df_prec.index, 'PREC'] = df_prec['N'].values

    Ta_path = output_dir / "TA.txt"
    PREC_path = output_dir / "PREC.txt"
    PET_path = output_dir / "PET.txt"

    # export precipitation to .txt
    df_PREC = pd.DataFrame(index=prec_10mins.index, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PREC'])
    df_PREC['YYYY'] = df_PREC.index.year.values
    df_PREC['MM'] = df_PREC.index.month.values
    df_PREC['DD'] = df_PREC.index.day.values
    df_PREC['hh'] = df_PREC.index.hour.values
    df_PREC['mm'] = df_PREC.index.minute.values
    df_PREC['PREC'] = prec_10mins['PREC'].values
    df_PREC.to_csv(PREC_path, header=True, index=False, sep="\t")

    # export air temperature to .txt
    df_TA = pd.DataFrame(index=df_ta.index, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'TA'])
    df_TA['YYYY'] = df_TA.index.year.values
    df_TA['MM'] = df_TA.index.month.values
    df_TA['DD'] = df_TA.index.day.values
    df_TA['hh'] = df_TA.index.hour.values
    df_TA['mm'] = 0
    df_TA['TA'] = df_ta.loc[:, 'T'].values
    df_TA.to_csv(Ta_path, header=True, index=False, sep="\t")

    # export potential evapotranspiration to .txt
    df_PET = pd.DataFrame(index=df_pet.index, columns=['YYYY', 'MM', 'DD', 'hh', 'mm', 'PET', 'R_R0'])
    df_PET['YYYY'] = df_PET.index.year.values
    df_PET['MM'] = df_PET.index.month.values
    df_PET['DD'] = df_PET.index.day.values
    df_PET['hh'] = df_PET.index.hour.values
    df_PET['mm'] = 0
    df_PET['PET'] = df_pet.loc[:, 'ET'].values
    df_PET['R_R0'] = 0
    df_PET.to_csv(PET_path, header=True, index=False, sep="\t")


write_meteo_txt_from_WeatherDB(base_path, base_path)
