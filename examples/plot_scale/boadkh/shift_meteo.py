from pathlib import Path
import pandas as pd
import numpy as np



base_path = Path(__file__).parent


stations = ["freiburg", "lahr", "muellheim", 
            "stockach", "gottmadingen", "weingarten", 
            "eppingen-elsenz", "bruchsal-heidelsheim", "bretten", 
            "ehingen-kirchen", "merklingen", "hayingen", 
            "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
stations = ["stockach", "gottmadingen", "weingarten", 
            "eppingen-elsenz", "bruchsal-heidelsheim", "bretten", 
            "ehingen-kirchen", "merklingen", "hayingen", 
            "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
years = np.arange(2013, 2017, 1, dtype=np.int32).tolist()

dict_shifted_data = {}
dict_shifts = {}

# --- read input data ------------------------------------------------------
for station in stations:
    print(station)
    dict_shifted_data[station] = {}
    dict_shifts[station] = {}

    path_to_dir = base_path / "input" / station / "2013-2022"
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
    df_PREC.index = [pd.to_datetime(f"{df_PREC.iloc[i, 0]} {df_PREC.iloc[i, 1]} {df_PREC.iloc[i, 2]} {df_PREC.iloc[i, 3]} {df_PREC.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_PREC.index))]
    df_PREC = df_PREC.loc[:, ["PREC"]]
    df_PREC.index = df_PREC.index.rename("Index")

    df_pet = pd.read_csv(
        PET_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_pet.index = [pd.to_datetime(f"{df_pet.iloc[i, 0]} {df_pet.iloc[i, 1]} {df_pet.iloc[i, 2]} {df_pet.iloc[i, 3]} {df_pet.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_pet.index))]
    df_pet = df_pet.loc[:, ["PET"]]
    df_pet.index = df_pet.index.rename("Index")

    df_ta = pd.read_csv(
        Ta_path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_ta.index = [pd.to_datetime(f"{df_ta.iloc[i, 0]} {df_ta.iloc[i, 1]} {df_ta.iloc[i, 2]} {df_ta.iloc[i, 3]} {df_ta.iloc[i, 4]}", format="%Y %m %d %H %M") for i in range(len(df_ta.index))]
    df_ta = df_ta.loc[:, ["TA", "TA_min", "TA_max"]]
    df_ta.index = df_ta.index.rename("Index")

    ll_prec_shifts = [df_PREC]
    ll_ta_shifts = [df_ta]
    ll_pet_shifts = [df_pet]
    for i, year in enumerate(years):
        dict_shifts[station][i] = {}
        df_prec1 = df_PREC.loc[:f"{year}-12-31 23:50", :]
        df_prec2 = df_PREC.loc[f"{year+1}-01-01 00:00":, :]
        df_ta1 = df_ta.loc[:f"{year}-12-31", :]
        df_ta2 = df_ta.loc[f"{year+1}-01-01":, :]
        df_pet1 = df_pet.loc[:f"{year}-12-31", :]
        df_pet2 = df_pet.loc[f"{year+1}-01-01":, :]

        df_prec_shift = pd.concat([df_prec2, df_prec1], axis=0)
        df_ta_shift = pd.concat([df_ta2, df_ta1], axis=0)
        df_pet_shift = pd.concat([df_pet2, df_pet1], axis=0)

        dict_shifts[station][i]["PRECIP"] = df_prec_shift
        dict_shifts[station][i]["TA"] = df_ta_shift
        dict_shifts[station][i]["PET"] = df_pet_shift

        ll_prec_shifts.append(df_prec_shift)
        ll_ta_shifts.append(df_ta_shift)
        ll_pet_shifts.append(df_pet_shift)

    df_prec_shifted = pd.concat(ll_prec_shifts, axis=0)
    df_ta_shifted = pd.concat(ll_ta_shifts, axis=0)
    df_pet_shifted = pd.concat(ll_pet_shifts, axis=0)
    prec_index = pd.date_range(start=str(df_prec_shifted.index[0]), periods=len(df_prec_shifted.index), freq="10min")
    ta_pet_index = pd.date_range(start=str(df_ta_shifted.index[0]), periods=len(df_ta_shifted.index), freq="D")
    df_prec_shifted.index = prec_index
    df_ta_shifted.index = ta_pet_index
    df_pet_shifted.index = ta_pet_index

    prec_index_full = pd.date_range(start=str(df_prec_shifted.index[0]), end=f"{ta_pet_index[-1].year}-12-31 23:50", freq="10min")
    ta_pet_index_full = pd.date_range(start=str(df_ta_shifted.index[0]), end=f"{ta_pet_index[-1].year}-12-31 23:50", freq="D")

    df_prec_shifted_full = pd.DataFrame(index=prec_index_full)
    df_ta_shifted_full = pd.DataFrame(index=ta_pet_index_full)
    df_pet_shifted_full = pd.DataFrame(index=ta_pet_index_full)

    df_prec_shifted1 = df_prec_shifted.loc[:f"{ta_pet_index[-1].year}-12-24 23:50", :]
    df_ta_shifted1 =  df_ta_shifted.loc[:f"{ta_pet_index[-1].year}-12-24", :]
    df_pet_shifted1 = df_pet_shifted.loc[:f"{ta_pet_index[-1].year}-12-24", :]

    df_prec_shifted2 = df_PREC.loc[f"{year+1}-01-01 00:00":f"{year+1}-01-07 23:50", :]
    df_prec_shifted2.index = df_prec_shifted_full.loc[f"{ta_pet_index[-1].year}-12-25 00:00":, :].index
    df_ta_shifted2 = df_ta.loc[f"{year+1}-01-01":f"{year+1}-01-07", :]
    df_ta_shifted2.index = df_ta_shifted_full.loc[f"{ta_pet_index[-1].year}-12-25":, :].index
    df_pet_shifted2 = df_pet.loc[f"{year+1}-01-01":f"{year+1}-01-07", :]
    df_pet_shifted2.index = df_pet_shifted_full.loc[f"{ta_pet_index[-1].year}-12-25":, :].index

    df_prec_shifted_full = df_prec_shifted_full.join(pd.concat([df_prec_shifted1, df_prec_shifted2], axis=0))
    df_ta_shifted_full = df_ta_shifted_full.join(pd.concat([df_ta_shifted1, df_ta_shifted2], axis=0))
    df_pet_shifted_full = df_pet_shifted_full.join(pd.concat([df_pet_shifted1, df_pet_shifted2], axis=0))

    dict_shifted_data[station]["PRECIP"] = df_prec_shifted_full
    dict_shifted_data[station]["TA"] = df_ta_shifted_full
    dict_shifted_data[station]["PET"] = df_pet_shifted_full

# --- write shifted input data to .txt -------------------------------------
for station in stations:
    data_precip = dict_shifted_data[station]["PRECIP"]
    data_ta = dict_shifted_data[station]["TA"]
    data_pet = dict_shifted_data[station]["PET"]

    path_dir = base_path / "input" / station
    idx_10mins = pd.date_range(start=str(data_precip.index[0]), end=str(data_precip.index[-1]), freq="10min")
    idx_daily = pd.date_range(start=str(data_ta.index[0]), end=str(data_ta.index[-1]), freq="d")
    df_PREC = pd.DataFrame(index=idx_10mins, columns=["YYYY", "MM", "DD", "hh", "mm", "PREC"])
    df_PREC["YYYY"] = data_precip.index.year.values
    df_PREC["MM"] = data_precip.index.month.values
    df_PREC["DD"] = data_precip.index.day.values
    df_PREC["hh"] = data_precip.index.hour.values
    df_PREC["mm"] = data_precip.index.minute.values
    df_PREC["PREC"] = data_precip["PREC"].values
    path_txt = path_dir / "PREC.txt"
    df_PREC.to_csv(path_txt, header=True, index=False, sep="\t")

    df_TA = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
    df_TA["YYYY"] = data_ta.index.year.values
    df_TA["MM"] = data_ta.index.month.values
    df_TA["DD"] = data_ta.index.day.values
    df_TA["hh"] = data_ta.index.hour.values
    df_TA["mm"] = data_ta.index.minute.values
    df_TA["TA"] = data_ta["TA"].values
    df_TA["TA_min"] = data_ta["TA_min"].values
    df_TA["TA_max"] = data_ta["TA_max"].values
    path_txt = path_dir / "TA.txt"
    df_TA.to_csv(path_txt, header=True, index=False, sep="\t")

    df_PET = pd.DataFrame(index=idx_daily, columns=["YYYY", "MM", "DD", "hh", "mm"])
    df_PET["YYYY"] = data_pet.index.year.values
    df_PET["MM"] = data_pet.index.month.values
    df_PET["DD"] = data_pet.index.day.values
    df_PET["hh"] = data_pet.index.hour.values
    df_PET["mm"] = data_pet.index.minute.values
    df_PET["PET"] = data_pet["PET"].values
    path_txt = path_dir / "PET.txt"
    df_PET.to_csv(path_txt, header=True, index=False, sep="\t")
