from pathlib import Path
import pandas as pd
import numpy as onp

CSV_DIR = Path(__file__).parent / "look_up_tables"

path_ilu = CSV_DIR / "land_use_dependent_interception.csv"
df_ilu = pd.read_csv(path_ilu, sep=";", na_values=-9999)
# ARR_ILU = onp.asarray(df_ilu.values, dtype=onp.float64)
ARR_ILU = df_ilu.values

path_is = CSV_DIR / "sealing_dependent_interception.csv"
df_is = pd.read_csv(path_is, sep=";", skiprows=1, na_values=-9999)
# ARR_IS = onp.asarray(df_is.values, dtype=onp.float64)
ARR_IS = df_is.values

path_mlms = CSV_DIR / "horizontal_macropore_flow_velocities.csv"
df_mlms = pd.read_csv(path_mlms, sep=";", skiprows=1, na_values=-9999)
# ARR_MLMS = onp.asarray(df_mlms.values, dtype=onp.float64)
ARR_MLMS = df_mlms.values

path_rdlu = CSV_DIR / "land_use_dependent_rooting_depth.csv"
df_rdlu = pd.read_csv(path_rdlu, sep=";", skiprows=1, na_values=-9999)
# ARR_RDLU = onp.asarray(df_rdlu.values, dtype=onp.float64)
ARR_RDLU = df_rdlu.values

path_cp = CSV_DIR / "crop_parameters.csv"
df_cp = pd.read_csv(path_cp, sep=";", na_values=-9999, skiprows=1, dtype=onp.float64)
# ARR_CP = onp.asarray(df_cp.values, dtype=onp.float64)
ARR_CP = df_cp.values

path_fert1 = CSV_DIR / "fertilization1.csv"
df_fert1 = pd.read_csv(path_fert1, sep=";", na_values=-9999, skiprows=1, dtype=onp.float64)
df_fert1.fillna(0, inplace=True)
# ARR_FERT1 = onp.asarray(df_fert1.values, dtype=onp.float64)
ARR_FERT1 = df_fert1.values

path_fert2 = CSV_DIR / "fertilization2.csv"
df_fert2 = pd.read_csv(path_fert2, sep=";", na_values=-9999, skiprows=1, dtype=onp.float64)
df_fert2.fillna(0, inplace=True)
# ARR_FERT2 = onp.asarray(df_fert2.values, dtype=onp.float64)
ARR_FERT2 = df_fert2.values

path_fert3 = CSV_DIR / "fertilization3.csv"
df_fert3 = pd.read_csv(path_fert3, sep=";", na_values=-9999, skiprows=1, dtype=onp.float64)
df_fert3.fillna(0, inplace=True)
# ARR_FERT3 = onp.asarray(df_fert3.values, dtype=onp.float64)
ARR_FERT3 = df_fert3.values

path_nup = CSV_DIR / "nitrogen_uptake.csv"
df_nup = pd.read_csv(path_nup, sep=";", na_values=-9999, skiprows=1, dtype=onp.float64)
df_nup.fillna(0, inplace=True)
# ARR_NUP = onp.asarray(df_nup.iloc[:, :-1].values, dtype=onp.float64)
ARR_NUP = df_nup.iloc[:, :-1].values

ARR_GC = onp.zeros((25, 13), dtype=onp.float64)
# print(ARR_ILU[:, 0].tolist())  
# required to build the docs
list0 = [0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 20.0, 98.0, 31.0, 32.0, 33.0, 40.0, 41.0, 50.0, 1001.0, 1002.0, 1003.0, 1004.0, 1005.0]
arr0 = onp.array(list0)
# print(ARR_ILU[:, 1:].tolist())
# required to build the docs
list1 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0], 
         [0.2, 0.2, 0.2, 0.3, 0.4, 0.7, 0.8, 0.8, 0.8, 0.3, 0.2, 0.2], 
         [0.4, 0.4, 0.4, 0.4, 0.6, 0.7, 0.8, 0.8, 0.8, 0.5, 0.4, 0.4], 
         [0.4, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4], 
         [0.4, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4], 
         [0.1, 0.1, 0.5, 1.4, 2.5, 3.9, 4.3, 4.3, 3.9, 2.8, 0.5, 0.1], 
         [2.0, 2.0, 2.2, 2.7, 3.2, 3.9, 4.1, 4.1, 3.9, 3.4, 2.2, 2.0], 
         [3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9, 3.9], 
         [0.4, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [2.0, 2.0, 2.2, 2.7, 3.2, 3.9, 4.1, 4.1, 3.9, 3.4, 2.2, 2.0], 
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
         [0.4, 0.4, 0.6, 0.8, 1.0, 1.2, 1.2, 1.0, 1.0, 0.6, 0.5, 0.4], 
         [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 
         [0.4, 0.4, 0.45, 0.6, 0.75, 0.75, 0.75, 0.75, 0.75, 0.45, 0.4, 0.4], 
         [0.4, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.5, 0.4], 
         [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 
         [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4], 
         [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.9], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0], 
         [0.1, 0.1, 0.1, 0.1, 0.6, 1.0, 0.9, 0.6, 0.3, 0.0, 0.0, 0.0]]
arr1 = onp.array(list1)
ARR_GC[:, 0] = arr0
ARR_GC[:, 1:] = 1 - 0.7 ** (arr1 / 0.2)

ARR_GCM = onp.zeros((25, 2))
ARR_GCM[:, 0] = arr0

ARR_GCM[:, 1] = onp.max(ARR_GC[:, 1:], axis=1)

SUMMER_CROPS = onp.array(
    [   501,
        502,
        503,
        504,
        505,
        506,
        507,
        508,
        509,
        510,
        511,
        512,
        513,
        514,
        515,
        516,
        517,
        518,
        519,
        520,
        521,
        522,
        523,
        524,
        525,
        526,
        527,
        528,
        529,
        530,
        531,
        532,
        533,
        534,
        535,
        536,
        537,
        538,
        539,
        540,
        541,
        542,
        543,
        544,
        545,
        546,
        547,
        548,
        549,
        550,
        551,
        552,
        553,
        554,
        555,
        561,
        562,
        563,
        565,
        567,
    ],
    dtype=onp.int32,
)
WINTER_CROPS = onp.array([556, 557, 558, 559, 560, 564, 579], dtype=onp.int32)
WINTER_CATCH_CROPS = onp.array([566, 568, 569, 570, 586, 587], dtype=onp.int32)
MULTI_YEAR_CROPS_INIT = onp.array([571, 572, 580, 583], dtype=onp.int32)
MULTI_YEAR_CROPS_CONT = onp.array([573, 574, 581, 582, 584, 585], dtype=onp.int32)
WINTER_MULTI_YEAR_CROPS_INIT = onp.array([572, 583], dtype=onp.int32)
WINTER_MULTI_YEAR_CROPS_CONT = onp.array([574, 581, 585, 590], dtype=onp.int32)
SUMMER_MULTI_YEAR_CROPS_INIT = onp.array([571, 580, 589], dtype=onp.int32)
SUMMER_MULTI_YEAR_CROPS_CONT = onp.array([582, 584], dtype=onp.int32)
SUMMER_MULTI_YEAR_CROPS_CONT_GROW = onp.array([573, 591], dtype=onp.int32)
WINTER_CROPS_FERT = onp.array([556, 557, 558, 559, 560, 564, 579, 580, 582, 584], dtype=onp.int32)

dict_crops = {
    536: "beetroot",
    539: "silage_corn",
    543: "summer wheat",
    556: "winter barley",
    557: "winter wheat",
    559: "winter rape",
    560: "triticale",
    563: "sugar beets",
    564: "winter green manure",
    565: "grass",
    566: "grass",
    571: "grass",
    572: "grass",
    573: "grass",
    574: "grass",
    580: "clover",
    581: "clover",
    582: "clover",
    583: "clover",
    599: "bare",
}
