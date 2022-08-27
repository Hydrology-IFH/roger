from pathlib import Path
import pandas as pd
import numpy as onp

CSV_DIR = Path(__file__).parent / "look_up_tables"

path_ilu = CSV_DIR / "intercept_land_use.csv"
df_ilu = pd.read_csv(path_ilu, sep=";")
ARR_ILU = onp.asarray(df_ilu.values)

path_is = CSV_DIR / "intercept_sealing.csv"
df_is = pd.read_csv(path_is, sep=";", skiprows=1)
ARR_IS = onp.asarray(df_is.values)

path_mlms = CSV_DIR / "mp_layer_manning_strickler.csv"
df_mlms = pd.read_csv(path_mlms, sep=";", skiprows=1)
ARR_MLMS = onp.asarray(df_mlms.values)

path_rdlu = CSV_DIR / "root_depth_land_use.csv"
df_rdlu = pd.read_csv(path_rdlu, sep=";", skiprows=1)
ARR_RDLU = onp.asarray(df_rdlu.values)

path_cp = CSV_DIR / "crop_parameters.csv"
df_cp = pd.read_csv(path_cp, sep=";", na_values=-9999, skiprows=1)
ARR_CP = onp.asarray(df_cp.values)

ARR_GC = onp.zeros((25, 13))
ARR_GC[:, 0] = ARR_ILU[:, 0]
ARR_GC[:, 1:] = 1 - 0.7**ARR_ILU[:, 1:]

ARR_GCM = onp.zeros((25, 2))
ARR_GCM[:, 0] = ARR_ILU[:, 0]

ARR_GCM[:, 1] = onp.max(ARR_GC[:, 1:], axis=1)

SUMMER_CROPS = onp.array([501, 502, 503, 504, 505, 506, 507, 508,
                          509, 510, 511, 512, 513, 514, 515, 516,
                          517, 518, 519, 520, 521, 522, 523, 524,
                          525, 526, 527, 528, 529, 530, 531, 532,
                          533, 534, 535, 536, 537, 538, 539, 540,
                          541, 542, 543, 544, 545, 546, 547, 548,
                          549, 550, 551, 552, 553, 554, 555, 561,
                          562, 563, 565, 567], dtype=int)
WINTER_CROPS = onp.array([556, 557, 558, 559, 560, 564], dtype=int)
WINTER_CATCH_CROPS = onp.array([566, 568, 569, 570], dtype=int)
MULTI_YEAR_CROPS_INIT = onp.array([571, 572], dtype=int)
MULTI_YEAR_CROPS_CONT = onp.array([573, 574], dtype=int)
WINTER_MULTI_YEAR_CROPS_INIT = onp.array([572], dtype=int)
WINTER_MULTI_YEAR_CROPS_CONT = onp.array([574], dtype=int)
SUMMER_MULTI_YEAR_CROPS_INIT = onp.array([571], dtype=int)
SUMMER_MULTI_YEAR_CROPS_CONT = onp.array([573], dtype=int)

dict_crops = {
              557: 'winter wheat',
              556: 'winter barley',
              563: 'sugar beets',
              559: 'winter rape',
              560: 'triticale',
              539: 'silage_corn',
              564: 'winter green manure',
              536: 'beetroot',
              571: 'grass',
              572: 'grass',
              573: 'grass',
              574: 'grass',
             }
