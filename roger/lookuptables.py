from pathlib import Path
import pandas as pd
from roger.core.operators import numpy as npx, update, at

CSV_DIR = Path(__file__).parent / "look_up_tables"

path_ilu = CSV_DIR / "intercept_land_use.csv"
df_ilu = pd.read_csv(path_ilu, sep=";")
ARR_ILU = npx.asarray(df_ilu.values)

path_is = CSV_DIR / "intercept_sealing.csv"
df_is = pd.read_csv(path_is, sep=";", skiprows=1)
ARR_IS = npx.asarray(df_is.values)

path_mlms = CSV_DIR / "mp_layer_manning_strickler.csv"
df_mlms = pd.read_csv(path_mlms, sep=";", skiprows=1)
ARR_MLMS = npx.asarray(df_mlms.values)

path_rdlu = CSV_DIR / "root_depth_land_use.csv"
df_rdlu = pd.read_csv(path_rdlu, sep=";", skiprows=1)
ARR_RDLU = npx.asarray(df_rdlu.values)

path_cp = CSV_DIR / "crop_parameters.csv"
df_cp = pd.read_csv(path_cp, sep=";", na_values=-9999, skiprows=1)
ARR_CP = npx.asarray(df_cp.values)

ARR_GC = npx.zeros((25, 13))
ARR_GC = update(
    ARR_GC,
    at[:, 0], ARR_ILU[:, 0],
)
ARR_GC = update(
    ARR_GC,
    at[:, 1:], 1 - 0.7**ARR_ILU[:, 1:],
)

ARR_GCM = npx.zeros((25, 2))
ARR_GCM = update(
    ARR_GCM,
    at[:, 0], ARR_ILU[:, 0],
)
ARR_GCM = update(
    ARR_GCM,
    at[:, 1], npx.max(ARR_GC[:, 1:], axis=1),
)

SUMMER_CROPS = npx.array([501, 502, 503, 504, 505, 506, 507, 508,
                          509, 510, 511, 512, 513, 514, 515, 516,
                          517, 518, 519, 520, 521, 522, 523, 524,
                          525, 526, 527, 528, 529, 530, 531, 532,
                          533, 534, 535, 536, 537, 538, 539, 540,
                          541, 542, 543, 544, 545, 546, 547, 548,
                          549, 550, 551, 552, 553, 554, 555, 561,
                          562, 563, 565, 567], dtype=int)
WINTER_CROPS = npx.array([556, 557, 558, 559, 560, 564], dtype=int)
WINTER_CATCH_CROPS = npx.array([566, 568, 569, 570], dtype=int)
MULTI_YEAR_CROPS_INIT = npx.array([571, 572], dtype=int)
MULTI_YEAR_CROPS_CONT = npx.array([573, 574], dtype=int)
WINTER_MULTI_YEAR_CROPS_INIT = npx.array([572], dtype=int)
WINTER_MULTI_YEAR_CROPS_CONT = npx.array([574], dtype=int)
SUMMER_MULTI_YEAR_CROPS_INIT = npx.array([571], dtype=int)
SUMMER_MULTI_YEAR_CROPS_CONT = npx.array([573], dtype=int)
