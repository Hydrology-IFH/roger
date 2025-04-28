from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import geopandas as gpd
import numpy as onp
import matplotlib as mpl
import seaborn as sns
import click
import warnings
warnings.filterwarnings('ignore')

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 9
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 9.0,
    },
)

_dict_ffid = {"winter-wheat_clover": "1_0",
              "winter-wheat_silage-corn": "2_0",
              "summer-wheat_winter-wheat": "3_0",
              "summer-wheat_clover_winter-wheat": "4_0",
              "winter-wheat_clover_silage-corn": "5_0",
              "winter-wheat_sugar-beet_silage-corn": "6_0",
              "summer-wheat_winter-wheat_silage-corn": "7_0",
              "summer-wheat_winter-wheat_winter-rape": "8_0",
              "winter-wheat_winter-rape": "9_0",
              "winter-wheat_soybean_winter-rape": "10_0",
              "sugar-beet_winter-wheat_winter-barley": "11_0", 
              "grain-corn_winter-wheat_winter-rape": "12_0", 
              "grain-corn_winter-wheat_winter-barley": "13_0",
              "grain-corn_winter-wheat_clover": "14_0",
              "miscanthus": "15_0",
              "bare-grass": "16_0",
              "winter-wheat_silage-corn_yellow-mustard": "2_1",
              "summer-wheat_winter-wheat_yellow-mustard": "3_1",
              "winter-wheat_sugar-beet_silage-corn_yellow-mustard": "6_1",
              "summer-wheat_winter-wheat_silage-corn_yellow-mustard": "7_1",
              "summer-wheat_winter-wheat_winter-rape_yellow-mustard": "8_1",
              "sugar-beet_winter-wheat_winter-barley_yellow-mustard": "11_1", 
              "grain-corn_winter-wheat_winter-rape_yellow-mustard": "12_1", 
              "grain-corn_winter-wheat_winter-barley_yellow-mustard": "13_1", 
}

_dict_ffid1 = {"winter-wheat_clover": 310,
              "winter-wheat_silage-corn": 311,
              "summer-wheat_winter-wheat": 312,
              "summer-wheat_clover_winter-wheat": 313,
              "winter-wheat_clover_silage-corn": 314,
              "winter-wheat_sugar-beet_silage-corn": 315,
              "summer-wheat_winter-wheat_silage-corn": 316,
              "summer-wheat_winter-wheat_winter-rape": 317,
              "winter-wheat_winter-rape": 318,
              "winter-wheat_soybean_winter-rape": 319,
              "sugar-beet_winter-wheat_winter-barley": 320, 
              "grain-corn_winter-wheat_winter-rape": 321, 
              "grain-corn_winter-wheat_winter-barley": 322,
              "grain-corn_winter-wheat_clover": 323,
              "miscanthus": 324,
              "bare-grass": 325,
              "winter-wheat_silage-corn_yellow-mustard": 351,
              "summer-wheat_winter-wheat_yellow-mustard": 352,
              "winter-wheat_sugar-beet_silage-corn_yellow-mustard": 353,
              "summer-wheat_winter-wheat_silage-corn_yellow-mustard": 354,
              "summer-wheat_winter-wheat_winter-rape_yellow-mustard": 355,
              "sugar-beet_winter-wheat_winter-barley_yellow-mustard": 356, 
              "grain-corn_winter-wheat_winter-rape_yellow-mustard": 357, 
              "grain-corn_winter-wheat_winter-barley_yellow-mustard": 358, 
}

_dict_var_names = {"q_hof": "QSUR",
                   "ground_cover": "GC",
                   "M_q_ss": "MPERC",
                   "C_q_ss": "CPERC",
                   "q_ss": "PERC",
}

_dict_fert = {"low": 1,
              "medium": 2,
              "high": 3,
}

_dict_crop_id = {"winter-wheat": 115,
                 "clover": 425,
                 "silage-corn": 411,
                 "summer-wheat": 116,
                 "sugar-beet": 603,
                 "winter-rape": 311,
                 "soybean": 330,
                 "grain-corn": 171,
                 "winter-barley": 131,
                 "grass": 591,
                 "miscanthus": 852,
                }

_dict_crop_id1 = {"winter-wheat": 411,
                 "clover": 412,
                 "silage-corn": 413,
                 "summer-wheat": 414,
                 "sugar-beet": 415,
                 "winter-rape": 416,
                 "soybean": 417,
                 "grain-corn": 418,
                 "winter-barley": 419,
                 "grass": 420,
                 "miscanthus": 421,
                }


_dict_clust_id = {51: 210,
                  52: 211,
                  53: 212,
                  54: 213,
                  55: 214,
                  56: 215,
                  57: 216,
                  58: 217,
                  59: 218,
                  510: 219,
                  511: 220,
                  512: 221,
                  513: 222,
                  514: 223,
                  515: 224,
                  516: 225,
                  517: 226,
                  518: 227,
                  519: 228,
                  520: 229,
}

_dict_lu_id = {"winter-wheat": [557],
               "clover": [583, 584, 585],
               "silage-corn": [539],
               "summer-wheat": [543],
               "sugar-beet": [563],
               "winter-rape": [559],
               "soybean": [541],
               "grain-corn": [525],
               "winter-barley": [556],
               "miscanthus": [589, 590, 591],
               "grass": [571, 573, 574],
               "bare": [599],
                }

_dict_crop_periods = {"winter-wheat_clover": {"winter-wheat": [557], "clover": [583, 584, 585]},
                      "winter-wheat_silage-corn": {"winter-wheat": [557], "silage-corn": [539]},
                      "summer-wheat_winter-wheat": {"winter-wheat": [557], "summer-wheat": [543]},
                      "summer-wheat_clover_winter-wheat": {"winter-wheat": [557], "summer-wheat": [543], "clover": [583, 584, 585]},
                      "winter-wheat_clover_silage-corn": {"winter-wheat": [557], "silage-corn": [539], "clover": [583, 584, 585]},
                      "winter-wheat_sugar-beet_silage-corn": {"winter-wheat": [557], "silage-corn": [539], "sugar-beet": [563]},
                      "summer-wheat_winter-wheat_silage-corn": {"winter-wheat": [557], "summer-wheat": [543], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_winter-rape": {"winter-wheat": [557], "summer-wheat": [543], "winter-rape": [559]},
                      "winter-wheat_winter-rape": {"winter-wheat": [557], "winter-rape": [559]},
                      "winter-wheat_soybean_winter-rape": {"winter-wheat": [557], "winter-rape": [559], "soybean": [541]},
                      "sugar-beet_winter-wheat_winter-barley": {"winter-wheat": [557], "winter-barley": [556], "sugar-beet": [563]}, 
                      "grain-corn_winter-wheat_winter-rape": {"grain-corn": [525], "winter-wheat": [557], "winter-rape": [559]},
                      "grain-corn_winter-wheat_winter-barley": {"grain-corn": [525], "winter-wheat": [557], "winter-barley": [556]},
                      "grain-corn_winter-wheat_clover": {"grain-corn": [525], "winter-wheat": [557], "clover": [583, 584, 585]},
                      "miscanthus": {"miscanthus": [589, 590, 591]},
                      "bare-grass": {"grass": [599, 571, 573, 574]},
                      "winter-wheat_silage-corn_yellow-mustard": {"winter-wheat": [557], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543]},
                      "winter-wheat_sugar-beet_silage-corn_yellow-mustard": {"winter-wheat": [557], "silage-corn": [539], "sugar-beet": [563]},
                      "summer-wheat_winter-wheat_silage-corn_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543], "silage-corn": [539]},
                      "summer-wheat_winter-wheat_winter-rape_yellow-mustard": {"winter-wheat": [557], "summer-wheat": [543], "winter-rape": [559]},
                      "sugar-beet_winter-wheat_winter-barley_yellow-mustard": {"winter-wheat": [557], "winter-barley": [556], "sugar-beet": [563]},  
                      "grain-corn_winter-wheat_winter-rape_yellow-mustard": {"grain-corn": [525], "winter-wheat": [557], "winter-rape": [559]}, 
                      "grain-corn_winter-wheat_winter-barley_yellow-mustard": {"grain-corn": [525], "winter-wheat": [557], "winter-barley": [556]}, 
}

_dict_locations = {"freiburg": 111,
                   "lahr": 112,
                   "muellheim": 113,
                   "stockach": 114,
                   "gottmadingen": 115,
                   "weingarten": 116,
                   "eppingen-elsenz": 117,
                   "bruchsal-heidelsheim": 118,
                   "bretten": 119,
                   "ehingen-kirchen": 120,
                   "merklingen": 121,
                   "hayingen": 122,
                   "kupferzell": 123,
                   "oehringen": 124,
                   "vellberg-kleinaltdorf": 125,
}

locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]

crop_rotation_scenarios = ["winter-wheat_clover",
                           "winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat",
                           "summer-wheat_clover_winter-wheat",
                           "winter-wheat_clover_silage-corn",
                           "winter-wheat_sugar-beet_silage-corn",
                           "summer-wheat_winter-wheat_silage-corn",
                           "summer-wheat_winter-wheat_winter-rape",
                           "winter-wheat_winter-rape",
                           "winter-wheat_soybean_winter-rape",
                           "sugar-beet_winter-wheat_winter-barley", 
                           "grain-corn_winter-wheat_winter-rape", 
                           "grain-corn_winter-wheat_winter-barley",
                           "grain-corn_winter-wheat_clover",
                           "miscanthus",
                           "bare-grass",
                           "winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_yellow-mustard",
                           "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                           "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                           "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

fertilization_intensities = ["low", "medium", "high"]

_dict_fertilization_intensities = {"low": 501, 
                                   "medium": 502, 
                                   "high": 503}


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg

@click.option("-td", "--tmp-dir", type=str, default=Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output")
@click.command("main")
def main(tmp_dir):
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output"
    # base_path_output = Path(__file__).parent / "output"

    df_areas = pd.read_csv(base_path / "output" / "areas.csv", sep=";")
    # group by clust_id
    df_areas = df_areas.groupby(["clust_id"]).sum().loc[:, ["area"]]
    df_areas.index = [int(idx.replace('-', '')) for idx in df_areas.index]
    df_areas = df_areas.sort_index(ascending=True)
    # calculate area share of clust_id 
    df_areas.loc[:, "area_share"] = df_areas["area"] / df_areas["area"].sum()

    file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values.csv"
    df = pd.read_csv(file, sep=";")

    df_per_crop = pd.DataFrame(index=_dict_crop_id.keys(), columns=["MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    df_per_crop.index.name = "crop"
    for key in _dict_crop_id.keys():
        df1 = df.copy()
        cond = (df1["CID"] == _dict_crop_id[key])
        df1 = df1.loc[cond, :]
        df2 = df1.loc[:, ["SID", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
        df2 = df2.sort_index(ascending=True)

        for key1 in ["MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
            df_per_crop.loc[key, key1] = onp.sum(df2[key1] * df_areas["area_share"].values)

    df_per_crop = df_per_crop.fillna(-9999)
    file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop.csv"
    df_per_crop.to_csv(file, sep=";", index=True, header=True)


if __name__ == "__main__":
    main()