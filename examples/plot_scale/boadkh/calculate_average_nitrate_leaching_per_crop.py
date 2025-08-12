from pathlib import Path
import pandas as pd
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

_dict_ffid_rev = {v: k for k, v in _dict_ffid.items()}


_dict_crop_id = {"winter-wheat": 115,
                 "clover": 422,
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

crop_rotation_scenarios_without_mustard = ["winter-wheat_silage-corn",
                                           "summer-wheat_winter-wheat",
                                           "winter-wheat_sugar-beet_silage-corn",
                                           "summer-wheat_winter-wheat_silage-corn",
                                           "summer-wheat_winter-wheat_winter-rape",
                                           "sugar-beet_winter-wheat_winter-barley", 
                                           "grain-corn_winter-wheat_winter-rape", 
                                           "grain-corn_winter-wheat_winter-barley"]

crop_rotation_scenarios_with_mustard = ["winter-wheat_silage-corn_yellow-mustard",
                                        "summer-wheat_winter-wheat_yellow-mustard",
                                        "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                                        "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                                        "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                                        "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                                        "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                                        "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

fertilization_intensities = ["low", "medium", "high"]

@click.option("-td", "--tmp-dir", type=str, default=Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output")
@click.command("main")
def main(tmp_dir):
    base_path = Path(__file__).parent
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output"
    # base_path_output = Path(__file__).parent / "output"

    df_areas = pd.read_csv(base_path / "output" / "areas.csv", sep=";")
    df_areas["clust_id"] = df_areas["clust_id"].str.replace("-", "").astype(int)
    # # group by clust_id
    # df_areas = df_areas.groupby(["clust_id"]).sum().loc[:, ["area"]]
    # df_areas.index = [int(idx.replace('-', '')) for idx in df_areas.index]
    # df_areas = df_areas.sort_index(ascending=True)
    # # calculate area share of clust_id 
    # df_areas.loc[:, "area_share"] = df_areas["area"] / df_areas["area"].sum()

    file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values.csv"
    df = pd.read_csv(file, sep=";")

    # df_per_crop = pd.DataFrame(index=_dict_crop_id.keys(), columns=["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    # df_per_crop.index.name = "crop"
    # for key in _dict_crop_id.keys():
    #     df1 = df.copy()
    #     cond = (df1["CID"] == _dict_crop_id[key]) & onp.isin(df1["FFID"].values, list(_dict_ffid_rev.keys())[:16])
    #     df1 = df1.loc[cond, :]
    #     df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
    #     df2 = df2.sort_index(ascending=True)

    #     for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
    #         df_per_crop.loc[key, key1] = onp.sum(df2[key1] * df_areas["area_share"].values)

    # df_per_crop = df_per_crop.fillna(-9999)
    # file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop.csv"
    # df_per_crop.to_csv(file, sep=";", index=True, header=True)

    # df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios, columns=["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    # df_per_crop_rotation.index.name = "crop_rotation"
    # for key in crop_rotation_scenarios:
    #     df1 = df.copy()
    #     cond = (df1["FFID"] == _dict_ffid[key])
    #     df1 = df1.loc[cond, :]
    #     df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
    #     df2 = df2.sort_index(ascending=True)

    #     for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
    #         df_per_crop_rotation.loc[key, key1] = onp.sum(df2[key1] * df_areas["area_share"].values)

    # df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    # file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop_rotation.csv"
    # df_per_crop_rotation.to_csv(file, sep=";", index=True, header=True)

    # # averaged per region
    # regions = ["upper rhine valley", "lake constance", "kraichgau", "hohenlohe", "alb-danube"]
    # df_per_crop = pd.DataFrame(index=_dict_crop_id.keys(), columns=["region", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    # df_per_crop.index.name = "crop"
    # for region in regions:
    #     cond = (df_areas["region"] == region)
    #     df_areas_region = df_areas.loc[cond, :]
    #     df_areas_region["area_share"] = df_areas_region["area"] / df_areas_region["area"].sum()
    #     for key in _dict_crop_id.keys():
    #         df1 = df.copy()
    #         cond = (df1["CID"] == _dict_crop_id[key]) & onp.isin(df1["FFID"].values, list(_dict_ffid_rev.keys())[:16]) & (df1["agr_region"] == region)
    #         df1 = df1.loc[cond, :]
    #         df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
    #         df2 = df2.sort_index(ascending=True)
    #         df_per_crop.loc[key, "region"] = region

    #         for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
    #             df_per_crop.loc[key, key1] = onp.sum(df2[key1] * df_areas_region["area_share"].values)

    # df_per_crop = df_per_crop.fillna(-9999)
    # file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop_and_region.csv"
    # df_per_crop.to_csv(file, sep=";", index=True, header=True)

    # df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios, columns=["region", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    # df_per_crop_rotation.index.name = "crop_rotation"
    # for region in regions:
    #     cond = (df_areas["region"] == region)
    #     df_areas_region = df_areas.loc[cond, :]
    #     df_areas_region["area_share"] = df_areas_region["area"] / df_areas_region["area"].sum()
    #     for key in crop_rotation_scenarios:
    #         df1 = df.copy()
    #         cond = (df1["FFID"] == _dict_ffid[key]) & (df1["agr_region"] == region)
    #         df1 = df1.loc[cond, :]
    #         df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
    #         df2 = df2.sort_index(ascending=True)

    #         df_per_crop_rotation.loc[key, "region"] = region

    #         for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
    #             df_per_crop_rotation.loc[key, key1] = onp.sum(df2[key1] * df_areas_region["area_share"].values)

    # df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    # file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop_rotation_and_region.csv"
    # df_per_crop_rotation.to_csv(file, sep=";", index=True, header=True)

    # # averaged per subregion
    # subregions = ["freiburg", "lahr", "muellheim",
    #               "stockach", "gottmadingen", "weingarten",
    #               "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
    #               "ehingen-kirchen", "merklingen", "hayingen",
    #               "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
    # df_per_crop = pd.DataFrame(index=_dict_crop_id.keys(), columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
    # df_per_crop.index.name = "crop"
    # for subregion in subregions:
    #     cond = (df_areas["location"] == subregion)
    #     df_areas_subregion = df_areas.loc[cond, :]
    #     df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
    #     for key in _dict_crop_id.keys():
    #         df1 = df.copy()
    #         cond = (df1["CID"] == _dict_crop_id[key]) & onp.isin(df1["FFID"].values, list(_dict_ffid_rev.keys())[:16]) & (df1["stationsna"] == subregion)
    #         df1 = df1.loc[cond, :]
    #         df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
    #         df2 = df2.sort_index(ascending=True)
    #         df_per_crop.loc[key, "subregion"] = subregion

    #         for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
    #             df_per_crop.loc[key, key1] = onp.sum(df2[key1] * df_areas_subregion["area_share"].values)

    # df_per_crop = df_per_crop.fillna(-9999)
    # file = base_path_output / "data_nitrate_leaching" / "nitrate_leaching_values_per_crop_and_subregion.csv"
    # df_per_crop.to_csv(file, sep=";", index=True, header=True)


    subregions = ["freiburg", "lahr", "muellheim",
                  "stockach", "gottmadingen", "weingarten",
                  "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                  "ehingen-kirchen", "merklingen", "hayingen",
                  "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
    
    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.sum(df2[key1][cond] * df_areas_subregion["area_share"].values)

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "avg_nitrate_leaching_values_per_crop_rotation_and_subregion_alldata.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.nanmin(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    file = base_path_output / "data_nitrate_leaching" / "min_nitrate_leaching_values_per_crop_rotation_and_subregio_alldata.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=True, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.max(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)

    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "max_nitrate_leaching_values_per_crop_rotation_and_subregion_alldata.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)


    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_without_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.sum(df2[key1][cond] * df_areas_subregion["area_share"].values)

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "avg_nitrate_leaching_values_per_crop_rotation_and_subregion.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_without_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.nanmin(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    file = base_path_output / "data_nitrate_leaching" / "min_nitrate_leaching_values_per_crop_rotation_and_subregion.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=True, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_without_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.max(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)

    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "max_nitrate_leaching_values_per_crop_rotation_and_subregion.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_without_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.sum(df2[key1][cond] * df_areas_subregion["area_share"].values)

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "avg_nitrate_leaching_values_per_crop_rotation_and_subregion_mustard.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)


    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_with_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.nanmin(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "min_nitrate_leaching_values_per_crop_rotation_and_subregion_mustard.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)

    ll_dfs = []
    for subregion in subregions:
        df_per_crop_rotation = pd.DataFrame(index=crop_rotation_scenarios_without_mustard, columns=["subregion", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"])
        df_per_crop_rotation.index.name = "crop_rotation"
        cond = (df_areas["location"] == subregion)
        df_areas_subregion = df_areas.loc[cond, :]
        df_areas_subregion.index = df_areas_subregion["clust_id"]
        df_areas_subregion = df_areas_subregion.sort_index(ascending=True)
        df_areas_subregion["area_share"] = df_areas_subregion["area"] / df_areas_subregion["area"].sum()
        for key in crop_rotation_scenarios_with_mustard:
            df1 = df.copy()
            cond = (df1["FFID"] == _dict_ffid[key]) & (df1["stationsna"] == subregion)
            df1 = df1.loc[cond, :]
            df2 = df1.loc[:, ["SID", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]].groupby(["SID"]).mean()
            df2 = df2.sort_index(ascending=True)

            df_per_crop_rotation.loc[key, "subregion"] = subregion

            cond = df2.index.isin(df_areas_subregion.index)
            for key1 in ["QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]:
                df_per_crop_rotation.loc[key, key1] = onp.max(df2[key1][cond])

        ll_dfs.append(df_per_crop_rotation)

    df_per_crop_rotation = pd.concat(ll_dfs, axis=0)
    df_per_crop_rotation = df_per_crop_rotation.fillna(-9999)
    df_per_crop_rotation["crop_rotation"] = df_per_crop_rotation.index
    df_per_crop_rotation = df_per_crop_rotation.loc[:, ["subregion", "crop_rotation", "QSUR", "PERC", "MPERC_N1", "MPERC_N2", "MPERC_N3", "CPERC_N1", "CPERC_N2", "CPERC_N3"]]
    file = base_path_output / "data_nitrate_leaching" / "max_nitrate_leaching_values_per_crop_rotation_and_subregion_mustard.csv"
    df_per_crop_rotation.to_csv(file, sep=";", index=False, header=True)

if __name__ == "__main__":
    main()