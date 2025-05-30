import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as np
import numpy as onp
from matplotlib.patches import Patch
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")

sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "legend.fontsize": 9.0,
        "legend.title_fontsize": 10.0,
    },
)


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg

def repeat_by_areashare(values, area_share):
    ll = [np.repeat(val, int(np.round(area_share[i], 0)))for i, val in enumerate(values)]
    return np.concatenate(ll)


base_path = Path(__file__).parent
# directory of results
base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# identifiers for crop rotation scenarios
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

_dict_crop_to_ger = {"winter-wheat": "Winterweizen",
                     "clover": "Kleegras",
                     "silage-corn": "Silomais",
                     "summer-wheat": "Sommerweizen",
                     "sugar-beet": "Zuckerrübe",
                     "winter-rape": "Winterraps",
                     "soybean": "Sojabohne",
                     "grain-corn": "Körnermais",
                     "winter-barley": "Wintergerste",
                     "grass": "Brache",
                     "miscanthus": "Miscanthus",
                    }

# identifiers for simulations
locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]

_dict_location = {"freiburg": "Freiburg",
                  "lahr": "Lahr",
                  "muellheim": "Müllheim",
                  "stockach": "Stockach",
                  "gottmadingen": "Gottmadingen",
                  "weingarten": "Weingarten",
                  "eppingen-elsenz": "Eppingen",
                  "bruchsal-heidelsheim": "Bruchsal",
                  "bretten": "Bretten",
                  "ehingen-kirchen": "Ehingen",
                  "merklingen": "Merklingen",
                  "hayingen": "Hayingen",
                  "kupferzell": "Kupferzell",
                  "oehringen": "Öhringen",
                  "vellberg-kleinaltdorf": "Vellberg"}

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

crop_rotation_scenarios_mustard = ["winter-wheat_silage-corn",
                                   "summer-wheat_winter-wheat",
                                   "winter-wheat_sugar-beet_silage-corn",
                                   "summer-wheat_winter-wheat_silage-corn",
                                   "summer-wheat_winter-wheat_winter-rape",
                                   "sugar-beet_winter-wheat_winter-barley", 
                                   "grain-corn_winter-wheat_winter-rape", 
                                   "grain-corn_winter-wheat_winter-barley",
                                   "winter-wheat_silage-corn_yellow-mustard",
                                   "summer-wheat_winter-wheat_yellow-mustard",
                                   "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                                   "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                                   "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                                   "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                                   "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                                   "grain-corn_winter-wheat_winter-barley_yellow-mustard"]

fertilization_intensities = ["niedrig", "mittel", "hoch"]

_lab_unit_annual = {
    "M_q_ss": "$NO_3$-N [kg N/Jahr/ha]",
    "C_q_ss": "$NO_3$ [mg/l]",
    "q_ss": "GW-Neubildung\n [mm/Jahr]",
    "q_hof": "Oberflaechenabfluss\n [mm/Jahr]",
    "prec": "Niederschlag\n [mm/Jahr]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()
df_areas = pd.read_csv(base_path / "output" / "areas.csv", sep=";")

# load simulated fluxes and states
dict_fluxes_states = {}
for location in locations:
    dict_fluxes_states[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
        dict_fluxes_states[location][crop_rotation_scenario] = {}
        output_hm_file = (
            base_path_output
            / "svat_crop"
            / f"SVATCROP_{location}_{crop_rotation_scenario}.nc"
        )
        ds_fluxes_states = xr.open_dataset(output_hm_file, engine="h5netcdf")
        # assign date
        days = ds_fluxes_states["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        date = num2date(
            days,
            units=f"days since {ds_fluxes_states['Time'].attrs['time_origin']}",
            calendar="standard",
            only_use_cftime_datetimes=False,
        )
        ds_fluxes_states = ds_fluxes_states.assign_coords(Time=("Time", date))
        dict_fluxes_states[location][crop_rotation_scenario] = ds_fluxes_states

# # load nitrogen loads and concentrations
# dict_nitrate = {}
# for location in locations:
#     dict_nitrate[location] = {}
#     for crop_rotation_scenario in crop_rotation_scenarios:
#         dict_nitrate[location][crop_rotation_scenario] = {}
#         for fertilization_intensity in fertilization_intensities:
#             dict_nitrate[location][crop_rotation_scenario][fertilization_intensity] = {}
#             output_nitrate_file = (
#                 base_path_output
#                 / "svat_crop_nitrate"
#                 / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
#             )
#             ds_nitrate = xr.open_dataset(output_nitrate_file, engine="h5netcdf")
#             # assign date
#             days = ds_nitrate["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
#             date = num2date(
#                 days,
#                 units=f"days since {ds_nitrate['Time'].attrs['time_origin']}",
#                 calendar="standard",
#                 only_use_cftime_datetimes=False,
#             )
#             ds_nitrate = ds_nitrate.assign_coords(Time=("Time", date))
#             dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] = ds_nitrate

# # file = base_path_output / "dict_nitrate.pickle"
# # with open(file, 'wb') as handle:
# #     pickle.dump(dict_nitrate, handle, protocol=pickle.HIGHEST_PROTOCOL)


# # plot weighted average annual sum
# vars_sim = ["M_q_ss", "C_q_ss"]
# for var_sim in vars_sim:
#         ll_df = []
#         for location in locations:
#             for crop_rotation_scenario in crop_rotation_scenarios:
#                 for fertilization_intensity in fertilization_intensities:
#                     if var_sim == "M_q_ss":
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual sum
#                         df_ann_avg1 = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
#                         cond = (df_areas["location"] == location)
#                         df_area_share = df_areas.loc[cond, :]
#                         data = df_params.loc[cond1, "CLUST_ID"].to_frame()
#                         data.loc[:, "area_share"] = 0.0
#                         for clust_id in clust_ids:
#                             cond1 = (df_area_share["clust_id"] == clust_id)
#                             if cond1.any():
#                                 cond2 = (data["CLUST_ID"] == clust_id)
#                                 data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
#                         data = repeat_by_areashare(df_ann_avg1.values, data["area_share"].values)
#                         df_ann_avg = pd.DataFrame(data=data).mean().to_frame()
#                         df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                         df_ann_avg.loc[:, "location"] = _dict_location[location]
#                         df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
#                         if location in ["freiburg", "lahr", "muellheim"]:
#                             df_ann_avg.loc[:, 'region'] = "Oberrheingraben"
#                         elif location in ["stockach", "gottmadingen", "weingarten"]:
#                             df_ann_avg.loc[:, 'region'] = "Bodensee"
#                         elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
#                             df_ann_avg.loc[:, 'region'] = "Kraichgau"
#                         elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
#                             df_ann_avg.loc[:, 'region'] = "Alb-Donau"
#                         elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
#                             df_ann_avg.loc[:, 'region'] = "Hohenlohe"
#                         ll_df.append(df_ann_avg)
#                         ylim = (0, 60)
#                     elif var_sim == "C_q_ss":
#                         ds = dict_fluxes_states[location][crop_rotation_scenario]
#                         sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#                         sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual mean
#                         df_avg1 = df.sum(axis=0).to_frame()
#                         cond = (df_areas["location"] == location)
#                         df_area_share = df_areas.loc[cond, :]
#                         data = df_params.loc[cond1, "CLUST_ID"].to_frame()
#                         data.loc[:, "area_share"] = 0.0
#                         for clust_id in clust_ids:
#                             cond1 = (df_area_share["clust_id"] == clust_id)
#                             if cond1.any():
#                                 cond2 = (data["CLUST_ID"] == clust_id)
#                                 data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
#                         data = repeat_by_areashare(df_avg1.values, data["area_share"].values)
#                         df_avg = pd.DataFrame(data=data).mean().to_frame()
#                         df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                         df_avg.loc[:, "location"] = _dict_location[location]
#                         df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
#                         if location in ["freiburg", "lahr", "muellheim"]:
#                             df_avg.loc[:, 'region'] = "Oberrheingraben"
#                         elif location in ["stockach", "gottmadingen", "weingarten"]:
#                             df_avg.loc[:, 'region'] = "Bodensee"
#                         elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
#                             df_avg.loc[:, 'region'] = "Kraichgau"
#                         elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
#                             df_avg.loc[:, 'region'] = "Alb-Donau"
#                         elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
#                             df_avg.loc[:, 'region'] = "Hohenlohe"
#                         ll_df.append(df_avg)
#                         ylim = (0, 75)
#         df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
#         df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
#         fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
#         sns.barplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, errorbar=None, palette="PuRd")
#         axes.legend().set_visible(False)
#         if var_sim == "C_q_ss":
#             axes.axhline(y=50, color="red", linestyle="--", linewidth=2)
#         axes.set_xlabel("")
#         axes.set_ylabel(_lab_unit_annual[var_sim])
#         axes.set_ylim(ylim)
#         axes.set_xlabel("")
#         plt.xticks(rotation=45)
#         fig.tight_layout()
#         file = base_path_figs / f"barplot_average_{var_sim}_locations_weighted.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

#         df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
#         df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
#         fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
#         sns.barplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, errorbar=None, palette="PuRd")
#         axes.legend().set_visible(False)
#         if var_sim == "C_q_ss":
#             axes.axhline(y=50, color="red", linestyle="--", linewidth=2)
#         axes.set_xlabel("")
#         axes.set_ylabel(_lab_unit_annual[var_sim])
#         axes.set_ylim(ylim)
#         axes.set_xlabel("")
#         plt.xticks(rotation=33)
#         fig.tight_layout()
#         file = base_path_figs / f"barplot_average_{var_sim}_regions_weighted.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

#         df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
#         df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
#         fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
#         sns.barplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, errorbar=None, palette="PuRd")
#         axes.legend().set_visible(False)
#         if var_sim == "C_q_ss":
#             axes.axhline(y=50, color="red", linestyle="--", linewidth=2)
#         axes.set_xlabel("")
#         axes.set_ylabel(_lab_unit_annual[var_sim])
#         axes.set_ylim(ylim)
#         axes.set_xlabel("")
#         plt.xticks(rotation=33)
#         fig.tight_layout()
#         file = base_path_figs / f"barplot_average_{var_sim}_crop_rotations_weighted.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)


# plot weighted average annual sum
vars_sim = ["q_ss", "q_hof", "prec"]
vars_sim = ["prec"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                ds = dict_fluxes_states[location][crop_rotation_scenario]
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                cond1 = (df_params["CLUST_flag"] == 1)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                # calculate annual sum
                df_ann_avg1 = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                cond = (df_areas["location"] == location)
                df_area_share = df_areas.loc[cond, :]
                data = df_params.loc[cond1, "CLUST_ID"].to_frame()
                data.loc[:, "area_share"] = 0.0
                for clust_id in clust_ids:
                    cond1 = (df_area_share["clust_id"] == clust_id)
                    if cond1.any():
                        cond2 = (data["CLUST_ID"] == clust_id)
                        data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
                data = repeat_by_areashare(df_ann_avg1.values, data["area_share"].values)
                df_ann_avg = pd.DataFrame(data=data).mean().to_frame()
                df_ann_avg.loc[:, "location"] = _dict_location[location]
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_ann_avg.loc[:, 'region'] = "Oberrheingraben"
                elif location in ["stockach", "gottmadingen", "weingarten"]:
                    df_ann_avg.loc[:, 'region'] = "Bodensee"
                elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                    df_ann_avg.loc[:, 'region'] = "Kraichgau"
                elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                    df_ann_avg.loc[:, 'region'] = "Alb-Donau"
                elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                    df_ann_avg.loc[:, 'region'] = "Hohenlohe"
                ll_df.append(df_ann_avg)
                if var_sim == "q_ss":
                    ylim = (0, 600)
                    color = "grey"
                elif var_sim == "q_hof":
                    ylim = (0, 100)
                    color = "grey"
                elif var_sim == "prec":
                    ylim = (0, 1100)
                    color = "blue"
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.barplot(data=df_ann_avg_long, x="location", y="vals", ax=axes, errorbar=None, color=color)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"barplot_average_{var_sim}_locations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.barplot(data=df_ann_avg_long, x="region", y="vals", ax=axes,  errorbar=None, color=color)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"barplot_average_{var_sim}_regions_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation"], value_name="vals").loc[:, ["crop_rotation", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.barplot(data=df_ann_avg_long, x="crop_rotation", y="vals", ax=axes, errorbar=None, color=color)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"barplot_average_{var_sim}_crop_rotations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/data_nitrate_leaching") / "nitrate_leaching_values_per_crop.csv"
df = pd.read_csv(file, sep=";")

df1 = df.loc[:, ["crop", "QSUR"]]
df1.columns = ["crop", "vals"]
df1["variable"] = "QSUR"
df1["fertilization_intensity"] = np.nan

df2 = df.loc[:, ["crop", "PERC"]]
df2.columns = ["crop", "vals"]
df2["variable"] = "PERC"
df2["fertilization_intensity"] = np.nan

df3 = df.loc[:, ["crop", "MPERC_N1"]]
df3.columns = ["crop", "vals"]
df3["variable"] = "MPERC"
df3["fertilization_intensity"] = "niedrig"

df4 = df.loc[:, ["crop", "MPERC_N2"]]
df4.columns = ["crop", "vals"]
df4["variable"] = "MPERC"
df4["fertilization_intensity"] = "mittel"

df5 = df.loc[:, ["crop", "MPERC_N3"]]
df5.columns = ["crop", "vals"]
df5["variable"] = "MPERC"
df5["fertilization_intensity"] = "hoch"

df6 = df.loc[:, ["crop", "CPERC_N1"]]
df6.columns = ["crop", "vals"]
df6["variable"] = "CPERC"
df6["fertilization_intensity"] = "niedrig"

df7 = df.loc[:, ["crop", "CPERC_N2"]]
df7.columns = ["crop", "vals"]
df7["variable"] = "CPERC"
df7["fertilization_intensity"] = "mittel"

df8 = df.loc[:, ["crop", "CPERC_N3"]]
df8.columns = ["crop", "vals"]
df8["variable"] = "CPERC"
df8["fertilization_intensity"] = "hoch"

df_long = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
df_long["crop"] = [_dict_crop_to_ger[crop] for crop in df_long["crop"].values]

fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=False)
cond = (df_long["variable"] == "QSUR")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", ax=axes[0], errorbar=None, color="grey")
axes[0].set_xlabel("")
axes[0].set_ylabel("OA [mm/Jahr]")

cond = (df_long["variable"] == "PERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", ax=axes[1], errorbar=None, color="grey")
axes[1].set_xlabel("")
axes[1].set_ylabel("GWN [mm/Jahr]")

cond = (df_long["variable"] == "CPERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", hue="fertilization_intensity", ax=axes[2], errorbar=None, palette="PuRd")
axes[2].legend().set_visible(False)
axes[2].set_xlabel("")
axes[2].set_ylabel("$NO_3$-N [kg N/ha/Jahr]")

cond = (df_long["variable"] == "MPERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", hue="fertilization_intensity", ax=axes[3], errorbar=None, palette="PuRd")
axes[3].legend().set_visible(False)
axes[3].set_xlabel("")
axes[3].set_ylim(0, 60)
axes[3].axhline(y=50, color="red", linestyle="--", linewidth=2)
axes[3].axhline(y=37.5, color="orange", linestyle="--", linewidth=2)
axes[3].set_ylabel("$NO_3$ [mg/l]")
plt.xticks(rotation=33)

fig.tight_layout()
file = base_path_figs / "barplot_per_crop_types.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=False)
cond = (df_long["variable"] == "PERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", ax=axes[0], errorbar=None, color="grey")
axes[0].set_xlabel("")
axes[0].set_ylabel("GW-Neubildung\n [mm/Jahr]")
axes[0].text(
    0.97,
    0.92,
    "(a)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0].transAxes,
)

cond = (df_long["variable"] == "CPERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", hue="fertilization_intensity", ax=axes[1], errorbar=None, palette="PuRd")
axes[1].legend(title="N-Düngeintensität", loc="upper center", fontsize=8, title_fontsize=9, frameon=False, ncol=3)
axes[1].set_ylim(0, 50)
axes[1].set_xlabel("")
axes[1].set_ylabel("$NO_3$-N in\n GW-Neubildung\n [kg N/ha/Jahr]")
axes[1].text(
    0.97,
    0.92,
    "(b)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1].transAxes,
)

cond = (df_long["variable"] == "MPERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", hue="fertilization_intensity", ax=axes[2], errorbar=None, palette="PuRd")
axes[2].legend().set_visible(False)
axes[2].set_xlabel("")
axes[2].set_ylim(0, 60)
axes[2].axhline(y=50, color="red", linestyle="--", linewidth=2)
axes[2].axhline(y=37.5, color="orange", linestyle="--", linewidth=2)
axes[2].set_ylabel("$NO_3$ in\n GW-Neubildung\n [mg/l]")
axes[2].text(
    0.97,
    0.92,
    "(c)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[2].transAxes,
)
plt.xticks(rotation=33)

fig.tight_layout()
file = base_path_figs / "barplot_per_crop_types_.png"
fig.savefig(file, dpi=300)
plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(6, 4), sharex=True, sharey=False)
cond = (df_long["variable"] == "PERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", ax=axes[0], errorbar=None, color="grey")
axes[0].set_xlabel("")
axes[0].set_ylabel("GW-Neubildung\n [mm/Jahr]")
axes[0].text(
    0.97,
    0.92,
    "(a)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[0].transAxes,
)

cond = (df_long["variable"] == "CPERC")
df_long_ = df_long.loc[cond, :]
sns.barplot(data=df_long_, x="crop", y="vals", hue="fertilization_intensity", ax=axes[1], errorbar=None, palette="PuRd")
axes[1].legend(title="N-Düngeintensität", loc="upper center", fontsize=8, title_fontsize=9, frameon=False, ncol=3)
axes[1].set_ylim(0, 75)
axes[1].axhline(y=50, color="red", linestyle="--", linewidth=2)
axes[1].axhline(y=37.5, color="orange", linestyle="--", linewidth=2)
axes[1].set_xlabel("")
axes[1].set_ylabel("$NO_3$ in\n GW-Neubildung\n [mg/l]")
axes[1].text(
    0.97,
    0.92,
    "(b)",
    fontsize=9,
    horizontalalignment="center",
    verticalalignment="center",
    transform=axes[1].transAxes,
)
plt.xticks(rotation=45)

fig.tight_layout()
file = base_path_figs / "barplot_per_crop_types__.png"
fig.savefig(file, dpi=300)
plt.close(fig)