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

_ffids_mustard = ["2_0", "3_0", "6_0", "7_0", "8_0", "11_0", "12_0", "13_0",
                  "2_1", "3_1", "6_1", "7_1", "8_1", "11_1", "12_1", "13_1"]

_ffids_no_mustard = ["2_0", "3_0", "6_0", "7_0", "8_0", "11_0", "12_0", "13_0"]

_ffids_with_mustard = ["2_1", "3_1", "6_1", "7_1", "8_1", "11_1", "12_1", "13_1"]

_dict_intercropping_effects = {'low Nfert & no intercropping': 1.0,
                               'low Nfert & intercropping': 1.01,
                               'medium Nfert & no intercropping': 2.0,
                               'medium Nfert & intercropping': 2.01,
                               'high Nfert & no intercropping': 3,
                               'high Nfert & intercropping': 3.01
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

_dict_crop_id_rev = {115: "winter wheat",
                     425: "clover",
                     411: "silage corn",
                     116: "summer wheat",
                     603: "sugar beet",
                     311: "winter rape",
                     330: "soybean",
                     171: "grain corn",
                     131: "winter barley",
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

fertilization_intensities = ["low", "medium", "high"]

_lab_unit_daily = {
    "M_q_ss": "PERC-$NO_3$\n [kg N/Tag/ha]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]"
}

_lab_unit_annual = {
    "M_q_ss": "PERC-$NO_3$-N\n [kg N/Jahr/ha]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]",
    "q_ss": "PERC\n [mm/Jahr]",
    "q_hof": "$Q_{sur}$\n [mm/Jahr]",
}

_lab_unit_total = {
    "M_q_ss": "PERC-$NO_3$-N\n [kg N]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]"
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
    for crop_rotation_scenario in crop_rotation_scenarios:
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

# load nitrogen loads and concentrations
dict_nitrate = {}
for location in locations:
    dict_nitrate[location] = {}
    for crop_rotation_scenario in crop_rotation_scenarios:
        dict_nitrate[location][crop_rotation_scenario] = {}
        for fertilization_intensity in fertilization_intensities:
            dict_nitrate[location][crop_rotation_scenario][fertilization_intensity] = {}
            output_nitrate_file = (
                base_path_output
                / "svat_crop_nitrate"
                / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
            )
            ds_nitrate = xr.open_dataset(output_nitrate_file, engine="h5netcdf")
            # assign date
            days = ds_nitrate["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
            date = num2date(
                days,
                units=f"days since {ds_nitrate['Time'].attrs['time_origin']}",
                calendar="standard",
                only_use_cftime_datetimes=False,
            )
            ds_nitrate = ds_nitrate.assign_coords(Time=("Time", date))
            dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] = ds_nitrate

# # plot annual sums
# vars_sim = ["M_q_ss"]
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for fertilization_intensity in fertilization_intensities:
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             for var_sim in vars_sim:
#                 sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                 cond1 = (df_params["CLUST_flag"] == 1)
#                 df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                 # calculate annual sum
#                 df_ann = df.resample("YE").sum().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
#                 df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
    
#                 fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#                 sns.boxplot(data=df_ann_long, x="year", y="vals", ax=ax, color="red", whis=(1, 99), showfliers=False)
#                 ax.set_xlabel("Time [Year]")
#                 ax.set_ylabel(_lab_unit_annual[var_sim])
#                 ax.set_ylim(0, )
#                 fig.tight_layout()
#                 file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.png"
#                 fig.savefig(file, dpi=300)
#                 plt.close(fig)

# vars_sim = ["M_q_ss"]
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for var_sim in vars_sim:
#             ll_df = []
#             for fertilization_intensity in fertilization_intensities:
#                 ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                 sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                 cond1 = (df_params["CLUST_flag"] == 1)
#                 df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                 # calculate annual sum
#                 df_ann = df.resample("YE").sum().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
#                 df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
#                 ll_df.append(df_ann_long)

#         df_ann_long = pd.concat(ll_df)
    
#         fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#         sns.boxplot(data=df_ann_long, x="year", y="vals", hue="Nfert", ax=ax, whis=(1, 99), palette="RdPu", showfliers=False)
#         ax.set_xlabel("Time [Year]")
#         ax.set_ylabel(_lab_unit_annual[var_sim])
#         ax.set_ylim(0, )
#         plt.legend(title="N-Fertilization intensity", frameon=False)
#         fig.tight_layout()
#         file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

# # plot average annual sum for each location
# colors = sns.color_palette("RdPu_r", n_colors=len(fertilization_intensities))
# vars_sim = ["M_q_ss", "C_q_ss"]
# for var_sim in vars_sim:
#     for location in locations:
#         fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)
#         for i, fertilization_intensity in enumerate(fertilization_intensities[::-1]):
#             ll_df = []
#             for crop_rotation_scenario in crop_rotation_scenarios:
#                 if var_sim == "M_q_ss":
#                     ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                     sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                     cond1 = (df_params["CLUST_flag"] == 1)
#                     df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                     # calculate annual sum
#                     df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
#                     df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
#                     ll_df.append(df_ann_avg)
#                     ylim = (0, 80)
#                 elif var_sim == "C_q_ss":
#                     ds = dict_fluxes_states[location][crop_rotation_scenario]
#                     sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#                     ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                     sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#                     sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
#                     cond1 = (df_params["CLUST_flag"] == 1)
#                     df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                     # calculate annual mean
#                     df_avg = df.sum(axis=0).to_frame()
#                     df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
#                     ll_df.append(df_avg)
#                     ylim = (0, 150)
#             df_ann_avg = pd.concat(ll_df)
#             df_ann_avg_long = df_ann_avg.melt(id_vars="crop_rotation", value_name="vals").loc[:, ["crop_rotation", "vals"]]        
#             if var_sim == "C_q_ss":
#                 axes[i].axhline(y=50, color='r', linestyle='--', lw=2)            
#             sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", ax=axes[i], whis=(1, 99), showfliers=False, color=colors[i])
#             axes[i].set_xlabel("")
#             axes[i].set_ylabel(_lab_unit_annual[var_sim])
#             axes[i].set_ylim(ylim)
#         axes[-1].set_xlabel("Crop rotation")
#         plt.xticks(rotation=90)
#         fig.tight_layout()
#         file = base_path_figs / f"boxplot_annual_average_{var_sim}_{location}.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

# plot average annual sum
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                for fertilization_intensity in fertilization_intensities:
                    if var_sim == "M_q_ss":
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual sum
                        df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                        df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_ann_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_ann_avg.loc[:, 'region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_ann_avg.loc[:, 'region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_ann_avg.loc[:, 'region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_ann_avg.loc[:, 'region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_ann_avg.loc[:, 'region'] = "Hohenlohe"
                        ll_df.append(df_ann_avg)
                        ylim = (0, 80)
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg = df.sum(axis=0).to_frame()
                        df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_avg.loc[:, "location"] = _dict_location[location]
                        df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_avg.loc[:, 'region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_avg.loc[:, 'region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_avg.loc[:, 'region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_avg.loc[:, 'region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_avg.loc[:, 'region'] = "Hohenlohe"
                        ll_df.append(df_avg)
                        ylim = (0, 120)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2) 
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_locations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2) 
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_regions.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2) 
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crop_rotations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

# plot weighted average annual sum
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                for fertilization_intensity in fertilization_intensities:
                    if var_sim == "M_q_ss":
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
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
                        df_ann_avg = pd.DataFrame(data=data)
                        df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_ann_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_ann_avg.loc[:, 'region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_ann_avg.loc[:, 'region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_ann_avg.loc[:, 'region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_ann_avg.loc[:, 'region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_ann_avg.loc[:, 'region'] = "Hohenlohe"
                        ll_df.append(df_ann_avg)
                        ylim = (0, 70)
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg1 = df.sum(axis=0).to_frame()
                        cond = (df_areas["location"] == location)
                        df_area_share = df_areas.loc[cond, :]
                        data = df_params.loc[cond1, "CLUST_ID"].to_frame()
                        data.loc[:, "area_share"] = 0.0
                        for clust_id in clust_ids:
                            cond1 = (df_area_share["clust_id"] == clust_id)
                            if cond1.any():
                                cond2 = (data["CLUST_ID"] == clust_id)
                                data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
                        data = repeat_by_areashare(df_avg1.values, data["area_share"].values)
                        df_avg = pd.DataFrame(data=data)
                        df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_avg.loc[:, "location"] = _dict_location[location]
                        df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_avg.loc[:, 'region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_avg.loc[:, 'region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_avg.loc[:, 'region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_avg.loc[:, 'region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_avg.loc[:, 'region'] = "Hohenlohe"
                        ll_df.append(df_avg)
                        ylim = (0, 70)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2)  
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_locations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2)  
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_regions_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2)  
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crop_rotations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

# plot weighted average annual sum per crop type
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                # extract the values for the single crop periods within the crop rotation
                crop_ids = _dict_crop_periods[crop_rotation_scenario]
                for crop_id in crop_ids:
                    ds = dict_fluxes_states[location][crop_rotation_scenario]
                    lu_id = ds["lu_id"].isel(x=0, y=0).values[1:]
                    df_lu_id = pd.DataFrame(index=ds["Time"].values[1:], data=lu_id)
                    df_lu_id.columns = ["lu_id"]
                    cond = onp.isin(df_lu_id.values, onp.array([586, 587, 599])).flatten()
                    df_lu_id.loc[cond, "lu_id"] = onp.nan
                    df_lu_id = df_lu_id.bfill()
                    df_lu_id.loc[:, 'mask'] = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_lu_id[crop_id]).flatten()
                    df_lu_id['new'] = df_lu_id['mask'].gt(df_lu_id['mask'].shift(fill_value=False))
                    df_lu_id['crop_period'] = (df_lu_id.new + 0).cumsum() * df_lu_id['mask']
                    df_lu_id['day'] = 1
                    df_lu_id.replace(onp.nan, 0, inplace=True)
                    df_lu_id['crop_period'] = df_lu_id['crop_period'].replace(0, onp.nan)
                    df_lu_id = df_lu_id.drop(columns=['mask', 'new'])
                    df_lu_id = df_lu_id.astype({"lu_id": int, "crop_period": float, "day": float})
                    cond_crop = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_crop_periods[crop_rotation_scenario][crop_id]).flatten()
                    for fertilization_intensity in fertilization_intensities:
                        if var_sim == "M_q_ss":
                            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                            sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                            cond1 = (df_params["CLUST_flag"] == 1)
                            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                            df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values
                            df = df.loc[cond_crop, :]
                            # calculate sum per crop period
                            weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                            weight = onp.where(weight > 1, 1, weight)
                            df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis]
                            # calculate average
                            df_ann_avg1 = df_cp.mean(axis=0).to_frame()
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
                            df_ann_avg = pd.DataFrame(data=data)
                            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            df_ann_avg.loc[:, "location"] = _dict_location[location]
                            df_ann_avg.loc[:, "crop"] = crop_id
                            if location in ["freiburg", "lahr", "muellheim"]:
                                df_ann_avg.loc[:, 'region'] = "Oberrhein"
                            elif location in ["stockach", "gottmadingen", "weingarten"]:
                                df_ann_avg.loc[:, 'region'] = "Bodensee"
                            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                                df_ann_avg.loc[:, 'region'] = "Kraichgau"
                            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                                df_ann_avg.loc[:, 'region'] = "Alb-Donau"
                            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                                df_ann_avg.loc[:, 'region'] = "Hohenlohe"
                            ll_df.append(df_ann_avg)
                            ylim = (0, 70)
                        elif var_sim == "C_q_ss":
                            ds = dict_fluxes_states[location][crop_rotation_scenario]
                            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:][:, cond_crop]
                            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1][:, cond_crop] * 4.427  # convert nitrate-nitrogen to nitrate
                            sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                            cond1 = (df_params["CLUST_flag"] == 1)
                            df = pd.DataFrame(data=sim_vals.T).loc[:, cond1]
                            # calculate annual mean
                            df_avg1 = df.sum(axis=0).to_frame()
                            cond = (df_areas["location"] == location)
                            df_area_share = df_areas.loc[cond, :]
                            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
                            data.loc[:, "area_share"] = 0.0
                            for clust_id in clust_ids:
                                cond1 = (df_area_share["clust_id"] == clust_id)
                                if cond1.any():
                                    cond2 = (data["CLUST_ID"] == clust_id)
                                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
                            data = repeat_by_areashare(df_avg1.values, data["area_share"].values)
                            df_avg = pd.DataFrame(data=data)
                            df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            df_avg.loc[:, "location"] = _dict_location[location]
                            df_avg.loc[:, "crop"] = crop_id
                            if location in ["freiburg", "lahr", "muellheim"]:
                                df_avg.loc[:, 'region'] = "Oberrhein"
                            elif location in ["stockach", "gottmadingen", "weingarten"]:
                                df_avg.loc[:, 'region'] = "Bodensee"
                            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                                df_avg.loc[:, 'region'] = "Kraichgau"
                            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                                df_avg.loc[:, 'region'] = "Alb-Donau"
                            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                                df_avg.loc[:, 'region'] = "Hohenlohe"
                            ll_df.append(df_avg)
                            ylim = (0, 100)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop", "fertilization_intensity"], value_name="vals").loc[:, ["crop", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2)  
        sns.boxplot(data=df_ann_avg_long, x="crop", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crops_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# plot average annual sum
vars_sim = ["q_ss", "q_hof"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                ds = dict_fluxes_states[location][crop_rotation_scenario]
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                cond1 = (df_params["CLUST_flag"] == 1)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                # calculate annual sum
                df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                df_ann_avg.loc[:, "location"] = _dict_location[location]
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_ann_avg.loc[:, 'region'] = "Oberrhein"
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
                    ylim = (0, 800)
                elif var_sim == "q_hof":
                    ylim = (0, 200)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_locations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_regions.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation"], value_name="vals").loc[:, ["crop_rotation", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crop_rotations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# plot weighted average annual sum
vars_sim = ["q_ss", "q_hof"]
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
                df_ann_avg = pd.DataFrame(data=data)
                df_ann_avg.loc[:, "location"] = _dict_location[location]
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_ann_avg.loc[:, 'region'] = "Oberrhein"
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
                    ylim = (0, 800)
                elif var_sim == "q_hof":
                    ylim = (0, 200)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_locations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_regions_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation"], value_name="vals").loc[:, ["crop_rotation", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crop_rotations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# plot weighted average annual sum per crop type
vars_sim = ["q_ss", "q_hof"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                # extract the values for the single crop periods within the crop rotation
                crop_ids = _dict_crop_periods[crop_rotation_scenario]
                for crop_id in crop_ids:
                    ds = dict_fluxes_states[location][crop_rotation_scenario]
                    lu_id = ds["lu_id"].isel(x=0, y=0).values[1:]
                    df_lu_id = pd.DataFrame(index=ds["Time"].values[1:], data=lu_id)
                    df_lu_id.columns = ["lu_id"]
                    cond = onp.isin(df_lu_id.values, onp.array([586, 587, 599])).flatten()
                    df_lu_id.loc[cond, "lu_id"] = onp.nan
                    df_lu_id = df_lu_id.bfill()
                    df_lu_id.loc[:, 'mask'] = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_lu_id[crop_id]).flatten()
                    df_lu_id['new'] = df_lu_id['mask'].gt(df_lu_id['mask'].shift(fill_value=False))
                    df_lu_id['crop_period'] = (df_lu_id.new + 0).cumsum() * df_lu_id['mask']
                    df_lu_id['day'] = 1
                    df_lu_id.replace(onp.nan, 0, inplace=True)
                    df_lu_id['crop_period'] = df_lu_id['crop_period'].replace(0, onp.nan)
                    df_lu_id = df_lu_id.drop(columns=['mask', 'new'])
                    df_lu_id = df_lu_id.astype({"lu_id": int, "crop_period": float, "day": float})
                    cond_crop = onp.isin(df_lu_id.loc[:, "lu_id"].values, _dict_crop_periods[crop_rotation_scenario][crop_id]).flatten()

                    ds = dict_fluxes_states[location][crop_rotation_scenario]
                    sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                    cond1 = (df_params["CLUST_flag"] == 1)
                    df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                    df.loc[:, 'crop_period'] = df_lu_id.loc[:, 'crop_period'].values
                    df = df.loc[cond_crop, :]
                    # calculate sum per crop period
                    weight = 365/df_lu_id.groupby("crop_period").sum()["day"].values
                    weight = onp.where(weight > 1, 1, weight)
                    df_cp = df.groupby("crop_period").sum() * weight[:, onp.newaxis]
                    # calculate annual sum
                    df_ann_avg1 = df_cp.mean(axis=0).to_frame()
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
                    df_ann_avg = pd.DataFrame(data=data)
                    df_ann_avg.loc[:, "location"] = _dict_location[location]
                    df_ann_avg.loc[:, "crop"] = crop_id
                    if location in ["freiburg", "lahr", "muellheim"]:
                        df_ann_avg.loc[:, 'region'] = "Oberrhein"
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
                        ylim = (0, 800)
                    elif var_sim == "q_hof":
                        ylim = (0, 200)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["crop"], value_name="vals").loc[:, ["crop", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_crops_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# plot relative nitrate leaching
ll_df = []
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals1 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            sim_vals2 = ds["Nfert"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            cond1 = (df_params["CLUST_flag"] == 1)
            df1 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals1.T).loc[:, cond1]
            df2 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals2.T).loc[:, cond1]
            # calculate annual sum
            df1_ann_avg = df1.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            df_ann_avg = (df1_ann_avg / df2_ann_avg) * 100
            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
            df_ann_avg.loc[:, "location"] = _dict_location[location]
            df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
            if location in ["freiburg", "lahr", "muellheim"]:
                df_ann_avg.loc[:, 'region'] = "Oberrhein"
            elif location in ["stockach", "gottmadingen", "weingarten"]:
                df_ann_avg.loc[:, 'region'] = "Bodensee"
            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                df_ann_avg.loc[:, 'region'] = "Kraichgau"
            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                df_ann_avg.loc[:, 'region'] = "Alb-Donau"
            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                df_ann_avg.loc[:, 'region'] = "Hohenlohe"
            ll_df.append(df_ann_avg)
            ylim = (0, )
df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("Location")
plt.xticks(rotation=45)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_locations.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_regions.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_crop_rotations.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot weighted relative nitrate leaching
ll_df = []
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals1 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            sim_vals2 = ds["Nfert"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            cond1 = (df_params["CLUST_flag"] == 1)
            df1 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals1.T).loc[:, cond1]
            df2 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals2.T).loc[:, cond1]
            # calculate annual sum
            df1_ann_avg = df1.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data1 = repeat_by_areashare(df1_ann_avg.values, data["area_share"].values)
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data2 = repeat_by_areashare(df2_ann_avg.values, data["area_share"].values)
            df_ann_avg = pd.DataFrame(data=(data1 / data2) * 100)
            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
            df_ann_avg.loc[:, "location"] = _dict_location[location]
            df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
            if location in ["freiburg", "lahr", "muellheim"]:
                df_ann_avg.loc[:, 'region'] = "Oberrhein"
            elif location in ["stockach", "gottmadingen", "weingarten"]:
                df_ann_avg.loc[:, 'region'] = "Bodensee"
            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                df_ann_avg.loc[:, 'region'] = "Kraichgau"
            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                df_ann_avg.loc[:, 'region'] = "Alb-Donau"
            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                df_ann_avg.loc[:, 'region'] = "Hohenlohe"
            ll_df.append(df_ann_avg)
            ylim = (0, )
df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("Location")
plt.xticks(rotation=45)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_locations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_regions_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC-$NO_3$/$N_{fert}$ [%]")
axes.set_ylim(0, 50)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrate_leaching_crop_rotations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)


# plot weighted relative nitrogen uptake
ll_df = []
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals1_ = ds["M_transp"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            sim_vals2_ = ds["nh4_up"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            sim_vals1 = sim_vals1_ + sim_vals2_
            sim_vals2 = ds["Nfert"].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
            cond1 = (df_params["CLUST_flag"] == 1)
            df1 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals1.T).loc[:, cond1]
            df2 = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals2.T).loc[:, cond1]
            # calculate annual sum
            df1_ann_avg = df1.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data1 = repeat_by_areashare(df1_ann_avg.values, data["area_share"].values)
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data2 = repeat_by_areashare(df2_ann_avg.values, data["area_share"].values)
            df_ann_avg = pd.DataFrame(data=(data1 / data2) * 100)
            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
            df_ann_avg.loc[:, "location"] = _dict_location[location]
            df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
            if location in ["freiburg", "lahr", "muellheim"]:
                df_ann_avg.loc[:, 'region'] = "Oberrhein"
            elif location in ["stockach", "gottmadingen", "weingarten"]:
                df_ann_avg.loc[:, 'region'] = "Bodensee"
            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                df_ann_avg.loc[:, 'region'] = "Kraichgau"
            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                df_ann_avg.loc[:, 'region'] = "Alb-Donau"
            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                df_ann_avg.loc[:, 'region'] = "Hohenlohe"
            ll_df.append(df_ann_avg)
            ylim = (0, )
df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("$N_{uptake}$/$N_{fert}$ [%]")
axes.set_ylim(10, 100)
axes.set_xlabel("Location")
plt.xticks(rotation=45)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrogen_uptake_locations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("$N_{uptake}$/$N_{fert}$ [%]")
axes.set_ylim(10, 100)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrogen_uptake_regions_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("$N_{uptake}$/$N_{fert}$ [%]")
axes.set_ylim(10, 100)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_nitrogen_uptake_crop_rotations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot weighted relative groundwater recharge
ll_df = []
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            sim_vals2 = ds["prec"].isel(y=0).values[:, 1:]
            cond1 = (df_params["CLUST_flag"] == 1)
            df1 = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals1.T).loc[:, cond1]
            df2 = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals2.T).loc[:, cond1]
            # calculate annual sum
            df1_ann_avg = df1.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data1 = repeat_by_areashare(df1_ann_avg.values, data["area_share"].values)
            df2_ann_avg = df2.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            cond1 = (df_params["CLUST_flag"] == 1)
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data2 = repeat_by_areashare(df2_ann_avg.values, data["area_share"].values)
            df_ann_avg = pd.DataFrame(data=(data1 / data2) * 100)
            df_ann_avg.loc[:, "location"] = _dict_location[location]
            df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
            if location in ["freiburg", "lahr", "muellheim"]:
                df_ann_avg.loc[:, 'region'] = "Oberrhein"
            elif location in ["stockach", "gottmadingen", "weingarten"]:
                df_ann_avg.loc[:, 'region'] = "Bodensee"
            elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                df_ann_avg.loc[:, 'region'] = "Kraichgau"
            elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                df_ann_avg.loc[:, 'region'] = "Alb-Donau"
            elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                df_ann_avg.loc[:, 'region'] = "Hohenlohe"
            ll_df.append(df_ann_avg)
df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="location", y="vals", color="grey", ax=axes, whis=(1, 99), showfliers=False)
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC/PRECIP [%]")
axes.set_ylim(25, 75)
axes.set_xlabel("Location")
plt.xticks(rotation=45)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_gw_recharge_locations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="region", y="vals", color="grey", ax=axes, whis=(1, 99), showfliers=False)
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC/PRECIP [%]")
axes.set_ylim(25, 75)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_gw_recharge_regions_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation"]]
df_ann_avg_long = df_ann_avg.melt(id_vars=["crop_rotation"], value_name="vals").loc[:, ["crop_rotation", "vals"]]        
fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", color="grey", ax=axes, whis=(1, 99), showfliers=False)
axes.legend().set_visible(False)
axes.set_xlabel("")
axes.set_ylabel("PERC/PRECIP [%]")
axes.set_ylim(25, 75)
axes.set_xlabel("")
plt.xticks(rotation=33)
fig.tight_layout()
file = base_path_figs / "boxplot_annual_average_relative_gw_recharge_crop_rotations_weighted.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot average annual sum considering crop rotations with and without mustard
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_mustard:
                for fertilization_intensity in fertilization_intensities:
                    if var_sim == "M_q_ss":
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual sum
                        df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_ann_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & no intercropping']
                        else:
                            df_ann_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & intercropping']
                        df_ann_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_ann_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_ann_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_ann_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_ann_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_ann_avg['region'] = "Hohenlohe"
                        ll_df.append(df_ann_avg)
                        ylim = (0, 70)
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg = df.sum(axis=0).to_frame()
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & no intercropping']
                        else:
                            df_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & intercropping']
                        df_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_avg['region'] = "Hohenlohe"
                        ll_df.append(df_avg)
                        ylim = (0, 80)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2) 
            axes.axhline(y=37.5, color='orange', linestyle='--', lw=2)    
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd", hue_norm=(0, 4))
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_intercropping_effects_locations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_long = df_ann_avg.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 4), sharex=True, sharey=True)
        if var_sim == "C_q_ss":
            axes.axhline(y=50, color='r', linestyle='--', lw=2)   
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd", hue_norm=(0, 4))
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        axes.set_ylim(ylim)
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_intercropping_effects_regions.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

# plot difference between average annual sum considering crop rotations with and without mustard
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        ll_df_mustard = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_mustard:
                for fertilization_intensity in fertilization_intensities:
                    if var_sim == "M_q_ss":
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual sum
                        df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                        df_ann_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_ann_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_ann_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_ann_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_ann_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_ann_avg['region'] = "Hohenlohe"
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df.append(df_ann_avg)
                        else:
                            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df_mustard.append(df_ann_avg)
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg = df.sum(axis=0).to_frame()
                        df_avg.loc[:, "location"] = _dict_location[location]
                        df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_avg['region'] = "Hohenlohe"
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df.append(df_avg)
                        else:
                            df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df_mustard.append(df_avg)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_locations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_regions.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_crop_rotations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# plot weighted difference between average annual sum considering crop rotations with and without mustard
vars_sim = ["M_q_ss", "C_q_ss"]
for var_sim in vars_sim:
        ll_df = []
        ll_df_mustard = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_mustard:
                for fertilization_intensity in fertilization_intensities:
                    if var_sim == "M_q_ss":
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
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
                        df_ann_avg = pd.DataFrame(data=data)
                        df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_ann_avg.loc[:, "location"] = _dict_location[location]
                        df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_ann_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_ann_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_ann_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_ann_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_ann_avg['region'] = "Hohenlohe"
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df.append(df_ann_avg)
                        else:
                            df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df_mustard.append(df_ann_avg)
                    elif var_sim == "C_q_ss":
                        ds = dict_fluxes_states[location][crop_rotation_scenario]
                        sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
                        ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                        sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
                        sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
                        cond1 = (df_params["CLUST_flag"] == 1)
                        df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                        # calculate annual mean
                        df_avg1 = df.sum(axis=0).to_frame()
                        cond = (df_areas["location"] == location)
                        df_area_share = df_areas.loc[cond, :]
                        data = df_params.loc[cond1, "CLUST_ID"].to_frame()
                        data.loc[:, "area_share"] = 0.0
                        for clust_id in clust_ids:
                            cond1 = (df_area_share["clust_id"] == clust_id)
                            if cond1.any():
                                cond2 = (data["CLUST_ID"] == clust_id)
                                data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
                        data = repeat_by_areashare(df_avg1.values, data["area_share"].values)
                        df_avg = pd.DataFrame(data=data)
                        df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                        df_avg.loc[:, "location"] = _dict_location[location]
                        df_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                        if location in ["freiburg", "lahr", "muellheim"]:
                            df_avg['region'] = "Oberrhein"
                        elif location in ["stockach", "gottmadingen", "weingarten"]:
                            df_avg['region'] = "Bodensee"
                        elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                            df_avg['region'] = "Kraichgau"
                        elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                            df_avg['region'] = "Alb-Donau"
                        elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                            df_avg['region'] = "Hohenlohe"
                        if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                            df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df.append(df_avg)
                        else:
                            df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
                            ll_df_mustard.append(df_avg)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "location", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_locations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "region", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["region", "fertilization_intensity"], value_name="vals").loc[:, ["region", "fertilization_intensity", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        if var_sim == "C_q_ss":
            axes.set_ylim(-10, 10)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_regions_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "crop_rotation", "fertilization_intensity"]]
        df_ann_avg_diff = df_ann_avg_mustard.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["crop_rotation", "fertilization_intensity"], value_name="vals").loc[:, ["crop_rotation", "fertilization_intensity", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
        axes.legend().set_visible(False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        if var_sim == "C_q_ss":
            axes.set_ylim(-10, 10)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_crop_rotations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


vars_sim = ["q_ss", "q_hof"]
for var_sim in vars_sim:
        ll_df = []
        ll_df_mustard = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_mustard:
                ds = dict_fluxes_states[location][crop_rotation_scenario]
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                cond1 = (df_params["CLUST_flag"] == 1)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                # calculate annual sum
                df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                df_ann_avg.loc[:, "location"] = _dict_location[location]
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_ann_avg['region'] = "Oberrhein"
                elif location in ["stockach", "gottmadingen", "weingarten"]:
                    df_ann_avg['region'] = "Bodensee"
                elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                    df_ann_avg['region'] = "Kraichgau"
                elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                    df_ann_avg['region'] = "Alb-Donau"
                elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                    df_ann_avg['region'] = "Hohenlohe"
                if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                    ll_df.append(df_ann_avg)
                else:
                    ll_df_mustard.append(df_ann_avg)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "location"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_locations.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "region"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_regions.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


vars_sim = ["q_ss", "q_hof"]
for var_sim in vars_sim:
        ll_df = []
        ll_df_mustard = []
        for location in locations:
            for crop_rotation_scenario in crop_rotation_scenarios_mustard:
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
                df_ann_avg = pd.DataFrame(data=data)
                df_ann_avg.loc[:, "location"] = _dict_location[location]
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                if location in ["freiburg", "lahr", "muellheim"]:
                    df_ann_avg['region'] = "Oberrhein"
                elif location in ["stockach", "gottmadingen", "weingarten"]:
                    df_ann_avg['region'] = "Bodensee"
                elif location in ["eppingen-elsenz", "bruchsal-heidelsheim", "bretten"]:
                    df_ann_avg['region'] = "Kraichgau"
                elif location in ["ehingen-kirchen", "merklingen", "hayingen"]:
                    df_ann_avg['region'] = "Alb-Donau"
                elif location in ["kupferzell", "oehringen", "vellberg-kleinaltdorf"]:
                    df_ann_avg['region'] = "Hohenlohe"
                if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
                    ll_df.append(df_ann_avg)
                else:
                    ll_df_mustard.append(df_ann_avg)
        df_ann_avg = pd.concat(ll_df).loc[:, [0, "location"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "location"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["location"], value_name="vals").loc[:, ["location", "vals"]]        
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="location", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        # axes.set_ylim(ylim)
        axes.set_xlabel("Location")
        plt.xticks(rotation=45)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_locations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "region"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "region"]]
        df_ann_avg_diff = df_ann_avg.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["region"], value_name="vals").loc[:, ["region", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="region", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        if var_sim == "q_ss":
            axes.set_ylim(-70, 40)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_regions_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        df_ann_avg = pd.concat(ll_df).loc[:, [0, "crop_rotation"]]
        df_ann_avg_mustard = pd.concat(ll_df_mustard).loc[:, [0, "crop_rotation"]]
        df_ann_avg_diff = df_ann_avg_mustard.copy()        
        df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])
        df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["crop_rotation"], value_name="vals").loc[:, ["crop_rotation", "vals"]]       
        fig, axes = plt.subplots(1, 1, figsize=(6, 3), sharex=True, sharey=True)
        sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", color="#969696", ax=axes, whis=(1, 99), showfliers=False)
        axes.set_xlabel("")
        axes.set_ylabel(_lab_unit_annual[var_sim])
        if var_sim == "q_ss":
            axes.set_ylim(-70, 40)
        plt.xticks(rotation=0)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_intercropping_effects_crop_rotations_weighted.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)



# # plot average annual sum considering crop rotations with and without mustard
# vars_sim = ["M_q_ss", "C_q_ss"]
# for var_sim in vars_sim:
#     for i, _crop_rotation_scenario in enumerate(crop_rotation_scenarios_without_mustard):
#         crop_rotation_scenarios1 = [_crop_rotation_scenario, crop_rotation_scenarios_with_mustard[i]]
#         fig, axes = plt.subplots(1, 1, figsize=(3, 2), sharex=True, sharey=True)
#         ll_df = []
#         for location in locations:
#             for crop_rotation_scenario in crop_rotation_scenarios1:
#                 for fertilization_intensity in fertilization_intensities:
#                     if var_sim == "M_q_ss":
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual sum
#                         df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
#                         if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
#                             df_ann_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & no intercropping']
#                         else:
#                             df_ann_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & intercropping']
#                         df_ann_avg.loc[:, "location"] = _dict_location[location]
#                         ll_df.append(df_ann_avg)
#                     elif var_sim == "C_q_ss":
#                         ds = dict_fluxes_states[location][crop_rotation_scenario]
#                         sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#                         sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual mean
#                         df_avg = df.sum(axis=0).to_frame()
#                         if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
#                             df_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & no intercropping']
#                         else:
#                             df_avg.loc[:, "fertilization_intensity"] = _dict_intercropping_effects[f'{fertilization_intensity} Nfert & intercropping']
#                         df_avg.loc[:, "location"] = _dict_location[location]
#                         ll_df.append(df_avg)
#         df_ann_avg = pd.concat(ll_df)
#         df_ann_avg_long = df_ann_avg.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
#         if var_sim == "C_q_ss":
#             axes.axhline(y=50, color='r', linestyle='--', lw=2)   
#             axes.axhline(y=37.5, color='orange', linestyle='--', lw=2) 
#         sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd", hue_norm=(0, 4))
#         axes.legend().set_visible(False)
#         axes.set_xlabel("")
#         axes.set_ylabel(_lab_unit_annual[var_sim])
#         axes.set_ylim(0,)
#         axes.set_xlabel("Location")
#         plt.xticks(rotation=45)
#         fig.tight_layout()
#         file = base_path_figs / f"boxplot_annual_average_{var_sim}_{_crop_rotation_scenario}_intercropping_effects.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

# # plot difference between average annual sum considering crop rotations with and without mustard
# vars_sim = ["M_q_ss", "C_q_ss"]
# for var_sim in vars_sim:
#     for i, _crop_rotation_scenario in enumerate(crop_rotation_scenarios_without_mustard):
#         crop_rotation_scenarios1 = [_crop_rotation_scenario, crop_rotation_scenarios_with_mustard[i]]
#         fig, axes = plt.subplots(1, 1, figsize=(3, 2), sharex=True, sharey=True)
#         ll_df = []
#         ll_df_mustard = []
#         for location in locations:
#             for crop_rotation_scenario in crop_rotation_scenarios1:
#                 for fertilization_intensity in fertilization_intensities:
#                     if var_sim == "M_q_ss":
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01 # convert from mg/m2 to kg/ha
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual sum
#                         df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
#                         df_ann_avg.loc[:, "location"] = _dict_location[location]
#                         if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
#                             df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                             ll_df.append(df_ann_avg)
#                         else:
#                             df_ann_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                             ll_df_mustard.append(df_ann_avg)
#                     elif var_sim == "C_q_ss":
#                         ds = dict_fluxes_states[location][crop_rotation_scenario]
#                         sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#                         ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                         sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#                         sim_vals = onp.where(sim_vals1 > 0.01, (sim_vals2/sim_vals1) * (sim_vals1/onp.sum(sim_vals1, axis=-1)[:, onp.newaxis]), onp.nan)
#                         cond1 = (df_params["CLUST_flag"] == 1)
#                         df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#                         # calculate annual mean
#                         df_avg = df.sum(axis=0).to_frame()
#                         df_avg.loc[:, "location"] = _dict_location[location]
#                         if crop_rotation_scenario in crop_rotation_scenarios_without_mustard:
#                             df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                             ll_df.append(df_avg)
#                         else:
#                             df_avg.loc[:, "fertilization_intensity"] = fertilization_intensity
#                             ll_df_mustard.append(df_avg)
#         df_ann_avg = pd.concat(ll_df)
#         df_ann_avg_mustard = pd.concat(ll_df_mustard)
#         df_ann_avg_diff = df_ann_avg.copy()        
#         df_ann_avg_diff.iloc[:, 0] = -(df_ann_avg.iloc[:, 0] - df_ann_avg_mustard.iloc[:, 0])   
#         df_ann_avg_long = df_ann_avg_diff.melt(id_vars=["location", "fertilization_intensity"], value_name="vals").loc[:, ["location", "fertilization_intensity", "vals"]]        
#         sns.boxplot(data=df_ann_avg_long, x="location", y="vals", hue="fertilization_intensity", ax=axes, whis=(1, 99), showfliers=False, palette="PuRd")
#         axes.legend().set_visible(False)
#         axes.set_xlabel("")
#         axes.set_ylabel(_lab_unit_annual[var_sim])
#         if df_ann_avg_diff.iloc[:, 0].max() < 0:
#             axes.set_ylim(0, )
#         axes.set_xlabel("Location")
#         plt.xticks(rotation=45)
#         fig.tight_layout()
#         file = base_path_figs / f"boxplot_difference_annual_average_{var_sim}_{_crop_rotation_scenario}_intercropping_effects.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

# legend_elements_mustard = [Patch(facecolor='#e5f5f9', edgecolor='k',
#                            label='no intercropping'),
#                            Patch(facecolor='#2ca25f', edgecolor='k',
#                                  label='intercropping'),
#                            Patch(facecolor='#f1eef6', edgecolor='k',
#                                  label='low'),
#                            Patch(facecolor='#d4b9da', edgecolor='k',
#                                  label='low'),
#                            Patch(facecolor='#c994c7', edgecolor='k',
#                                  label='medium'),
#                            Patch(facecolor='#df65b0', edgecolor='k',
#                                  label='medium'),
#                            Patch(facecolor='#dd1c77', edgecolor='k',
#                                  label='high'),
#                            Patch(facecolor='#980043', edgecolor='k',
#                                  label='high')]

# legend_elements= [Patch(facecolor='#e7e1ef', edgecolor='k',
#                           label='low'),
#                   Patch(facecolor='#c994c7', edgecolor='k',
#                           label='medium'),
#                   Patch(facecolor='#dd1c77', edgecolor='k',
#                         label='high')]

# _dict_crop_id = {"winter-wheat": 115,
#                  "clover": 425,
#                  "silage-corn": 411,
#                  "summer-wheat": 116,
#                  "sugar-beet": 603,
#                  "winter-rape": 311,
#                  "soybean": 330,
#                  "grain-corn": 171,
#                  "winter-barley": 131,
#                 }

# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/data_freiburg_nitrate_leaching") / "nitrate_leaching.csv"
# df = pd.read_csv(file, sep=";")

# vars_sim = ["q_hof", "q_ss", "C_q_ss", "M_q_ss"]
# fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=False)
# for i, var_sim in enumerate(vars_sim):
#     if var_sim in ["M_q_ss", "C_q_ss"]:
#         ll_df = []
#         for fertilization_intensity in fertilization_intensities:
#             df1 = pd.DataFrame()
#             cond = (df["CID"] > 0)
#             df1.loc[:, 'vals'] = df.loc[cond, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}']
#             df1.loc[:, 'fertilization_intensity'] = fertilization_intensity
#             df1.loc[:, 'FFID'] = df.loc[cond, 'FFID']
#             df1.loc[:, 'CID'] = df.loc[cond, 'CID']
#             ll_df.append(df1)
#         data = pd.concat(ll_df, ignore_index=True)
#         if var_sim == "C_q_ss":
#             axes[i].axhline(y=50, color='red', linestyle='--', lw=2)
#             axes[i].axhline(y=37.5, color='orange', linestyle='--', lw=2)
#         sns.boxplot(data=data, x="CID", y="vals", hue="fertilization_intensity", ax=axes[i], whis=(1, 99), showfliers=False, palette="PuRd", order=[115, 131, 311, 411, 171, 116, 603])
#         axes[i].legend().set_visible(False)
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel(_lab_unit_annual[var_sim])
#         axes[i].set_ylim(0, )
#     else:
#         data = pd.DataFrame()
#         cond = (df["CID"] > 0)
#         data.loc[:, 'vals'] = df.loc[cond, f'{_dict_var_names[var_sim]}']
#         data.loc[:, 'FFID'] = df.loc[cond, 'FFID']
#         data.loc[:, 'CID'] = df.loc[cond, 'CID']
#         sns.boxplot(data=data, x="CID", y="vals", ax=axes[i], whis=(1, 99), showfliers=False, order=[115, 131, 311, 411, 171, 116, 603])
#         axes[i].legend().set_visible(False)
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel(_lab_unit_annual[var_sim])
#         axes[i].set_ylim(0, )
# axes[-1].set_xlabel("")
# xticklabels1 = axes[-1].get_xticklabels()
# xticklabels = [f"{_dict_crop_id_rev[int(x.get_text())]}" for x in xticklabels1]
# axes[-1].set_xticklabels(xticklabels)
# plt.xticks(rotation=45)
# fig.legend(handles=legend_elements, loc='upper center', ncol=4, frameon=False, title="Fertilization intensity")
# plt.xticks(rotation=45)
# fig.tight_layout()
# fig.subplots_adjust(top=0.9)
# file = base_path_figs / "leaching_for_crop_types.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)


# vars_sim = ["q_hof", "q_ss", "C_q_ss", "M_q_ss"]
# fig, axes = plt.subplots(4, 1, figsize=(6, 6), sharex=True, sharey=False)
# for i, var_sim in enumerate(vars_sim):
#     if var_sim in ["M_q_ss", "C_q_ss"]:
#         ll_df = []
#         for fertilization_intensity in fertilization_intensities:
#             df1 = pd.DataFrame()
#             cond = (df["CID"] > 0)
#             df1.loc[:, 'vals'] = df.loc[cond, f'{_dict_var_names[var_sim]}_N{_dict_fert[fertilization_intensity]}']
#             df1.loc[:, 'fertilization_intensity'] = fertilization_intensity
#             df1.loc[:, 'FFID'] = df.loc[cond, 'FFID']
#             df1.loc[:, 'CID'] = df.loc[cond, 'CID']
#             cond1 = onp.isin(df1["FFID"].values, _ffids_mustard).flatten()
#             df2 = df1.loc[cond1, :].copy()
#             cond2 = onp.isin(df2["FFID"].values, _ffids_with_mustard).flatten()
#             df2.loc[:, 'mustard'] = '_0'
#             df2.loc[cond2, 'mustard'] = '_1'
#             df2.loc[:, 'fertilization_intensity_mustard'] = df2.loc[:, 'fertilization_intensity'] + df2.loc[:, 'mustard']
#             ll_df.append(df2)
#         data = pd.concat(ll_df, ignore_index=True)
#         if var_sim == "C_q_ss":
#             axes[i].axhline(y=50, color='red', linestyle='--', lw=2)
#             axes[i].axhline(y=37.5, color='orange', linestyle='--', lw=2)
#         sns.boxplot(data=data, x="CID", y="vals", hue="fertilization_intensity_mustard", ax=axes[i], whis=(1, 99), showfliers=False, palette="PuRd", order=[115, 131, 311, 411, 171, 116, 603])
#         axes[i].legend().set_visible(False)
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel(_lab_unit_annual[var_sim])
#         axes[i].set_ylim(0, )
#     else:
#         data = pd.DataFrame()
#         cond = (df["CID"] > 0)
#         df1.loc[:, 'vals'] = df.loc[cond, f'{_dict_var_names[var_sim]}']
#         df1.loc[:, 'FFID'] = df.loc[cond, 'FFID']
#         df1.loc[:, 'CID'] = df.loc[cond, 'CID']
#         cond1 = onp.isin(df1["FFID"].values, _ffids_mustard).flatten()
#         data = df1.loc[cond1, :].copy()
#         cond2 = onp.isin(data["FFID"].values, _ffids_with_mustard).flatten()
#         data.loc[:, 'mustard'] = 'no intercropping'
#         data.loc[cond2, 'mustard'] = 'intercropping'
#         sns.boxplot(data=data, x="CID", y="vals", hue="mustard", ax=axes[i], whis=(1, 99), showfliers=False, palette="BuGn", order=[115, 131, 311, 411, 171, 116, 603])
#         axes[i].legend().set_visible(False)
#         axes[i].set_xlabel("")
#         axes[i].set_ylabel(_lab_unit_annual[var_sim])
#         axes[i].set_ylim(0, )
# axes[-1].set_xlabel("")
# xticklabels1 = axes[-1].get_xticklabels()
# xticklabels = [f"{_dict_crop_id_rev[int(x.get_text())]}" for x in xticklabels1]
# axes[-1].set_xticklabels(xticklabels)
# fig.legend(handles=legend_elements_mustard, loc='upper center', ncol=4, frameon=False, title="                           Fertilization intensity")
# plt.xticks(rotation=45)
# fig.tight_layout()
# fig.subplots_adjust(top=0.88)
# file = base_path_figs / "leaching_for_crop_types_mustard.png"
# fig.savefig(file, dpi=300)
# plt.close(fig)
