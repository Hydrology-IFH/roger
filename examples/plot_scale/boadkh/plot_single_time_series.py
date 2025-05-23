import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as np
import geopandas as gpd
import numpy as onp
from matplotlib.patches import Patch
import matplotlib as mpl
import seaborn as sns
import pickle

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

# mpl.rcParams["font.size"] = 8
# mpl.rcParams["axes.titlesize"] = 8
# mpl.rcParams["axes.labelsize"] = 9
# mpl.rcParams["xtick.labelsize"] = 8
# mpl.rcParams["ytick.labelsize"] = 8
# mpl.rcParams["legend.fontsize"] = 8
# mpl.rcParams["legend.title_fontsize"] = 9

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")
# sns.plotting_context(
#     "paper",
#     font_scale=1,
#     rc={
#         "font.size": 8.0,
#         "axes.labelsize": 9.0,
#         "axes.titlesize": 8.0,
#         "xtick.labelsize": 8.0,
#         "ytick.labelsize": 8.0,
#         "legend.fontsize": 8.0,
#         "legend.title_fontsize": 9.0,
#     },
# )

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
    "M_q_ss": "PERC-$NO_3$\n [kg N/day/ha]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]",
    "q_ss": "PERC\n [mm/day]"
}

_lab_unit_annual = {
    "M_q_ss": "PERC-$NO_3$-N\n [kg N/year/ha]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]",
    "q_ss": "PERC\n [mm/year]",
    "q_hof": "$Q_{sur}$\n [mm/year]",
}

_lab_unit_total = {
    "M_q_ss": "PERC-$NO_3$-N\n [kg N]",
    "C_q_ss": "PERC-$NO_3$\n [mg/l]"
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)

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

# plot daily values
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0, x=36).values[1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0, x=36).values[1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals)
            df = df.loc["2014-01-01":"2022-12-31"]
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            ax.plot(df.index, df.values, color="black", linewidth=1)
            ax.set_xlabel("Time [Year]")
            ax.set_ylabel(_lab_unit_daily["C_q_ss"])
            ax.set_xlim(df.index[0], df.index[-1])
            ax.set_ylim(0, )
            plt.xticks(rotation=33)
            fig.tight_layout()
            file = base_path_figs / f"trace_C_q_ss_{location}_{crop_rotation_scenario}_{fertilization_intensity}_.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        ds = dict_fluxes_states[location][crop_rotation_scenario]
        sim_vals = ds["ground_cover"].isel(y=0, x=36).values[1:]
        df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
        df = df.loc["2014-01-01":"2022-12-31"]
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(df.index, df.values, color="black", linewidth=1)
        ax.set_xlabel("Time [Year]")
        ax.set_ylabel("canopy cover [-]")
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, )
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"trace_canopy_cover_{location}_{crop_rotation_scenario}_.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        ds = dict_fluxes_states[location][crop_rotation_scenario]
        sim_vals = ds["z_root"].isel(y=0, x=36).values[1:] / 10
        df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
        df = df.loc["2014-01-01":"2022-12-31"]
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(df.index, df.values, color="black", linewidth=1)
        ax.set_xlabel("Time [Year]")
        ax.set_ylabel("root depth [cm]")
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, )
        ax.invert_yaxis()
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"trace_root_depth_{location}_{crop_rotation_scenario}_.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)


# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for fertilization_intensity in fertilization_intensities:
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             sim_vals = ds["C_q_ss"].isel(y=0, x=36).values[1:] * 4.427  # convert nitrate-nitrogen to nitrate
#             df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
#             df = df.loc["2014-01-01":"2023-12-31"]
#             fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#             ax.plot(df.index, df.values, color="black", linewidth=1)
#             ax.set_xlabel("Time [Year]")
#             ax.set_ylabel(_lab_unit_daily["C_q_ss"])
#             ax.set_xlim(df.index[0], df.index[-1])
#             ax.set_ylim(0, )
#             plt.xticks(rotation=33)
#             fig.tight_layout()
#             file = base_path_figs / f"trace_C_q_ss_{location}_{crop_rotation_scenario}__.png"
#             fig.savefig(file, dpi=300)
#             plt.close(fig)

# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for fertilization_intensity in fertilization_intensities:
#             fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             sim_vals = ds["C_q_ss"].isel(y=0, x=36).values[1:] * 4.427  # convert nitrate-nitrogen to nitrate
#             # sim_vals = onp.where(sim_vals > 200, onp.nan, sim_vals)                  
#             df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
#             df = df.loc["2016-01-01":"2017-12-31"]
#             ax[1].plot(df.index, df.values, color="#c994c7", linewidth=1)
#             ax[1].set_xlabel("Time [Year]")
#             ax[1].set_ylabel(_lab_unit_daily[var_sim])
#             ax[1].set_xlim(df.index[0], df.index[-1])
#             ax[1].set_ylim(0, )
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals = ds["q_ss"].isel(y=0, x=36).values[1:]               
#             df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
#             df = df.loc["2016-01-01":"2017-12-31"]
#             ax[0].plot(df.index, df.values, color="#9ecae1", linewidth=1)
#             ax[0].set_ylabel(_lab_unit_daily["q_ss"])
#             ax[0].set_xlim(df.index[0], df.index[-1])
#             ax[0].set_ylim(0, )
#             plt.xticks(rotation=33)
#             fig.tight_layout()
#             file = base_path_figs / f"trace_perc_nitrate_{location}_{crop_rotation_scenario}_2016_2017_.png"
#             fig.savefig(file, dpi=300)
#             plt.close(fig)

for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0, x=36).values[1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0, x=36).values[1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.1, sim_vals2/sim_vals1, onp.nan)
            # sim_vals = onp.where(sim_vals > 200, onp.nan, sim_vals)                  
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals)
            df = df.loc["2016-01-01":"2017-12-31"]
            ax[1].plot(df.index, df.values, color="#c994c7", linewidth=1)
            ax[1].set_xlabel("Time [Year]")
            ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
            ax[1].set_xlim(df.index[0], df.index[-1])
            ax[1].set_ylim(0, )
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds["q_ss"].isel(y=0, x=36).values[1:]               
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
            df = df.loc["2016-01-01":"2017-12-31"]
            ax[0].plot(df.index, df.values, color="#9ecae1", linewidth=1)
            ax[0].set_ylabel(_lab_unit_daily["q_ss"])
            ax[0].set_xlim(df.index[0], df.index[-1])
            ax[0].set_ylim(0, )
            plt.xticks(rotation=33)
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_nitrate_{location}_{crop_rotation_scenario}_{fertilization_intensity}_2016_2017.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0, x=36).values[1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0, x=36).values[1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.1, sim_vals2/sim_vals1, onp.nan)
            # sim_vals = onp.where(sim_vals > 200, onp.nan, sim_vals)                  
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals)
            df = df.loc["2017-01-01":"2018-12-31"]
            ax[1].plot(df.index, df.values, color="#c994c7", linewidth=1)
            ax[1].set_xlabel("Time [Year]")
            ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
            ax[1].set_xlim(df.index[0], df.index[-1])
            ax[1].set_ylim(0, )
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds["q_ss"].isel(y=0, x=36).values[1:]               
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
            df = df.loc["2017-01-01":"2018-12-31"]
            ax[0].plot(df.index, df.values, color="#9ecae1", linewidth=1)
            ax[0].set_ylabel(_lab_unit_daily["q_ss"])
            ax[0].set_xlim(df.index[0], df.index[-1])
            ax[0].set_ylim(0, )
            plt.xticks(rotation=33)
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_nitrate_{location}_{crop_rotation_scenario}_{fertilization_intensity}_2017_2018.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)


for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0, x=36).values[1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0, x=36).values[1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.1, sim_vals2/sim_vals1, onp.nan)
            # sim_vals = onp.where(sim_vals > 200, onp.nan, sim_vals)                  
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals)
            df = df.loc["2019-01-01":"2020-12-31"]
            ax[1].plot(df.index, df.values, color="#c994c7", linewidth=1)
            ax[1].set_xlabel("Time [Year]")
            ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
            ax[1].set_xlim(df.index[0], df.index[-1])
            ax[1].set_ylim(0, )
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds["q_ss"].isel(y=0, x=36).values[1:]               
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals)
            df = df.loc["2019-01-01":"2020-12-31"]
            ax[0].plot(df.index, df.values, color="#9ecae1", linewidth=1)
            ax[0].set_ylabel(_lab_unit_daily["q_ss"])
            ax[0].set_xlim(df.index[0], df.index[-1])
            ax[0].set_ylim(0, )
            plt.xticks(rotation=33)
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_nitrate_{location}_{crop_rotation_scenario}_{fertilization_intensity}_2019_2020.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)




# colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for i, fertilization_intensity in enumerate(fertilization_intensities):
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#             sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
#             cond1 = (df_params["CLUST_flag"] == 1)
#             df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#             df = df.loc["2014-01-01":"2022-12-31", :]
#             vals = df.values.T
#             idx = np.argmax(np.nanmax(vals, axis=1))

#             fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
#             color = "blue"
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals = ds["q_ss"].isel(y=0).values
#             cond1 = (df_params["CLUST_flag"] == 1)
#             df = pd.DataFrame(index=ds["Time"].values, data=sim_vals.T).loc[:, cond1]
#             df = df.loc["2014-01-01":"2022-12-31", :]
#             vals = df.values.T[idx, :]
#             ax[0].plot(df.index, vals, color=color, label="Median", linewidth=1)
#             ax[0].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
#             ax[0].set_ylabel('PERC [mm/day]')
#             ax[0].set_xlim(df.index[0], df.index[-1])
#             ax[0].set_ylim(0, 1)

#             color = colors[i]
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#             sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
#             cond1 = (df_params["CLUST_flag"] == 1)
#             df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#             df = df.loc["2014-01-01":"2022-12-31", :]
#             vals = df.values.T[idx, :]
#             ax[1].plot(df.index, vals, color=color, label="Median", linewidth=1)
#             ax[1].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
#             ax[1].set_xlabel("Time [Year]")
#             ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
#             ax[1].set_xlim(df.index[0], df.index[-1])
#             ax[1].set_ylim(0, 1000)
#             fig.tight_layout()
#             file = base_path_figs / f"trace_perc_flux_nitrate_conc_{location}_{crop_rotation_scenario}_{fertilization_intensity}_{idx}.png"
#             fig.savefig(file, dpi=300)
#             plt.close(fig)
