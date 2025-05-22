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
    ncols = values.shape[1]
    nrows = values.shape[0]
    ll = []
    for i in range(ncols):
        reps = int(np.round(area_share[i], 0))
        if reps > 0:
            array = np.zeros((nrows, reps))
            for j in range(reps):
                array[:, j] = values[:, i]
            ll.append(array)
    return np.concatenate(ll, axis=1)


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

locations = ["freiburg"]

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

crop_rotation_scenarios = ["winter-wheat_winter-rape"]

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
    "C_q_ss": "PERC-$NO_3$\n [mg/l]"
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
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()
df_areas = pd.read_csv(base_path / "output" / "areas.csv", sep=";")

# # load linkage between BK50 and cropland clusters
# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
# df_link_bk50_cluster_cropland = pd.read_hdf(file)

# # load BK50 shapefile for Freiburg
# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "BK50_acker_freiburg.gpkg"
# gdf_bk50 = gpd.read_file(file, include_fields=["SHP_ID", "area"]).loc[:, ["SHP_ID", "area"]]

# # get unique cluster ids for cropland
# cond = onp.isin(df_link_bk50_cluster_cropland.index.values, gdf_bk50["SHP_ID"].values)
# clust_ids = onp.unique(df_link_bk50_cluster_cropland.loc[cond, "CLUST_ID"].values).astype(str)

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

# file = base_path_output / "dict_fluxes_states.pickle"
# with open(file, 'wb') as handle:
#     pickle.dump(dict_fluxes_states, handle, protocol=pickle.HIGHEST_PROTOCOL)

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

# file = base_path_output / "dict_nitrate.pickle"
# with open(file, 'wb') as handle:
#     pickle.dump(dict_nitrate, handle, protocol=pickle.HIGHEST_PROTOCOL)

# plot daily values
colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for i, fertilization_intensity in enumerate(fertilization_intensities):
            fig, ax = plt.subplots(1, 1, figsize=(6, 2))
            color = colors[i]
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals3 = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            sim_vals = onp.where((sim_vals3 > 250), onp.nan, sim_vals3)
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
            df = df.loc["2014-01-01":"2022-12-31", :]
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data = repeat_by_areashare(df.values, data["area_share"].values)
            df1 = pd.DataFrame(data=data, index=df.index)
            vals = df1.values.T
            median_vals = onp.nanmedian(vals, axis=0)
            min_vals = onp.nanmin(vals, axis=0)
            max_vals = onp.nanmax(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax.fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor=None,
                facecolor=color,
                alpha=0.33,
                label="Min-Max interval",
            )
            # ax.fill_between(
            #     df.index,
            #     p5_vals,
            #     p95_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=0.66,
            #     label="95% interval",
            # )
            # ax.fill_between(
            #     df.index,
            #     p25_vals,
            #     p75_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=1,
            #     label="75% interval",
            # )
            ax.plot(df.index, median_vals, color=color, label="Median", linewidth=1)
            ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
            ax.set_xlabel("Time [Year]")
            ax.set_ylabel(_lab_unit_daily["C_q_ss"])
            ax.set_xlim(df.index[0], df.index[-1])
            ax.set_ylim(0, )
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_nitrate_conc_{location}_{crop_rotation_scenario}_{fertilization_intensity}Nfert.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)


colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for i, fertilization_intensity in enumerate(fertilization_intensities):
            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            color = "blue"
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds["q_ss"].isel(y=0).values
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values, data=sim_vals.T).loc[:, cond1]
            df = df.loc["2014-01-01":"2022-12-31", :]
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data = repeat_by_areashare(df.values, data["area_share"].values)
            df1 = pd.DataFrame(data=data, index=df.index)
            vals = df1.values.T
            median_vals = onp.nanmedian(vals, axis=0)
            min_vals = onp.nanmin(vals, axis=0)
            max_vals = onp.nanmax(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax[0].fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor=color,
                facecolor=color,
                alpha=0.33,
                label="Min-Max interval",
            )
            # ax[0].fill_between(
            #     df.index,
            #     p5_vals,
            #     p95_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=0.66,
            #     label="95% interval",
            # )
            # ax[0].fill_between(
            #     df.index,
            #     p25_vals,
            #     p75_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=1,
            #     label="75% interval",
            # )
            ax[0].plot(df.index, median_vals, color=color, label="Median", linewidth=1)
            ax[0].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
            ax[0].set_ylabel('PERC [mm/day]')
            ax[0].set_xlim(df.index[0], df.index[-1])
            ax[0].set_ylim(0, )

            color = colors[i]
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
            df = df.loc["2014-01-01":"2022-12-31", :]
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data = repeat_by_areashare(df.values, data["area_share"].values)
            df1 = pd.DataFrame(data=data, index=df.index)
            vals = df1.values.T
            median_vals = onp.nanmedian(vals, axis=0)
            min_vals = onp.nanmin(vals, axis=0)
            max_vals = onp.nanmax(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            ax[1].fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor=None,
                facecolor=color,
                alpha=0.33,
                label="Min-Max interval",
            )
            # ax[1].fill_between(
            #     df.index,
            #     p5_vals,
            #     p95_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=0.66,
            #     label="95% interval",
            # )
            # ax[1].fill_between(
            #     df.index,
            #     p25_vals,
            #     p75_vals,
            #     edgecolor=color,
            #     facecolor=color,
            #     alpha=1,
            #     label="75% interval",
            # )
            ax[1].plot(df.index, median_vals, color=color, label="Median", linewidth=1)
            ax[1].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
            ax[1].set_xlabel("Time [Year]")
            ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
            ax[1].set_xlim(df.index[0], df.index[-1])
            ax[1].set_ylim(0, 200)
            ax[i].axhline(y=50, color='r', linestyle='--', lw=1) 
            ax[i].axhline(y=37.5, color='orange', linestyle='--', lw=1)   
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_flux_nitrate_conc_{location}_{crop_rotation_scenario}_{fertilization_intensity}Nfert.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for i, fertilization_intensity in enumerate(fertilization_intensities):
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
            df = df.loc["2014-01-01":"2022-12-31", :]
            maxcol = np.argmax(np.nanmax(df.values, axis=0))

            fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
            color = "blue"
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds["q_ss"].isel(y=0).values
            df = pd.DataFrame(index=ds["Time"].values, data=sim_vals.T)
            df = df.loc["2014-01-01":"2022-12-31", :]
            df = df.iloc[:, maxcol].to_frame()
            ax[0].plot(df.index, df.values, color=color, linewidth=1)
            ax[0].set_ylabel('PERC [mm/day]')
            ax[0].set_xlim(df.index[0], df.index[-1])
            ax[0].set_ylim(0, )

            color = colors[i]
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            df_perc = pd.DataFrame(index=ds["Time"].values, data=sim_vals.T)
            df_perc = df_perc.loc["2014-01-01":"2022-12-31", :]
            df_perc = df_perc.iloc[:, maxcol].to_frame()
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals3 = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            sim_vals = onp.where((sim_vals3 > 250), onp.nan, sim_vals3)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T)
            df = df.loc["2014-01-01":"2022-12-31", :]
            df = df.iloc[:, maxcol].to_frame()
            avg = onp.nanmean(df.values)
            avg_weighted = onp.nansum(onp.where(df_perc.values > 0.01, df.values * (df_perc.values/onp.sum(df_perc.values)), 0))
            ax[1].plot(df.index, df.values, color=color, linewidth=1)
            ax[1].axhline(avg, color="black", linestyle="--")
            ax[1].axhline(avg_weighted, color=color, linestyle="-")
            ax[1].set_xlabel("Time [Year]")
            ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
            ax[1].set_xlim(df.index[0], df.index[-1])
            ax[1].set_ylim(0, )
            fig.tight_layout()
            file = base_path_figs / f"trace_perc_flux_nitrate_conc_{location}_{crop_rotation_scenario}_{fertilization_intensity}Nfert_{maxcol}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
for crop_rotation_scenario in crop_rotation_scenarios:
    for i, fertilization_intensity in enumerate(fertilization_intensities):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ll_dfs = []
        color = colors[i]
        for location in locations:
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
            df = df.loc["2014-01-01":"2022-12-31", :]
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data = repeat_by_areashare(df.values, data["area_share"].values)
            df1 = pd.DataFrame(data=data, index=df.index)
            ll_dfs.append(df1)
        df = pd.concat(ll_dfs, axis=1)
        vals = df.values.T
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        median_vals = onp.nanmedian(vals, axis=0)
        min_vals = onp.nanmin(vals, axis=0)
        max_vals = onp.nanmax(vals, axis=0)
        p5_vals = onp.nanquantile(vals, 0.05, axis=0)
        p25_vals = onp.nanquantile(vals, 0.25, axis=0)
        p75_vals = onp.nanquantile(vals, 0.75, axis=0)
        p95_vals = onp.nanquantile(vals, 0.95, axis=0)
        ax.fill_between(
            df.index,
            min_vals,
            max_vals,
            edgecolor=None,
            facecolor=color,
            alpha=0.33,
            label="Min-Max interval",
        )
        # ax.fill_between(
        #     df.index,
        #     p5_vals,
        #     p95_vals,
        #     edgecolor=color,
        #     facecolor=color,
        #     alpha=0.66,
        #     label="95% interval",
        # )
        # ax.fill_between(
        #     df.index,
        #     p25_vals,
        #     p75_vals,
        #     edgecolor=color,
        #     facecolor=color,
        #     alpha=1,
        #     label="75% interval",
        # )
        ax.plot(df.index, median_vals, color=color, label="Median", linewidth=1)
        ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.7, 1.3))
        ax.set_xlabel("Time [Year]")
        ax.set_ylabel(_lab_unit_daily["C_q_ss"])
        ax.set_xlim(df.index[0], df.index[-1])
        ax.set_ylim(0, )
        fig.tight_layout()
        file = base_path_figs / f"trace_perc_nitrate_conc_{crop_rotation_scenario}_{fertilization_intensity}Nfert.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

# colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))
# for crop_rotation_scenario in crop_rotation_scenarios:
#     color = "blue"
#     fig, ax = plt.subplots(2, 1, figsize=(6, 4), sharex=True)
#     ll_dfs_flux = []
#     for location in locations:
#         ds = dict_fluxes_states[location][crop_rotation_scenario]
#         sim_vals = ds["q_ss"].isel(y=0).values[:, 1:]
#         cond1 = (df_params["CLUST_flag"] == 1)
#         df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#         df = df.loc["2014-01-01":"2022-12-31", :]
#         cond = (df_areas["location"] == location)
#         df_area_share = df_areas.loc[cond, :]
#         data = df_params.loc[cond1, "CLUST_ID"].to_frame()
#         data.loc[:, "area_share"] = 0.0
#         for clust_id in clust_ids:
#             cond1 = (df_area_share["clust_id"] == clust_id)
#             if cond1.any():
#                 cond2 = (data["CLUST_ID"] == clust_id)
#                 data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
#         data = repeat_by_areashare(df.values, data["area_share"].values)
#         df1 = pd.DataFrame(data=data, index=df.index)
#         ll_dfs_flux.append(df1)
#     df = pd.concat(ll_dfs_flux, axis=1)
#     vals = df.values.T
#     median_vals = onp.nanmedian(vals, axis=0)
#     min_vals = onp.nanmin(vals, axis=0)
#     max_vals = onp.nanmax(vals, axis=0)
#     p5_vals = onp.nanquantile(vals, 0.05, axis=0)
#     p25_vals = onp.nanquantile(vals, 0.25, axis=0)
#     p75_vals = onp.nanquantile(vals, 0.75, axis=0)
#     p95_vals = onp.nanquantile(vals, 0.95, axis=0)
#     ax[0].fill_between(
#         df.index,
#         min_vals,
#         max_vals,
#         edgecolor=None,
#         facecolor=color,
#         alpha=0.33,
#         label="Min-Max interval",
#     )
#     # ax[0].fill_between(
#     #     df.index,
#     #     p5_vals,
#     #     p95_vals,
#     #     edgecolor=color,
#     #     facecolor=color,
#     #     alpha=0.66,
#     #     label="95% interval",
#     # )
#     # ax[0].fill_between(
#     #     df.index,
#     #     p25_vals,
#     #     p75_vals,
#     #     edgecolor=color,
#     #     facecolor=color,
#     #     alpha=1,
#     #     label="75% interval",
#     # )
#     ax[0].plot(df.index, median_vals, color=color, label="Median", linewidth=1)
#     ax[0].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
#     ax[0].set_ylabel('PERC [mm/day]')
#     ax[0].set_xlim(df.index[0], df.index[-1])
#     ax[0].set_ylim(0, )

#     ll_dfs_conc = []
#     for i, fertilization_intensity in enumerate(fertilization_intensities):
#         for location in locations:
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             ds = dict_fluxes_states[location][crop_rotation_scenario]
#             sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
#             sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
#             cond1 = (df_params["CLUST_flag"] == 1)
#             df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
#             df = df.loc["2014-01-01":"2022-12-31", :]
#             cond = (df_areas["location"] == location)
#             df_area_share = df_areas.loc[cond, :]
#             data = df_params.loc[cond1, "CLUST_ID"].to_frame()
#             data.loc[:, "area_share"] = 0.0
#             for clust_id in clust_ids:
#                 cond1 = (df_area_share["clust_id"] == clust_id)
#                 if cond1.any():
#                     cond2 = (data["CLUST_ID"] == clust_id)
#                     data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
#             data = repeat_by_areashare(df.values, data["area_share"].values)
#             df1 = pd.DataFrame(data=data, index=df.index)
#             ll_dfs_conc.append(df1)
#         df = pd.concat(ll_dfs_conc, axis=1)
#         vals = df.values.T
#         median_vals = onp.nanmedian(vals, axis=0)
#         min_vals = onp.nanmin(vals, axis=0)
#         max_vals = onp.nanmax(vals, axis=0)
#         p5_vals = onp.nanquantile(vals, 0.05, axis=0)
#         p25_vals = onp.nanquantile(vals, 0.25, axis=0)
#         p75_vals = onp.nanquantile(vals, 0.75, axis=0)
#         p95_vals = onp.nanquantile(vals, 0.95, axis=0)
#         ax[1].fill_between(
#             df.index,
#             min_vals,
#             max_vals,
#             edgecolor=None,
#             facecolor=color,
#             alpha=0.33,
#             label="Min-Max interval",
#         )
#         # ax[1].fill_between(
#         #     df.index,
#         #     p5_vals,
#         #     p95_vals,
#         #     edgecolor=color,
#         #     facecolor=color,
#         #     alpha=0.66,
#         #     label="95% interval",
#         # )
#         # ax[1].fill_between(
#         #     df.index,
#         #     p25_vals,
#         #     p75_vals,
#         #     edgecolor=color,
#         #     facecolor=color,
#         #     alpha=1,
#         #     label="75% interval",
#         # )
#         ax[1].plot(df.index, median_vals, color=color, label="Median", linewidth=1)
#         ax[1].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
#         ax[1].set_xlabel("Time [Year]")
#         ax[1].set_ylabel(_lab_unit_daily["C_q_ss"])
#         ax[1].set_xlim(df.index[0], df.index[-1])
#         ax[1].set_ylim(0, 300)
#         fig.tight_layout()
#         file = base_path_figs / f"trace_perc_flux_nitrate_conc_{location}_{crop_rotation_scenario}.png"
#         fig.savefig(file, dpi=300)
#         plt.close(fig)

colors = sns.color_palette("RdPu", n_colors=len(fertilization_intensities))[::-1]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        for i, fertilization_intensity in enumerate(fertilization_intensities[::-1]):
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals1 = ds["q_ss"].isel(y=0).values[:, 1:]
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            sim_vals2 = ds["M_q_ss"].isel(y=0).values[:, 1:-1] * 4.427  # convert nitrate-nitrogen to nitrate
            sim_vals = onp.where(sim_vals1 > 0.01, sim_vals2/sim_vals1, onp.nan)
            cond1 = (df_params["CLUST_flag"] == 1)
            df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
            df = df.loc["2014-01-01":"2022-12-31", :]
            cond = (df_areas["location"] == location)
            df_area_share = df_areas.loc[cond, :]
            data = df_params.loc[cond1, "CLUST_ID"].to_frame()
            data.loc[:, "area_share"] = 0.0
            for clust_id in clust_ids:
                cond1 = (df_area_share["clust_id"] == clust_id)
                if cond1.any():
                    cond2 = (data["CLUST_ID"] == clust_id)
                    data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
            data = repeat_by_areashare(df.values, data["area_share"].values)
            df1 = pd.DataFrame(data=data, index=df.index)
            vals = df1.values.T
            median_vals = onp.nanmedian(vals, axis=0)
            min_vals = onp.nanmin(vals, axis=0)
            max_vals = onp.nanmax(vals, axis=0)
            p5_vals = onp.nanquantile(vals, 0.05, axis=0)
            p25_vals = onp.nanquantile(vals, 0.25, axis=0)
            p75_vals = onp.nanquantile(vals, 0.75, axis=0)
            p95_vals = onp.nanquantile(vals, 0.95, axis=0)
            axs[i].fill_between(
                df.index,
                min_vals,
                max_vals,
                edgecolor=None,
                facecolor=colors[i],
                alpha=0.33,
                label="Min-Max interval",
            )
            # ax.fill_between(
            #     df.index,
            #     p5_vals,
            #     p95_vals,
            #     edgecolor=colors[i],
            #     facecolor=colors[i],
            #     alpha=0.66,
            #     label="95% interval",
            # )
            # ax.fill_between(
            #     df.index,
            #     p25_vals,
            #     p75_vals,
            #     edgecolor=colors[i],
            #     facecolor=colors[i],
            #     alpha=1,
            #     label="75% interval",
            # )
            axs[i].plot(df.index, median_vals, color=colors[i], label="Median", linewidth=1.2)
            axs[i].set_xlabel("")
            axs[i].set_ylabel(_lab_unit_daily["C_q_ss"])
            axs[i].set_xlim(df.index[0], df.index[-1])
            axs[i].set_ylim(0, 200)
            axs[i].axhline(y=50, color='r', linestyle='--', lw=1) 
            axs[i].axhline(y=37.5, color='orange', linestyle='--', lw=1)   
            axs[i].legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.8, 1.0))
        axs[-1].set_xlabel("Time [Year]")
        fig.tight_layout()
        file = base_path_figs / f"trace_perc_nitrate_conc_{location}_{crop_rotation_scenario}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

# plot daily values
vars_sim = ["M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        for fertilization_intensity in fertilization_intensities:
            ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
            for var_sim in vars_sim:
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:-1] * 0.01
                cond1 = (df_params["CLUST_flag"] == 1)
                df = pd.DataFrame(index=ds["Time"].values[1:-1], data=sim_vals.T).loc[:, cond1]
                df = df.loc["2014-01-01":"2022-12-31", :]
                cond = (df_areas["location"] == location)
                df_area_share = df_areas.loc[cond, :]
                data = df_params.loc[cond1, "CLUST_ID"].to_frame()
                data.loc[:, "area_share"] = 0.0
                for clust_id in clust_ids:
                    cond1 = (df_area_share["clust_id"] == clust_id)
                    if cond1.any():
                        cond2 = (data["CLUST_ID"] == clust_id)
                        data.loc[cond2, "area_share"] = df_area_share.loc[cond1, "area_share"].values[0] * 100
                data = repeat_by_areashare(df.values, data["area_share"].values)
                df1 = pd.DataFrame(data=data, index=df.index)
                vals = df1.values.T
                fig, ax = plt.subplots(1, 1, figsize=(6, 2))
                median_vals = onp.nanmedian(vals, axis=0)
                min_vals = onp.nanmin(vals, axis=0)
                max_vals = onp.nanmax(vals, axis=0)
                p5_vals = onp.nanquantile(vals, 0.05, axis=0)
                p25_vals = onp.nanquantile(vals, 0.25, axis=0)
                p75_vals = onp.nanquantile(vals, 0.75, axis=0)
                p95_vals = onp.nanquantile(vals, 0.95, axis=0)
                ax.fill_between(
                    df.index,
                    min_vals,
                    max_vals,
                    edgecolor=None,
                    facecolor='grey',
                    alpha=0.33,
                    label="Min-Max interval",
                )
                # ax.fill_between(
                #     df.index,
                #     p5_vals,
                #     p95_vals,
                #     edgecolor='grey',
                #     facecolor='grey',
                #     alpha=0.66,
                #     label="95% interval",
                # )
                # ax.fill_between(
                #     df.index,
                #     p25_vals,
                #     p75_vals,
                #     edgecolor='grey',
                #     facecolor='grey',
                #     alpha=1,
                #     label="75% interval",
                # )
                ax.plot(df.index, median_vals, color="black", label="Median", linewidth=1)
                ax.legend(frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.8, 1.27))
                ax.set_xlabel("Time [Year]")
                ax.set_ylabel(_lab_unit_daily[var_sim])
                ax.set_xlim(df.index[0], df.index[-1])
                ax.set_ylim(0,)
                fig.tight_layout()
                file = base_path_figs / f"trace_{var_sim}_{location}_{crop_rotation_scenario}_{fertilization_intensity}Nfert.png"
                fig.savefig(file, dpi=300)
                plt.close(fig)
