import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import geopandas as gpd
import numpy as onp
import matplotlib as mpl
import seaborn as sns

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


def nanmeanweighted(y, w, axis=None):
    w1 = w / onp.nansum(w, axis=axis)
    w2 = onp.where(onp.isnan(w), 0, w1)
    w3 = onp.where(onp.isnan(y), 0, w2)
    y1 = onp.where(onp.isnan(y), 0, y)
    wavg = onp.sum(y1 * w3, axis=axis) / onp.sum(w3, axis=axis)

    return wavg


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
              "winter-wheat_silage-corn_yellow-mustard": "2_1",
              "summer-wheat_winter-wheat_yellow-mustard": "3_1",
              "winter-wheat_sugar-beet_silage-corn_yellow-mustard": "6_1",
              "summer-wheat_winter-wheat_silage-corn_yellow-mustard": "7_1",
              "summer-wheat_winter-wheat_winter-rape_yellow-mustard": "8_1",
              "sugar-beet_winter-wheat_winter-barley_yellow-mustard": "11_1", 
              "grain-corn_winter-wheat_winter-rape_yellow-mustard": "12_1", 
              "grain-corn_winter-wheat_winter-barley_yellow-mustard": "13_1", 
}

# identifiers for simulations
locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
locations = [
    "freiburg",
]

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

fertilization_intensities = ["low", "medium", "high"]

_lab_unit_daily = {
    "M_q_ss": "PERC-$NO_3$\n [kg $NO_3$-N/day/ha]",
}

_lab_unit_annual = {
    "M_q_ss": "PERC-$NO_3$\n  [kg $NO_3$-N/year/ha]",
}

_lab_unit_total = {
    "M_q_ss": "PERC-$NO_3$ [kg $NO_3$-N]"
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

# # load linkage between BK50 and cropland clusters
# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
# df_link_bk50_cluster_cropland = pd.read_hdf(file)

# # load BK50 shapefile for Freiburg
# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "BK50_acker_freiburg.gpkg"
# gdf_bk50 = gpd.read_file(file, include_fields=["SHP_ID", "area"]).loc[:, ["SHP_ID", "area"]]

# # get unique cluster ids for cropland
# cond = onp.isin(df_link_bk50_cluster_cropland.index.values, gdf_bk50["SHP_ID"].values)
# clust_ids = onp.unique(df_link_bk50_cluster_cropland.loc[cond, "CLUST_ID"].values).astype(str)

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

# # plot daily values
# vars_sim = ["M_q_ss"]
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for fertilization_intensity in fertilization_intensities:
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             for var_sim in vars_sim:
#                 sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01
#                 cond1 = (df_params["CLUST_flag"] == 1)
#                 df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#                 vals = df.values.T * 0.01
#                 fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#                 median_vals = onp.median(vals, axis=0)
#                 min_vals = onp.min(vals, axis=0)
#                 max_vals = onp.max(vals, axis=0)
#                 p5_vals = onp.nanquantile(vals, 0.05, axis=0)
#                 p25_vals = onp.nanquantile(vals, 0.25, axis=0)
#                 p75_vals = onp.nanquantile(vals, 0.75, axis=0)
#                 p95_vals = onp.nanquantile(vals, 0.95, axis=0)
#                 ax.fill_between(
#                     df.index,
#                     min_vals,
#                     max_vals,
#                     edgecolor='grey',
#                     facecolor='grey',
#                     alpha=0.33,
#                     label="Min-Max interval",
#                 )
#                 ax.fill_between(
#                     df.index,
#                     p5_vals,
#                     p95_vals,
#                     edgecolor='grey',
#                     facecolor='grey',
#                     alpha=0.66,
#                     label="95% interval",
#                 )
#                 ax.fill_between(
#                     df.index,
#                     p25_vals,
#                     p75_vals,
#                     edgecolor='grey',
#                     facecolor='grey',
#                     alpha=1,
#                     label="75% interval",
#                 )
#                 ax.plot(df.index, median_vals, color="black", label="Median", linewidth=1)
#                 ax.legend(frameon=False, loc="upper right", ncol=4, bbox_to_anchor=(0.93, 1.19))
#                 ax.set_xlabel("Time [Year]")
#                 ax.set_ylabel(_lab_unit_daily[var_sim])
#                 ax.set_xlim(df.index[0], df.index[-1])
#                 fig.tight_layout()
#                 file = base_path_figs / f"trace_{var_sim}_{location}_{crop_rotation_scenario}.png"
#                 fig.savefig(file, dpi=250)
#                 plt.close(fig)

# # plot annual sums
# vars_sim = ["M_q_ss"]
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for fertilization_intensity in fertilization_intensities:
#             ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#             for var_sim in vars_sim:
#                 sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
#                 cond1 = (df_params["CLUST_flag"] == 1)
#                 df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#                 # calculate annual sum
#                 df_ann = df.resample("YE").sum().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
#                 df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
    
#                 fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#                 sns.boxplot(data=df_ann_long, x="year", y="vals", ax=ax, color="red", whis=(5, 95), showfliers=False)
#                 ax.set_xlabel("Time [Year]")
#                 ax.set_ylabel(_lab_unit_annual[var_sim])
#                 ax.set_ylim(0, )
#                 fig.tight_layout()
#                 file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.png"
#                 fig.savefig(file, dpi=250)
#                 plt.close(fig)

# vars_sim = ["M_q_ss"]
# for crop_rotation_scenario in crop_rotation_scenarios:
#     for location in locations:
#         for var_sim in vars_sim:
#             ll_df = []
#             for fertilization_intensity in fertilization_intensities:
#                 ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
#                 sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
#                 cond1 = (df_params["CLUST_flag"] == 1)
#                 df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
#                 # calculate annual sum
#                 df_ann = df.resample("YE").sum().iloc[:-1, :]
#                 df_ann.loc[:, "year"] = df_ann.index.year
#                 df_ann_long = df_ann.melt(id_vars="year", value_name="vals", var_name='Nfert')
#                 df_ann_long.loc[:, "Nfert"] = f'{fertilization_intensity}'
#                 ll_df.append(df_ann_long)

#         df_ann_long = pd.concat(ll_df)
    
#         fig, ax = plt.subplots(1, 1, figsize=(6, 2))
#         sns.boxplot(data=df_ann_long, x="year", y="vals", hue="Nfert", ax=ax, whis=(5, 95), palette="magma_r", showfliers=False)
#         ax.set_xlabel("Time [Year]")
#         ax.set_ylabel(_lab_unit_annual[var_sim])
#         ax.set_ylim(0, )
#         plt.legend(title="N-Fertilization intensity", frameon=False)
#         fig.tight_layout()
#         file = base_path_figs / f"boxplot_{var_sim}_{location}_{crop_rotation_scenario}.png"
#         fig.savefig(file, dpi=250)
#         plt.close(fig)

# plot average annual sum
colors = sns.color_palette("magma_r", n_colors=len(fertilization_intensities))
vars_sim = ["M_q_ss"]
for var_sim in vars_sim:
    for location in locations:
        fig, axes = plt.subplots(3, 1, figsize=(6, 6), sharex=True, sharey=True)
        for i, fertilization_intensity in enumerate(fertilization_intensities):
            ll_df = []
            for crop_rotation_scenario in crop_rotation_scenarios:
                ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert'] 
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:] * 0.01 # convert from mg/m2 to kg/ha
                cond1 = (df_params["CLUST_flag"] == 1)
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T).loc[:, cond1]
                # calculate annual sum
                df_ann_avg = df.resample("YE").sum().iloc[:-1, :].mean(axis=0).to_frame()
                df_ann_avg.loc[:, "crop_rotation"] = _dict_ffid[crop_rotation_scenario]
                ll_df.append(df_ann_avg)
            df_ann_avg = pd.concat(ll_df)
            df_ann_avg_long = df_ann_avg.melt(id_vars="crop_rotation", value_name="vals").loc[:, ["crop_rotation", "vals"]]        
            sns.boxplot(data=df_ann_avg_long, x="crop_rotation", y="vals", ax=axes[i], whis=(5, 95), showfliers=False, color=colors[i])
            axes[i].set_xlabel("")
            axes[i].set_ylabel(_lab_unit_total[var_sim])
            axes[i].set_ylim(0, 50)
        axes[-1].set_xlabel("Crop rotation")
        plt.xticks(rotation=33)
        fig.tight_layout()
        file = base_path_figs / f"boxplot_annual_average_{var_sim}_{location}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)


