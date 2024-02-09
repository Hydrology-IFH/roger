import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import geopandas as gpd
import numpy as onp
import matplotlib as mpl
import seaborn as sns
import h5py

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

# identifiers for simulations
locations = ["freiburg", "lahr", "muellheim", 
             "stockach", "gottmadingen", "weingarten",
             "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
             "ehingen-kirchen", "merklingen", "hayingen",
             "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
# locations = [
#     "freiburg",
# ]

crop_rotation_scenarios = ["winter-wheat_clover",
                           "winter-wheat_corn",
                           "winter-wheat_winter-rape",
                           "summer-wheat_winter-wheat", 
                           "summer-wheat_clover_winter-wheat",
                           "winter-wheat_clover_corn",
                           "winter-wheat_sugar-beet_corn",
                           "summer-wheat_winter-wheat_corn",
                           "summer-wheat_winter-wheat_winter-rape", 
                           "winter-wheat_winter-rape",
                           "winter-wheat_winter-grain-pea_winter-rape", 
                           "winter-wheat_corn_yellow-mustard", 
                           "summer-wheat_winter-wheat_corn_yellow-mustard",
                           "winter-wheat_sugar-beet_corn_yellow-mustard",
                           "summer-wheat_winter-wheat_corn_yellow-mustard", 
                           "summer-wheat_winter-wheat_winter-rape_yellow-mustard"]
crop_rotation_scenarios = ["winter-wheat_corn"]

# load buffers for assigning the simulations to the meteorological stations
file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "buffer30km_NBiomasseBW_assigment.gpkg"
gdf_buffer = gpd.read_file(file, include_fields=["fid", "station_id", "stationsna", "agr_region"])

# load linkage between BK50 and cropland clusters
file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)

# prepare shapefiles for each location and crop rotation scenario
for crop_rotation_scenario in crop_rotation_scenarios:
    new_file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/svat_crop") / f"surface_runoff_ground_cover_{crop_rotation_scenario}.gpkg"
    if not os.path.exists(new_file):
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "BK50_NBiomasseBW.gpkg"
        gdf = gpd.read_file(file, include_fields=["fid", "SHP_ID"], mask=gdf_buffer)
        gdf.to_file(new_file, driver="GPKG")

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

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

# aggregate ground cover to average annual mean
vars_sim = ["ground_cover", "q_hof"]
for location in locations:
    for crop_rotation_scenario in crop_rotation_scenarios:
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/svat_crop") / f"surface_runoff_ground_cover_{crop_rotation_scenario}.gpkg"
        gdf = gpd.read_file(file, include_fields=["fid", "SHP_ID"], mask=gdf_buffer[gdf_buffer.stationsna==location],)
        gdf['stationsna'] = location
        gdf['stationsna'] = gdf['stationsna'].astype('str')
        gdf['agr_region'] = gdf_buffer[gdf_buffer.stationsna==location].agr_region.values[0]
        gdf['agr_region'] = gdf['agr_region'].astype('str')
        for var_sim in vars_sim:
            gdf[f'{var_sim}'] = None  # initialize field, float, two decimals
            gdf[f'{var_sim}'] = gdf[f'{var_sim}'].astype('float64')
            gdf[f'{var_sim}'] = gdf[f'{var_sim}'].round(decimals=2)
            ds = dict_fluxes_states[location][crop_rotation_scenario]
            sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
            df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
            if var_sim == "ground_cover":
                # calculate annual mean
                df_ann = df.resample("YE").mean()
            else:
                # calculate annual sum
                df_ann = df.resample("YE").sum()
            # calculate average
            df_ann_avg = df_ann.mean(axis=0).to_frame()
            # assign aggregated values to polygons
            for clust_id in clust_ids:
                cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 1)
                val = df_ann_avg.loc[cond1, :].values[0][0]
                cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                cond2 = gdf["SHP_ID"].isin(shp_ids)
                if cond2.any():
                    gdf.loc[cond2, f'{var_sim}'] = val
        gdf.to_file(file, layer=f"{location}_{crop_rotation_scenario}", driver="GPKG")