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

_dict_var_names = {"q_hof": "Qsur",
                   "ground_cover": "GC",
                   "M_q_ss": "NO3PERC"
}


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

# # load buffers for assigning the simulations to the meteorological stations
# file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "buffer30km_NBiomasseBW_assignment.gpkg"
# gdf_buffer = gpd.read_file(file, include_fields=["fid", "station_id", "stationsna", "agr_region"])

# load linkage between BK50 and cropland clusters
file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)

# prepare shapefiles for each location and crop rotation scenario
for crop_rotation_scenario in crop_rotation_scenarios:
    new_file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/data_for_nitrate_leaching") / f"FFID_{_dict_ffid[crop_rotation_scenario]}.gpkg"
    if not os.path.exists(new_file):
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "BK50_acker_freiburg.gpkg"
        gdf = gpd.read_file(file, include_fields=["fid", "SHP_ID"])
        gdf.to_file(new_file, driver="GPKG")

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

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

# and aggregate nitrate leaching to average annual sum
vars_sim = ["M_q_ss"]
for crop_rotation_scenario in crop_rotation_scenarios:
    for location in locations:
        file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh/output/data_for_nitrate_leaching") / f"FFID_{_dict_ffid[crop_rotation_scenario]}.gpkg"
        # gdf = gpd.read_file(file, include_fields=["fid", "SHP_ID"], mask=gdf_buffer[gdf_buffer.stationsna==location])
        gdf = gpd.read_file(file, include_fields=["fid", "SHP_ID"])
        for var_sim in vars_sim:
            for fertilization_intensity in fertilization_intensities:
                gdf[f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'] = None  # initialize field, float, two decimals
                gdf[f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'] = gdf[f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'].astype('float64')
                gdf[f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'] = gdf[f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'].round(decimals=2)
                ds = dict_nitrate[location][crop_rotation_scenario][f'{fertilization_intensity}_Nfert']
                sim_vals = ds[var_sim].isel(y=0).values[:, 1:]
                df = pd.DataFrame(index=ds["Time"].values[1:], data=sim_vals.T)
                # calculate annual sum
                df_ann = df.resample("YE").sum() * 0.01  # convert from mg/m2 to kg/ha
                # calculate average
                df_ann_avg = df_ann.mean(axis=0).to_frame()
                # assign aggregated values to polygons
                for clust_id in clust_ids:
                    cond1 = (df_params["CLUST_ID"] == clust_id) & (df_params["CLUST_flag"] == 2)
                    val = df_ann_avg.loc[cond1, :].values[0][0]
                    cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
                    shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
                    cond2 = gdf["SHP_ID"].isin(shp_ids)
                    if cond2.any():
                        gdf.loc[cond2, f'{_dict_var_names[var_sim]}_{fertilization_intensity}Nfert_avg'] = val
    gdf = gdf.to_crs("EPSG:25832")
    gdf.to_file(file, driver="GPKG")