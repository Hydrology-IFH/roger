import os
from pathlib import Path
import xarray as xr
import numpy as onp
import yaml
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from cftime import num2date

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

_lab_unit_ann = {
    "theta": r"$\theta$ [-]",
    "prec": r"PRECIP [mm/year]",
    "aet": r"AET [mm/year]",
    "pet": r"PET [mm/year]",
    "q_ss": r"PERC [mm/year]",
    "q_sub": r"$Q_{sub}$ [mm/year]",
    "cpr_ss": r"$CPR$ [mm/year]",
    "inf": r"$INF_{MP}$ [mm/year]",
    "inf_mat": r"$INF_{MAT}$ [mm/year]",
    "inf_mp": r"$INF_{MP}$ [mm/year]",
    "inf_sc": r"$INF_{SC}$ [mm/year]",
}

_dict_vars = {"aet": "ET",
              "pet": "ET pot",
              "prec": "N",
              "inf": "Inf Gesamt",
              "inf_mat": "Inf Mtrx",
              "inf_mp": "Inf MP",
              "inf_sc": "Inf TR",
              "q_ss": "TP",
              "q_sub": "ZA Gesamt",
              "q_hof": "OA HOF",
              "q_sof": "OA SOF",
              "q_sur": "OA Gesamt",
              "cpr_ss": "kap.A.",
              }


cell_width = 25
ny = 404
nx = 356
grid_extent = (0, nx*cell_width, 0, ny*cell_width)

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# directory of output
base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/StressRes/Moehlin") / "output" / "oneD"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)

# load configuration file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)


# load RoGeR model parameters
params_hm_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_hm_file, engine="h5netcdf")
dem = ds_params["dgm"].values
mask = onp.isfinite(dem)

# make discrete colormaps
cmap1 = mpl.colormaps.get_cmap('viridis_r').resampled(15)
cmap2 = mpl.colormaps.get_cmap('PuOr').resampled(10)
cmap3 = mpl.colormaps.get_cmap('PuOr').resampled(8)

# load hydrological simulations
states_hm_file = base_path_output / "ONED_Moehlin_total.nc"
ds_sim1 = xr.open_dataset(states_hm_file, engine="h5netcdf")

states_hm_file = base_path_output / "RoGeR_WBM_1D" / "total.nc"
ds_sim2 = xr.open_dataset(states_hm_file, engine="h5netcdf")

cmap4 = mpl.colormaps.get_cmap('viridis_r').resampled(3)
vals1 = onp.where(mask, ds_params['lanu'].values, onp.nan)
vals2 = onp.where((vals1 < 10) | (vals1 > 13), onp.nan, vals1)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(vals2, extent=grid_extent, cmap=cmap4, zorder=2, aspect='equal')
plt.colorbar(im, ax=ax, shrink=0.7, label="")
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.grid(zorder=-1)
plt.tight_layout()
file = base_path_figs / "forest.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# vars_sim = ["prec", "pet", "aet", "cpr_ss", "inf", "inf_mat", "inf_mp", "inf_sc", "q_hof", "q_ss", "q_sub"]
# for var_sim in vars_sim:
#     mask = (ds_sim1[var_sim].values <= 0)
#     vals1 = onp.where(mask, onp.nan, ds_sim1[var_sim].values)
#     fig, ax = plt.subplots(figsize=(6,5))
#     im = ax.imshow(vals1, extent=grid_extent, cmap=cmap1, zorder=2, aspect='equal', vmin=0, vmax=1500)
#     plt.colorbar(im, ax=ax, shrink=0.7, label="[mm/year]")
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.grid(zorder=-1)
#     plt.tight_layout()
#     file = base_path_figs / f"{var_sim}_average_annual_sum.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)

# vars_sim = ["prec", "pet", "aet", "cpr_ss", "inf", "inf_mat", "inf_mp", "inf_sc", "q_hof", "q_ss", "q_sub", "q_sof"]
# for var_sim in vars_sim:
#     mask = (ds_sim2[_dict_vars[var_sim]].values <= 0)
#     vals2 = onp.where(mask, onp.nan, ds_sim2[_dict_vars[var_sim]].values)
#     fig, ax = plt.subplots(figsize=(6,5))
#     im = ax.imshow(vals2, extent=grid_extent, cmap=cmap1, zorder=2, aspect='equal', vmin=0, vmax=1500)
#     plt.colorbar(im, ax=ax, shrink=0.7, label="[mm/year]")
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.grid(zorder=-1)
#     plt.tight_layout()
#     file = base_path_figs / f"{var_sim}_average_annual_sum_roger_legacy.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)

# vars_sim = ["prec", "pet", "aet", "cpr_ss", "inf", "inf_mat", "inf_mp", "inf_sc"]
# # plot absolute difference
# for var_sim in vars_sim:
#     mask = (ds_sim1[var_sim].values <= 0)
#     vals1 = onp.where(mask, onp.nan, ds_sim1[var_sim].values)
#     vals2 = onp.where(mask, onp.nan, ds_sim2[_dict_vars[var_sim]].values)
#     diff_vals = vals1 - vals2
#     fig, ax = plt.subplots(figsize=(6,5))
#     im = ax.imshow(diff_vals, extent=grid_extent, cmap=cmap2, zorder=2, aspect='equal', vmin=-500, vmax=500)
#     plt.colorbar(im, ax=ax, shrink=0.7, label="[mm]")
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.grid(zorder=-1)
#     plt.tight_layout()
#     file = base_path_figs / f"{var_sim}_absolute_difference.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)

# # plot relative difference
# for var_sim in vars_sim:
#     mask = (ds_sim1[var_sim].values <= 0)
#     vals1 = onp.where(mask, onp.nan, ds_sim1[var_sim].values)
#     vals2 = onp.where(mask, onp.nan, ds_sim2[_dict_vars[var_sim]].values)
#     diff_vals = (vals1 - vals2) / vals2
#     fig, ax = plt.subplots(figsize=(6,5))
#     im = ax.imshow(diff_vals, extent=grid_extent, cmap=cmap3, zorder=2, aspect='equal', vmin=-0.8, vmax=0.8)
#     plt.colorbar(im, ax=ax, shrink=0.7, label="[-]")
#     plt.xlabel('Distance in x-direction [m]')
#     plt.ylabel('Distance in y-direction [m]')
#     plt.grid(zorder=-1)
#     plt.tight_layout()
#     file = base_path_figs / f"{var_sim}_relative_difference.png"
#     fig.savefig(file, dpi=300)
#     plt.close(fig)
    
x1 = 200
y1 = 200
pet_weight = ds_params["F_et"].isel(x=x1, y=y1).values/100
file = base_path / "input" / "PET.txt"
df_pet = pd.read_csv(
    file,
    sep=r"\s+",
    skiprows=0,
    header=0,
    parse_dates=[[0, 1, 2, 3, 4]],
    index_col=0,
    na_values=-9999,
)
df_pet.index = pd.to_datetime(df_pet.index, format="%Y %m %d %H %M")
df_pet.index = df_pet.index.rename("Index")

print(onp.sum(df_pet["PET"].values) * pet_weight)

states_hm_file = base_path_output / "ONED_Moehlin.nc"
ds_sim3 = xr.open_dataset(states_hm_file, engine="h5netcdf")
days = (ds_sim3['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date = num2date(
    days,
    units=f"days since {ds_sim3['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
ds_sim3 = ds_sim3.assign_coords(Time=("Time", date))
vals = ds_sim3["pet"].isel(x=x1, y=y1).values

print(onp.sum(vals))