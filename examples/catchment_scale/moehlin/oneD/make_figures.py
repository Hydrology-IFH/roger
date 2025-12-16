import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import pandas as pd
import numpy as onp
import yaml
import matplotlib as mpl
import seaborn as sns

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11
mpl.rcParams["legend.fontsize"] = 11
mpl.rcParams["legend.title_fontsize"] = 12
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 11.0,
        "axes.labelsize": 12.0,
        "axes.titlesize": 11.0,
        "xtick.labelsize": 11.0,
        "ytick.labelsize": 11.0,
        "legend.fontsize": 11.0,
        "legend.title_fontsize": 11.0,
    },
)

_lab_unit_sum = {
    "theta": r"$\theta$ [-]",
    "prec": r"PRECIP [mm]",
    "aet": r"AET [mm]",
    "transp": r"TRANSP [mm]",
    "evap_soil": r"$EVAP_{soil}$ [mm]",
    "q_ss": r"PERC [mm]",
    "q_sub": r"$Q_{sub}$ [mm]",
    "inf_mp": r"$INF_{MP}$ [mm]",
}

_lab_unit_ann = {
    "theta": r"$\theta$ [-]",
    "prec": r"PRECIP [mm/year]",
    "aet": r"AET [mm/year]",
    "transp": r"TRANSP [mm/year]",
    "evap_soil": r"$EVAP_{soil}$ [mm/year]",
    "q_ss": r"PERC [mm/year]",
    "q_sub": r"$Q_{sub}$ [mm/year]",
    "inf_mp": r"$INF_{MP}$ [mm/year]",
}

_lab_unit_daily = {
    "theta": r"$\theta$ [-]",
    "prec": r"PRECIP [mm/day]",
    "aet": r"AET [mm/day]",
    "transp": r"TRANSP [mm/day]",
    "evap_soil": r"$EVAP_{soil}$ [mm/day]",
    "q_ss": r"PERC [mm/day]",
    "q_sub": r"$Q_{sub}$ [mm/day]",
    "inf_mp": r"$INF_{MP}$ [mm/day]",
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

def xr_annual_avg(ds, var):
    """
    weight by days in each month
    """
    # Determine the month length
    month_length = ds.Time.dt.days_in_month

    # Calculate the weights
    wgts = month_length.groupby("Time.year") / month_length.groupby("Time.year").sum()

    # Make sure the weights in each year add up to 1
    onp.testing.assert_allclose(wgts.groupby("Time.year").sum(xr.ALL_DIMS), 1.0)

    # Subset our dataset for our variable
    obs = ds[var]

    # Setup our masking for nan values
    cond = obs.isnull()
    ones = xr.where(cond, 0.0, 1.0)

    # Calculate the numerator
    obs_sum = (obs * wgts).resample(Time="AS").sum(dim="Time")

    # Calculate the denominator
    ones_out = (ones * wgts).resample(Time="AS").sum(dim="Time")

    # Return the weighted average
    darr = obs_sum / ones_out
    vals = onp.where(darr.values == -9999, onp.nan, darr.values)

    return vals


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
params_hm_file = base_path / "parameters_25_polygons.nc"
ds_params = xr.open_dataset(params_hm_file, engine="h5netcdf")
dem = ds_params["dgm"].values
lu_id = ds_params["lanu"].values
mask = onp.isfinite(dem)
mask_forest = onp.where(onp.isin(lu_id, [10, 11, 12, 13, 15]), True, False)
mask_cropland = onp.where(onp.isin(lu_id, [5, 6, 7]), True, False)
mask_grass = onp.where(onp.isin(lu_id, [8]), True, False)
mask_sealed = onp.where(onp.isin(lu_id, [0, 100]), True, False)
print(onp.unique(lu_id))

area = onp.round(onp.sum(mask) * cell_width**2, 1) / 1e6
print(f"Area: {area} km²")
area_forest = onp.round(onp.sum(mask_forest) * cell_width**2, 1) / 1e6
print(f"Forest area: {area_forest} km²")
area_cropland = onp.round(onp.sum(mask_cropland) * cell_width**2, 1) / 1e6
print(f"Cropland area: {area_cropland} km²")
area_grass = onp.round(onp.sum(mask_grass) * cell_width**2, 1) / 1e6
print(f"Grassland area: {area_grass} km²")
area_sealed = onp.round(onp.sum(mask_sealed) * cell_width**2, 1) / 1e6
print(f"Sealed area: {area_sealed} km²")

# plot elevation
dem = ds_params["dgm"].values
fig, ax = plt.subplots(figsize=(6,5))
fig.patch.set_alpha(0)
plt.imshow(dem, extent=grid_extent, cmap='terrain', zorder=1, aspect='equal')
# plt.colorbar(label='Elevation [m]', shrink=0.8)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
plt.colorbar(label='Höhe [m ü. NN]', shrink=0.8)
plt.grid(zorder=0)
plt.xlabel('Distanz in x-Richtung [m]')
plt.ylabel('Distanz in y-Richtung [m]')
plt.tight_layout()
file = base_path_figs / "elevation.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot soil depth
soil_depth = ds_params["GRUND"].values/100
fig, ax = plt.subplots(figsize=(6,5))
fig.patch.set_alpha(0)
plt.imshow(soil_depth, extent=grid_extent, cmap='Oranges', zorder=1, aspect='equal')
# plt.colorbar(label='Soil depth [m]', shrink=0.8)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
plt.colorbar(label='Bodenmächtigkeit [m]', shrink=0.8)
plt.grid(zorder=0)
plt.xlabel('Distanz in x-Richtung [m]')
plt.ylabel('Distanz in y-Richtung [m]')
plt.tight_layout()
file = base_path_figs / "soil_depth.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot slope
slope = ds_params["slope"].values
fig, ax = plt.subplots(figsize=(6,5))
fig.patch.set_alpha(0)
plt.imshow(slope, extent=grid_extent, cmap='Oranges', zorder=1, aspect='equal', vmin=0, vmax=45)
# plt.colorbar(label='Soil depth [m]', shrink=0.8)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
plt.colorbar(label='Hangneigung [%]', shrink=0.8)
plt.grid(zorder=0)
plt.xlabel('Distanz in x-Richtung [m]')
plt.ylabel('Distanz in y-Richtung [m]')
plt.tight_layout()
file = base_path_figs / "slope.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot groundwater level
dem = ds_params["gwfa_gew"].values/100
fig, ax = plt.subplots(figsize=(6,5))
fig.patch.set_alpha(0)
plt.imshow(dem, extent=grid_extent, cmap='Oranges', zorder=1, aspect='equal', vmin=0, vmax=10)
# plt.colorbar(label='Groundwater level [m]', shrink=0.8)
# plt.grid(zorder=0)
# plt.xlabel('Distance in x-direction [m]')
# plt.ylabel('Distance in y-direction [m]')
plt.colorbar(label='GW-Flurabstand [m]', shrink=0.8)
plt.grid(zorder=0)
plt.xlabel('Distanz in x-Richtung [m]')
plt.ylabel('Distanz in y-Richtung [m]')
plt.tight_layout()
file = base_path_figs / "gw_level.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# plot land use
land_use = onp.copy(ds_params["lanu"].values)
land_use[:, :] = onp.nan
land_use = onp.where(mask_forest, 1, land_use)
land_use = onp.where(mask_grass, 2, land_use)
land_use = onp.where(mask_sealed, 3, land_use)
land_use = onp.where(mask_cropland, 4, land_use)
cmap4 = mpl.colormaps.get_cmap('RdPu').resampled(4)
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(land_use, extent=grid_extent, cmap=cmap4, zorder=2, aspect='equal', vmin=1, vmax=5)
cbar = plt.colorbar(im, ax=ax, shrink=0.7, label="")
cbar.ax.invert_yaxis()
plt.xlabel('Distanz in x-Richtung [m]')
plt.ylabel('Distanz in y-Richtung [m]')
plt.grid(zorder=-1)
plt.tight_layout()
file = base_path_figs / "land_use.png"
fig.savefig(file, dpi=300)
plt.close(fig)

states_hm_file = base_path_output / "raster" / "ONED_Moehlin_total.nc"
ds_sim1 = xr.open_dataset(states_hm_file, engine="h5netcdf")

# states_hm_file = base_path_output / "RoGeR_WBM_1D" / "total.nc"
# ds_sim2 = xr.open_dataset(states_hm_file, engine="h5netcdf")

cmap1 = mpl.colormaps.get_cmap('viridis_r').resampled(6)
vars_sim = ["cpr_ss", "q_ss", "q_sub"]
for var_sim in vars_sim:
    # mask = (ds_sim2[_dict_vars[var_sim]].values <= 0)
    # vals2 = onp.where(mask, onp.nan, ds_sim2[_dict_vars[var_sim]].values)
    mask = (ds_sim1[var_sim].values <= 0)
    vals2 = onp.where(mask, onp.nan, ds_sim1[var_sim].values)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(vals2, extent=grid_extent, cmap=cmap1, zorder=2, aspect='equal', vmin=0, vmax=600)
    plt.colorbar(im, ax=ax, shrink=0.7, label="[mm/Jahr]")
    plt.xlabel('Distanz in x-Richtung [m]')
    plt.ylabel('Distanz in y-Richtung [m]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / f"{var_sim}_average_annual_sum.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

vars_sim = ["cpr_ss", "q_ss", "q_sub"]
for var_sim in vars_sim:
    # mask = (ds_sim2[_dict_vars[var_sim]].values <= 0)
    # vals2 = onp.where(mask, onp.nan, ds_sim2[_dict_vars[var_sim]].values)
    mask = (ds_sim1[var_sim].values <= 0)
    vals2 = onp.where(mask, onp.nan, ds_sim1[var_sim].values)
    ll_df = []
    df = pd.DataFrame(vals2[mask_forest])
    df["x"] = "Wald"
    df["land_use"] = 1
    ll_df.append(df)
    df = pd.DataFrame(vals2[mask_grass])
    df["x"] = "Gruenland"
    df["land_use"] = 2
    ll_df.append(df)
    df = pd.DataFrame(vals2[mask_sealed])
    df["x"] = "versiegelte\n Flaechen"
    df["land_use"] = 3
    ll_df.append(df)
    df = pd.DataFrame(vals2[mask_cropland])
    df["x"] = "Landwirtschaft"
    df["land_use"] = 4
    ll_df.append(df)
    data = pd.concat(ll_df)
    data.columns = ["y", "x", "land_use"]
    data.index = range(len(data.index))
    
    fig, axes = plt.subplots(figsize=(4,3))
    sns.boxplot(data=data, x="x", y="y", hue="land_use", ax=axes, whis=(5, 95), showfliers=False, palette="RdPu", hue_norm=(1, 4))
    axes.set_ylabel("[mm/Jahr]")
    axes.set_xlabel("")
    axes.legend().set_visible(False)
    axes.set_ylim(0, 800)
    plt.xticks(rotation=33)
    plt.tight_layout()
    file = base_path_figs / f"{var_sim}_per_land_use.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


