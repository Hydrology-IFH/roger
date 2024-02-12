import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import numpy as onp
import yaml
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
base_path_output = base_path / "output" / "2000_2023"
if not os.path.exists(base_path_output):
    os.mkdir(base_path_output)

# load configuration file
file_config = base_path / "config.yml"
with open(file_config, "r") as file:
    config = yaml.safe_load(file)


# load RoGeR model parameters
params_hm_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_hm_file, engine="h5netcdf")

# plot elevation
dem = ds_params["dgm"].values
fig, ax = plt.subplots(figsize=(6,5))
fig.patch.set_alpha(0)
plt.imshow(dem, extent=grid_extent, cmap='terrain', zorder=1, aspect='equal')
plt.colorbar(label='Elevation [m]', shrink=0.8)
plt.grid(zorder=0)
plt.xlabel('Distance in x-direction [m]')
plt.ylabel('Distance in y-direction [m]')
plt.tight_layout()
file = base_path_figs / "elevation.png"
fig.savefig(file, dpi=300)
plt.close(fig)

# load hydrological simulations
states_hm_file = base_path_output / f"{config['identifier']}.nc"
ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")
mask_file = base_path / "mask.nc"
mask = onp.isfinite(dem)

# assign date
days = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date = num2date(
    days,
    units=f"days since {ds_sim['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
ds_sim = ds_sim.assign_coords(Time=("Time", date))

