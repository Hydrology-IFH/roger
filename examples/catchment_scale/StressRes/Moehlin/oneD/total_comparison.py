import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import datetime
import numpy as onp
import yaml
import matplotlib as mpl
import seaborn as sns
import h5netcdf
import roger

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
    "q_ss": r"PERC [mm/year]",
    "q_sub": r"$Q_{sub}$ [mm/year]",
    "cpr_ss": r"$CPR$ [mm/year]",
    "inf": r"$INF_{MP}$ [mm/year]",
    "inf_mat": r"$INF_{MAT}$ [mm/year]",
    "inf_mp": r"$INF_{MP}$ [mm/year]",
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
params_hm_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_hm_file, engine="h5netcdf")
dem = ds_params["dgm"].values
mask = onp.isfinite(dem)


# variables to be aggregated
vars_sim = ["aet", "q_ss", "q_sub", "cpr_ss", "inf", "inf_mat", "inf_mp", "inf_sc"]


# load hydrological simulations
states_hm_file = base_path_output / "ONED_Moehlin_annual_avg.nc"
ds_sim1 = xr.open_dataset(states_hm_file, engine="h5netcdf")

# assign date
days = (ds_sim1['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date = num2date(
    days,
    units=f"days since {ds_sim1['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)

# load hydrological simulations
states_hm_file = base_path_output / "RoGeR_WBM_1D" / "ONED_Moehlin_annual_avg.nc"
ds_sim2 = xr.open_dataset(states_hm_file, engine="h5netcdf")

# assign date
days = (ds_sim2['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date = num2date(
    days,
    units=f"days since {ds_sim2['Time'].attrs['time_origin']}",
    calendar="standard",
    only_use_cftime_datetimes=False,
)
