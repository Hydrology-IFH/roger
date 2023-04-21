import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import datetime
import glob
import h5netcdf
import roger
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.title_fontsize'] = 9
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 8.0,
                                                'axes.labelsize': 9.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 8.0,
                                                'ytick.labelsize': 8.0,
                                                'legend.fontsize': 8.0,
                                                'legend.title_fontsize': 9.0})

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "output"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# identifiers for simulations
locations = ['freiburg', 'kupferzell']
land_cover_scenarios = ['corn', 'corn_catch_crop', 'crop_rotation', 'grass']
climate_scenarios = ['CCCma-CanESM2_CCLM4-8-17', 'MPI-M-MPI-ESM-LR_RCA4']
periods = ['1985-2005', '2040-2060', '2080-2100']

file = base_path / "input" / "freiburg" / "MPI-M-MPI-ESM-LR_RCA4" / "2080-2100" / "PREC.txt"
df = pd.read_csv(file, skiprows=0, sep='\t', na_values='')
print(df['PREC'].sum())

# # load simulations
# dict_sim = {}
# for location in locations:
#     dict_sim[location] = {}
#     dict_sim[location]['grass'] = {}
#     for period in periods:
#         dict_sim[location]['grass'][period] = {}
#         for climate_scenario in climate_scenarios:
#             output_hm_file = base_path_results / "svat" / f"SVAT_{location}_grass_{climate_scenario}_{period}.nc"
#             ds_sim_hm = xr.open_dataset(output_hm_file, engine="h5netcdf")
#             # assign date
#             days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
#             date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
#             ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
#             dict_sim[location]['grass'][period][climate_scenario] = ds_sim_hm

# # plot time series
# vars_sim = ['aet', 'theta', 'q_ss']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(2, 3, sharex='col', figsize=(6, 3))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_sim[location]['grass'][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#             sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#             sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
#             sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#             sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)

#             ds_MPIM = dict_sim[location]['grass'][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#             sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#             sim_vals_5_MPIM = onp.nanquantile(sim_vals_MPIM, 0.05, axis=0)
#             sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
#             sim_vals_95_MPIM = onp.nanquantile(sim_vals_MPIM, 0.95, axis=0)

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#             axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_5_CanESM, sim_vals_95_CanESM, color='red',
#                                     edgecolor=None, alpha=0.2)

#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='orange', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='orange', lw=1)
#             axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_5_MPIM, sim_vals_95_MPIM, color='orange',
#                                     edgecolor=None, alpha=0.2)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel(labs._Y_LABS_DAILY[var_sim])
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_grass.png"
#     fig.savefig(file, dpi=250)

# # plot cumulated time series
# vars_sim = ['aet', 'q_ss']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(2, 3, sharex='col', figsize=(6, 3))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_sim[location]['grass'][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#             sim_vals_avg_CanESM = onp.cumsum(onp.nanmean(sim_vals_CanESM, axis=0))
#             sim_vals_5_CanESM = onp.cumsum(onp.nanquantile(sim_vals_CanESM, 0.05, axis=0))
#             sim_vals_50_CanESM = onp.cumsum(onp.nanmedian(sim_vals_CanESM, axis=0))
#             sim_vals_95_CanESM = onp.cumsum(onp.nanquantile(sim_vals_CanESM, 0.95, axis=0))
#             sim_vals_prec = ds_CanESM['prec'].isel(y=0).values
#             print('prec: ', onp.sum(sim_vals_prec[0, :]))
#             print(var_sim, ': ', onp.sum(sim_vals_CanESM[0, :]))

#             ds_MPIM = dict_sim[location]['grass'][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#             sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#             sim_vals_5_MPIM = onp.cumsum(onp.nanquantile(sim_vals_MPIM, 0.05, axis=0))
#             sim_vals_50_MPIM = onp.cumsum(onp.nanmedian(sim_vals_MPIM, axis=0))
#             sim_vals_95_MPIM = onp.cumsum(onp.nanquantile(sim_vals_MPIM, 0.95, axis=0))

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#             axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_5_CanESM, sim_vals_95_CanESM, color='red',
#                                     edgecolor=None, alpha=0.2)

#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='orange', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='orange', lw=1)
#             axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_5_MPIM, sim_vals_95_MPIM, color='orange',
#                                     edgecolor=None, alpha=0.2)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel(labs._Y_LABS_CUM[var_sim])
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_grass_cumulated.png"
#     fig.savefig(file, dpi=250)