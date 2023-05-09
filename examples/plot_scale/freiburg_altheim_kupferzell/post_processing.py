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
locations = ['freiburg', 'altheim', 'kupferzell']
Locations = ['Freiburg', 'Altheim', 'Kupferzell']
land_cover_scenarios = ['grass', 'corn', 'corn_catch_crop', 'crop_rotation']
climate_scenarios = ['CCCma-CanESM2_CCLM4-8-17', 'MPI-M-MPI-ESM-LR_RCA4']
periods = ['1985-2005', '2040-2060', '2080-2100']
start_dates = [datetime.date(1985, 1, 1), datetime.date(2040, 1, 1), datetime.date(2080, 1, 1)]
end_dates = [datetime.date(2004, 12, 31), datetime.date(2059, 12, 31), datetime.date(2099, 12, 31)]

# load simulations
dict_sim = {}
for location in locations:
    dict_sim[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_sim[location][land_cover_scenario] = {}
        for period in periods:
            dict_sim[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                output_hm_file = base_path_results / "svat" / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                ds_sim_hm = xr.open_dataset(output_hm_file, engine="h5netcdf")
                # assign date
                days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
                date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
                ds_sim_hm = ds_sim_hm.assign_coords(Time=("Time", date_sim_hm))
                dict_sim[location][land_cover_scenario][period][climate_scenario] = ds_sim_hm

# # plot time series
# vars_sim = ['theta']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#                 sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#                 sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)

#                 ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#                 sim_vals_5_MPIM = onp.nanquantile(sim_vals_MPIM, 0.05, axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
#                 sim_vals_95_MPIM = onp.nanquantile(sim_vals_MPIM, 0.95, axis=0)

#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#                 axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_5_CanESM, sim_vals_95_CanESM, color='red',
#                                         edgecolor=None, alpha=0.2)

#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_5_MPIM, sim_vals_95_MPIM, color='blue',
#                                         edgecolor=None, alpha=0.2)
#                 axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#                 axes[-1, j].set_xlabel('Time [year]')
#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_DAILY[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}.png"
#         fig.savefig(file, dpi=250)

# plot cumulated time series
vars_sim = ['transp', 'evap_soil', 'q_ss']
for land_cover_scenario in land_cover_scenarios:
    for var_sim in vars_sim:
        fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
        for i, location in enumerate(locations):
            for j, period in enumerate(periods):
                ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_avg_CanESM = onp.nanmean(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
                sim_vals_min_CanESM = onp.nanmin(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
                sim_vals_50_CanESM = onp.nanmedian(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
                sim_vals_max_CanESM = onp.nanmax(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)

                ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_avg_MPIM = onp.nanmean(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
                sim_vals_min_MPIM = onp.nanmin(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
                sim_vals_50_MPIM = onp.nanmedian(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
                sim_vals_max_MPIM = onp.nanmax(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)

                axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
                axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
                axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color='red',
                                        edgecolor=None, alpha=0.2)

                axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
                axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
                axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color='blue',
                                        edgecolor=None, alpha=0.2)
                axes[i, j].set_xlim(start_dates[j], end_dates[j])
                axes[i, j].set_ylim(0,)
                axes[-1, j].set_xlabel('Time [year]')
                # axes[-1, j].tick_params(axis='x', rotation=33)
                axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
                axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
            axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
        fig.tight_layout()
        file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated.png"
        fig.savefig(file, dpi=250)
        plt.close('all')

# plot cumulated precipitation
# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[i, j].set_ylim(0,)
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel('PRECIP [mm]')
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_cumulated.png"
#     fig.savefig(file, dpi=250)
#     plt.close('all')


# plot soil water content time series
vars_sim = ['theta']
for land_cover_scenario in land_cover_scenarios:
    for var_sim in vars_sim:
        fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
        for i, location in enumerate(locations):
            for j, period in enumerate(periods):
                ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
                sim_vals_min_CanESM = onp.nanmin(sim_vals_CanESM, axis=0)
                sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
                sim_vals_max_CanESM = onp.nanmax(sim_vals_CanESM, axis=0)

                ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
                sim_vals_min_MPIM = onp.nanmin(sim_vals_MPIM, axis=0)
                sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
                sim_vals_max_MPIM = onp.nanmax(sim_vals_MPIM, axis=0)

                axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
                axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
                axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color='red',
                                        edgecolor=None, alpha=0.2)

                axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
                axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
                axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color='blue',
                                        edgecolor=None, alpha=0.2)
                axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
                axes[-1, j].set_xlabel('Time [year]')
                # axes[-1, j].tick_params(axis='x', rotation=33)
                axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
                axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
            axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_DAILY[var_sim]))
        fig.tight_layout()
        file = base_path_figs / f"{var_sim}_{land_cover_scenario}.png"
        fig.savefig(file, dpi=250)
        plt.close('all')


# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#             sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#             axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='-', color='red', lw=1)
#             axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='-', color='blue', lw=1)
#             axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#             axes[i, j].set_ylim(0,)
#             axes[-1, j].set_xlabel('Time [year]')
#         axes[i, 0].set_ylabel('%s\nPRECIP [mm]' % (Locations[i]))
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_cumulated.png"
#     fig.savefig(file, dpi=250)
#     plt.close('all')

vars_sim = ['transp', 'evap_soil', 'q_ss']
for land_cover_scenario in land_cover_scenarios:
    for var_sim in vars_sim:
        fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
        for i, location in enumerate(locations):
            for j, period in enumerate(periods):
                for x in range(675):
                    ds_CanESM = dict_sim[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                    sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=x).values
                    sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

                    ds_MPIM = dict_sim[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                    sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=x).values
                    sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

                    axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='-', color='red', lw=1)
                    axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='-', color='blue', lw=1)
                axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
                axes[i, j].set_ylim(0,)
                axes[-1, j].set_xlabel('Time [year]')
                # axes[-1, j].tick_params(axis='x', rotation=33)
                axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
                axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
            axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
        fig.tight_layout()
        file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated_trace.png"
        fig.savefig(file, dpi=250)
        plt.close('all')