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

# load model parameters
csv_file = base_path / 'parameters.csv'
df_params = pd.read_csv(csv_file, sep=';', skiprows=1)

# load simulated fluxes and states
dict_fluxes_states = {}
for location in locations:
    dict_fluxes_states[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_fluxes_states[location][land_cover_scenario] = {}
        for period in periods:
            dict_fluxes_states[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                try:
                    output_hm_file = base_path_results / "svat" / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                    ds_fluxes_states = xr.open_dataset(output_hm_file, engine="h5netcdf")
                    # assign date
                    days = (ds_fluxes_states['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
                    date = num2date(days, units=f"days since {ds_fluxes_states['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
                    ds_fluxes_states = ds_fluxes_states.assign_coords(Time=("Time", date))
                    dict_fluxes_states[location][land_cover_scenario][period][climate_scenario] = ds_fluxes_states
                except KeyError:
                    print(f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc")

# # plot time series
# vars_sim = ['theta']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#                 sim_vals_5_CanESM = onp.nanquantile(sim_vals_CanESM, 0.05, axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#                 sim_vals_95_CanESM = onp.nanquantile(sim_vals_CanESM, 0.95, axis=0)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
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

# # plot cumulated time series of fluxes
# vars_sim = ['transp', 'evap_soil', 'q_ss']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_min_CanESM = onp.nanmin(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_max_CanESM = onp.nanmax(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_min_MPIM = onp.nanmin(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_max_MPIM = onp.nanmax(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)

#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#                 axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color='red',
#                                         edgecolor=None, alpha=0.2)

#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color='blue',
#                                         edgecolor=None, alpha=0.2)
#                 axes[i, j].set_xlim(start_dates[j], end_dates[j])
#                 axes[i, j].set_ylim(0,)
#                 axes[-1, j].set_xlabel('Time [year]')
#                 # axes[-1, j].tick_params(axis='x', rotation=33)
#                 axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                 axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))

#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated.png"
#         fig.savefig(file, dpi=250)
#         plt.close('all')

_LABS = {'evap_soil': r'$EVAP_{soil}$',
         'transp': r'$TRANSP$',
         'q_ss': r'$PERC$',
         'theta': r'$\theta$',
         'grass': 'Grass', 
         'corn': 'Corn', 
         'corn_catch_crop': 'Corn + CC', 
         'crop_rotation': 'Crop Rot.',
        }
vars_sim = ['evap_soil', 'transp', 'q_ss']
for var_sim in vars_sim:
    fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey='row', figsize=(6, 4.5))
    for i, location in enumerate(locations):
        for j, period in enumerate(periods):
            ll_df_fluxes = []
            for land_cover_scenario in land_cover_scenarios:
                ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_sum_CanESM = onp.sum(sim_vals_CanESM, axis=1)

                ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_sum_MPIM = onp.sum(sim_vals_MPIM, axis=1)

                df1 = pd.DataFrame()
                df1.loc[:, 'flux_sum'] = sim_vals_sum_CanESM.flatten()
                df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
                df1.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
                df1.loc[:, 'flux'] = _LABS[var_sim]

                df2 = pd.DataFrame()
                df2.loc[:, 'flux_sum'] = sim_vals_sum_MPIM.flatten()
                df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
                df2.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
                df2.loc[:, 'flux'] = _LABS[var_sim]

                df = pd.concat([df1, df2], ignore_index=True)
                ll_df_fluxes.append(df)

            df_fluxes = pd.concat(ll_df_fluxes, ignore_index=True)
            sns.boxplot(x="land_cover_scenario", y="flux_sum", hue="climate_scenario", palette=["red", "blue"],
            data=df_fluxes, ax=axes[i, j], showfliers=False)
            axes[i, j].legend([],[], frameon=False)
            axes[0, j].set_title(f'{period}')
            axes[i, j].set_ylabel('')
            axes[i, j].set_xlabel('')
            axes[i, j].tick_params(axis='x', rotation=33)
        axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
    fig.tight_layout()
    file = base_path_figs / f"{var_sim}_boxplot.png"
    fig.savefig(file, dpi=250)
    plt.close('all')


vars_sim = ['theta']
for var_sim in vars_sim:
    fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey='row', figsize=(6, 4.5))
    for i, location in enumerate(locations):
        for j, period in enumerate(periods):
            ll_df_fluxes = []
            for land_cover_scenario in land_cover_scenarios:
                ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                sim_vals_sum_CanESM = onp.nanmean(sim_vals_CanESM, axis=1)

                ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                sim_vals_sum_MPIM = onp.nanmean(sim_vals_MPIM, axis=1)

                df1 = pd.DataFrame()
                df1.loc[:, 'state_avg'] = sim_vals_sum_CanESM.flatten()
                df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
                df1.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
                df1.loc[:, 'state'] = _LABS[var_sim]

                df2 = pd.DataFrame()
                df2.loc[:, 'state_avg'] = sim_vals_sum_MPIM.flatten()
                df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
                df2.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
                df2.loc[:, 'state'] = _LABS[var_sim]

                df = pd.concat([df1, df2], ignore_index=True)
                ll_df_fluxes.append(df)

            df_fluxes = pd.concat(ll_df_fluxes, ignore_index=True)
            sns.boxplot(x="land_cover_scenario", y="state_avg", hue="climate_scenario", palette=["red", "blue"],
            data=df_fluxes, ax=axes[i, j], showfliers=False)
            axes[i, j].legend([],[], frameon=False)
            axes[0, j].set_title(f'{period}')
            axes[i, j].set_ylabel('')
            axes[i, j].set_xlabel('')
            axes[i, j].tick_params(axis='x', rotation=33)
        axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
    fig.tight_layout()
    file = base_path_figs / f"{var_sim}_boxplot.png"
    fig.savefig(file, dpi=250)
    plt.close('all')



# plot cumulated precipitation
# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
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


# # plot soil water content time series
# vars_sim = ['theta']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(sim_vals_CanESM, axis=0)
#                 sim_vals_min_CanESM = onp.nanmin(sim_vals_CanESM, axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(sim_vals_CanESM, axis=0)
#                 sim_vals_max_CanESM = onp.nanmax(sim_vals_CanESM, axis=0)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(sim_vals_MPIM, axis=0)
#                 sim_vals_min_MPIM = onp.nanmin(sim_vals_MPIM, axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(sim_vals_MPIM, axis=0)
#                 sim_vals_max_MPIM = onp.nanmax(sim_vals_MPIM, axis=0)

#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#                 axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color='red',
#                                         edgecolor=None, alpha=0.2)

#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color='blue',
#                                         edgecolor=None, alpha=0.2)
#                 axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#                 axes[-1, j].set_xlabel('Time [year]')
#                 # axes[-1, j].tick_params(axis='x', rotation=33)
#                 axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                 axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_DAILY[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}.png"
#         fig.savefig(file, dpi=250)
#         plt.close('all')


# vars_sim = ['prec']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#             sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=0).values
#             sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#             ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
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

# vars_sim = ['transp', 'evap_soil', 'q_ss']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 for x in range(675):
#                     ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                     sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=x).values
#                     sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#                     ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                     sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=x).values
#                     sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#                     axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='-', color='red', lw=1)
#                     axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#                 axes[i, j].set_ylim(0,)
#                 axes[-1, j].set_xlabel('Time [year]')
#                 # axes[-1, j].tick_params(axis='x', rotation=33)
#                 axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                 axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated_trace.png"
#         fig.savefig(file, dpi=250)
#         plt.close('all')


# load simulated tracer concentrations and water ages
land_cover_scenarios = ['grass']
dict_conc_ages = {}
for location in locations:
    dict_conc_ages[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_conc_ages[location][land_cover_scenario] = {}
        for period in periods:
            dict_conc_ages[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                output_tm_file = base_path_results / "svat_transport" / f"SVATTRANSPORT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                ds_conc_ages = xr.open_dataset(output_tm_file, engine="h5netcdf", decode_times=False)
                # assign date
                date = num2date(ds_conc_ages['Time'].values, units=f"days since {ds_conc_ages['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
                ds_conc_ages = ds_conc_ages.assign_coords(Time=("Time", date))
                dict_conc_ages[location][land_cover_scenario][period][climate_scenario] = ds_conc_ages

# # plot water ages
# vars_sim1 = [['rt10_s', 'rt50_s', 'rt90_s'], ['tt10_transp', 'tt50_transp', 'tt90_transp'], ['tt10_q_ss', 'tt50_q_ss', 'tt90_q_ss']]
# colorr = ['#fcae91', '#fb6a4a', '#cb181d']
# colorb = ['#bdd7e7', '#6baed6', '#2171b5']
# for vars_sim in vars_sim1:
#     for land_cover_scenario in land_cover_scenarios:
#             fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, period in enumerate(periods):
#                     for jj, var_sim in enumerate(vars_sim):
#                         ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                         sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                         if var_sim == 'tt90_transp':
#                             sim_vals_CanESM = onp.where(sim_vals_CanESM > 500, 500, sim_vals_CanESM)
#                         else:
#                             sim_vals_CanESM = onp.where(sim_vals_CanESM > 1490, onp.nan, sim_vals_CanESM)
#                         sim_vals_avg_CanESM = onp.mean(sim_vals_CanESM, axis=0)
#                         sim_vals_min_CanESM = onp.min(sim_vals_CanESM, axis=0)
#                         sim_vals_50_CanESM = onp.median(sim_vals_CanESM, axis=0)
#                         sim_vals_max_CanESM = onp.max(sim_vals_CanESM, axis=0)

#                         ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                         sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                         if var_sim == 'tt90_transp':
#                             sim_vals_MPIM = onp.where(sim_vals_MPIM > 500, 500, sim_vals_MPIM)
#                         else:
#                             sim_vals_MPIM = onp.where(sim_vals_MPIM > 1490, onp.nan, sim_vals_MPIM)
#                         sim_vals_avg_MPIM = onp.mean(sim_vals_MPIM, axis=0)
#                         sim_vals_min_MPIM = onp.min(sim_vals_MPIM, axis=0)
#                         sim_vals_50_MPIM = onp.median(sim_vals_MPIM, axis=0)
#                         sim_vals_max_MPIM = onp.max(sim_vals_MPIM, axis=0)

#                         axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color=colorr[jj], lw=1)
#                         axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color=colorr[jj], lw=1)
#                         axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color=colorr[jj],
#                                                 edgecolor=None, alpha=0.2)

#                         axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color=colorb[jj], lw=1)
#                         axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color=colorb[jj], lw=1)
#                         axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color=colorb[jj],
#                                                 edgecolor=None, alpha=0.2)
#                         axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#                         axes[i, j].set_ylim(0, 1500)
#                         axes[-1, j].set_xlabel('Time [year]')
#                         # axes[-1, j].tick_params(axis='x', rotation=33)
#                         axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                         axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
#                         if var_sim == 'tt90_transp':
#                             axes[i, j].set_ylim(0, 500)
#                         else:
#                             axes[i, j].set_ylim(0, 1500)
#                 axes[i, 0].set_ylabel('%s\n[days]' % (Locations[i]))
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim.split('90')[0]}{''.join(var_sim.split('90')[1:])}_{land_cover_scenario}.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')

_LABS = {'rt10': r'$RT_{10}$',
         'rt50': r'$RT_{50}$',
         'rt90': r'$RT_{90}$',
         'tt10': r'$TT_{10}$',
         'tt50': r'$TT_{50}$',
         'tt90': r'$TT_{90}$',
          }
vars_sim1 = [['rt10_s', 'rt50_s', 'rt90_s'], ['tt10_transp', 'tt50_transp', 'tt90_transp'], ['tt10_q_ss', 'tt50_q_ss', 'tt90_q_ss']]
for vars_sim in vars_sim1:
    for land_cover_scenario in land_cover_scenarios:
            fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
            for i, location in enumerate(locations):
                for j, period in enumerate(periods):
                    ll_df_ages = []
                    for jj, var_sim in enumerate(vars_sim):
                        ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
                        sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
                        if var_sim == 'tt90_transp':
                            sim_vals_CanESM = onp.where(sim_vals_CanESM > 500, 500, sim_vals_CanESM)
                        else:
                            sim_vals_CanESM = onp.where(sim_vals_CanESM > 1490, onp.nan, sim_vals_CanESM)

                        ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
                        sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
                        if var_sim == 'tt90_transp':
                            sim_vals_MPIM = onp.where(sim_vals_MPIM > 500, 500, sim_vals_MPIM)
                        else:
                            sim_vals_MPIM = onp.where(sim_vals_MPIM > 1490, onp.nan, sim_vals_MPIM)

                        df1 = pd.DataFrame()
                        df1.loc[:, 'ages'] = sim_vals_CanESM.flatten()
                        df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
                        df1.loc[:, 'land_cover_scenario'] = land_cover_scenario
                        df1.loc[:, 'age_metric'] = f"{_LABS[var_sim.split('_')[0]]}"

                        df2 = pd.DataFrame()
                        df2.loc[:, 'ages'] = sim_vals_MPIM.flatten()
                        df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
                        df2.loc[:, 'land_cover_scenario'] = land_cover_scenario
                        df2.loc[:, 'age_metric'] = f"{_LABS[var_sim.split('_')[0]]}"

                        df = pd.concat([df1, df2], ignore_index=True)
                        ll_df_ages.append(df)

                    df_ages = pd.concat(ll_df_ages, ignore_index=True)

                    sns.boxplot(x="age_metric", y="ages", hue="climate_scenario", palette=["red", "blue"],
                    data=df_ages, ax=axes[i, j], showfliers=False)
                    axes[i, j].legend([],[], frameon=False)
                    axes[0, j].set_title(f'{period}')
                    axes[i, j].set_ylabel('')
                    axes[i, j].set_xlabel('')
                    axes[i, j].set_ylim(0,)
                axes[i, 0].set_ylabel('%s\nT [days]' % (Locations[i]))
            fig.tight_layout()
            file = base_path_figs / f"{var_sim.split('90')[0]}{''.join(var_sim.split('90')[1:])}_{land_cover_scenario}_boxplot.png"
            fig.savefig(file, dpi=250)
            plt.close('all')


# # plot tracer mass of fluxes
# _Y_LABS = {'M_transp': r'$M_{TRANSP}$ [mg]',
#            'M_q_ss': r'$M_{PERC}$ [mg]',
#            }
# vars_sim = ['M_transp', 'M_q_ss']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_avg_CanESM = onp.nanmean(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_min_CanESM = onp.nanmin(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_50_CanESM = onp.nanmedian(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)
#                 sim_vals_max_CanESM = onp.nanmax(onp.cumsum(sim_vals_CanESM, axis=1), axis=0)

#                 ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_avg_MPIM = onp.nanmean(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_min_MPIM = onp.nanmin(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_50_MPIM = onp.nanmedian(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)
#                 sim_vals_max_MPIM = onp.nanmax(onp.cumsum(sim_vals_MPIM, axis=1), axis=0)

#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='--', color='red', lw=1)
#                 axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_50_CanESM, ls='-', color='red', lw=1)
#                 axes[i, j].fill_between(ds_CanESM['Time'].values, sim_vals_min_CanESM, sim_vals_max_CanESM, color='red',
#                                         edgecolor=None, alpha=0.2)

#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='--', color='blue', lw=1)
#                 axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_50_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].fill_between(ds_MPIM['Time'].values, sim_vals_min_MPIM, sim_vals_max_MPIM, color='blue',
#                                         edgecolor=None, alpha=0.2)
#                 axes[i, j].set_xlim(start_dates[j], end_dates[j])
#                 axes[i, j].set_ylim(0,)
#                 axes[-1, j].set_xlabel('Time [year]')
#                 # axes[-1, j].tick_params(axis='x', rotation=33)
#                 axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                 axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))

#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], _Y_LABS[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated.png"
#         fig.savefig(file, dpi=250)
#         plt.close('all')

# vars_sim = ['M_transp', 'M_q_ss']
# for land_cover_scenario in land_cover_scenarios:
#     for var_sim in vars_sim:
#         fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#         for i, location in enumerate(locations):
#             for j, period in enumerate(periods):
#                 for x in range(675):
#                     ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                     sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0, x=x).values
#                     sim_vals_avg_CanESM = onp.cumsum(sim_vals_CanESM)

#                     ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                     sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0, x=x).values
#                     sim_vals_avg_MPIM = onp.cumsum(sim_vals_MPIM)

#                     axes[i, j].plot(ds_CanESM['Time'].values, sim_vals_avg_CanESM, ls='-', color='red', lw=1)
#                     axes[i, j].plot(ds_MPIM['Time'].values, sim_vals_avg_MPIM, ls='-', color='blue', lw=1)
#                 axes[i, j].set_xlim(ds_MPIM['Time'].values[0], ds_MPIM['Time'].values[-1])
#                 axes[i, j].set_ylim(0,)
#                 axes[-1, j].set_xlabel('Time [year]')
#                 # axes[-1, j].tick_params(axis='x', rotation=33)
#                 axes[-1, j].xaxis.set_major_locator(mpl.dates.YearLocator(5, month = 1, day = 1))
#                 axes[-1, j].xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
#             axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], _Y_LABS[var_sim]))
#         fig.tight_layout()
#         file = base_path_figs / f"{var_sim}_{land_cover_scenario}_cumulated_trace.png"
#         fig.savefig(file, dpi=250)
#         plt.close('all')


