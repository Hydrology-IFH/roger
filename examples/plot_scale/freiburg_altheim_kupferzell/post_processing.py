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
import statsmodels.api as sm
from statsmodels.graphics.api import abline_plot
import statsmodels.tools.eval_measures as smem
from scipy import stats

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
locations = ["freiburg", "altheim", "kupferzell"]
Locations = ["Freiburg", "Altheim", "Kupferzell"]
land_cover_scenarios = ["grass", "corn", "corn_catch_crop", "crop_rotation"]
Land_cover_scenarios = ["Grass", "Corn", "Corn & catch crop", "Crop rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2005", "2040-2060", "2080-2100"]
start_dates = [datetime.date(1985, 1, 1), datetime.date(2040, 1, 1), datetime.date(2080, 1, 1)]
end_dates = [datetime.date(2004, 12, 31), datetime.date(2059, 12, 31), datetime.date(2099, 12, 31)]

_lab = {
    "q_ss": "PERC",
    "transp": "TRANSP",
    "evap_soil": "$EVAP_{soil}$",
    "theta": r"$\theta$",
    "tt10_transp": "$TT_{10-TRANSP}$",
    "tt50_transp": "$TT_{50-TRANSP}$",
    "tt90_transp": "$TT_{90-TRANSP}$",
    "tt10_q_ss": "$TT_{10-PERC}$",
    "tt50_q_ss": "$TT_{50-PERC}$",
    "tt90_q_ss": "$TT_{90-PERC}$",
    "rt10_s": "$RT_{10}$",
    "rt50_s": "$RT_{50}$",
    "rt90_s": "$RT_{90}$",
    "M_transp": "$M_{TRANSP}$",
    "M_q_ss": "$M_{PERC}$",
    "dAvg": r"$\overline{\Delta}$",
    "dIPR": r"$\Delta IPR$",
    "dSum": r"$\Delta\sum$",
}

_lab_unit1 = {
    "q_ss": "PERC [mm/day]",
    "transp": "TRANSP [mm/day]",
    "evap_soil": "$EVAP_{soil}$ [mm/day]",
    "theta": r"$\theta$ [-]",
    "tt10_transp": "$TT_{10-TRANSP}$ [days]",
    "tt50_transp": "$TT_{50-TRANSP}$ [days]",
    "tt90_transp": "$TT_{90-TRANSP}$ [days]",
    "tt10_q_ss": "$TT_{10-PERC}$ [days]",
    "tt50_q_ss": "$TT_{50-PERC}$ [days]",
    "tt90_q_ss": "$TT_{90-PERC}$ [days]",
    "rt10_s": "$RT_{10}$ [days]",
    "rt50_s": "$RT_{50}$ [days]",
    "rt90_s": "$RT_{90}$ [days]",
    "M_transp": "$M_{TRANSP}$ [mg]",
    "M_q_ss": "$M_{PERC}$ [mg]",
    "theta_ac": r"$\theta_{ac}$ [-]",
    "theta_ufc": r"$\theta_{ufc}$ [-]",
    "theta_pwp": r"$\theta_{pwp}$ [-]",
    "ks": "$k_s$ [mm/day]",
}

_lab_unit2 = {
    "q_ss": "PERC [mm]",
    "transp": "TRANSP [mm]",
    "evap_soil": "$EVAP_{soil}$ [mm]",
    "theta": r"$\theta$ [-]",
    "tt10_transp": "$TT_{10-TRANSP}$ [days]",
    "tt50_transp": "$TT_{50-TRANSP}$ [days]",
    "tt90_transp": "$TT_{90-TRANSP}$ [days]",
    "tt10_q_ss": "$TT_{10-PERC}$ [days]",
    "tt50_q_ss": "$TT_{50-PERC}$ [days]",
    "tt90_q_ss": "$TT_{90-PERC}$ [days]",
    "rt10_s": "$RT_{10}$ [days]",
    "rt50_s": "$RT_{50}$ [days]",
    "rt90_s": "$RT_{90}$ [days]",
    "M_transp": "$M_{TRANSP}$ [mg]",
    "M_q_ss": "$M_{PERC}$ [mg]",
}

# load model parameters
csv_file = base_path / "parameters.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond_soil_depth_300 = df_params.loc[:, "z_soil"].values == 300
cond_soil_depth_600 = df_params.loc[:, "z_soil"].values == 600
cond_soil_depth_900 = df_params.loc[:, "z_soil"].values == 900
cond_soil_depth = onp.copy(cond_soil_depth_300)
cond_soil_depth[:] = True
soil_depths = ["all", "shallow", "medium", "deep"]
_soil_depths = {
    "all": cond_soil_depth,
    "shallow": cond_soil_depth_300,
    "medium": cond_soil_depth_600,
    "deep": cond_soil_depth_900,
}

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
                    output_hm_file = (
                        base_path_results
                        / "svat"
                        / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
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
                    dict_fluxes_states[location][land_cover_scenario][period][climate_scenario] = ds_fluxes_states
                except KeyError:
                    print(f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc")

# load simulated tracer concentrations and water ages
dict_conc_ages = {}
for location in locations:
    dict_conc_ages[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_conc_ages[location][land_cover_scenario] = {}
        for period in periods:
            dict_conc_ages[location][land_cover_scenario][period] = {}
            for climate_scenario in climate_scenarios:
                output_tm_file = (
                    base_path_results
                    / "svat_transport"
                    / f"SVATTRANSPORT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                )
                ds_conc_ages = xr.open_dataset(output_tm_file, engine="h5netcdf", decode_times=False)
                # assign date
                date = num2date(
                    ds_conc_ages["Time"].values,
                    units=f"days since {ds_conc_ages['Time'].attrs['time_origin']}",
                    calendar="standard",
                    only_use_cftime_datetimes=False,
                )
                ds_conc_ages = ds_conc_ages.assign_coords(Time=("Time", date))
                dict_conc_ages[location][land_cover_scenario][period][climate_scenario] = ds_conc_ages

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

# _LABS = {'evap_soil': r'$EVAP_{soil}$',
#          'transp': r'$TRANSP$',
#          'q_ss': r'$PERC$',
#          'theta': r'$\theta$',
#          'grass': 'Grass',
#          'corn': 'Corn',
#          'corn_catch_crop': 'Corn + CC',
#          'crop_rotation': 'Crop Rot.',
#         }
# vars_sim = ['evap_soil', 'transp', 'q_ss']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey='row', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ll_df_fluxes = []
#             for land_cover_scenario in land_cover_scenarios:
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_sum_CanESM = onp.sum(sim_vals_CanESM, axis=1)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_sum_MPIM = onp.sum(sim_vals_MPIM, axis=1)

#                 df1 = pd.DataFrame()
#                 df1.loc[:, 'flux_sum'] = sim_vals_sum_CanESM.flatten()
#                 df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
#                 df1.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
#                 df1.loc[:, 'flux'] = _LABS[var_sim]

#                 df2 = pd.DataFrame()
#                 df2.loc[:, 'flux_sum'] = sim_vals_sum_MPIM.flatten()
#                 df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                 df2.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
#                 df2.loc[:, 'flux'] = _LABS[var_sim]

#                 df = pd.concat([df1, df2], ignore_index=True)
#                 ll_df_fluxes.append(df)

#             df_fluxes = pd.concat(ll_df_fluxes, ignore_index=True)
#             sns.boxplot(x="land_cover_scenario", y="flux_sum", hue="climate_scenario", palette=["red", "blue"],
#             data=df_fluxes, ax=axes[i, j], showfliers=False)
#             axes[i, j].legend([],[], frameon=False)
#             axes[0, j].set_title(f'{period}')
#             axes[i, j].set_ylabel('')
#             axes[i, j].set_xlabel('')
#             axes[i, j].tick_params(axis='x', rotation=33)
#         axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_boxplot.png"
#     fig.savefig(file, dpi=250)
#     plt.close('all')


# vars_sim = ['theta']
# for var_sim in vars_sim:
#     fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey='row', figsize=(6, 4.5))
#     for i, location in enumerate(locations):
#         for j, period in enumerate(periods):
#             ll_df_fluxes = []
#             for land_cover_scenario in land_cover_scenarios:
#                 ds_CanESM = dict_fluxes_states[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                 sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                 sim_vals_sum_CanESM = onp.nanmean(sim_vals_CanESM, axis=1)

#                 ds_MPIM = dict_fluxes_states[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                 sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                 sim_vals_sum_MPIM = onp.nanmean(sim_vals_MPIM, axis=1)

#                 df1 = pd.DataFrame()
#                 df1.loc[:, 'state_avg'] = sim_vals_sum_CanESM.flatten()
#                 df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
#                 df1.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
#                 df1.loc[:, 'state'] = _LABS[var_sim]

#                 df2 = pd.DataFrame()
#                 df2.loc[:, 'state_avg'] = sim_vals_sum_MPIM.flatten()
#                 df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                 df2.loc[:, 'land_cover_scenario'] = _LABS[land_cover_scenario]
#                 df2.loc[:, 'state'] = _LABS[var_sim]

#                 df = pd.concat([df1, df2], ignore_index=True)
#                 ll_df_fluxes.append(df)

#             df_fluxes = pd.concat(ll_df_fluxes, ignore_index=True)
#             sns.boxplot(x="land_cover_scenario", y="state_avg", hue="climate_scenario", palette=["red", "blue"],
#             data=df_fluxes, ax=axes[i, j], showfliers=False)
#             axes[i, j].legend([],[], frameon=False)
#             axes[0, j].set_title(f'{period}')
#             axes[i, j].set_ylabel('')
#             axes[i, j].set_xlabel('')
#             axes[i, j].tick_params(axis='x', rotation=33)
#         axes[i, 0].set_ylabel('%s\n%s' % (Locations[i], labs._Y_LABS_CUM[var_sim]))
#     fig.tight_layout()
#     file = base_path_figs / f"{var_sim}_boxplot.png"
#     fig.savefig(file, dpi=250)
#     plt.close('all')

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

# _LABS = {'rt10': r'$RT_{10}$',
#          'rt50': r'$RT_{50}$',
#          'rt90': r'$RT_{90}$',
#          'tt10': r'$TT_{10}$',
#          'tt50': r'$TT_{50}$',
#          'tt90': r'$TT_{90}$',
#           }
# vars_sim1 = [['rt10_s', 'rt50_s', 'rt90_s'], ['tt10_transp', 'tt50_transp', 'tt90_transp'], ['tt10_q_ss', 'tt50_q_ss', 'tt90_q_ss']]
# for vars_sim in vars_sim1:
#     for land_cover_scenario in land_cover_scenarios:
#             fig, axes = plt.subplots(len(locations), len(periods), sharex='col', sharey=True, figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, period in enumerate(periods):
#                     ll_df_ages = []
#                     for jj, var_sim in enumerate(vars_sim):
#                         ds_CanESM = dict_conc_ages[location][land_cover_scenario][period]['CCCma-CanESM2_CCLM4-8-17']
#                         sim_vals_CanESM = ds_CanESM[var_sim].isel(y=0).values
#                         if var_sim == 'tt90_transp':
#                             sim_vals_CanESM = onp.where(sim_vals_CanESM > 500, 500, sim_vals_CanESM)
#                         else:
#                             sim_vals_CanESM = onp.where(sim_vals_CanESM > 1490, onp.nan, sim_vals_CanESM)

#                         ds_MPIM = dict_conc_ages[location][land_cover_scenario][period]['MPI-M-MPI-ESM-LR_RCA4']
#                         sim_vals_MPIM = ds_MPIM[var_sim].isel(y=0).values
#                         if var_sim == 'tt90_transp':
#                             sim_vals_MPIM = onp.where(sim_vals_MPIM > 500, 500, sim_vals_MPIM)
#                         else:
#                             sim_vals_MPIM = onp.where(sim_vals_MPIM > 1490, onp.nan, sim_vals_MPIM)

#                         df1 = pd.DataFrame()
#                         df1.loc[:, 'ages'] = sim_vals_CanESM.flatten()
#                         df1.loc[:, 'climate_scenario'] = 'CCCma-CanESM2_CCLM4-8-17'
#                         df1.loc[:, 'land_cover_scenario'] = land_cover_scenario
#                         df1.loc[:, 'age_metric'] = f"{_LABS[var_sim.split('_')[0]]}"

#                         df2 = pd.DataFrame()
#                         df2.loc[:, 'ages'] = sim_vals_MPIM.flatten()
#                         df2.loc[:, 'climate_scenario'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                         df2.loc[:, 'land_cover_scenario'] = land_cover_scenario
#                         df2.loc[:, 'age_metric'] = f"{_LABS[var_sim.split('_')[0]]}"

#                         df = pd.concat([df1, df2], ignore_index=True)
#                         ll_df_ages.append(df)

#                     df_ages = pd.concat(ll_df_ages, ignore_index=True)

#                     sns.boxplot(x="age_metric", y="ages", hue="climate_scenario", palette=["red", "blue"],
#                     data=df_ages, ax=axes[i, j], showfliers=False)
#                     axes[i, j].legend([],[], frameon=False)
#                     axes[0, j].set_title(f'{period}')
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                     axes[i, j].set_ylim(0,)
#                 axes[i, 0].set_ylabel('%s\nT [days]' % (Locations[i]))
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim.split('90')[0]}{''.join(var_sim.split('90')[1:])}_{land_cover_scenario}_boxplot.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')


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

# calculate mean and percentiles
dict_statistics = {}
dict_deltas = {}
vars_sim = ["transp", "q_ss"]
for location in locations:
    dict_statistics[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_statistics[location][land_cover_scenario] = {}
        for climate_scenario in climate_scenarios:
            dict_statistics[location][land_cover_scenario][climate_scenario] = {}
            for period in periods:
                dict_statistics[location][land_cover_scenario][climate_scenario][period] = {}
                ds = dict_fluxes_states[location][land_cover_scenario][period][climate_scenario]
                for var_sim in vars_sim:
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                    sim_vals = ds[var_sim].isel(y=0).values
                    df = pd.DataFrame(
                        index=range(sim_vals.shape[0]), columns=["Avg", "Sum", "p10", "p50", "p90", "IPR"]
                    )
                    df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                    df.loc[:, "Sum"] = onp.nansum(sim_vals, axis=-1)
                    # df.loc[:, 'p10'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.1, axis=-1)
                    # df.loc[:, 'p50'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.5, axis=-1)
                    # df.loc[:, 'p90'] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.9, axis=-1)
                    df.loc[:, "p10"] = onp.quantile(sim_vals, 0.1, axis=-1)
                    df.loc[:, "p50"] = onp.quantile(sim_vals, 0.5, axis=-1)
                    df.loc[:, "p90"] = onp.quantile(sim_vals, 0.9, axis=-1)
                    df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

vars_sim = ["theta"]
for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period in periods:
                ds = dict_fluxes_states[location][land_cover_scenario][period][climate_scenario]
                for var_sim in vars_sim:
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                    sim_vals = ds[var_sim].isel(y=0).values
                    df = pd.DataFrame(index=range(sim_vals.shape[0]), columns=["Avg", "p10", "p50", "p90", "IPR"])
                    df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                    df.loc[:, "p10"] = onp.quantile(sim_vals, 0.1, axis=-1)
                    df.loc[:, "p50"] = onp.quantile(sim_vals, 0.5, axis=-1)
                    df.loc[:, "p90"] = onp.quantile(sim_vals, 0.9, axis=-1)
                    df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

vars_sim = [
    "rt10_s",
    "rt50_s",
    "rt90_s",
    "tt10_transp",
    "tt50_transp",
    "tt90_transp",
    "tt10_q_ss",
    "tt50_q_ss",
    "tt90_q_ss",
]
vars_sim = ["rt50_s", "tt50_q_ss", "tt50_transp"]
for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period in periods:
                ds = dict_conc_ages[location][land_cover_scenario][period][climate_scenario]
                for var_sim in vars_sim:
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                    sim_vals = ds[var_sim].isel(y=0).values
                    df = pd.DataFrame(index=range(sim_vals.shape[0]), columns=["Avg", "p10", "p50", "p90", "IPR"])
                    df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                    # TODO: calculate flux-weighted average
                    df.loc[:, "p10"] = onp.nanquantile(sim_vals, 0.1, axis=-1)
                    df.loc[:, "p50"] = onp.nanquantile(sim_vals, 0.5, axis=-1)
                    df.loc[:, "p90"] = onp.nanquantile(sim_vals, 0.9, axis=-1)
                    df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

vars_sim = ["M_q_ss", "M_transp"]
for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period in periods:
                ds = dict_conc_ages[location][land_cover_scenario][period][climate_scenario]
                for var_sim in vars_sim:
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = {}
                    sim_vals = ds[var_sim].isel(y=0).values
                    df = pd.DataFrame(
                        index=range(sim_vals.shape[0]), columns=["Avg", "Sum", "p10", "p50", "p90", "IPR"]
                    )
                    df.loc[:, "Avg"] = onp.nanmean(sim_vals, axis=-1)
                    df.loc[:, "Sum"] = onp.nansum(sim_vals, axis=-1)
                    df.loc[:, "p10"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.1, axis=-1)
                    df.loc[:, "p50"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.5, axis=-1)
                    df.loc[:, "p90"] = onp.quantile(onp.cumsum(sim_vals, axis=-1), 0.9, axis=-1)
                    df.loc[:, "IPR"] = df.loc[:, "p90"] - df.loc[:, "p10"]
                    dict_statistics[location][land_cover_scenario][climate_scenario][period][var_sim] = df

# calculate deltas of mean and interpercentile range for near future and far future
vars_sim = ["transp", "q_ss", "M_q_ss", "M_transp"]
for location in locations:
    dict_deltas[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_deltas[location][land_cover_scenario] = {}
        for climate_scenario in climate_scenarios:
            dict_deltas[location][land_cover_scenario][climate_scenario] = {}
            for var_sim in vars_sim:
                dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = {}
                df_statistics_ref = dict_statistics[location][land_cover_scenario][climate_scenario]["1985-2005"][
                    var_sim
                ]
                df_statistics_nf = dict_statistics[location][land_cover_scenario][climate_scenario]["2040-2060"][
                    var_sim
                ]
                df_statistics_ff = dict_statistics[location][land_cover_scenario][climate_scenario]["2080-2100"][
                    var_sim
                ]
                df_deltas = pd.DataFrame(
                    index=df_statistics_ref.index,
                    columns=["dAvg_nf", "dSum_nf", "dIPR_nf", "dAvg_ff", "dSum_ff", "dIPR_ff"],
                )
                df_deltas.loc[:, "dAvg_nf"] = (
                    (df_statistics_nf["Avg"].values - df_statistics_ref["Avg"].values) / df_statistics_ref["Avg"].values
                ) * 100
                df_deltas.loc[:, "dAvg_ff"] = (
                    (df_statistics_ff["Avg"].values - df_statistics_ref["Avg"].values) / df_statistics_ref["Avg"].values
                ) * 100
                df_deltas.loc[:, "dSum_nf"] = (
                    (df_statistics_nf["Sum"].values - df_statistics_ref["Sum"].values) / df_statistics_ref["Sum"].values
                ) * 100
                df_deltas.loc[:, "dSum_ff"] = (
                    (df_statistics_ff["Sum"].values - df_statistics_ref["Sum"].values) / df_statistics_ref["Sum"].values
                ) * 100
                df_deltas.loc[:, "dIPR_nf"] = (
                    (df_statistics_nf["IPR"].values - df_statistics_ref["IPR"].values) / df_statistics_ref["IPR"].values
                ) * 100
                df_deltas.loc[:, "dIPR_ff"] = (
                    (df_statistics_ff["IPR"].values - df_statistics_ref["IPR"].values) / df_statistics_ref["IPR"].values
                ) * 100
                dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = df_deltas

vars_sim = ["theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for var_sim in vars_sim:
                dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = {}
                df_statistics_ref = dict_statistics[location][land_cover_scenario][climate_scenario]["1985-2005"][
                    var_sim
                ]
                df_statistics_nf = dict_statistics[location][land_cover_scenario][climate_scenario]["2040-2060"][
                    var_sim
                ]
                df_statistics_ff = dict_statistics[location][land_cover_scenario][climate_scenario]["2080-2100"][
                    var_sim
                ]
                df_deltas = pd.DataFrame(
                    index=df_statistics_ref.index, columns=["dAvg_nf", "dIPR_nf", "dAvg_ff", "dIPR_ff"]
                )
                df_deltas.loc[:, "dAvg_nf"] = (
                    (df_statistics_nf["Avg"].values - df_statistics_ref["Avg"].values) / df_statistics_ref["Avg"].values
                ) * 100
                df_deltas.loc[:, "dAvg_ff"] = (
                    (df_statistics_ff["Avg"].values - df_statistics_ref["Avg"].values) / df_statistics_ref["Avg"].values
                ) * 100
                df_deltas.loc[:, "dIPR_nf"] = (
                    (df_statistics_nf["IPR"].values - df_statistics_ref["IPR"].values) / df_statistics_ref["IPR"].values
                ) * 100
                df_deltas.loc[:, "dIPR_ff"] = (
                    (df_statistics_ff["IPR"].values - df_statistics_ref["IPR"].values) / df_statistics_ref["IPR"].values
                ) * 100
                dict_deltas[location][land_cover_scenario][climate_scenario][var_sim] = df_deltas

# vars_sim = ['transp', 'q_ss', 'M_q_ss', 'M_transp']
# deltas = ['dSum', 'dIPR']
# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim].loc[:, ['dSum_nf', 'dIPR_nf']]
#                     df_deltas_canesm_nf.columns = deltas
#                     df_deltas_canesm_nf.loc[:, 'Period'] = 'NF/Ref'
#                     df_deltas_canesm_nf.loc[:, 'Climate model'] = 'CCCma-CanESM2_CCLM4-8-17'
#                     df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim].loc[:, ['dSum_ff', 'dIPR_ff']]
#                     df_deltas_canesm_ff.columns = deltas
#                     df_deltas_canesm_ff.loc[:, 'Period'] = 'FF/Ref'
#                     df_deltas_canesm_ff.loc[:, 'Climate model'] = 'CCCma-CanESM2_CCLM4-8-17'

#                     df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim].loc[:, ['dSum_nf', 'dIPR_nf']]
#                     df_deltas_mpiesm_nf.columns = deltas
#                     df_deltas_mpiesm_nf.loc[:, 'Period'] = 'NF/Ref'
#                     df_deltas_mpiesm_nf.loc[:, 'Climate model'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                     df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim].loc[:, ['dSum_ff', 'dIPR_ff']]
#                     df_deltas_mpiesm_ff.columns = deltas
#                     df_deltas_mpiesm_ff.loc[:, 'Period'] = 'FF/Ref'
#                     df_deltas_mpiesm_ff.loc[:, 'Climate model'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                     df_deltas = pd.concat([df_deltas_canesm_nf.loc[cond, :], df_deltas_canesm_ff.loc[cond, :], df_deltas_mpiesm_nf.loc[cond, :], df_deltas_mpiesm_ff.loc[cond, :]], ignore_index=True)
#                     df_deltas_long = pd.melt(df_deltas, id_vars=['Period', 'Climate model'], value_vars=[delta], ignore_index=False)
#                     sns.boxplot(x="Period", y="value", hue="Climate model", palette=["red", "blue"],
#                                 data=df_deltas_long, ax=axes[i, j], showfliers=False)
#                     axes[i, j].legend([],[], frameon=False)
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                     axes[i, j].tick_params(axis='x', rotation=33)
#                 axes[i, 0].set_ylabel(f'{Locations[i]}\n{_lab[delta]} {_lab[var_sim]} [%]')
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim}_{delta}_{soil_depth}_boxplot.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')

# vars_sim = ['transp', 'q_ss', 'theta', 'rt50_s', 'tt50_q_ss', 'tt50_transp', 'M_q_ss', 'M_transp']
# deltas = ['dAvg', 'dIPR']
# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     df_deltas_canesm_nf = dict_deltas[location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim].loc[:, ['dAvg_nf', 'dIPR_nf']]
#                     df_deltas_canesm_nf.columns = deltas
#                     df_deltas_canesm_nf.loc[:, 'Period'] = 'NF/Ref'
#                     df_deltas_canesm_nf.loc[:, 'Climate model'] = 'CCCma-CanESM2_CCLM4-8-17'
#                     df_deltas_canesm_ff = dict_deltas[location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim].loc[:, ['dAvg_ff', 'dIPR_ff']]
#                     df_deltas_canesm_ff.columns = deltas
#                     df_deltas_canesm_ff.loc[:, 'Period'] = 'FF/Ref'
#                     df_deltas_canesm_ff.loc[:, 'Climate model'] = 'CCCma-CanESM2_CCLM4-8-17'

#                     df_deltas_mpiesm_nf = dict_deltas[location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim].loc[:, ['dAvg_nf', 'dIPR_nf']]
#                     df_deltas_mpiesm_nf.columns = deltas
#                     df_deltas_mpiesm_nf.loc[:, 'Period'] = 'NF/Ref'
#                     df_deltas_mpiesm_nf.loc[:, 'Climate model'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                     df_deltas_mpiesm_ff = dict_deltas[location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim].loc[:, ['dAvg_ff', 'dIPR_ff']]
#                     df_deltas_mpiesm_ff.columns = deltas
#                     df_deltas_mpiesm_ff.loc[:, 'Period'] = 'FF/Ref'
#                     df_deltas_mpiesm_ff.loc[:, 'Climate model'] = 'MPI-M-MPI-ESM-LR_RCA4'
#                     df_deltas = pd.concat([df_deltas_canesm_nf.loc[cond, :], df_deltas_canesm_ff.loc[cond, :], df_deltas_mpiesm_nf.loc[cond, :], df_deltas_mpiesm_ff.loc[cond, :]], ignore_index=True)
#                     df_deltas_long = pd.melt(df_deltas, id_vars=['Period', 'Climate model'], value_vars=[delta], ignore_index=False)
#                     sns.boxplot(x="Period", y="value", hue="Climate model", palette=["red", "blue"],
#                                 data=df_deltas_long, ax=axes[i, j], showfliers=False)
#                     axes[i, j].legend([],[], frameon=False)
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                     axes[i, j].tick_params(axis='x', rotation=33)
#                 axes[i, 0].set_ylabel(f'{Locations[i]}\n{_lab[delta]} {_lab[var_sim]} [%]')
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim}_{delta}_{soil_depth}_boxplot.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')

# # plot distributions of single sites
# vars_sim = ['q_ss', 'transp']
# for var_sim in vars_sim:
#     for climate_scenario in climate_scenarios:
#         for x in [0, 226, 451]:
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     ds_ref = dict_fluxes_states[location][land_cover_scenario]['1985-2005'][climate_scenario]
#                     ds_nf = dict_fluxes_states[location][land_cover_scenario]['2040-2060'][climate_scenario]
#                     ds_ff = dict_fluxes_states[location][land_cover_scenario]['2080-2100'][climate_scenario]
#                     sim_vals_ref = ds_ref[var_sim].isel(x=x, y=0).values.flatten()
#                     sim_vals_nf = ds_nf[var_sim].isel(x=x, y=0).values.flatten()
#                     sim_vals_ff = ds_ff[var_sim].isel(x=x, y=0).values.flatten()
#                     sns.kdeplot(data=sim_vals_ref, ax=axes[i, j], fill=False, color='grey', lw=2, clip=(0,25))
#                     sns.kdeplot(data=sim_vals_nf, ax=axes[i, j], fill=False, color='#9e9ac8', lw=1.5, ls='-.', clip=(0,25))
#                     sns.kdeplot(data=sim_vals_ff, ax=axes[i, j], fill=False, color='#3f007d', lw=1, ls='--', clip=(0,25))
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[i, j].set_xlim(0, )
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                     axes[i, j].tick_params(axis='x', rotation=33)
#                     axes[-1, j].set_xlabel(f'{_lab_unit2[var_sim]}')
#                 axes[i, 0].set_ylabel('Density [-]')
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim}_kde_{climate_scenario}_{x}.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')

# vars_sim = ['theta']
# for var_sim in vars_sim:
#     for climate_scenario in climate_scenarios:
#         for x in [0, 226, 451]:
#             fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex=True, sharey=True, figsize=(6, 4.5))
#             for i, location in enumerate(locations):
#                 for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                     ds_ref = dict_fluxes_states[location][land_cover_scenario]['1985-2005'][climate_scenario]
#                     ds_nf = dict_fluxes_states[location][land_cover_scenario]['2040-2060'][climate_scenario]
#                     ds_ff = dict_fluxes_states[location][land_cover_scenario]['2080-2100'][climate_scenario]
#                     sim_vals_ref = ds_ref[var_sim].isel(x=x, y=0).values.flatten()
#                     sim_vals_nf = ds_nf[var_sim].isel(x=x, y=0).values.flatten()
#                     sim_vals_ff = ds_ff[var_sim].isel(x=x, y=0).values.flatten()
#                     sns.kdeplot(data=sim_vals_ref, ax=axes[i, j], fill=False, color='grey', lw=2, clip=(0.1, 0.7))
#                     sns.kdeplot(data=sim_vals_nf, ax=axes[i, j], fill=False, color='#9e9ac8', lw=1.5, ls='-.', clip=(0.1, 0.7))
#                     sns.kdeplot(data=sim_vals_ff, ax=axes[i, j], fill=False, color='#3f007d', lw=1, ls='--', clip=(0.1, 0.7))
#                     axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                     axes[i, j].set_ylabel('')
#                     axes[i, j].set_xlabel('')
#                     axes[i, j].tick_params(axis='x', rotation=33)
#                     axes[-1, j].set_xlabel(f'{_lab_unit2[var_sim]}')
#                 axes[i, 0].set_ylabel('Density [-]')
#             fig.tight_layout()
#             file = base_path_figs / f"{var_sim}_kde_{climate_scenario}_{x}.png"
#             fig.savefig(file, dpi=250)
#             plt.close('all')

# identify impacted sites based on GLM
# for param in ['theta_pwp', 'theta_ufc', 'theta_ac', 'ks']:
#     vals = df_params.loc[:, param].values
#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.hist(vals, bins=25, color='black')
#     ax.set_xlabel(f'{_lab_unit1[param]}')
#     fig.tight_layout()
#     file = base_path_figs / f"{param}_hist.png"
#     fig.savefig(file, dpi=250)

dict_glm = {}
vars_sim = ["transp", "q_ss", "theta", "rt50_s", "tt50_q_ss", "tt50_transp"]
deltas = ["dAvg", "dIPR"]
vars_sim = ["q_ss"]
deltas = ["dAvg"]
for location in locations:
    dict_glm[location] = {}
    for land_cover_scenario in land_cover_scenarios:
        dict_glm[location][land_cover_scenario] = {}
        for climate_scenario in climate_scenarios:
            dict_glm[location][land_cover_scenario][climate_scenario] = {}
            for var_sim in vars_sim:
                dict_glm[location][land_cover_scenario][climate_scenario][var_sim] = {}
                for delta in deltas:
                    dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta] = {}
                    for soil_depth in soil_depths:
                        dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth] = {}
                        cond = _soil_depths[soil_depth]
                        for future in ["nf", "ff"]:
                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ] = {}
                            y = (
                                dict_deltas[location][land_cover_scenario][climate_scenario][var_sim]
                                .loc[cond, f"{delta}_{future}"]
                                .to_frame()
                            )
                            y = pd.DataFrame(data=y, dtype=onp.float64)
                            x = df_params.loc[cond, "theta_pwp":]
                            x = sm.add_constant(x)
                            # standardize the parameters
                            x_std = x.copy()
                            for param in ["theta_pwp", "theta_ufc", "theta_ac", "ks"]:
                                x_std.loc[:, param] = (x_std.loc[:, param] - x_std.loc[:, param].mean()) / x_std.loc[
                                    :, param
                                ].std()

                            # fit the GLM model
                            glm = sm.GLM(y, x_std, family=sm.families.Gaussian())
                            res = glm.fit()
                            nobs = res.nobs
                            yhat = res.mu

                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ]["params"] = res.params
                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ]["RMSE"] = smem.rmse(y, yhat)
                            ll_values = glm.loglike(res.params, scale=res.scale)
                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ]["AIC"] = smem.aic(ll_values, nobs, res.params.shape[0])
                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ]["AICC"] = smem.aicc(ll_values, nobs, res.params.shape[0])
                            dict_glm[location][land_cover_scenario][climate_scenario][var_sim][delta][soil_depth][
                                future
                            ]["BIC"] = smem.bic(ll_values, nobs, res.params.shape[0])

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(yhat, y, color='black', s=4)
                            # line_fit = sm.OLS(y, sm.add_constant(yhat, prepend=True)).fit()
                            # abline_plot(model_results=line_fit, ax=ax, color='black')
                            # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (RoGeR) [%]')
                            # ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM) [%]')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_line_fit.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(yhat, res.resid_pearson, color='black', s=4)
                            # ax.set_ylabel(f'{_lab_unit1[var_sim]} (RoGeR) [%]')
                            # ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM)')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(x.loc[:, 'theta_ufc'].values, y, color='black', s=4)
                            # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                            # ax.set_xlabel(f'{_lab_unit1["theta_ufc"]}')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ufc.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(x.loc[:, 'theta_ac'].values, y, color='black', s=4)
                            # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                            # ax.set_xlabel(f'{_lab_unit1["theta_ac"]}')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_ac.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(x.loc[:, 'theta_pwp'].values, y, color='black', s=4)
                            # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                            # ax.set_xlabel(f'{_lab_unit1["theta_pwp"]}')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_theta_pwp.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # ax.scatter(x.loc[:, 'ks'].values, y, color='black', s=4)
                            # ax.set_ylabel(f'{_lab[delta]}{_lab_unit1[var_sim]} [%]')
                            # ax.set_xlabel(f'{_lab_unit1["ks"]}')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_ks.png"
                            # fig.savefig(file, dpi=250)

                            # fig, ax = plt.subplots(figsize=(3, 3))
                            # resid = res.resid_deviance.copy()
                            # resid_std = stats.zscore(resid)
                            # ax.hist(resid_std, bins=25, color='black')
                            # ax.set_xlabel(f'{_lab[delta]}{_lab_unit1[var_sim]} (GLM)')
                            # fig.tight_layout()
                            # file = base_path_figs / f"{var_sim}_{delta}_{land_cover_scenario}_{climate_scenario}_{soil_depth}_{future}_residuals_hist.png"
                            # fig.savefig(file, dpi=250)
                            # plt.close('all')

import pickle
# Store data (serialize)
with open(base_path_figs / 'glm_results.pkl', 'wb') as handle:
    pickle.dump(dict_glm, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Load data (deserialize)
# with open(base_path_figs / 'glm_results.pkl', 'rb') as handle:
#     dict_glm = pickle.load(handle)

# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             for future in ["nf", "ff"]:
#                 fig, axes = plt.subplots(
#                     len(locations), len(land_cover_scenarios), sharex="col", sharey="row", figsize=(6, 4.5)
#                 )
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         values = dict_glm[location][land_cover_scenario]["CCCma-CanESM2_CCLM4-8-17"][var_sim][delta][
#                             soil_depth
#                         ][future]["params"][1:]
#                         df_params_canesm = pd.DataFrame(index=range(len(values)), columns=["value", "Parameter"])
#                         df_params_canesm.loc[:, "value"] = values
#                         df_params_canesm.loc[:, "Parameter"] = [
#                             r"$\theta_{pwp}$",
#                             r"$\theta_{ufc}$",
#                             r"$\theta_{ac}$",
#                             r"$k_s$",
#                         ]
#                         df_params_canesm.loc[:, "Climate model"] = "CCCma-CanESM2_CCLM4-8-17"

#                         values = dict_glm[location][land_cover_scenario]["MPI-M-MPI-ESM-LR_RCA4"][var_sim][delta][
#                             soil_depth
#                         ][future]["params"][1:]
#                         df_params_mpiesm = pd.DataFrame(index=range(len(values)), columns=["value", "Parameter"])
#                         df_params_mpiesm.loc[:, "value"] = values
#                         df_params_mpiesm.loc[:, "Parameter"] = [
#                             r"$\theta_{pwp}$",
#                             r"$\theta_{ufc}$",
#                             r"$\theta_{ac}$",
#                             r"$k_s$",
#                         ]
#                         df_params_mpiesm.loc[:, "Climate model"] = "MPI-M-MPI-ESM-LR_RCA4"

#                         df_params = pd.concat([df_params_canesm, df_params_mpiesm], ignore_index=True)
#                         sns.barplot(
#                             x="Parameter",
#                             y="value",
#                             hue="Climate model",
#                             palette=["red", "blue"],
#                             data=df_params,
#                             ax=axes[i, j],
#                             errorbar=None
#                         )
#                         axes[i, j].legend([], [], frameon=False)
#                         axes[0, j].set_title(f"{Land_cover_scenarios[j]}")
#                         axes[i, j].set_ylabel("")
#                         axes[i, j].set_xlabel("")
#                         axes[i, j].tick_params(axis="x", rotation=33)
#                     axes[i, 0].set_ylabel(f"{Locations[i]}\n{_lab[delta]} {_lab[var_sim]} [%]")
#                 fig.tight_layout()
#                 file = base_path_figs / f"{var_sim}_{delta}_{soil_depth}_{future}_barplot.png"
#                 fig.savefig(file, dpi=250)
#                 plt.close("all")


# norm = mpl.colors.Normalize(vmin=-1, vmax=1)
# for var_sim in vars_sim:
#     for delta in deltas:
#         for soil_depth in soil_depths:
#             cond = _soil_depths[soil_depth]
#             for future in ['nf', 'ff']:
#                 fig, axes = plt.subplots(len(locations), len(land_cover_scenarios), sharex='col', sharey='row', figsize=(6, 4.5))
#                 # axes for colorbar
#                 axl = fig.add_axes([0.88, 0.3, 0.02, 0.58])
#                 cb1 = mpl.colorbar.ColorbarBase(axl, cmap='PuOr', norm=norm,
#                                                 orientation='vertical',
#                                                 ticks=[-1, -0.5, 0, 0.5, 1])
#                 cb1.ax.invert_yaxis()
#                 cb1.set_label(r'[-]')
#                 for i, location in enumerate(locations):
#                     for j, land_cover_scenario in enumerate(land_cover_scenarios):
#                         params_cm = [r'$\theta_{pwp}$ (CanESM)', r'$\theta_{pwp}$ (MPIM)', r'$\theta_{ufc}$ (CanESM)', r'$\theta_{ufc}$ (MPIM)', r'$\theta_{ac}$ (CanESM)', r'$\theta_{ac}$ (MPIM)', r'$k_s$ (CanESM)', r'$k_s$ (MPIM)']
#                         params_canesm = [r'$\theta_{pwp}$ (CanESM)', r'$\theta_{ufc}$ (CanESM)', r'$\theta_{ac}$ (CanESM)', r'$k_s$ (CanESM)']
#                         params_mpim = [r'$\theta_{pwp}$ (MPIM)', r'$\theta_{ufc}$ (MPIM)', r'$\theta_{ac}$ (MPIM)', r'$k_s$ (MPIM)']
#                         df_params = pd.DataFrame(columns=params_cm)
#                         for ii, var_sim in enumerate(vars_sim):
#                             df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_canesm] = dict_glm[location][land_cover_scenario]['CCCma-CanESM2_CCLM4-8-17'][var_sim][delta][soil_depth][future]['params'][1:]
#                             df_params.loc[f'{_lab[delta]}{_lab[var_sim]}', params_mpim] = dict_glm[location][land_cover_scenario]['MPI-M-MPI-ESM-LR_RCA4'][var_sim][delta][soil_depth][future]['params'][1:]
#                         sns.heatmap(df_params, ax=ax, vmin=-1, vmax=1, cmap='PuOr', cbar=False)
#                         axes[0, j].set_title(f'{Land_cover_scenarios[j]}')
#                         axes[-1, j].set_yticklabels(params_cm, rotation=33)
#                         axes[i, j].set_ylabel('')
#                         axes[i, j].set_xlabel('')
#                     axes[i, 0].set_ylabel(f'{Locations[i]}')
#                     axes[i, 0].set_yticklabels(df_params.index.tolist(), rotation=0)
#                 fig.tight_layout()
#                 file = base_path_figs / f"{var_sim}_{delta}_{soil_depth}_{future}_heatmap.png"
#                 fig.savefig(file, dpi=250)
#                 plt.close('all')
