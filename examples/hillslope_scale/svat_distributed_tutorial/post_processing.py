import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import numpy as onp
import copy
from PIL import Image
import imageio
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
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load hydrologic simulations
states_hm_file = base_path / "svat_distributed" / "states_hm.nc"
ds_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
# assign date
days_hm = (ds_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_hm = num2date(days_hm, units=f"days since {ds_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_hm = ds_hm.assign_coords(Time=("Time", date_hm))

# load transport simulations
states_tm_file = base_path / "svat_oxygen18_distributed" / "states_power.nc"
ds_tm = xr.open_dataset(states_tm_file, engine="h5netcdf", decode_times=False)
# assign date
days_tm = ds_tm['Time'].values
date_tm = num2date(days_tm, units="days since 2019-10-31", calendar='standard', only_use_cftime_datetimes=False)
ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

# plot spatially distributed soil moisture and median travel time of percolation at different dates
fig, axes = plt.subplots(2, 2, figsize=(3, 4))
axes[0, 0].imshow(ds_hm['theta'].isel(Time=770).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[0, 0].set_xticks(onp.arange(-.5, 11, 5))
axes[0, 0].set_yticks(onp.arange(-.5, 23, 5))
axes[0, 0].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[0, 0].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[0, 0].set_xlabel('[m]')
axes[0, 0].set_ylabel('[m]')
axes[0, 0].set_title(str(ds_hm['Time'].values[770]).split('T')[0])
axes[0, 1].imshow(ds_hm['theta'].isel(Time=1006).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[0, 1].set_xticks(onp.arange(-.5, 11, 5))
axes[0, 1].set_yticks(onp.arange(-.5, 23, 5))
axes[0, 1].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[0, 1].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[0, 1].set_xlabel('[m]')
axes[0, 1].set_title(str(ds_hm['Time'].values[1006]).split('T')[0])
cmap = copy.copy(plt.cm.get_cmap('Blues'))
norm = mpl.colors.Normalize(vmin=0.2, vmax=0.4)
axl1 = fig.add_axes([0.8, 0.62, 0.02, 0.31])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.2, 0.3, 0.4])
cb1.ax.set_yticklabels(['<0.2', '0.3', '>0.4'])
cb1.set_label(r'$\theta$ [-]')

axes[1, 0].imshow(ds_tm['tt50_q_ss'].isel(Time=770).values.T, origin="lower", cmap='Purples_r', vmin=100, vmax=900)
axes[1, 0].set_xticks(onp.arange(-.5, 11, 5))
axes[1, 0].set_yticks(onp.arange(-.5, 23, 5))
axes[1, 0].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[1, 0].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[1, 0].set_xlabel('[m]')
axes[1, 0].set_ylabel('[m]')
axes[1, 0].set_title(str(ds_tm['Time'].values[770]).split('T')[0])
axes[1, 1].imshow(ds_tm['tt50_q_ss'].isel(Time=1006).values.T, origin="lower", cmap='Purples_r', vmin=100, vmax=900)
axes[1, 1].set_xticks(onp.arange(-.5, 11, 5))
axes[1, 1].set_yticks(onp.arange(-.5, 23, 5))
axes[1, 1].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[1, 1].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[1, 1].set_xlabel('[m]')
axes[1, 1].set_title(str(ds_tm['Time'].values[1006]).split('T')[0])
cmap = copy.copy(plt.cm.get_cmap('Purples_r'))
norm = mpl.colors.Normalize(vmin=100, vmax=900)
axl2 = fig.add_axes([0.8, 0.11, 0.02, 0.31])
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[100, 300, 600, 900])
cb2.ax.set_yticklabels(['<100', '300', '600', '>900'])
cb2.ax.invert_yaxis()
cb2.set_label(r'$TT_{50}$ [days]')
fig.subplots_adjust(left=0.06, bottom=0.1, top=0.94, right=0.9, wspace=-0.3, hspace=0.5)
file = base_path_figs / "theta_and_tt50.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "theta_and_tt50.pdf"
fig.savefig(file, dpi=250)

# plot fluxes and isotopic signals of a single grid cell
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].bar(date_hm, ds_hm['prec'].isel(x=0, y=0).values, width=.1, color='blue', align='edge', edgecolor='blue')
axes[0, 0].set_ylabel(r'PRECIP [mm/day]')
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[0, 0].tick_params(axis='x', labelrotation=60)

axes[0, 1].plot(date_tm, ds_tm['C_iso_in'].isel(x=0, y=0).values, '-', color='blue', lw=1)
axes[0, 1].set_xlim(date_tm[0], date_tm[-1])
axes[0, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[0, 1].tick_params(axis='x', labelrotation=60)


axes[1, 0].plot(date_hm, ds_hm['transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 0].plot(date_hm, ds_hm['evap_soil'].isel(x=0, y=0).values, '--', color='#c2e699', lw=0.8)
axes[1, 0].set_xlim(date_hm[0], date_hm[-1])
axes[1, 0].set_ylabel(r'ET [mm/day]')
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 0].tick_params(axis='x', labelrotation=60)

axes[2, 0].axvline(date_hm[150], color='red', alpha=0.5)
axes[2, 0].axvline(date_hm[340], color='red', alpha=0.5)
axes[2, 0].axvline(date_hm[770], color='red', alpha=0.5)
axes[2, 0].axvline(date_hm[1006], color='red', alpha=0.5)
axes[2, 0].plot(date_hm, ds_hm['theta'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 0].set_xlim(date_hm[0], date_hm[-1])
axes[2, 0].set_ylabel(r'$\theta$ [-]')
axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 0].tick_params(axis='x', labelrotation=60)

axes[3, 0].axvline(date_hm[150], color='red', alpha=0.5)
axes[3, 0].axvline(date_hm[340], color='red', alpha=0.5)
axes[3, 0].axvline(date_hm[770], color='red', alpha=0.5)
axes[3, 0].axvline(date_hm[1006], color='red', alpha=0.5)
axes[3, 0].plot(date_hm, ds_hm['q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 0].set_xlim(date_hm[0], date_hm[-1])
axes[3, 0].set_ylabel(r'PERC [mm/day]')
axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 0].set_xlabel(r'Time [year-month]')
axes[3, 0].tick_params(axis='x', labelrotation=60)

axes[1, 1].plot(date_tm, ds_tm['C_iso_transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 1].plot(date_tm, ds_tm['C_iso_evap_soil'].isel(x=0, y=0).values, '--', color='#c2e699', lw=0.8)
axes[1, 1].set_xlim(date_tm[0], date_tm[-1])
axes[1, 1].set_ylim(-15, -5)
axes[1, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 1].tick_params(axis='x', labelrotation=60)

axes[2, 1].axvline(date_hm[150], color='red', alpha=0.5)
axes[2, 1].axvline(date_hm[340], color='red', alpha=0.5)
axes[2, 1].axvline(date_hm[770], color='red', alpha=0.5)
axes[2, 1].axvline(date_hm[1006], color='red', alpha=0.5)
axes[2, 1].plot(date_tm, ds_tm['C_iso_s'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 1].set_xlim(date_tm[0], date_tm[-1])
axes[2, 1].set_ylim(-15, -5)
axes[2, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 1].tick_params(axis='x', labelrotation=60)

axes[3, 1].axvline(date_hm[150], color='red', alpha=0.5)
axes[3, 1].axvline(date_hm[340], color='red', alpha=0.5)
axes[3, 1].axvline(date_hm[770], color='red', alpha=0.5)
axes[3, 1].axvline(date_hm[1006], color='red', alpha=0.5)
axes[3, 1].plot(date_tm, ds_tm['C_iso_q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 1].set_xlim(date_tm[0], date_tm[-1])
axes[3, 1].set_ylim(-15, -5)
axes[3, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[3, 1].set_xlabel(r'Time [year-month]')
axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 1].tick_params(axis='x', labelrotation=60)

axes[0, 2].set_axis_off()

axes[1, 2].fill_between(date_tm, ds_tm['tt25_transp'].isel(x=0, y=0).values, ds_tm['tt75_transp'].isel(x=0, y=0).values, color='#31a354',
                        edgecolor=None, alpha=0.2)
axes[1, 2].plot(date_tm, ds_tm['tt50_transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 2].set_xlim(date_tm[0], date_tm[-1])
axes[1, 2].set_ylabel(r'age [days]')
axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 2].tick_params(axis='x', labelrotation=60)

axes[2, 2].axvline(date_hm[150], color='red', alpha=0.5)
axes[2, 2].axvline(date_hm[340], color='red', alpha=0.5)
axes[2, 2].axvline(date_hm[770], color='red', alpha=0.5)
axes[2, 2].axvline(date_hm[1006], color='red', alpha=0.5)
axes[2, 2].fill_between(date_tm, ds_tm['rt25_s'].isel(x=0, y=0).values, ds_tm['rt75_s'].isel(x=0, y=0).values, color='brown',
                        edgecolor=None, alpha=0.2)
axes[2, 2].plot(date_tm, ds_tm['rt50_s'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 2].set_xlim(date_tm[0], date_tm[-1])
axes[2, 2].set_ylabel(r'age [days]')
axes[2, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 2].tick_params(axis='x', labelrotation=60)

axes[3, 2].axvline(date_hm[150], color='red', alpha=0.5)
axes[3, 2].axvline(date_hm[340], color='red', alpha=0.5)
axes[3, 2].axvline(date_hm[770], color='red', alpha=0.5)
axes[3, 2].axvline(date_hm[1006], color='red', alpha=0.5)
axes[3, 2].fill_between(date_tm, ds_tm['tt25_q_ss'].isel(x=0, y=0).values, ds_tm['tt75_q_ss'].isel(x=0, y=0).values, color='grey',
                        edgecolor=None, alpha=0.2)
axes[3, 2].plot(date_tm, ds_tm['tt50_q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 2].set_xlim(date_tm[0], date_tm[-1])
axes[3, 2].set_ylabel(r'age [days]')
axes[3, 2].set_xlabel(r'Time [year-month]')
axes[3, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 2].tick_params(axis='x', labelrotation=60)

fig.subplots_adjust(left=0.1, bottom=0.13, top=0.95, right=0.98, hspace=0.6, wspace=0.42)
file = base_path_figs / "ts_single_grid_cell.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "ts_single_grid_cell.pdf"
fig.savefig(file, dpi=250)

# plot flux distributions, isotopic distributions and age distributions of all grid cells at wet and dry conditions
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=770).values.flatten(), color='#c2e699', bins=10, range=(0, 1), align='mid')
axes[0, 0].set_xlabel(r'$EVAP_{soil}$ [mm/day]')
axes[0, 0].set_ylabel('# grid cells')

axes[1, 0].hist(ds_hm['transp'].isel(Time=770).values.flatten(), color='#31a354', bins=40, range=(0, 4), align='mid')
axes[1, 0].set_xlabel(r'$TRANSP$ [mm/day]')
axes[1, 0].set_ylabel('# grid cells')

axes[2, 0].hist(ds_hm['theta'].isel(Time=770).values.flatten(), color='brown', bins=20, range=(0.2, 0.4), align='mid')
axes[2, 0].set_xlabel(r'$\theta$ [-]')
axes[2, 0].set_ylabel('# grid cells')

axes[3, 0].hist(ds_hm['q_ss'].isel(Time=770).values.flatten(), color='grey', bins=30, range=(0, 15), align='mid')
axes[3, 0].set_xlabel(r'$PERC$ [mm/day]')
axes[3, 0].set_ylabel('# grid cells')

axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=770).values.flatten(), color='#c2e699', bins=24, range=(-12, -6), align='mid')
axes[0, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=770).values.flatten(), color='#31a354', bins=24, range=(-12, -6), align='mid')
axes[1, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=770).values.flatten(), color='brown', bins=24, range=(-12, -6), align='mid')
axes[2, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=770).values.flatten(), color='grey', bins=24, range=(-12, -6), align='mid')
axes[3, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[0, 2].set_axis_off()

axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=770).values.flatten(), color='#31a354', bins=50, range=(0, 600), align='mid')
axes[1, 2].set_xlabel(r'$TT_{50-TRANSP}$ [days]')

axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=770).values.flatten(), color='brown', bins=50, range=(0, 600), align='mid')
axes[2, 2].set_xlabel(r'$RT_{50-\theta}$ [days]')

axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=770).values.flatten(), color='grey', bins=50, range=(0, 600), align='mid')
axes[3, 2].set_xlabel(r'$TT_{50-PERC}$ [days]')

fig.tight_layout()
file = base_path_figs / "dist_states_wet.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "dist_states_wet.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=1006).values.flatten(), color='#c2e699', bins=10, range=(0, 1))
axes[0, 0].set_xlabel(r'$EVAP_{soil}$ [mm/day]')
axes[0, 0].set_ylabel('# grid cells')

axes[1, 0].hist(ds_hm['transp'].isel(Time=1006).values.flatten(), color='#31a354', bins=40, range=(0, 4))
axes[1, 0].set_xlabel(r'$TRANSP$ [mm/day]')
axes[1, 0].set_ylabel('# grid cells')

axes[2, 0].hist(ds_hm['theta'].isel(Time=1006).values.flatten(), color='brown', bins=20, range=(0.2, 0.4))
axes[2, 0].set_xlabel(r'$\theta$ [-]')
axes[2, 0].set_ylabel('# grid cells')

axes[3, 0].hist(ds_hm['q_ss'].isel(Time=1006).values.flatten(), color='grey', bins=30, range=(0, 15))
axes[3, 0].set_xlabel(r'$PERC$ [mm/day]')
axes[3, 0].set_ylabel('# grid cells')

axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=1006).values.flatten(), color='#c2e699', bins=24, range=(-12, -6))
axes[0, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=1006).values.flatten(), color='#31a354', bins=24, range=(-12, -6))
axes[1, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=1006).values.flatten(), color='brown', bins=24, range=(-12, -6))
axes[2, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=1006).values.flatten(), color='grey', bins=24, range=(-12, -6))
axes[3, 1].set_xlabel(r'$\delta^{18}$O [‰]')

axes[0, 2].set_axis_off()

axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=1006).values.flatten(), color='#31a354', bins=50, range=(0, 600))
axes[1, 2].set_xlabel(r'$TT_{50-TRANSP}$ [days]')

axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=1006).values.flatten(), color='brown', bins=50, range=(0, 600))
axes[2, 2].set_xlabel(r'$RT_{50-\theta}$ [days]')

axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=1006).values.flatten(), color='grey', bins=50, range=(0, 600))
axes[3, 2].set_xlabel(r'$TT_{50-PERC}$ [days]')

fig.tight_layout()
file = base_path_figs / "dist_states_dry.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "dist_states_dry.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=770).values.flatten(), color='#253494', bins=10, range=(0, 1), align='mid', alpha=0.5)
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=10, range=(0, 1), align='mid', alpha=0.5)
axes[0, 0].set_xlabel(r'$EVAP_{soil}$ [mm/day]')
axes[0, 0].set_ylabel('# grid cells')
axes[0, 0].text(0.95, 1.12, '(a)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[0, 0].transAxes)

axes[1, 0].hist(ds_hm['transp'].isel(Time=770).values.flatten(), color='#253494', bins=40, range=(0, 4), align='mid', alpha=0.5)
axes[1, 0].hist(ds_hm['transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=40, range=(0, 4), align='mid', alpha=0.5)
axes[1, 0].set_xlabel(r'$TRANSP$ [mm/day]')
axes[1, 0].set_ylabel('# grid cells')
axes[1, 0].text(0.95, 1.12, '(b)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 0].transAxes)

axes[2, 0].hist(ds_hm['theta'].isel(Time=770).values.flatten(), color='#253494', bins=20, range=(0.2, 0.4), align='mid', alpha=0.5)
axes[2, 0].hist(ds_hm['theta'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=20, range=(0.2, 0.4), align='mid', alpha=0.5)
axes[2, 0].set_xlabel(r'$\theta$ [-]')
axes[2, 0].set_ylabel('# grid cells')
axes[2, 0].text(0.95, 1.12, '(c)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 0].transAxes)

axes[3, 0].hist(ds_hm['q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=30, range=(0, 15), align='mid', alpha=0.5)
axes[3, 0].hist(ds_hm['q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=30, range=(0, 15), align='mid', alpha=0.5)
axes[3, 0].set_xlabel(r'$PERC$ [mm/day]')
axes[3, 0].set_ylabel('# grid cells')
axes[3, 0].text(0.95, 1.12, '(d)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 0].transAxes)

axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=770).values.flatten(), color='#253494', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[0, 1].set_xlabel(r'$\delta^{18}$$O_{EVAP_{soil}}$ [‰]')
axes[0, 1].text(0.95, 1.12, '(e)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[0, 1].transAxes)

axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=770).values.flatten(), color='#253494', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[1, 1].set_xlabel(r'$\delta^{18}$$O_{TRANSP}$ [‰]')
axes[1, 1].text(0.95, 1.12, '(f)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 1].transAxes)

axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=770).values.flatten(), color='#253494', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[2, 1].set_xlabel(r'$\delta^{18}$$O_{\theta}$ [‰]')
axes[2, 1].text(0.95, 1.12, '(g)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 1].transAxes)

axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=24, range=(-12, -6), align='mid', alpha=0.5)
axes[3, 1].set_xlabel(r'$\delta^{18}$$O_{PERC}$ [‰]')
axes[3, 1].text(0.95, 1.12, '(h)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 1].transAxes)

axes[0, 2].set_axis_off()

axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 600), align='mid', alpha=0.5, label=r'wet condtions ($9^{th}$ Dec 2021)')
axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 600), align='mid', alpha=0.5, label=r'dry condtions ($8^{th}$ Aug 2022)')
axes[1, 2].set_xlabel(r'$TT_{50-TRANSP}$ [days]')
axes[1, 2].legend(frameon=False, loc='upper left', bbox_to_anchor=(0.0, 1.6))
axes[1, 2].text(0.95, 1.12, '(i)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 2].transAxes)

axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 600), align='mid', alpha=0.5)
axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 600), align='mid', alpha=0.5)
axes[2, 2].set_xlabel(r'$RT_{50-\theta}$ [days]')
axes[2, 2].text(0.95, 1.12, '(j)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 2].transAxes)

axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 600), align='mid', alpha=0.5)
axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 600), align='mid', alpha=0.5)
axes[3, 2].set_xlabel(r'$TT_{50-PERC}$ [days]')
axes[3, 2].text(0.95, 1.12, '(k)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 2].transAxes)

fig.tight_layout()
file = base_path_figs / "dist_states_wet_dry.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "dist_states_wet_dry.pdf"
fig.savefig(file, dpi=250)

# plot cumulated distributions of fluxes, isotopic signal and median age of all grid cells at dry, normal and wet conditions
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=340).values.flatten(), color='#41b6c4', bins=50, range=(0, 1), histtype='step', cumulative=True)
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 1), histtype='step', cumulative=True)
axes[0, 0].hist(ds_hm['evap_soil'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 1), histtype='step', cumulative=True)
axes[0, 0].set_xlim(0, 1)
axes[0, 0].set_xlabel(r'$EVAP_{soil}$ [mm/day]')
axes[0, 0].set_ylabel('# grid cells')
axes[0, 0].text(0.95, 1.12, '(a)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[0, 0].transAxes)

axes[1, 0].hist(ds_hm['transp'].isel(Time=340).values.flatten(), color='#41b6c4', bins=40, range=(0, 4), histtype='step', cumulative=True)
axes[1, 0].hist(ds_hm['transp'].isel(Time=770).values.flatten(), color='#253494', bins=40, range=(0, 4), histtype='step', cumulative=True)
axes[1, 0].hist(ds_hm['transp'].isel(Time=150).values.flatten(), color='#fed976', bins=40, range=(0, 4), histtype='step', cumulative=True)
axes[1, 0].hist(ds_hm['transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=40, range=(0, 4), histtype='step', cumulative=True)
axes[1, 0].set_xlim(0, 4)
axes[1, 0].set_xlabel(r'$TRANSP$ [mm/day]')
axes[1, 0].set_ylabel('# grid cells')
axes[1, 0].text(0.95, 1.12, '(b)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 0].transAxes)

axes[2, 0].hist(ds_hm['theta'].isel(Time=340).values.flatten(), color='#41b6c4', bins=40, range=(0.2, 0.4), histtype='step', cumulative=True)
axes[2, 0].hist(ds_hm['theta'].isel(Time=770).values.flatten(), color='#253494', bins=40, range=(0.2, 0.4), histtype='step', cumulative=True)
axes[2, 0].hist(ds_hm['theta'].isel(Time=150).values.flatten(), color='#fed976', bins=40, range=(0.2, 0.4), histtype='step', cumulative=True)
axes[2, 0].hist(ds_hm['theta'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=40, range=(0.2, 0.4), histtype='step', cumulative=True)
axes[2, 0].set_xlim(0.2, 0.4)
axes[2, 0].set_xlabel(r'$\theta$ [-]')
axes[2, 0].set_ylabel('# grid cells')
axes[2, 0].text(0.95, 1.12, '(c)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 0].transAxes)

axes[3, 0].hist(ds_hm['q_ss'].isel(Time=340).values.flatten(), color='#41b6c4', bins=60, range=(0, 15), histtype='step', cumulative=True)
axes[3, 0].hist(ds_hm['q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=60, range=(0, 15), histtype='step', cumulative=True)
axes[3, 0].hist(ds_hm['q_ss'].isel(Time=150).values.flatten(), color='#fed976', bins=60, range=(0, 15), histtype='step', cumulative=True)
axes[3, 0].hist(ds_hm['q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=60, range=(0, 15), histtype='step', cumulative=True)
axes[3, 0].set_xlim(0, 10)
axes[3, 0].set_xlabel(r'$PERC$ [mm/day]')
axes[3, 0].set_ylabel('# grid cells')
axes[3, 0].text(0.95, 1.12, '(d)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 0].transAxes)

axes[3, 0].fill([0, 0.4, 0.4, 0], [0, 0, 288, 288], color='grey', alpha=0.5) 

axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=340).values.flatten(), color='#41b6c4', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=770).values.flatten(), color='#253494', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[0, 1].hist(ds_tm['C_iso_evap_soil'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[0, 1].set_xlim(-12, -6)
axes[0, 1].set_xlabel(r'$\delta^{18}$$O_{EVAP_{soil}}$ [‰]')
axes[0, 1].text(0.95, 1.12, '(e)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[0, 1].transAxes)

axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=340).values.flatten(), color='#41b6c4', bins=70, range=(-14, -7), histtype='step', cumulative=True)
axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=770).values.flatten(), color='#253494', bins=70, range=(-14, -7), histtype='step', cumulative=True)
axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=150).values.flatten(), color='#fed976', bins=70, range=(-14, -7), histtype='step', cumulative=True)
axes[1, 1].hist(ds_tm['C_iso_transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=70, range=(-14, -7), histtype='step', cumulative=True)
axes[1, 1].set_xlim(-14, -7)
axes[1, 1].set_xlabel(r'$\delta^{18}$$O_{TRANSP}$ [‰]')
axes[1, 1].text(0.95, 1.12, '(f)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 1].transAxes)

axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=340).values.flatten(), color='#41b6c4', bins=40, range=(-12, -8), histtype='step', cumulative=True)
axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=770).values.flatten(), color='#253494', bins=40, range=(-12, -8), histtype='step', cumulative=True)
axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=150).values.flatten(), color='#fed976', bins=40, range=(-12, -8), histtype='step', cumulative=True)
axes[2, 1].hist(ds_tm['C_iso_s'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=40, range=(-12, -8), histtype='step', cumulative=True)
axes[2, 1].set_xlim(-12, -8)
axes[2, 1].set_xlabel(r'$\delta^{18}$$O_{\theta}$ [‰]')
axes[2, 1].text(0.95, 1.12, '(g)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 1].transAxes)

axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=340).values.flatten(), color='#41b6c4', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=150).values.flatten(), color='#fed976', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[3, 1].hist(ds_tm['C_iso_q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=60, range=(-12, -6), histtype='step', cumulative=True)
axes[3, 1].set_xlim(-12, -8)
axes[3, 1].set_xlabel(r'$\delta^{18}$$O_{PERC}$ [‰]')
axes[3, 1].text(0.95, 1.12, '(h)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 1].transAxes)

axes[0, 2].set_axis_off()

axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=340).values.flatten(), color='#41b6c4', bins=50, range=(0, 50), histtype='step', cumulative=True)
axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 50), histtype='step', cumulative=True)
axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=150).values.flatten(), color='#fed976', bins=50, range=(0, 50), histtype='step', cumulative=True)
axes[1, 2].hist(ds_tm['tt50_transp'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 50), histtype='step', cumulative=True)
axes[1, 2].set_xlim(0, 50)
axes[1, 2].set_xlabel(r'$TT_{50-TRANSP}$ [days]')
axes[1, 2].text(0.95, 1.12, '(i)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[1, 2].transAxes)
axes[1, 2].plot([], [], color='#fed976', label='transition to dry condtions\n($29^{th}$ Mar 2020)')
axes[1, 2].plot([], [], color='#fd8d3c', label='dry condtions\n($8^{th}$ Aug 2022)')
axes[1, 2].plot([], [], color='#41b6c4', label='transition to wet condtions\n($5^{th}$ Oct 2020)')
axes[1, 2].plot([], [], color='#253494', label='wet condtions\n($9^{th}$ Dec 2021)')
lines, labels = axes[1, 2].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper right', frameon=False, bbox_to_anchor=(0.98, 1.005))

axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=340).values.flatten(), color='#41b6c4', bins=600, range=(0, 600), histtype='step', cumulative=True)
axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=770).values.flatten(), color='#253494', bins=600, range=(0, 600), histtype='step', cumulative=True)
axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=150).values.flatten(), color='#fed976', bins=600, range=(0, 600), histtype='step', cumulative=True)
axes[2, 2].hist(ds_tm['rt50_s'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=600, range=(0, 600), histtype='step', cumulative=True)
axes[2, 2].set_xlim(100, 600)
axes[2, 2].set_xlabel(r'$RT_{50-\theta}$ [days]')
axes[2, 2].text(0.95, 1.12, '(j)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[2, 2].transAxes)

axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=340).values.flatten(), color='#41b6c4', bins=800, range=(0, 800), histtype='step', cumulative=True)
axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=800, range=(0, 800), histtype='step', cumulative=True)
axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=150).values.flatten(), color='#fed976', bins=800, range=(0, 800), histtype='step', cumulative=True)
axes[3, 2].hist(ds_tm['tt50_q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=800, range=(0, 800), histtype='step', cumulative=True)
axes[3, 2].set_xlim(200, 800)
axes[3, 2].set_xlabel(r'$TT_{50-PERC}$ [days]')
axes[3, 2].text(0.95, 1.12, '(k)', fontsize=9, horizontalalignment='center',
                verticalalignment='center', transform=axes[3, 2].transAxes)

inset = fig.add_axes([0.225, 0.15, 0.11, 0.08])
inset.hist(ds_hm['q_ss'].isel(Time=340).values.flatten(), color='#41b6c4', bins=40, range=(0, 0.4), histtype='step', cumulative=True)
inset.hist(ds_hm['q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=40, range=(0, 0.4), histtype='step', cumulative=True)
inset.hist(ds_hm['q_ss'].isel(Time=150).values.flatten(), color='#fed976', bins=40, range=(0, 0.4), histtype='step', cumulative=True)
inset.hist(ds_hm['q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=40, range=(0, 0.4), histtype='step', cumulative=True)
inset.set_xlim(0, 0.4)

fig.subplots_adjust(left=0.1, bottom=0.1, top=0.95, right=0.98, hspace=0.6, wspace=0.25)
file = base_path_figs / "cumulated_dist_states_dry_normal_wet.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "cumulated_dist_states_dry_normal_wet.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(figsize=(3, 2))
axes.hist(ds_hm['q_ss'].isel(Time=340).values.flatten(), color='#41b6c4', bins=50, range=(0, 0.5), histtype='step', cumulative=True)
axes.hist(ds_hm['q_ss'].isel(Time=770).values.flatten(), color='#253494', bins=50, range=(0, 0.5), histtype='step', cumulative=True)
axes.hist(ds_hm['q_ss'].isel(Time=150).values.flatten(), color='#fed976', bins=50, range=(0, 0.5), histtype='step', cumulative=True)
axes.hist(ds_hm['q_ss'].isel(Time=1006).values.flatten(), color='#fd8d3c', bins=50, range=(0, 0.5), histtype='step', cumulative=True)
axes.set_xlim(0, 0.5)
axes.set_xlabel(r'$PERC$ [mm/day]')
axes.set_ylabel('# grid cells')
fig.tight_layout()
file = base_path_figs / "cumulated_dist_perc_inset.png"
fig.savefig(file, dpi=250)

# load hydrologic model parameters
params_file = base_path / "svat_distributed" / "parameters.nc"
ds_params = xr.open_dataset(params_file, engine="h5netcdf")

fig, axes = plt.subplots(1, 1, figsize=(2, 2.2))
axes.imshow(ds_params['ks'].values.T, origin="lower", cmap='Greys', vmin=5, vmax=15)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=5, vmax=15)
axl1 = fig.add_axes([0.745, 0.2, 0.04, 0.72])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[5, 10, 15])
cb1.ax.set_yticklabels(['5', '10', '15'])
cb1.set_label(r'$k_s$ [mm/hour]')
fig.subplots_adjust(left=0.25, bottom=0.12, top=1.0, right=0.68)
file = base_path_figs / "ks_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "ks_grid.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.2))
axes.imshow(ds_params['dmpv'].values.T, origin="lower", cmap='Greys', vmin=50, vmax=100)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=50, vmax=100)
axl1 = fig.add_axes([0.7, 0.2, 0.04, 0.74])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[50, 75, 100])
cb1.ax.set_yticklabels(['50', '75', '100'])
cb1.set_label(r'$\rho_{mpv}$ [1/$m^2$]')
fig.subplots_adjust(left=0.22, bottom=0.18, top=0.98, right=0.68)
file = base_path_figs / "dmpv_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "dmpv_grid.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.2))
axes.imshow(ds_params['lmpv'].values.T, origin="lower", cmap='Greys', vmin=200, vmax=500)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=200, vmax=500)
axl1 = fig.add_axes([0.7, 0.2, 0.04, 0.74])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[200, 350, 500])
cb1.ax.set_yticklabels(['200', '350', '500'])
cb1.set_label(r'$l_{mpv}$ [mm]')
fig.subplots_adjust(left=0.22, bottom=0.18, top=0.98, right=0.68)
file = base_path_figs / "lmpv_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "lmpv_grid.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 2.2))
axes.imshow(ds_params['z_soil'].values.T/1000, origin="lower", cmap='Greys', vmin=1, vmax=1.2)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=1, vmax=1.2)
axl1 = fig.add_axes([0.745, 0.2, 0.04, 0.72])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[1, 1.1, 1.2])
cb1.ax.set_yticklabels(['1', '1.1', '1.2'])
cb1.set_label(r'$z_{soil}$ [m]')
fig.subplots_adjust(left=0.25, bottom=0.12, top=1.0, right=0.68)
file = base_path_figs / "zsoil_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.2))
axes.imshow(ds_params['theta_ac'].values.T, origin="lower", cmap='Greys', vmin=0.08, vmax=0.12)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.08, vmax=0.12)
axl1 = fig.add_axes([0.7, 0.2, 0.04, 0.74])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.08, 0.1, 0.12])
cb1.ax.set_yticklabels(['0.08', '0.1', '0.12'])
cb1.set_label(r'$\theta_{ac}$ [-]')
fig.subplots_adjust(left=0.22, bottom=0.18, top=0.98, right=0.68)
file = base_path_figs / "thetaac_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "thetaac_grid.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.2))
axes.imshow(ds_params['theta_ufc'].values.T, origin="lower", cmap='Greys', vmin=0.08, vmax=0.12)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.08, vmax=0.12)
axl1 = fig.add_axes([0.7, 0.2, 0.04, 0.74])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.08, 0.1, 0.12])
cb1.ax.set_yticklabels(['0.08', '0.1', '0.12'])
cb1.set_label(r'$\theta_{ufc}$ [-]')
fig.subplots_adjust(left=0.22, bottom=0.18, top=0.98, right=0.68)
file = base_path_figs / "thetaufc_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "thetaufc_grid.pdf"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2.4, 2.2))
axes.imshow(ds_params['theta_pwp'].values.T, origin="lower", cmap='Greys', vmin=0.18, vmax=0.22)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.18, vmax=0.22)
axl1 = fig.add_axes([0.7, 0.2, 0.04, 0.74])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.18, 0.2, 0.22])
cb1.ax.set_yticklabels(['0.18', '0.2', '0.22'])
cb1.set_label(r'$\theta_{pwp}$ [-]')
fig.subplots_adjust(left=0.22, bottom=0.18, top=0.98, right=0.68)
file = base_path_figs / "thetapwp_grid.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "thetapwp_grid.pdf"
fig.savefig(file, dpi=250)

# # make GIF for Online-Documentation
# frames = []
# # reading png image
# file = base_path_figs / "front_gif.png"
# front = Image.open(file)
# frames.append(front.resize((1000, 500)))
# for t in range(1, 1000):
#     fig, axes = plt.subplots(1, 2, figsize=(4, 2))
#     axes[0].imshow(ds_hm['theta'].isel(Time=t).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
#     axes[0].set_xticks(onp.arange(-.5, 11, 5))
#     axes[0].set_yticks(onp.arange(-.5, 23, 5))
#     axes[0].set_xticklabels(onp.arange(0, 12, 5) * 5)
#     axes[0].set_yticklabels(onp.arange(0, 24, 5) * 5)
#     axes[0].set_xlabel('[m]')
#     axes[0].set_ylabel('[m]')
#     cmap = copy.copy(plt.cm.get_cmap('Blues'))
#     norm = mpl.colors.Normalize(vmin=0.2, vmax=0.4)
#     axl1 = fig.add_axes([0.31, 0.215, 0.02, 0.65])
#     cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
#                                     orientation='vertical',
#                                     ticks=[0.2, 0.3, 0.4])
#     cb1.ax.set_yticklabels(['<0.2', '0.3', '>0.4'])
#     cb1.set_label(r'Soil water content [-]')

#     axes[1].imshow(ds_tm['tt50_q_ss'].isel(Time=t).values.T, origin="lower", cmap='Purples_r', vmin=100, vmax=900)
#     axes[1].set_xticks(onp.arange(-.5, 11, 5))
#     axes[1].set_yticks(onp.arange(-.5, 23, 5))
#     axes[1].set_xticklabels(onp.arange(0, 12, 5) * 5)
#     axes[1].set_yticklabels(onp.arange(0, 24, 5) * 5)
#     axes[1].set_xlabel('[m]')
#     axes[1].set_ylabel('[m]')
#     cmap = copy.copy(plt.cm.get_cmap('Purples_r'))
#     norm = mpl.colors.Normalize(vmin=100, vmax=900)
#     axl2 = fig.add_axes([0.8, 0.215, 0.02, 0.65])
#     cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm,
#                                     orientation='vertical',
#                                     ticks=[100, 300, 600, 900])
#     cb2.ax.set_yticklabels(['<100', '300', '600', '>900'])
#     cb2.ax.invert_yaxis()
#     cb2.set_label('Median travel time\n of percolation [days]')
#     fig.suptitle(f't = {t}d', fontsize=9)
#     fig.subplots_adjust(wspace=0.2, left=0.0, bottom=0.2)
#     file = base_path_figs / f"t{t}.png"
#     fig.savefig(file, dpi=250)
#     plt.close('all')
#     img = imageio.v2.imread(file)
#     frames.append(img)
    
# file = base_path_figs / "theta_and_tt.gif"
# imageio.mimsave(file,
#                 frames,
#                 fps = 7)

plt.close('all')