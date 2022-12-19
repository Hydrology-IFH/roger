import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import numpy as onp
import copy
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 6
mpl.rcParams['axes.titlesize'] = 6
mpl.rcParams['axes.labelsize'] = 7
mpl.rcParams['xtick.labelsize'] = 6
mpl.rcParams['ytick.labelsize'] = 6
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['legend.title_fontsize'] = 7
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 6.0,
                                                'axes.labelsize': 7.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 6.0,
                                                'ytick.labelsize': 6.0,
                                                'legend.fontsize': 6.0,
                                                'legend.title_fontsize': 7.0})

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
states_hm_file = base_path / "svat_oxygen18_distributed" / "states_advection-dispersion.nc"
ds_tm = xr.open_dataset(states_hm_file, engine="h5netcdf", decode_times=False)
# assign date
days_tm = ds_tm['Time'].values
date_tm = num2date(days_tm, units="days since 2019-10-31", calendar='standard', only_use_cftime_datetimes=False)
ds_tm = ds_tm.assign_coords(Time=("Time", date_tm))

# plot spatially distributed soil moisture and age statistic at different dates
fig, axes = plt.subplots(2, 2, figsize=(4, 4))
axes[0, 0].imshow(ds_hm['theta'].isel(Time=772).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[0, 0].set_xticks(onp.arange(-.5, 11, 5))
axes[0, 0].set_yticks(onp.arange(-.5, 23, 5))
axes[0, 0].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[0, 0].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[0, 0].set_xlabel('[m]')
axes[0, 0].set_ylabel('[m]')
axes[0, 0].set_title(str(ds_hm['Time'].values[772]).split('T')[0])
axes[0, 1].imshow(ds_hm['theta'].isel(Time=1004).values.T, origin="lower", cmap='Blues', vmin=0.2, vmax=0.4)
axes[0, 1].set_xticks(onp.arange(-.5, 11, 5))
axes[0, 1].set_yticks(onp.arange(-.5, 23, 5))
axes[0, 1].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[0, 1].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[0, 1].set_xlabel('[m]')
axes[0, 1].set_title(str(ds_hm['Time'].values[1004]).split('T')[0])
cmap = copy.copy(plt.cm.get_cmap('Blues'))
norm = mpl.colors.Normalize(vmin=0.2, vmax=0.4)
axl1 = fig.add_axes([0.75, 0.58, 0.02, 0.29])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.2, 0.3, 0.4])
cb1.ax.set_yticklabels(['<0.2', '0.3', '>0.4'])
cb1.set_label(r'$\theta$ [-]')

axes[1, 0].imshow(ds_tm['tt50_q_ss'].isel(Time=772).values.T, origin="lower", cmap='Purples_r', vmin=100, vmax=900)
axes[1, 0].set_xticks(onp.arange(-.5, 11, 5))
axes[1, 0].set_yticks(onp.arange(-.5, 23, 5))
axes[1, 0].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[1, 0].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[1, 0].set_xlabel('[m]')
axes[1, 0].set_ylabel('[m]')
axes[1, 0].set_title(str(ds_tm['Time'].values[772]).split('T')[0])
axes[1, 1].imshow(ds_tm['tt50_q_ss'].isel(Time=1004).values.T, origin="lower", cmap='Purples_r', vmin=100, vmax=900)
axes[1, 1].set_xticks(onp.arange(-.5, 11, 5))
axes[1, 1].set_yticks(onp.arange(-.5, 23, 5))
axes[1, 1].set_xticklabels(onp.arange(0, 12, 5) * 5)
axes[1, 1].set_yticklabels(onp.arange(0, 24, 5) * 5)
axes[1, 1].set_xlabel('[m]')
axes[1, 1].set_title(str(ds_tm['Time'].values[1004]).split('T')[0])
cmap = copy.copy(plt.cm.get_cmap('Purples_r'))
norm = mpl.colors.Normalize(vmin=100, vmax=900)
axl2 = fig.add_axes([0.75, 0.13, 0.02, 0.29])
cb2 = mpl.colorbar.ColorbarBase(axl2, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[100, 300, 600, 900])
cb2.ax.set_yticklabels(['<100', '300', '600', '>900'])
cb2.ax.invert_yaxis()
cb2.set_label(r'$TT_{50}$ [days]')
fig.subplots_adjust(wspace=-0.5, hspace=0.5)
file = base_path_figs / "theta_and_tt50.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "theta_and_tt50.pdf"
fig.savefig(file, dpi=250)

# plot fluxes and isotopic signals of a single grid cell
fig, axes = plt.subplots(4, 3, figsize=(6, 5))
axes[0, 0].bar(date_hm, ds_hm['prec'].isel(x=0, y=0).values, width=.1, color='blue', align='edge', edgecolor='blue')
axes[0, 0].set_ylabel(r'PREC [mm/day]')
axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[0, 0].tick_params(axis='x', labelrotation=30)

axes[0, 1].plot(date_tm, ds_tm['C_iso_in'].isel(x=0, y=0).values, '-', color='blue', lw=1)
axes[0, 1].set_xlim(date_tm[0], date_tm[-1])
axes[0, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[0, 1].tick_params(axis='x', labelrotation=30)


axes[1, 0].plot(date_hm, ds_hm['transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 0].plot(date_hm, ds_hm['evap_soil'].isel(x=0, y=0).values, '--', color='#c2e699', lw=0.8)
axes[1, 0].set_xlim(date_hm[0], date_hm[-1])
axes[1, 0].set_ylabel(r'ET [mm/day]')
axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 0].tick_params(axis='x', labelrotation=30)

axes[2, 0].axvline(date_hm[772], color='red', alpha=0.5)
axes[2, 0].axvline(date_hm[1004], color='red', alpha=0.5)
axes[2, 0].plot(date_hm, ds_hm['theta'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 0].set_xlim(date_hm[0], date_hm[-1])
axes[2, 0].set_ylabel(r'$\theta$ [-]')
axes[2, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 0].tick_params(axis='x', labelrotation=30)

axes[3, 0].axvline(date_hm[772], color='red', alpha=0.5)
axes[3, 0].axvline(date_hm[1004], color='red', alpha=0.5)
axes[3, 0].plot(date_hm, ds_hm['q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 0].set_xlim(date_hm[0], date_hm[-1])
axes[3, 0].set_ylabel(r'PERC [mm/day]')
axes[3, 0].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 0].set_xlabel(r'Time [year-month]')
axes[3, 0].tick_params(axis='x', labelrotation=30)

axes[1, 1].plot(date_tm, ds_tm['C_iso_transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 1].plot(date_tm, ds_tm['C_iso_evap_soil'].isel(x=0, y=0).values, '--', color='#c2e699', lw=0.8)
axes[1, 1].set_xlim(date_tm[0], date_tm[-1])
axes[1, 1].set_ylim(-15, -5)
axes[1, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 1].tick_params(axis='x', labelrotation=30)

axes[2, 1].axvline(date_hm[772], color='red', alpha=0.5)
axes[2, 1].axvline(date_hm[1004], color='red', alpha=0.5)
axes[2, 1].plot(date_tm, ds_tm['C_iso_s'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 1].set_xlim(date_tm[0], date_tm[-1])
axes[2, 1].set_ylim(-15, -5)
axes[2, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[2, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 1].tick_params(axis='x', labelrotation=30)

axes[3, 1].axvline(date_hm[772], color='red', alpha=0.5)
axes[3, 1].axvline(date_hm[1004], color='red', alpha=0.5)
axes[3, 1].plot(date_tm, ds_tm['C_iso_q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 1].set_xlim(date_tm[0], date_tm[-1])
axes[3, 1].set_ylim(-15, -5)
axes[3, 1].set_ylabel(r'$\delta^{18}$O [‰]')
axes[3, 1].set_xlabel(r'Time [year-month]')
axes[3, 1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 1].tick_params(axis='x', labelrotation=30)

axes[0, 2].set_axis_off()

axes[1, 2].fill_between(date_tm, ds_tm['tt25_transp'].isel(x=0, y=0).values, ds_tm['tt75_transp'].isel(x=0, y=0).values, color='#31a354',
                        edgecolor=None, alpha=0.2)
axes[1, 2].plot(date_tm, ds_tm['tt50_transp'].isel(x=0, y=0).values, '-', color='#31a354', lw=1)
axes[1, 2].set_xlim(date_tm[0], date_tm[-1])
axes[1, 2].set_ylabel(r'age [days]')
axes[1, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[1, 2].tick_params(axis='x', labelrotation=30)

axes[2, 2].axvline(date_hm[772], color='red', alpha=0.5)
axes[2, 2].axvline(date_hm[1004], color='red', alpha=0.5)
axes[2, 2].fill_between(date_tm, ds_tm['rt25_s'].isel(x=0, y=0).values, ds_tm['rt75_s'].isel(x=0, y=0).values, color='brown',
                        edgecolor=None, alpha=0.2)
axes[2, 2].plot(date_tm, ds_tm['rt50_s'].isel(x=0, y=0).values, '-', color='brown', lw=1)
axes[2, 2].set_xlim(date_tm[0], date_tm[-1])
axes[2, 2].set_ylabel(r'age [days]')
axes[2, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[2, 2].tick_params(axis='x', labelrotation=30)

axes[3, 2].axvline(date_hm[772], color='red', alpha=0.5)
axes[3, 2].axvline(date_hm[1004], color='red', alpha=0.5)
axes[3, 2].fill_between(date_tm, ds_tm['tt25_q_ss'].isel(x=0, y=0).values, ds_tm['tt75_q_ss'].isel(x=0, y=0).values, color='grey',
                        edgecolor=None, alpha=0.2)
axes[3, 2].plot(date_tm, ds_tm['tt50_q_ss'].isel(x=0, y=0).values, '-', color='grey', lw=1)
axes[3, 2].set_xlim(date_tm[0], date_tm[-1])
axes[3, 2].set_ylabel(r'age [days]')
axes[3, 2].set_xlabel(r'Time [year-month]')
axes[3, 2].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
axes[3, 2].tick_params(axis='x', labelrotation=30)

fig.tight_layout()
file = base_path_figs / "ts_single_grid_cell.png"
fig.savefig(file, dpi=250)
file = base_path_figs / "ts_single_grid_cell.pdf"
fig.savefig(file, dpi=250)


# load hydrologic model parameters
params_file = base_path / "svat_distributed" / "parameters.nc"
ds_params = xr.open_dataset(params_file, engine="h5netcdf")

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['ks'].values.T, origin="lower", cmap='Greys', vmin=5, vmax=15)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=5, vmax=15)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[5, 10, 15])
cb1.ax.set_yticklabels(['5', '10', '15'])
cb1.set_label(r'$k_s$ [mm/hour]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "ks_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['dmpv'].values.T, origin="lower", cmap='Greys', vmin=50, vmax=100)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=50, vmax=100)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[50, 75, 100])
cb1.ax.set_yticklabels(['50', '75', '100'])
cb1.set_label(r'$\rho_{mpv}$ [1/$m^2$]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "dmpv_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['lmpv'].values.T, origin="lower", cmap='Greys', vmin=200, vmax=500)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=200, vmax=500)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[200, 350, 500])
cb1.ax.set_yticklabels(['200', '350', '500'])
cb1.set_label(r'$l_{mpv}$ [mm]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "lmpv_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['z_soil'].values.T/1000, origin="lower", cmap='Greys', vmin=1, vmax=1.2)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=1, vmax=1.2)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[1, 1.1, 1.2])
cb1.ax.set_yticklabels(['1', '1.1', '1.2'])
cb1.set_label(r'$z_{soil}$ [m]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "zsoil_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['theta_ac'].values.T, origin="lower", cmap='Greys', vmin=0.08, vmax=0.12)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.08, vmax=0.12)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.08, 0.1, 0.12])
cb1.ax.set_yticklabels(['0.08', '0.1', '0.12'])
cb1.set_label(r'$\theta_{ac}$ [-]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "thetaac_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['theta_ufc'].values.T, origin="lower", cmap='Greys', vmin=0.08, vmax=0.12)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.08, vmax=0.12)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.08, 0.1, 0.12])
cb1.ax.set_yticklabels(['0.08', '0.1', '0.12'])
cb1.set_label(r'$\theta_{ufc}$ [-]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "thetaufc_grid.png"
fig.savefig(file, dpi=250)

fig, axes = plt.subplots(1, 1, figsize=(2, 4))
axes.imshow(ds_params['theta_pwp'].values.T, origin="lower", cmap='Greys', vmin=0.18, vmax=0.22)
axes.set_xticks(onp.arange(-.5, 11, 5))
axes.set_yticks(onp.arange(-.5, 23, 5))
axes.set_xticklabels(onp.arange(0, 12, 5) * 5)
axes.set_yticklabels(onp.arange(0, 24, 5) * 5)
axes.set_xlabel('[m]')
axes.set_ylabel('[m]')
cmap = copy.copy(plt.cm.get_cmap('Greys'))
norm = mpl.colors.Normalize(vmin=0.18, vmax=0.22)
axl1 = fig.add_axes([0.745, 0.3, 0.04, 0.48])
cb1 = mpl.colorbar.ColorbarBase(axl1, cmap=cmap, norm=norm,
                                orientation='vertical',
                                ticks=[0.18, 0.2, 0.22])
cb1.ax.set_yticklabels(['0.18', '0.2', '0.22'])
cb1.set_label(r'$\theta_{pwp}$ [-]')
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.68)
file = base_path_figs / "thetapwp_grid.png"
fig.savefig(file, dpi=250)

plt.close('all')
