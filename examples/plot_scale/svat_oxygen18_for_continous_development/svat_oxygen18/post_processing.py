import os
import glob
import h5netcdf
import datetime
from pathlib import Path
import numpy as onp
import xarray as xr
import roger
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

_LABS_TM = {'complete-mixing': 'CM',
            'piston': 'PI',
            'advection-dispersion': 'AD',
            'older-prefrence': 'OP',
            'power': 'POW',
            'preferential + advection-dispersion': 'PF-AD',
            'preferential': 'PF'}


base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

tm_structures = ['advection-dispersion',
                 'older-preference',
                 'preferential',
                 'preferential + advection-dispersion',
                 'power']
# merge results into single file
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    path = str(base_path / f"SVATTRANSPORT_{tms}_deterministic.*.nc")
    diag_files = glob.glob(path)
    states_tm_file = base_path / f"states_{tms}.nc"
    if not os.path.exists(states_tm_file):
        with h5netcdf.File(states_tm_file, 'w', decode_vlen_strings=False) as f:
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title=f'RoGeR {tm_structure} transport model simulations',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='First timestep (t=0) contains initial values. Simulations start are written from second timestep (t=1) to last timestep (t=N).',
                model_structure=f'SVAT {tm_structure} transport model with free drainage',
                sas_solver='deterministic',
                roger_version=f'{roger.__version__}'
            )
            # collect dimensions
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    # set dimensions with a dictionary
                    if not dfs.split('/')[-1].split('.')[1] == 'constant':
                        dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time']), 'ages': len(df.variables['ages']), 'nages': len(df.variables['nages']), 'n_sas_params': len(df.variables['n_sas_params'])}
                        time = onp.array(df.variables.get('Time'))
            for dfs in diag_files:
                with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
                    if not f.dimensions:
                        f.dimensions = dict_dim
                        v = f.create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.create_variable('ages', ('ages',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(1, dict_dim["ages"]+1)
                        v = f.create_variable('nages', ('nages',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(0, dict_dim["nages"])
                        v = f.create_variable('n_sas_params', ('n_sas_params',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Number of SAS parameters'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(0, dict_dim["n_sas_params"])
                        v = f.create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                        units=var_obj.attrs["units"])
                        v[:] = time
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] > 2:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                            units=var_obj.attrs["units"])
                            del var_obj, vals
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions and var_obj.shape[0] <= 2:
                            v = f.create_variable(var_sim, ('x', 'y'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :] = vals.swapaxes(0, 2)[:, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                            units=var_obj.attrs["units"])
                            del var_obj, vals
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                            v = f.create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            v[:, :, :] = vals[:, :, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                            units=var_obj.attrs["units"])
                            del var_obj, vals
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                            units=var_obj.attrs["units"])
                            del var_obj, vals
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                            v = f.create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                            units=var_obj.attrs["units"])
                            del var_obj, vals

for i, tm_structure in enumerate(tm_structures):
    tms = tm_structure.replace(" ", "_")
    # load transport simulation
    states_tm_file = base_path / "svat_oxygen18_monte_carlo" / "deterministic" / "age_max_1500_days" / "optimized_with_KGE_multi" / f"states_{tms}_monte_carlo.nc"
    ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
    days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
    ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                '-', color='blue')
    ax.flatten()[0].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(df_obs.index[0], df_obs.index[-1])
    ax.flatten()[i+1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=idx_best, y=0).values, color='red', lw=1)
    ax.flatten()[i+1].plot(ds_hydrus_18O['Time'].values, ds_hydrus_18O['d18O_perc'].values, color='grey', lw=1)
    ax.flatten()[i+1].scatter(df_obs.index, df_obs.iloc[:, 0], color='blue', s=1)
    ax[i+1].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[i+1].set_ylim((-15, -5))
    ax.flatten()[i+1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
ax[-1].set_xlabel('Time [year]')
fig.tight_layout()
file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_optimized_with_{metric_for_opt}.png"
fig.savefig(file, dpi=250)

    fig, ax = plt.subplots(4, 1, sharey=False, figsize=(6, 5))
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        # load transport simulation
        states_tm_file = base_path / "svat_oxygen18_monte_carlo" / "deterministic" / "age_max_1500_days" / "optimized_with_KGE_multi" / f"states_{tms}_monte_carlo.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
        # join observations on simulations
        obs_vals = ds_obs['d18O_PERC'].isel(x=0, y=0).values
        sim_vals = ds_sim_tm['C_iso_q_ss'].isel(y=0).values
        sim_vals = onp.where((sim_vals > 0) | (sim_vals < -20), onp.nan, sim_vals)
        sim_vals_avg = onp.nanmean(sim_vals, axis=0)
        sim_vals_5 = onp.nanquantile(sim_vals, 0.05, axis=0)
        sim_vals_50 = onp.nanmedian(sim_vals, axis=0)
        sim_vals_95 = onp.nanquantile(sim_vals, 0.95, axis=0)
        sim_vals_hydrus = ds_hydrus_18O['d18O_perc_bs'].values
        ax.flatten()[i].plot(ds_sim_tm['Time'].values, sim_vals_avg, ls='--', color='red', lw=1)
        ax.flatten()[i].plot(ds_sim_tm['Time'].values, sim_vals_50, ls='-', color='red', lw=1)
        ax.flatten()[i].fill_between(ds_sim_tm['Time'].values, sim_vals_5, sim_vals_95, color='red',
                              edgecolor=None, alpha=0.2)
        ax.flatten()[i].plot(ds_hydrus_18O['Time'].values, sim_vals_hydrus, color='grey', lw=1)
        ax.flatten()[i].scatter(date_obs, obs_vals, color='blue', s=1)
        ax.flatten()[i].set_title(_LABS_TM[tm_structure])
        ax[i].set_ylabel(r'$\delta^{18}$O [‰]')
        ax.flatten()[i].set_ylim((-20, 0))
        ax.flatten()[i].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    ax[-1].set_xlabel('Time [year]')
    fig.tight_layout()
    file = base_path_figs / f"d18O_perc_sim_obs_tm_structures_conf_int_optimized_with_{metric_for_opt}.png"
    fig.savefig(file, dpi=250)
