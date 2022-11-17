import os
import glob
import h5netcdf
import datetime
from pathlib import Path
from cftime import num2date
from matplotlib.colors import Normalize
import matplotlib.cm as cm
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
tm_structures = ['advection-dispersion']
# merge results into single file
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    path = str(base_path / f"SVATOXYGEN18_{tms}_deterministic.*.nc")
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


tm_structure = 'advection-dispersion'
tms = tm_structure.replace(" ", "_")
# load transport simulation
states_tm_file = base_path / f"states_{tms}.nc"
ds_sim_tm = xr.open_dataset(states_tm_file, engine="h5netcdf")
days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim_tm = ds_sim_tm.assign_coords(Time=("Time", date_sim_tm))
params_tm_file = base_path / "parameters.nc"
ds_params = xr.open_dataset(params_tm_file, engine="h5netcdf", group=tm_structure)

b_transp = onp.unique(ds_params['b_transp'].isel(y=0).values).tolist()
a_q_rz = onp.unique(ds_params['a_q_rz'].isel(y=0).values).tolist()
rows = onp.where((ds_params['b_transp'].isel(y=0).values == 5) & (ds_params['a_q_rz'].isel(y=0).values == 2))[0]
cmap = cm.get_cmap('Reds')
norm = Normalize(vmin=2, vmax=10)

for b, a in zip(b_transp, a_q_rz):
    fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
    ax.flatten()[0].plot(ds_sim_tm['Time'].values,
                         ds_sim_tm['C_iso_in'].isel(x=0, y=0).values,
                         '-', color='black')
    ax.flatten()[0].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[0].set_ylim([-20, 0])
    ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_rz'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_q_ss'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    ax[4].set_xlabel('Time')
    fig.tight_layout()
    file = base_path_figs / f"d18O_drain_{tms}_b{b}_a{a}.png"
    fig.savefig(file, dpi=250)


for b, a in zip(b_transp, a_q_rz):
    fig, ax = plt.subplots(5, 1, sharey=False, figsize=(6, 6))
    for x in rows:
        ax.flatten()[0].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_evap_soil'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[0].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[0].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[1].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_transp'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[1].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[1].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[2].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_rz'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[2].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[2].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[3].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_cpr_rz'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[3].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[3].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    for x in rows:
        ax.flatten()[4].plot(ds_sim_tm['Time'].values, ds_sim_tm['C_iso_ss'].isel(x=x, y=0).values, color=cmap(ds_params['a_q_ss'].isel(x=x, y=0).values), lw=1)
    ax[4].set_ylabel(r'$\delta^{18}$O [‰]')
    ax.flatten()[4].set_xlim(ds_sim_tm['Time'].values[0], ds_sim_tm['Time'].values[-1])
    ax[4].set_xlabel('Time')
    fig.tight_layout()
    file = base_path_figs / f"d18O_uptake_{tms}_b{b}_a{a}.png"
    fig.savefig(file, dpi=250)
