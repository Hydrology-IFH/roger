from pathlib import Path
import os
import glob
import h5netcdf
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import roger

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# merge model output into single file
tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
years = onp.arange(1997, 2007).tolist()
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    for year in years:
        path = str(base_path / f'SVATTRANSPORT_{tms}_{year}.*.nc')
        diag_files = glob.glob(path)
        states_tm_file = base_path / "states_tm_bromide_benchmark.nc"
        with h5netcdf.File(states_tm_file, 'a', decode_vlen_strings=False) as f:
            if f"{tm_structure}-{year}" not in list(f.groups.keys()):
                f.create_group(f"{tm_structure}-{year}")
            f.attrs.update(
                date_created=datetime.datetime.today().isoformat(),
                title='RoGeR transport model results for bromide benchmark at Rietholzbach Lysimeter site',
                institution='University of Freiburg, Chair of Hydrology',
                references='',
                comment='',
                model_structure='SVAT transport model with free drainage',
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
                    if not f.groups[f"{tm_structure}-{year}"].dimensions:
                        f.groups[f"{tm_structure}-{year}"].dimensions = dict_dim
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('x', ('x',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Number of model run'
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["x"])
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('y', ('y',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = ''
                        v.attrs['units'] = ''
                        v[:] = onp.arange(dict_dim["y"])
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('ages', ('ages',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Water ages'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(1, dict_dim["ages"]+1)
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('nages', ('nages',), float, compression="gzip", compression_opts=1)
                        v.attrs['long_name'] = 'Water ages (cumulated)'
                        v.attrs['units'] = 'days'
                        v[:] = onp.arange(0, dict_dim["nages"])
                        v = f.groups[f"{tm_structure}-{year}"].create_variable('Time', ('Time',), float, compression="gzip", compression_opts=1)
                        var_obj = df.variables.get('Time')
                        v.attrs.update(time_origin=var_obj.attrs["time_origin"],
                                       units=var_obj.attrs["units"])
                        v[:] = time
                    for var_sim in list(df.variables.keys()):
                        var_obj = df.variables.get(var_sim)
                        if var_sim not in list(dict_dim.keys()) and ('Time', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            v[:, :, :] = vals.swapaxes(0, 2)
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'n_sas_params', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'n_sas_params'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            v[:, :, :] = vals[:, :, :, 0]
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'ages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'ages'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])
                        elif var_sim not in list(dict_dim.keys()) and ('Time', 'nages', 'y', 'x') == var_obj.dimensions:
                            v = f.groups[f"{tm_structure}-{year}"].create_variable(var_sim, ('x', 'y', 'Time', 'nages'), float, compression="gzip", compression_opts=1)
                            vals = onp.array(var_obj)
                            vals = vals.swapaxes(0, 3)
                            vals = vals.swapaxes(1, 2)
                            vals = vals.swapaxes(2, 3)
                            v[:, :, :, :] = vals
                            v.attrs.update(long_name=var_obj.attrs["long_name"],
                                           units=var_obj.attrs["units"])

# load simulation
states_hm_file = base_path / "states_hm.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

tm_structures = ['complete-mixing', 'piston',
                 'preferential', 'advection-dispersion',
                 'time-variant preferential',
                 'time-variant advection-dispersion']
years = onp.arange(1997, 2007).tolist()
cmap = cm.get_cmap('Greys')
norm = Normalize(vmin=onp.min(years), vmax=onp.max(years))
for tm_structure in tm_structures:
    tms = tm_structure.replace(" ", "_")
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    for year in years:
        # load simulation
        states_tm_file = base_path / "states_tm_bromide_benchmark.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, group=f"{tm_structure}-{year}", engine="h5netcdf")

        # plot simulated time series
        base_path_figs = base_path / "figures"

        # assign date
        days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        days_sim_tm = (ds_sim_tm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        date_sim_tm = num2date(days_sim_tm, units=f"days since {ds_sim_tm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
        ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))
        ds_sim_tm = ds_sim_tm.assign_coords(date=("Time", date_sim_tm))

        # plot percolation rate (in l/h) and bromide concentration (mmol/l)
        df_perc_br = pd.DataFrame(index=date_sim_hm, columns=['perc', 'Br_conc_mg', 'Br_conc_mmol'])
        # in liter per hour
        df_perc_br.loc[:, 'perc'] = ds_sim_hm.sel(date=slice(str(year), str(year + 1)))['q_ss'].isel(x=0, y=0).values * (3.14/24)
        # in mg per liter
        df_perc_br.loc[:, 'Br_conc_mg'] = ds_sim_tm.sel(date=slice(str(year), str(year + 1)))['C_q_ss'].isel(x=0, y=0).values * (1/3.14)
        # in mmol per liter
        df_perc_br.loc[:, 'Br_conc_mmol'] = df_perc_br.loc[:, 'Br_conc_mg'] / 79.904
        # daily samples from day 0 to day 220
        df_daily = df_perc_br.loc[:df_perc_br.index[315+220], 'Br_conc_mmol'].to_frame()
        # weekly samples after 220 days
        df_weekly = df_perc_br.loc[df_perc_br.index[316+220]:, 'Br_conc_mmol'].resample('7D').mean().to_frame()
        df_daily_weekly = pd.concat([df_daily, df_weekly])
        df_perc_br = df_perc_br.loc[:, 'perc':'Br_conc_mg'].join(df_daily_weekly)
        idx = range(len(df_perc_br.index))
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(idx, df_perc_br['Br_conc_mmol'], color='black', ls='-', colors=cmap(norm(year)))
        axes.set_ylabel('Br [mmol $l^{-1}$]')
        axes.set_xlabel('Time [days since injection]')
        axes.set_ylim(0,)
        axes.set_xlim(0,)
        ax2 = axes.twinx()
        ax2.plot(idx, df_perc_br['perc'], lw=1.5, color='black', ls=':')
        ax2.set_ylabel('Percolation [l $hour^{-1}$]')
        ax2.set_ylim(0,)
        file = f'perc_br_{tms}.png'
        path = base_path_figs / file
        fig.savefig(path, dpi=250)

        fig_year, axes_year = plt.subplots(1, 1, figsize=(10, 6))
        axes_year.plot(idx, df_perc_br['Br_conc_mmol'], color='black', ls='-')
        axes_year.set_ylabel('Br [mmol $l^{-1}$]')
        axes_year.set_xlabel('Time [hours since injection]')
        axes_year.set_ylim(0,)
        axes_year.set_xlim(0,)
        ax2_year = axes.twinx()
        ax2_year.plot(idx, df_perc_br['perc'], lw=1.5, color='black', ls=':')
        ax2_year.set_ylabel('Percolation [l $hour^{-1}$]')
        ax2_year.set_ylim(0,)
        file = f'perc_br_{tms}_{year}.png'
        path = base_path_figs / file
        fig_year.savefig(path, dpi=250)
