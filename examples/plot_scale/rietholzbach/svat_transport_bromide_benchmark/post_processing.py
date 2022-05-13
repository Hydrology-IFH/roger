from pathlib import Path
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)

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
        states_tm_file = base_path / "states_tm.nc"
        ds_sim_tm = xr.open_dataset(states_tm_file, group=f"{tm_structure}-{year}", engine="h5netcdf")

        # plot observed and simulated time series
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
        df_perc_br = df_perc_br.iloc[315:, :]
        idx = range(len(df_perc_br.index))
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes.plot(idx, df_perc_br['Br_conc_mmol'], color='black', ls='-', colors=cmap(norm(year)))
        axes.set_ylabel('Br [mmol $l^{-1}$]')
        axes.set_xlabel('Time [hours since injection]')
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
