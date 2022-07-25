from pathlib import Path
import os
import glob
import datetime
import h5netcdf
import xarray as xr
from cftime import num2date
import pandas as pd
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('ticks', {'xtick.major.size': 8, 'ytick.major.size': 8})
sns.set_context("paper", font_scale=1.5)

base_path = Path(__file__).parent
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

# load observations (measured data)
path_obs = base_path / "observations" / "rietholzbach_lysimeter.nc"
ds_obs = xr.open_dataset(path_obs, engine="h5netcdf")
# assign date
days_obs = (ds_obs['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_obs = num2date(days_obs, units=f"days since {ds_obs['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_obs = ds_obs.assign_coords(date=("Time", date_obs))
df_obs = pd.DataFrame(index=date_obs)
df_obs.loc[:, 'd18O_prec'] = ds_obs['d18O_PREC'].isel(x=0, y=0).values
df_obs.loc[:, 'd18O_perc'] = ds_obs['d18O_PERC'].isel(x=0, y=0).values

# measured oxygen-18 in precipitation and percolation
d18O_prec_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_prec'].values), 2)
d18O_perc_mean = onp.round(onp.nanmean(df_obs.loc[:, 'd18O_perc'].values), 2)
fig, axs = plt.subplots(2, 1, figsize=(10, 6))
axs[0].plot(df_obs.index,
            df_obs.loc[:, 'd18O_prec'].fillna(method='bfill'),
            '-', color='blue')
axs[0].plot(df_obs.index,
            df_obs.loc[:, 'd18O_prec'],
            '.', color='blue')
axs[0].set_ylabel(r'$\delta^{18}$O [‰]')
axs[0].set_ylim([-20,0])
axs[0].set_xlim(df_obs.index[0], df_obs.index[-1])
axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'].fillna(method='bfill'),
         '-', color='grey')
axs[1].plot(df_obs.index, df_obs.loc[:, 'd18O_perc'],
         '.', color='grey')
axs[1].set_ylabel(r'$\delta^{18}$O [‰]')
axs[1].set_xlabel('Time [year]')
axs[1].set_ylim([-20,0])
axs[1].set_xlim(df_obs.index[0], df_obs.index[-1])
fig.tight_layout()
fig.text(0.115, 0.92, "(a)", ha="center", va="center")
fig.text(0.89, 0.615, r"$\overline{\delta^{18}O}_{prec}$: %s" % (d18O_prec_mean), ha="center", va="center")
fig.text(0.89, 0.155, r"$\overline{\delta^{18}O}_{perc}$: %s" % (d18O_perc_mean), ha="center", va="center")
fig.text(0.115, 0.46, "(b)", ha="center", va="center")
path_png = base_path_figs / 'observed_d18O_prec_perc'
fig.savefig(path_png, dpi=250)
plt.close(fig=fig)

# load best monte carlo run
states_hm_file = base_path / "states_hm.nc"
ds_sim_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")

# assign date
days_sim_hm = (ds_sim_hm['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
date_sim_hm = num2date(days_sim_hm, units=f"days since {ds_sim_hm['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
ds_sim_hm = ds_sim_hm.assign_coords(date=("Time", date_sim_hm))
