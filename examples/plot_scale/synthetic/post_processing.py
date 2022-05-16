import os
from pathlib import Path
from cftime import num2date
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp

import roger.tools.evaluation as eval_utils
import roger.tools.labels as labs

base_path = Path(__file__).parent
# directory of results
base_path_results = base_path / "results"
if not os.path.exists(base_path_results):
    os.mkdir(base_path_results)
# directory of figures
base_path_figs = base_path / "figures"
if not os.path.exists(base_path_figs):
    os.mkdir(base_path_figs)

meteo_stations = ["breitnau", "ihringen"]
vars_sim = ['aet', 'evap_soil', 'inf_mat', 'inf_mp', 'inf_sc', 'q_hof',
            'q_sof', 'q_ss', 'q_sub', 'q_sub_mat', 'q_sub_mp', 'transp']
ll_df_sim_sum = []
ll_df_sim_sum_tot = []
for i, meteo_station in enumerate(meteo_stations):
    # load simulation
    states_hm_file = base_path / "states_hm.nc"
    ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf", group=meteo_station)

    # assign date
    days_sim = ds_sim.Time.values / onp.timedelta64(24 * 60 * 60, "s")
    time_origin = "2010-09-30 00:00:00"
    date_sim = num2date(days_sim, units=f"days since {time_origin}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    # sums per grid
    ds_sim_sum = ds_sim.sum(dim="Time")
    nx = ds_sim_sum.dims['x']  # number of rows
    df = pd.DataFrame(index=range(nx))
    for var_sim in vars_sim:
        df.loc[:, var_sim] = ds_sim_sum[var_sim].values.flatten()
        df.loc[:, 'variable'] = var_sim
        df.loc[:, 'meteo_station'] = meteo_station

    ll_df_sim_sum.append(df)

    # total sums
    ds_sim_sum_tot = ds_sim.sum()
    df = pd.DataFrame(index=["sum"])
    for var_sim in vars_sim:
        df.loc[:, var_sim] = ds_sim_sum_tot[var_sim].values
        df.loc[:, 'variable'] = var_sim
        df.loc[:, 'meteo_station'] = meteo_station

    ll_df_sim_sum_tot.append(df)

# concatenate dataframes
df_sim_sum = pd.concat(ll_df_sim_sum, sort=False)
df_sim_sum_tot = pd.concat(ll_df_sim_sum_tot, sort=False)

nrow = len(vars_sim)
ncol = len(meteo_stations)

fig1, ax1 = plt.subplots(1, ncol, sharey=True, figsize=(14, 7))
fig2, ax2 = plt.subplots(nrow, ncol, sharey=True, figsize=(14, 7))



for j in range(ncol):
    xlabel = labs._LABS[df_params.columns[j]]
    ax[-1, j].set_xlabel(xlabel)

ax[0, 0].set_ylabel('$KGE_{ET}$ [-]')
ax[1, 0].set_ylabel('$KGE_{PERC}$ [-]')
ax[2, 0].set_ylabel(r'$r_{\Delta S}$ [-]')
ax[3, 0].set_ylabel('$E_{multi}$\n [-]')

fig1.tight_layout()
file1 = base_path_figs / "sums_per_grid.png"
fig1.savefig(file1, dpi=250)

fig2.tight_layout()
file2 = base_path_figs / "total_sums.png"
fig2.savefig(file2, dpi=250)
