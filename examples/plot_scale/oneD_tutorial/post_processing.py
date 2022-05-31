import os
import glob
import h5netcdf
import datetime
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as onp
import seaborn as sns
import matplotlib.pyplot as plt
import roger.tools.labels as labs
import roger.tools.evaluation as eval_utils

sns.set_context("talk", font_scale=1)

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
path = str(base_path / "ONED.*.nc")
diag_files = glob.glob(path)
states_hm_file = base_path / "states_hm.nc"
with h5netcdf.File(states_hm_file, 'w', decode_vlen_strings=False) as f:
    f.attrs.update(
        date_created=datetime.datetime.today().isoformat(),
        title='RoGeR model results for a single event (Tutorial)',
        institution='University of Freiburg, Chair of Hydrology',
        references='',
        comment='1D model with free drainage'
    )
    for dfs in diag_files:
        with h5netcdf.File(dfs, 'r', decode_vlen_strings=False) as df:
            # set dimensions with a dictionary
            if not f.dimensions:
                dict_dim = {'x': len(df.variables['x']), 'y': len(df.variables['y']), 'Time': len(df.variables['Time'])}
                f.dimensions = dict_dim
                v = f.create_variable('x', ('x',), float)
                v.attrs['long_name'] = 'Model run'
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["x"])
                v = f.create_variable('y', ('y',), float)
                v.attrs['long_name'] = ''
                v.attrs['units'] = ''
                v[:] = onp.arange(dict_dim["y"])
                v = f.create_variable('Time', ('Time',), float)
                var_obj = df.variables.get('Time')
                v.attrs.update(units="hours")
                v[:] = onp.array(var_obj)
            for key in list(df.variables.keys()):
                var_obj = df.variables.get(key)
                if key not in list(f.dimensions.keys()) and var_obj.ndim == 3:
                    v = f.create_variable(key, ('x', 'y', 'Time'), float)
                    vals = onp.array(var_obj)
                    v[:, :, :] = vals.swapaxes(0, 2)
                    v.attrs.update(long_name=var_obj.attrs["long_name"],
                                   units=var_obj.attrs["units"])

# load simulation
states_hm_file = base_path / "states_hm.nc"
ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

# assign date
days_sim = (ds_sim['Time'].values / onp.timedelta64(60 * 60, "s"))
ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

# plot simulated time series
vars_sim = ["aet", "transp", "evap_soil", "inf_mat", "inf_mp", "inf_sc", "q_ss", "q_sub", "q_sub_mp", "q_sub_mat", "q_hof", "q_sof"]
for var_sim in vars_sim:
    sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
    df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
    df_sim.loc[:, var_sim] = sim_vals
    fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [days]')
    fig2 = eval_utils.plot_sim_cum(df_sim, y_lab=labs._Y_LABS_CUM[var_sim], x_lab='Time [days]')

vars_sim = ["theta"]
for var_sim in vars_sim:
    sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
    df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
    df_sim.loc[:, var_sim] = sim_vals
    fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._LABS[var_sim], x_lab='Time [days]')
