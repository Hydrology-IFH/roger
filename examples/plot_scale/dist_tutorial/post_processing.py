import os
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

# load simulation
states_hm_file = base_path / "states_hm.nc"
ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

# assign date
days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

# plot simulated time series
vars_sim = ["aet", "transp", "evap_soil", "inf_mat", "inf_mp", "inf_sc", "q_ss", "q_sub", "q_sub_mp", "q_sub_mat", "q_hof", "q_sof"]
for var_sim in vars_sim:
    sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
    df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
    df_sim.loc[:, var_sim] = sim_vals
    fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [days]')
    fig2 = eval_utils.plot_sim_cum(df_sim, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [days]')
