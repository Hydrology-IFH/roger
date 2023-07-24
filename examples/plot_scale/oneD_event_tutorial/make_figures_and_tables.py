import os
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as onp
import yaml
import click
import roger.tools.labels as labs
import roger.tools.evaluation as eval_utils


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of output
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load configuration file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # load hydrologic simulations
    states_hm_file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(states_hm_file, engine="h5netcdf")

    # assign date
    hours_sim = onp.around(ds_sim['Time'].values.astype('timedelta64[s]').astype(float) / (60 * 60), 3)
    ds_sim = ds_sim.assign_coords(date=("Time", hours_sim))

    # plot simulated time series
    vars_sim = ["inf_mat", "inf_mp", "inf_sc", "q_ss", "q_sub", "q_sub_mp", "q_sub_mat", "q_hof", "q_sof"]
    for var_sim in vars_sim:
        sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
        df_sim = pd.DataFrame(index=hours_sim, columns=[var_sim])
        df_sim.loc[:, var_sim] = sim_vals
        fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._Y_LABS_10mins[var_sim], x_lab='Time [hours]')
        fig2 = eval_utils.plot_sim_cum(df_sim, y_lab=labs._Y_LABS_CUM[var_sim], x_lab='Time [hours]')


    return


if __name__ == "__main__":
    main()