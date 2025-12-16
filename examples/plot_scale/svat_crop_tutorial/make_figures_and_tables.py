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
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

    # plot simulated time series
    vars_sim = ["transp", "evap_soil", "q_ss", "z_root", "ground_cover"]
    for var_sim in vars_sim:
        sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
        df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
        df_sim.loc[:, var_sim] = sim_vals
        fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._Y_LABS_DAILY[var_sim], x_lab='Time [days]')
        path = base_path_figs / f"{var_sim}.png"
        fig1.savefig(path, dpi=300)

    vars_sim = ["theta"]
    for var_sim in vars_sim:
        sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
        df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
        df_sim.loc[:, var_sim] = sim_vals
        fig1 = eval_utils.plot_sim(df_sim, y_lab=labs._LABS[var_sim], x_lab='Time [days]')
        path = base_path_figs / f"{var_sim}.png"
        fig1.savefig(path, dpi=300)

    # plot simulated cumulative time series
    vars_sim = ["transp", "evap_soil", "q_ss"]
    for var_sim in vars_sim:
        sim_vals = ds_sim[var_sim].isel(x=0, y=0).values
        df_sim = pd.DataFrame(index=days_sim, columns=[var_sim])
        df_sim.loc[:, var_sim] = sim_vals
        fig2 = eval_utils.plot_sim_cum(df_sim, y_lab=labs._Y_LABS_CUM[var_sim], x_lab='Time [days]')
        path = base_path_figs / f"{var_sim}_cumulated.png"
        fig2.savefig(path, dpi=300)

    return


if __name__ == "__main__":
    main()