import os
from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as onp
import matplotlib.pyplot as plt
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
    sim_file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")

    # assign date
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

    # calculate the total sum of the numerical error
    num_error = ds_sim["dC_num_error"].values
    num_error_sum = onp.nansum(num_error, axis=-1)
    num_error_sum1 = onp.where(num_error_sum <= 0, onp.nan, num_error_sum)

    # plot the numerical error
    fig, ax = plt.subplots(figsize=(6,4))
    grid_extent = (0, 80*25, 0, 53*25)
    im = ax.imshow(num_error_sum1, extent=grid_extent, cmap='viridis', zorder=2)
    plt.colorbar(im, ax=ax, shrink=0.7, label="Total numerical error [mm]")
    plt.xlabel('Distance in x-direction [m]')
    plt.ylabel('Distance in y-direction [m]')
    plt.grid(zorder=-1)
    plt.tight_layout()
    file = base_path_figs / "error_sum.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    # load the model parameters
    params_file = base_path / "parameters.nc"
    ds_params = xr.open_dataset(params_file, engine="h5netcdf")

    # numerical error and model parameters
    df_params_error = pd.DataFrame()
    params = ["lu_id", "sealing", "z_soil", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
    for param in params:
        df_params_error.loc[:, param] = ds_params[param].values.flatten()
    df_params_error.loc[:, 'num_error'] = num_error_sum.flatten()
    df_params_error.to_csv(base_path_figs / "params_error.csv", sep=";", index=False)    

    return


if __name__ == "__main__":
    main()