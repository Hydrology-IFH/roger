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

    # load hydrologic simulations
    meteo_stations = ["breitnau", "ihringen"]
    for meteo_station in meteo_stations:
        sim_file = base_path_output / "SVAT.nc"
        ds_sim = xr.open_dataset(sim_file, engine="h5netcdf", group=meteo_station)

        # assign date
        days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
        ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

        # calculate the total sum of the numerical error
        num_error = ds_sim["dS_num_error"].values
        num_error_sum = onp.nansum(num_error, axis=-1).flatten()
        num_error_sum1 = onp.where(num_error_sum <= 0, onp.nan, num_error_sum)

        # plot the numerical error
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(range(num_error_sum1.size), num_error_sum1, width=0.1, color="black", align="edge", edgecolor="black")
        ax.set_xlim(0, num_error_sum1.size)
        plt.xlabel("# grid")
        plt.ylabel("Total error [mm]")
        plt.tight_layout()
        file = base_path_figs / f"total_error_{meteo_station}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

        # load the model parameters
        params_file = base_path / "parameters.csv"
        df_params_error = pd.read_csv(params_file, sep=";", skiprows=1)
        df_params_error.loc[:, "num_error"] = num_error_sum1.flatten()
        df_params_error.to_csv(base_path_figs / f"params_error_{meteo_station}.csv", sep=";", index=False)

    return


if __name__ == "__main__":
    main()
