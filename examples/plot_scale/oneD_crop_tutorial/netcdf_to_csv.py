import os
from pathlib import Path
import numpy as onp
import pandas as pd
from cftime import num2date
import xarray as xr
import click
import yaml


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of simulation results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)

    # load configuration file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # load simulation
    file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(file, engine="h5netcdf")
    # assign date
    days_sim = ds_sim["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_sim = num2date(
        days_sim,
        units=f"days since {ds_sim['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    # write output to csv
    vars_sim = config["OUTPUT_RATE"] + config["OUTPUT_COLLECT"]
    for var_sim in vars_sim:
        df_var_sim = pd.DataFrame(index=date_sim[1:], columns=[f"sim_{x}" for x in range(config["nx"])])
        df_var_sim.index = pd.Series(df_var_sim.index).dt.floor("D")
        df_var_sim.iloc[:, :] = ds_sim[var_sim].isel(y=0).values.T[1:, :]
        df_var_sim.to_csv(base_path_output / f"{var_sim}.csv", sep=";", index=True, index_label="date")

    return


if __name__ == "__main__":
    main()
