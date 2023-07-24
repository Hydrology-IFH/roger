import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import yaml
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of results
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

    # merge model output into single file
    path = str(base_path_output / f"{config['identifier']}.*.nc")
    ds_sim = xr.open_dataset(path, engine="h5netcdf", decode_times=False)
    # assign date
    date_sim = num2date(ds_sim['Time'].values, units=f"days since {ds_sim['Time'].attrs['time_origin']}", calendar='standard', only_use_cftime_datetimes=False)
    ds_sim_tm = ds_sim.assign_coords(Time=("Time", date_sim))
    
    return


if __name__ == "__main__":
    main()
