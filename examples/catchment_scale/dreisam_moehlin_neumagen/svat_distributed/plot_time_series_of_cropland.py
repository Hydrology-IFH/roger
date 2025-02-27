import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import numpy as onp
import matplotlib.pyplot as plt
import yaml
import click
import matplotlib as mpl
import seaborn as sns
mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.title_fontsize'] = 9
sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 8.0,
                                                'axes.labelsize': 9.0,
                                                'axes.titlesize': 8.0,
                                                'xtick.labelsize': 8.0,
                                                'ytick.labelsize': 8.0,
                                                'legend.fontsize': 8.0,
                                                'legend.title_fontsize': 9.0})

_lab_unit2 = {
    "theta": r"$\theta$ [-]",
}


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

    # load the model parameters
    param_file = base_path / "parameters.nc"
    ds_param = xr.open_dataset(param_file)
    mask_crops = (ds_param["lanu"].values == 5)

    # load the hydrological simulations
    sim_file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")

    # assign date
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    date_sim = num2date(
        days_sim,
        units=f"days since {ds_sim['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_sim = ds_sim.assign_coords(date=("Time", date_sim))

    var_sim = "theta"
    vals = ds_sim[var_sim].values[:, :, :365]
    vals = onp.where((vals < 0) | (vals > 0.6), onp.nan, vals)
    vals = onp.where(mask_crops[:, :, onp.newaxis], vals, onp.nan)
    min_vals = onp.nanmin(onp.nanmin(vals, axis=0), axis=0)
    median_vals = onp.nanmedian(onp.nanmedian(vals, axis=0), axis=0)
    max_vals = onp.nanmax(onp.nanmax(vals, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(date_sim[:365], median_vals, label="median", color="red")
    ax.fill_between(date_sim[:365], min_vals, max_vals, color="red", alpha=0.3)
    ax.set_xlim(date_sim[0], date_sim[364])
    ax.set_ylabel(r"$\theta$ [-]")
    ax.set_xlabel("Time")
    fig.tight_layout()
    file = base_path_figs / f"{var_sim}_time_series_cropland.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)

    var_sim = "irr_demand"
    vals = ds_sim[var_sim].values[:, :, :365]
    vals = onp.where(vals < 0, onp.nan, vals)
    vals = onp.where(mask_crops[:, :, onp.newaxis], vals, onp.nan)
    min_vals = onp.nanmin(onp.nanmin(vals, axis=0), axis=0)
    median_vals = onp.nanmedian(onp.nanmedian(vals, axis=0), axis=0)
    max_vals = onp.nanmax(onp.nanmax(vals, axis=0), axis=0)
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(date_sim[:365], median_vals, label="median", color="red")
    ax.fill_between(date_sim[:365], min_vals, max_vals, color="red", alpha=0.3)
    ax.set_xlim(date_sim[0], date_sim[364])
    ax.set_ylabel("irrigation demand\n [mm/day]")
    ax.set_xlabel("Time")
    fig.tight_layout()
    file = base_path_figs / f"{var_sim}_time_series_cropland.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)
    return


if __name__ == "__main__":
    main()