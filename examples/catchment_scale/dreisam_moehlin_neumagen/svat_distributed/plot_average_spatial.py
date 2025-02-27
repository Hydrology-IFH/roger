import os
from pathlib import Path
import xarray as xr
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

    # load the hydrological simulations
    sim_file = base_path_output / f"{config['identifier']}.nc"
    ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")

    # assign date
    days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
    ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

    vars_sim = ["theta"]
    for var_sim in vars_sim:
        vals = onp.nanmean(ds_sim[var_sim].values[:, :, :365], axis=-1)
        vals = onp.where(vals < 0, onp.nan, vals)
        fig, ax = plt.subplots(figsize=(6,4))
        extent = (0, config["ny"]*25/1000, 0, config["nx"]*25/1000)
        im = ax.imshow(vals, cmap='viridis', zorder=2, vmin=0.1, vmax=0.6, extent=extent)
        plt.colorbar(im, ax=ax, shrink=0.7, label=_lab_unit2[var_sim])
        plt.xlabel('Distance in x-direction [km]')
        plt.ylabel('Distance in y-direction [km]')
        plt.grid(zorder=-1)
        plt.tight_layout()
        file = base_path_figs / f"{var_sim}_avg.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)
    return


if __name__ == "__main__":
    main()