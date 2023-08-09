import os
from pathlib import Path
import xarray as xr
from cftime import num2date
import matplotlib.dates as mdates
import numpy as onp
import click
import yaml
import matplotlib as mpl
import seaborn as sns
import roger.tools.labels as labs

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 9
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 9.0,
    },
)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
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

    # load hydrologic simulations
    states_hm_file = base_path_output / f"{config['identifier']}.nc"
    ds_hm = xr.open_dataset(states_hm_file, engine="h5netcdf")
    # assign date
    days_hm = ds_hm["Time"].values / onp.timedelta64(24 * 60 * 60, "s")
    date_hm = num2date(
        days_hm,
        units=f"days since {ds_hm['Time'].attrs['time_origin']}",
        calendar="standard",
        only_use_cftime_datetimes=False,
    )
    ds_hm = ds_hm.assign_coords(Time=("Time", date_hm))

    # analyse or plot the results
    # for example, plot the simulated time series
    vars_sim = ['transp', 'q_ss', 'q_sub', 'theta']
    for j, var_sim in enumerate(vars_sim):
        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        vals = ds_hm[var_sim].isel(y=0).values
        median_vals = onp.median(vals, axis=0)
        min_vals = onp.min(vals, axis=0)
        max_vals = onp.max(vals, axis=0)
        p5_vals = onp.nanquantile(vals, 0.05, axis=0)
        p25_vals = onp.nanquantile(vals, 0.25, axis=0)
        p75_vals = onp.nanquantile(vals, 0.75, axis=0)
        p95_vals = onp.nanquantile(vals, 0.95, axis=0)
        ax.fill_between(days_hm[1:], min_vals[1:], max_vals[1:], edgecolor='red', facecolor='red', alpha=.33, label='Min-Max interval')
        ax.fill_between(days_hm[1:], p5_vals[1:], p95_vals[1:], edgecolor='red', facecolor='red', alpha=.66, label='95% interval')
        ax.fill_between(days_hm[1:], p25_vals[1:], p75_vals[1:], edgecolor='red', facecolor='red', alpha=1, label='75% interval')
        ax.plot(days_hm[1:], median_vals[1:], color='black', label='Median', linewidth=1)
        ax.legend(frameon=False, loc='upper right', ncol=4, bbox_to_anchor=(0.93, 1.19))
        ax.set_xlabel('Time [days]')
        ax.set_ylabel(labs._Y_LABS_DAILY[var_sim])
        ax.set_xlim(days_hm[1], days_hm[-1])
        fig.tight_layout()
        file = base_path_figs / f"{var_sim}.png"
        fig.savefig(file, dpi=250)
        plt.close(fig)
    return


if __name__ == "__main__":
    main()
