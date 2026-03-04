import os
from pathlib import Path
import xarray as xr
import numpy as onp
import pandas as pd
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


@click.command("main")
def main():
    base_path = Path(__file__).parent
    # directory of output
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures" / "recharge"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load configuration file
    file_config = base_path / "config.yml"
    with open(file_config, "r") as file:
        config = yaml.safe_load(file)

    # load the model parameters
    param_file = base_path / "parameters_roger.nc"
    ds_param = xr.open_dataset(param_file)
    mask_crops = (ds_param["lanu"].values == 5)

    years = onp.arange(2013, 2023).tolist()
    for year in years:
        # load the hydrological simulations
        sim_file = base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc"
        ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        # assign date
        days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        ds_sim = ds_sim.assign_coords(date=("Time", days_sim))

        vals = onp.nansum(ds_sim["recharge"].values, axis=0)
        ds_sim.close()
        vals = onp.where(vals < 0, onp.nan, vals)
        fig, ax = plt.subplots(figsize=(6,4))
        extent = (ds_param.x.values[0] / 1000, ds_param.x.values[-1] / 1000, ds_param.y.values[-1] / 1000, ds_param.y.values[0] / 1000)
        im = ax.imshow(vals, cmap='viridis_r', zorder=2, extent=extent, vmin=0, vmax=600)
        plt.colorbar(im, ax=ax, shrink=0.7, label="Direkte GWN\n[mm/Jahr]")
        plt.xlabel('X-Koordinate')
        plt.ylabel('Y-Koordinate')
        plt.grid(zorder=-1)
        plt.tight_layout()
        file = base_path_figs / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_spatial_annual_sum_year{year}.png"
        fig.savefig(file, dpi=300)
        plt.close(fig)

    list_monthly_arrays = []
    for year in years:
        # load the hydrological simulations
        sim_file = base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc"
        ds_sim = xr.open_dataset(sim_file, engine="h5netcdf")

        # assign date
        days_sim = (ds_sim['Time'].values / onp.timedelta64(24 * 60 * 60, "s"))
        time = pd.date_range(start=f"{year}-01-31", periods=len(days_sim), freq="D")
        ds_sim["Time"] = time
        click.echo(ds_sim["Time"].values[0])

        vals = ds_sim["recharge"].resample(Time="ME").sum().values
        list_monthly_arrays.append(vals)
        ds_sim.close()

    vals_monthly = onp.concatenate(list_monthly_arrays, axis=0)
    vals_monthly = onp.where(vals_monthly < 0, onp.nan, vals_monthly)
    vals_monthly = onp.where(vals_monthly > 300, 300, vals_monthly)

    # plot time series of monthly recharge showing average, min and max
    vals_mean = onp.nanmean(onp.nanmean(vals_monthly, axis=2), axis=1)
    vals_median = onp.nanmedian(onp.nanmedian(vals_monthly, axis=2), axis=1)
    vals_99 = onp.nanpercentile(onp.nanpercentile(vals_monthly, 99, axis=2), 99, axis=1)
    vals_1 = onp.nanpercentile(onp.nanpercentile(vals_monthly, 1, axis=2), 1, axis=1)
    time = pd.date_range(start="2013-01-31", periods=vals_monthly.shape[0], freq="M")
    fig, axes = plt.subplots(figsize=(6, 2))
    axes.plot(time, vals_median, color='blue', label='median GWN')
    axes.plot(time, vals_mean, color='black', label='mittlere GWN', linestyle='--', lw=1)
    axes.fill_between(time, vals_1, vals_99, color='lightblue', alpha=0.5, label='1%-99% GWN')
    axes.set_xlim(time[0], time[-1])
    axes.set_xlabel('Zeit')
    axes.set_ylabel('Direkte GWN\n[mm/Monat]')
    plt.tight_layout()
    file = base_path_figs / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_monthly_timeseries.png"
    fig.savefig(file, dpi=300)
    plt.close(fig)


    return


if __name__ == "__main__":
    main()