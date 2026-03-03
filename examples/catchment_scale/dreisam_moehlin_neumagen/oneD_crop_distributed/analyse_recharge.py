import os
import xarray as xr
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt


@click.command("main")
def main():

    base_path = Path(__file__).parent

    # load model parameters
    model_parameters_file = base_path / "parameters_roger.nc"
    with xr.open_dataset(model_parameters_file) as infile:
        _mask = infile["maskCatch"].values == 1
        mask = _mask[np.newaxis, :, :]
        xcoords = infile["x"].values
        ycoords = infile["y"].values

    # directory of results
    base_path_output = base_path / "output"

    years = np.arange(2013, 2024)

    figures_folder = base_path / "figures" / "recharge"
    figures_folder.mkdir(parents=True, exist_ok=True)

    list_base_recharge = []
    list_base_recharge_annual_sum = []

    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        click.echo(f"Loading recharge data from {base_file}")
        ds_base = xr.open_dataset(base_file)
        base_recharge = ds_base["recharge"].values
        ds_base.close()
        base_recharge = np.where(mask, base_recharge, np.nan)
        base_recharge = np.where(base_recharge < 0, np.nan, base_recharge)
        list_base_recharge.append(base_recharge)
        base_recharge_annual_sum = np.sum(base_recharge, axis=0)
        base_recharge_annual_sum = np.where(base_recharge_annual_sum > 900, 900, base_recharge_annual_sum)
        click.echo(f"Annual sum of recharge for {year}: {np.nanmean(base_recharge_annual_sum)} mm/year")
        cond = np.isfinite(base_recharge_annual_sum)
        base_recharge_annual_sum_ = base_recharge_annual_sum[cond]
        list_base_recharge_annual_sum.append(base_recharge_annual_sum_.flatten())

    # make boxplot with annual sums of recharge
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.boxplot(list_base_recharge_annual_sum, showfliers=False, color='lightblue', medianprops={'color': 'black'})
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Direkte GWN\n[mm/Jahr]")
    ax.set_xticks(range(1, len(years) + 1))
    ax.set_xticklabels(years)
    ax.set_ylim(0, )
    fig.tight_layout()
    fig.savefig(figures_folder / "annual_recharge_boxplot.png")
    plt.close(fig)

    return


if __name__ == "__main__":
    main()