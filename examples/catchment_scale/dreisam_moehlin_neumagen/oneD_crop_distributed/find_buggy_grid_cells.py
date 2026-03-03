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
        lu_id = infile["lanu"].values
        z_soil = infile["GRUND"].values / 1000  # convert from mm to m
        slope = infile["slope"].values / 100  # convert from % to m/m
        sealing = infile["vers"].values / 100  # convert from % to m/m
        ks = infile["KS"].values  # in m/s

    # directory of results
    base_path_output = base_path / "output"

    years = np.arange(2013, 2024)

    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        click.echo(f"Loading recharge data from {base_file}")
        ds_base = xr.open_dataset(base_file)
        base_recharge = ds_base["recharge"].values
        ds_base.close()
        base_recharge = np.where(mask, base_recharge, np.nan)
        base_recharge = np.where(base_recharge < 0, np.nan, base_recharge)
        base_recharge_annual_sum = np.sum(base_recharge, axis=0)
        rows_cols = np.where(base_recharge_annual_sum > 1200)
        click.echo(f"Found {len(rows_cols[0])} buggy grid cells in year {year}")
        # loop over the rows and columns of the buggy grid cells and print the rows, columns, lu_id, z_soil, slope, sealing and ks values
        for row, col in zip(rows_cols[0], rows_cols[1]):
            click.echo(f"Row: {row}, Column: {col}, LU ID: {lu_id[row, col]}, soil depth: {z_soil[row, col]}, Slope: {slope[row, col]}, Sealing: {sealing[row, col]}, KS: {ks[row, col]}")

    return


if __name__ == "__main__":
    main()