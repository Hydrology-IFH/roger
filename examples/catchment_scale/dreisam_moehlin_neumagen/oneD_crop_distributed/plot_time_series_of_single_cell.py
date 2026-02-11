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
    base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"

    years = np.arange(2013, 2024)

    figures_folder = base_path / "figures" / "time_series"
    figures_folder.mkdir(parents=True, exist_ok=True)

    x = 1056
    y = 487
    for year in years:
        # if year == 2018:
        #     base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        #     ds_base = xr.open_dataset(base_file)
        #     base_recharge = ds_base["recharge"].values
        #     base_recharge = np.where(mask, base_recharge, np.nan)
        #     ds_base.close()
        #     scenario_file = str(base_path_output / f"recharge_summer-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        #     ds_scenario = xr.open_dataset(scenario_file)
        #     scenario_recharge = ds_scenario["recharge"].values
        #     scenario_recharge = np.where(mask, scenario_recharge, np.nan)
        #     ds_scenario.close()
        #     scenario_recharge = np.where(scenario_recharge < 0, np.nan, scenario_recharge)

        #     date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        #     base_recharge_ts = base_recharge[:, y, x]
        #     scenario_recharge_ts = scenario_recharge[:, y, x]

        #     fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        #     ax.plot(date_time, base_recharge_ts, label="Baseline", color="blue", linewidth=1.2, alpha=0.8)
        #     ax.plot(date_time, scenario_recharge_ts, label="Scenario", color="red", linewidth=1.5, alpha=0.9)
        #     ax.set_xlim(date_time[0], date_time[-1])
        #     ax.set_ylim(0,)
        #     # rotate xticklabels
        #     plt.xticks(rotation=33)
        #     ax.set_xlabel("Date")
        #     ax.set_ylabel("Recharge [mm/day]")
        #     output_file = figures_folder / f"recharge_time_series_{year}_summer-drought-magnitude2-duration0.png"
        #     fig.savefig(output_file, dpi=300, bbox_inches='tight')
        #     plt.close(fig)

        base_file = str(base_path_output / f"land_use_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_lu = ds_base["land_use"].values
        ds_base.close()
        base_lu = np.where(mask, base_lu, np.nan)
        date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        base_lu_ts = base_lu[:, y, x]

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(date_time, base_lu_ts, label="Baseline", color="black", linewidth=1.2, alpha=0.8)
        ax.set_xlim(date_time[0], date_time[-1])
        ax.set_ylim(500, 600)
        # rotate xticklabels
        plt.xticks(rotation=33)
        ax.set_xlabel("Date")
        ax.set_ylabel("Land Use")
        output_file = figures_folder / f"land_use_time_series_{year}_x{x}_y{y}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        base_file = str(base_path_output / f"ground_cover_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_ground_cover = ds_base["ground_cover"].values
        base_ground_cover = np.where(mask, base_ground_cover, np.nan)
        ds_base.close()
        date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        base_ground_cover_ts = base_ground_cover[:, y, x]

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(date_time, base_ground_cover_ts, label="Baseline", color="black", linewidth=1.2, alpha=0.8)
        ax.set_xlim(date_time[0], date_time[-1])
        ax.set_ylim(0, 1)
        # rotate xticklabels
        plt.xticks(rotation=33)
        ax.set_xlabel("Date")
        ax.set_ylabel("Ground Cover [-]")
        output_file = figures_folder / f"ground_cover_time_series_{year}_x{x}_y{y}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        base_file = str(base_path_output / f"root_depth_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_z_root = ds_base["z_root"].values
        base_z_root = np.where(mask, base_z_root, np.nan)
        ds_base.close()
        date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        base_z_root_ts = base_z_root[:, y, x]

        fig, ax = plt.subplots(1, 1, figsize=(6, 2))
        ax.plot(date_time, base_z_root_ts, label="Baseline", color="black", linewidth=1.2, alpha=0.8)
        ax.set_xlim(date_time[0], date_time[-1])
        ax.invert_yaxis()
        # rotate xticklabels
        plt.xticks(rotation=33)
        ax.set_xlabel("Date")
        ax.set_ylabel("Root Depth [mm]")
        output_file = figures_folder / f"root_depth_time_series_{year}_x{x}_y{y}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


    return


if __name__ == "__main__":
    main()