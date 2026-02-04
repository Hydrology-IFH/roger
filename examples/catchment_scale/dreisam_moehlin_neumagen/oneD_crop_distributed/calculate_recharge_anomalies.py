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
    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed") / "output"

    years = np.arange(2013, 2024)

    figures_folder = base_path / "figures" / "recharge_anomalies"
    figures_folder.mkdir(parents=True, exist_ok=True)

    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        scenario_file = str(base_path_output / f"recharge_spring-summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_scenario = xr.open_dataset(scenario_file)
        base_recharge = ds_base["recharge"].values
        scenario_recharge = ds_scenario["recharge"].values
        base_recharge = np.where(base_recharge < 0, np.nan, base_recharge)
        scenario_recharge = np.where(scenario_recharge < 0, np.nan, scenario_recharge)
        base_recharge_annual_sum = np.nansum(base_recharge, axis=0)
        scenario_recharge_annual_sum = np.nansum(scenario_recharge, axis=0)
        recharge_anomaly = ((scenario_recharge_annual_sum - base_recharge_annual_sum) / base_recharge_annual_sum) * 100
        recharge_anomaly_mm = scenario_recharge_annual_sum - base_recharge_annual_sum

        xcoords = ds_base["x"].values
        ycoords = ds_base["y"].values
        grid_extent = (xcoords[0], xcoords[-1], ycoords[0], ycoords[-1])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
        fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
        ax.set_xlabel('x-coordinate', fontsize=12)
        ax.set_ylabel('y-coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title(f"{year}", fontsize=14)
        output_file = figures_folder / f"recharge_anomalies_{year}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(recharge_anomaly_mm, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
        fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [mm]")
        ax.set_xlabel('x-coordinate', fontsize=12)
        ax.set_ylabel('y-coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title(f"{year}", fontsize=14)
        output_file = figures_folder / f"recharge_anomalies_mm_{year}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(base_recharge_annual_sum, cmap="viridis_r", vmin=0, vmax=500, extent=grid_extent)
        fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/year]")
        ax.set_xlabel('x-coordinate', fontsize=12)
        ax.set_ylabel('y-coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title(f"{year}", fontsize=14)
        output_file = figures_folder / f"base_recharge_{year}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == "__main__":
    main()