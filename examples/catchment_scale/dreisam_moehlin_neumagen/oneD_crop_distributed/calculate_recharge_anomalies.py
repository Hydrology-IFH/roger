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

    figures_folder = base_path / "figures" / "recharge_anomalies"
    figures_folder.mkdir(parents=True, exist_ok=True)

    list_base_recharge_annual_sum = []
    list_base_recharge_winter_sum = []
    list_base_recharge_spring_sum = []
    list_base_recharge_summer_sum = []
    list_base_recharge_autumn_sum = []
    list_base_recharge_march_sum = []
    list_base_recharge_april_sum = []
    list_base_recharge_may_sum = []
    list_base_recharge_june_sum = []
    list_base_recharge_july_sum = []
    list_base_recharge_august_sum = []

    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_recharge = ds_base["recharge"].values
        base_recharge = np.where(mask, base_recharge, np.nan)
        ds_base.close()
        base_recharge = np.where(base_recharge < 0, np.nan, base_recharge)
        base_recharge_annual_sum = np.sum(base_recharge, axis=0)
        list_base_recharge_annual_sum.append(base_recharge_annual_sum)
        date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
        cond_winter = (date_time.month == 12) | (date_time.month <= 2)
        cond_spring = (date_time.month >= 3) & (date_time.month <= 5)
        cond_summer = (date_time.month >= 6) & (date_time.month <= 8)
        cond_autumn = (date_time.month >= 9) & (date_time.month <= 11)
        base_recharge_winter_sum = np.sum(base_recharge[cond_winter, :, :], axis=0)
        base_recharge_spring_sum = np.sum(base_recharge[cond_spring, :, :], axis=0)
        base_recharge_summer_sum = np.sum(base_recharge[cond_summer, :, :], axis=0)
        base_recharge_autumn_sum = np.sum(base_recharge[cond_autumn, :, :], axis=0)
        list_base_recharge_winter_sum.append(base_recharge_winter_sum)
        list_base_recharge_spring_sum.append(base_recharge_spring_sum)
        list_base_recharge_summer_sum.append(base_recharge_summer_sum)
        list_base_recharge_autumn_sum.append(base_recharge_autumn_sum)
        list_base_recharge_march_sum.append(np.sum(base_recharge[date_time.month == 3, :, :], axis=0))
        list_base_recharge_april_sum.append(np.sum(base_recharge[date_time.month == 4, :, :], axis=0))
        list_base_recharge_may_sum.append(np.sum(base_recharge[date_time.month == 5, :, :], axis=0))
        list_base_recharge_june_sum.append(np.sum(base_recharge[date_time.month == 6, :, :], axis=0))
        list_base_recharge_july_sum.append(np.sum(base_recharge[date_time.month == 7, :, :], axis=0))
        list_base_recharge_august_sum.append(np.sum(base_recharge[date_time.month == 8, :, :], axis=0))

    base_recharge_annual_sum_avg = np.nanmean(np.stack(list_base_recharge_annual_sum, axis=0), axis=0)
    base_recharge_winter_sum_avg = np.nanmean(np.stack(list_base_recharge_winter_sum, axis=0), axis=0)
    base_recharge_spring_sum_avg = np.nanmean(np.stack(list_base_recharge_spring_sum, axis=0), axis=0)
    base_recharge_summer_sum_avg = np.nanmean(np.stack(list_base_recharge_summer_sum, axis=0), axis=0)
    base_recharge_autumn_sum_avg = np.nanmean(np.stack(list_base_recharge_autumn_sum, axis=0), axis=0)
    base_recharge_march_sum_avg = np.nanmean(np.stack(list_base_recharge_march_sum, axis=0), axis=0)
    base_recharge_april_sum_avg = np.nanmean(np.stack(list_base_recharge_april_sum, axis=0), axis=0)
    base_recharge_may_sum_avg = np.nanmean(np.stack(list_base_recharge_may_sum, axis=0), axis=0)
    base_recharge_june_sum_avg = np.nanmean(np.stack(list_base_recharge_june_sum, axis=0), axis=0)
    base_recharge_july_sum_avg = np.nanmean(np.stack(list_base_recharge_july_sum, axis=0), axis=0)
    base_recharge_august_sum_avg = np.nanmean(np.stack(list_base_recharge_august_sum, axis=0), axis=0)

    base_recharge_annual_sum_avg = np.where(base_recharge_annual_sum_avg < 0, np.nan, base_recharge_annual_sum_avg)
    base_recharge_winter_sum_avg = np.where(base_recharge_winter_sum_avg < 0, np.nan, base_recharge_winter_sum_avg)
    base_recharge_spring_sum_avg = np.where(base_recharge_spring_sum_avg < 0, np.nan, base_recharge_spring_sum_avg)
    base_recharge_summer_sum_avg = np.where(base_recharge_summer_sum_avg < 0, np.nan, base_recharge_summer_sum_avg)
    base_recharge_autumn_sum_avg = np.where(base_recharge_autumn_sum_avg < 0, np.nan, base_recharge_autumn_sum_avg)
    base_recharge_march_sum_avg = np.where(base_recharge_march_sum_avg < 0, np.nan, base_recharge_march_sum_avg)
    base_recharge_april_sum_avg = np.where(base_recharge_april_sum_avg < 0, np.nan, base_recharge_april_sum_avg)
    base_recharge_may_sum_avg = np.where(base_recharge_may_sum_avg < 0, np.nan, base_recharge_may_sum_avg)
    base_recharge_june_sum_avg = np.where(base_recharge_june_sum_avg < 0, np.nan, base_recharge_june_sum_avg)
    base_recharge_july_sum_avg = np.where(base_recharge_july_sum_avg < 0, np.nan, base_recharge_july_sum_avg)
    base_recharge_august_sum_avg = np.where(base_recharge_august_sum_avg < 0, np.nan, base_recharge_august_sum_avg)

    grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_annual_sum_avg, cmap="viridis_r", vmin=0, vmax=600, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/year]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_spring_sum_avg, cmap="viridis_r", vmin=0, vmax=200, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/3 months]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_spring.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_summer_sum_avg, cmap="viridis_r", vmin=0, vmax=200, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/3 months]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_summer.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_march_sum_avg, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_march.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_april_sum_avg, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_april.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_may_sum_avg, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_may.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_june_sum_avg, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_june.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(base_recharge_july_sum_avg, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
    fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
    ax.set_xlabel('x-coordinate', fontsize=12)
    ax.set_ylabel('y-coordinate', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    output_file = figures_folder / f"base_recharge_july.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)


    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_recharge = ds_base["recharge"].values
        base_recharge = np.where(mask, base_recharge, np.nan)
        ds_base.close()
        base_recharge_annual_sum = np.sum(base_recharge, axis=0)
        scenario_file = str(base_path_output / f"recharge_summer-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_scenario = xr.open_dataset(scenario_file)
        scenario_recharge = ds_scenario["recharge"].values
        scenario_recharge = np.where(mask, scenario_recharge, np.nan)
        ds_scenario.close()
        scenario_recharge = np.where(scenario_recharge < 0, np.nan, scenario_recharge)
        scenario_recharge_annual_sum = np.sum(scenario_recharge, axis=0)
        recharge_anomaly = ((scenario_recharge_annual_sum - base_recharge_annual_sum_avg) / base_recharge_annual_sum_avg) * 100
        recharge_anomaly_mm = scenario_recharge_annual_sum - base_recharge_annual_sum_avg

        grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
        fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
        ax.set_xlabel('x-coordinate', fontsize=12)
        ax.set_ylabel('y-coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title(f"{year}", fontsize=14)
        output_file = figures_folder / f"recharge_anomalies_{year}_summer-drought-magnitude2-duration0.png"
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
        output_file = figures_folder / f"recharge_anomalies_mm_{year}_summer-drought-magnitude2-duration0.png"
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

        if year == 2018:
            # calculate anomaly of summer recharge (June, July, August)
            date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
            cond_summer = (date_time.month >= 6) & (date_time.month <= 8)
            scenario_recharge_summer_sum = np.sum(scenario_recharge[cond_summer, :, :], axis=0)
            base_recharge_summer_sum = np.sum(base_recharge[cond_summer, :, :], axis=0)

            recharge_anomaly = ((scenario_recharge_summer_sum - base_recharge_summer_sum) / base_recharge_summer_sum) * 100
            recharge_anomaly_mm = scenario_recharge_summer_sum - base_recharge_summer_sum

            grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-summer_summer-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-summer_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_summer_sum, cmap="viridis_r", vmin=0, vmax=200, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/3 months]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-summer.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(scenario_recharge_summer_sum, cmap="viridis_r", vmin=0, vmax=200, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/3 months]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"scenario_recharge_{year}-summer.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_june = (date_time.month == 6)
            scenario_recharge_june_sum = np.sum(scenario_recharge[cond_june, :, :], axis=0)
            base_recharge_june_sum = np.sum(base_recharge[cond_june, :, :], axis=0)

            recharge_anomaly = ((scenario_recharge_june_sum - base_recharge_june_sum) / base_recharge_june_sum) * 100
            recharge_anomaly_mm = scenario_recharge_june_sum - base_recharge_june_sum

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-june_summer-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-june_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_june_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-june.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(scenario_recharge_june_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"scenario_recharge_{year}-june.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_july = (date_time.month == 7)
            scenario_recharge_july_sum = np.sum(scenario_recharge[cond_july, :, :], axis=0)
            base_recharge_july_sum = np.sum(base_recharge[cond_july, :, :], axis=0)
            recharge_anomaly = ((scenario_recharge_july_sum - base_recharge_july_sum_avg) / base_recharge_july_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_july_sum - base_recharge_july_sum_avg

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-july_summer-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-july_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_july_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-july.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_august = (date_time.month == 8)
            scenario_recharge_august_sum = np.sum(scenario_recharge[cond_august, :, :], axis=0)
            base_recharge_august_sum = np.sum(base_recharge[cond_august, :, :], axis=0)
            recharge_anomaly = ((scenario_recharge_august_sum - base_recharge_august_sum_avg) / base_recharge_august_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_august_sum - base_recharge_august_sum_avg
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-august_summer-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-august_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_august_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-august.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)


    for year in years:
        base_file = str(base_path_output / f"recharge_base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_base = xr.open_dataset(base_file)
        base_recharge = ds_base["recharge"].values
        base_recharge = np.where(mask, base_recharge, np.nan)
        ds_base.close()
        base_recharge = np.where(base_recharge < 0, np.nan, base_recharge)
        base_recharge_annual_sum = np.sum(base_recharge, axis=0)
        scenario_file = str(base_path_output / f"recharge_spring-drought-magnitude2-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
        ds_scenario = xr.open_dataset(scenario_file)
        scenario_recharge = ds_scenario["recharge"].values
        scenario_recharge = np.where(mask, scenario_recharge, np.nan)
        ds_scenario.close()
        scenario_recharge = np.where(scenario_recharge < 0, np.nan, scenario_recharge)
        scenario_recharge_annual_sum = np.sum(scenario_recharge, axis=0)
        recharge_anomaly = ((scenario_recharge_annual_sum - base_recharge_annual_sum_avg) / base_recharge_annual_sum_avg) * 100
        recharge_anomaly_mm = scenario_recharge_annual_sum - base_recharge_annual_sum_avg

        grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
        fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
        ax.set_xlabel('x-coordinate', fontsize=12)
        ax.set_ylabel('y-coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_title(f"{year}", fontsize=14)
        output_file = figures_folder / f"recharge_anomalies_{year}-summer_summer-drought-magnitude2-duration0.png"
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
        output_file = figures_folder / f"recharge_anomalies_mm_{year}-summer_summer-drought-magnitude2-duration0.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        if year == 2020:
            # calculate anomaly of summer recharge (June, July, August)
            date_time = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
            cond_spring = (date_time.month >= 3) & (date_time.month <= 5)
            scenario_recharge_spring_sum = np.sum(scenario_recharge[cond_spring, :, :], axis=0)
            base_recharge_spring_sum = np.sum(base_recharge[cond_spring, :, :], axis=0)

            recharge_anomaly = ((scenario_recharge_spring_sum - base_recharge_spring_sum_avg) / base_recharge_spring_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_spring_sum - base_recharge_spring_sum_avg

            grid_extent = (xcoords[0], xcoords[-1], ycoords[-1], ycoords[0])

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-spring_spring-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-spring_spring-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_spring_sum, cmap="viridis_r", vmin=0, vmax=200, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/3 months]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-spring.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_march = (date_time.month == 3)
            scenario_recharge_march_sum = np.sum(scenario_recharge[cond_march, :, :], axis=0)
            base_recharge_march_sum = np.sum(base_recharge[cond_march, :, :], axis=0)

            recharge_anomaly = ((scenario_recharge_march_sum - base_recharge_march_sum_avg) / base_recharge_march_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_march_sum - base_recharge_march_sum_avg
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-march_spring-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-june_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_june_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-march.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_april = (date_time.month == 4)
            scenario_recharge_april_sum = np.sum(scenario_recharge[cond_april, :, :], axis=0)
            base_recharge_april_sum = np.sum(base_recharge[cond_april, :, :], axis=0)
            recharge_anomaly = ((scenario_recharge_april_sum - base_recharge_april_sum_avg) / base_recharge_april_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_april_sum - base_recharge_april_sum_avg

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-april_spring-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-april_summer-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_april_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-april.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            cond_may = (date_time.month == 5)
            scenario_recharge_may_sum = np.sum(scenario_recharge[cond_may, :, :], axis=0)
            base_recharge_may_sum = np.sum(base_recharge[cond_may, :, :], axis=0)
            recharge_anomaly = ((scenario_recharge_may_sum - base_recharge_may_sum_avg) / base_recharge_may_sum_avg) * 100
            recharge_anomaly_mm = scenario_recharge_may_sum - base_recharge_may_sum_avg
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(recharge_anomaly, cmap="RdBu", vmin=-100, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge Anomaly [%]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"recharge_anomalies_{year}-may_spring-drought-magnitude2-duration0.png"
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
            output_file = figures_folder / f"recharge_anomalies_mm_{year}-may_spring-drought-magnitude2-duration0.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(base_recharge_may_sum, cmap="viridis_r", vmin=0, vmax=100, extent=grid_extent)
            fig.colorbar(ax.images[0], ax=ax, label="Recharge [mm/month]")
            ax.set_xlabel('x-coordinate', fontsize=12)
            ax.set_ylabel('y-coordinate', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            ax.set_title(f"{year}", fontsize=14)
            output_file = figures_folder / f"base_recharge_{year}-may.png"
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)


    return


if __name__ == "__main__":
    main()