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

    # load groundwater extraction data
    df_groundwater_extraction = pd.read_csv(base_path / "input" / "groundwater_extraction.csv", sep=";")
    df_groundwater_extraction["cell_y"] = df_groundwater_extraction["cell_y"].astype(int)
    df_groundwater_extraction["cell_x"] = df_groundwater_extraction["cell_x"].astype(int)
    df_groundwater_extraction["layer"] = df_groundwater_extraction["layer"].astype(int)
    df_groundwater_extraction["purpose"] = df_groundwater_extraction["purpose"].astype(str)
    df_groundwater_extraction["cell_y"] = df_groundwater_extraction["cell_y"].values - 1
    df_groundwater_extraction["cell_x"] = df_groundwater_extraction["cell_x"].values - 1
    df_groundwater_extraction["layer"] = df_groundwater_extraction["layer"].values - 1
    n_wells = len(df_groundwater_extraction)
    cond_drinking_water_supply = df_groundwater_extraction["purpose"].isin(['Badenova WW Ebnet', 'Badenova WW Hausen', 'Eigenwasserversorgung', 'oeffentliche Wasserversorgung']).values

    # load daily weights for drinking water supply wells to scale the pumping rates of the drinking water supply wells in the well package
    _daily_weights_drinking_water_supply = pd.read_csv(base_path / "input" / "daily_weights_drinking_water_supply.csv", sep=";", index_col=0)

    date_time = pd.date_range(start="2013-01-01", end="2023-12-31", freq="D")
    NDAYS = len(date_time)
    doys = date_time.dayofyear.values
    years = date_time.year.values

    daily_weights_drinking_water_supply = pd.DataFrame(index=date_time, columns=['weights'])
    for i in range(NDAYS):
        year = years[i]
        doy = doys[i]
        daily_weights_drinking_water_supply.iloc[i, 0] = _daily_weights_drinking_water_supply.loc[int(year), f"{int(doy)}"]

    df_drinking_water_well_extraction_daily = pd.DataFrame(index=date_time, columns=['well_extraction'])
    for i in range(NDAYS):
        year = years[i]
        extraction_year = df_groundwater_extraction.loc[cond_drinking_water_supply, f"{year}"].values.sum()
        df_drinking_water_well_extraction_daily.iloc[i, 0] = extraction_year * daily_weights_drinking_water_supply.iloc[i, 0]

    # save daily drinking water well extraction to csv
    file = base_path / "input" / "drinking_water_well_extraction_daily.csv"
    df_drinking_water_well_extraction_daily.to_csv(file, sep=";")

    # aggreate to monthly values
    df_drinking_water_well_extraction_monthly = df_drinking_water_well_extraction_daily.resample('ME').sum()

    # save monthly drinking water well extraction to csv
    file = base_path / "input" / "drinking_water_well_extraction_monthly.csv"
    df_drinking_water_well_extraction_monthly.to_csv(file, sep=";")

    # directory of results
    base_path_output = base_path / "output"

    years = np.arange(2013, 2024)

    figures_folder = base_path / "figures" / "recharge_anomalies"
    figures_folder.mkdir(parents=True, exist_ok=True)

    list_base_recharge_annual_sum = []
    list_base_recharge_winter_sum = []
    list_base_recharge_spring_sum = []
    list_base_recharge_summer_sum = []
    list_base_recharge_autumn_sum = []
    list_base_recharge_january_sum = []
    list_base_recharge_february_sum = []
    list_base_recharge_march_sum = []
    list_base_recharge_april_sum = []
    list_base_recharge_may_sum = []
    list_base_recharge_june_sum = []
    list_base_recharge_july_sum = []
    list_base_recharge_august_sum = []
    list_base_recharge_september_sum = []
    list_base_recharge_october_sum = []
    list_base_recharge_november_sum = []
    list_base_recharge_december_sum = []

    for year in years:
        base_file = str(base_path_output / f"recharge_base_2000-2024-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction_year{year}.nc")
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
        list_base_recharge_january_sum.append(np.sum(base_recharge[date_time.month == 1, :, :], axis=0))
        list_base_recharge_february_sum.append(np.sum(base_recharge[date_time.month == 2, :, :], axis=0))
        list_base_recharge_march_sum.append(np.sum(base_recharge[date_time.month == 3, :, :], axis=0))
        list_base_recharge_april_sum.append(np.sum(base_recharge[date_time.month == 4, :, :], axis=0))
        list_base_recharge_may_sum.append(np.sum(base_recharge[date_time.month == 5, :, :], axis=0))
        list_base_recharge_june_sum.append(np.sum(base_recharge[date_time.month == 6, :, :], axis=0))
        list_base_recharge_july_sum.append(np.sum(base_recharge[date_time.month == 7, :, :], axis=0))
        list_base_recharge_august_sum.append(np.sum(base_recharge[date_time.month == 8, :, :], axis=0))
        list_base_recharge_september_sum.append(np.sum(base_recharge[date_time.month == 9, :, :], axis=0))
        list_base_recharge_october_sum.append(np.sum(base_recharge[date_time.month == 10, :, :], axis=0))
        list_base_recharge_november_sum.append(np.sum(base_recharge[date_time.month == 11, :, :], axis=0))
        list_base_recharge_december_sum.append(np.sum(base_recharge[date_time.month == 12, :, :], axis=0))

    base_recharge_annual_sum_avg = np.nanmean(np.stack(list_base_recharge_annual_sum, axis=0), axis=0)
    base_recharge_winter_sum_avg = np.nanmean(np.stack(list_base_recharge_winter_sum, axis=0), axis=0)
    base_recharge_spring_sum_avg = np.nanmean(np.stack(list_base_recharge_spring_sum, axis=0), axis=0)
    base_recharge_summer_sum_avg = np.nanmean(np.stack(list_base_recharge_summer_sum, axis=0), axis=0)
    base_recharge_autumn_sum_avg = np.nanmean(np.stack(list_base_recharge_autumn_sum, axis=0), axis=0)
    base_recharge_january_sum_avg = np.nanmean(np.stack(list_base_recharge_january_sum, axis=0), axis=0)
    base_recharge_february_sum_avg = np.nanmean(np.stack(list_base_recharge_february_sum, axis=0), axis=0)
    base_recharge_march_sum_avg = np.nanmean(np.stack(list_base_recharge_march_sum, axis=0), axis=0)
    base_recharge_april_sum_avg = np.nanmean(np.stack(list_base_recharge_april_sum, axis=0), axis=0)
    base_recharge_may_sum_avg = np.nanmean(np.stack(list_base_recharge_may_sum, axis=0), axis=0)
    base_recharge_june_sum_avg = np.nanmean(np.stack(list_base_recharge_june_sum, axis=0), axis=0)
    base_recharge_july_sum_avg = np.nanmean(np.stack(list_base_recharge_july_sum, axis=0), axis=0)
    base_recharge_august_sum_avg = np.nanmean(np.stack(list_base_recharge_august_sum, axis=0), axis=0)
    base_recharge_september_sum_avg = np.nanmean(np.stack(list_base_recharge_september_sum, axis=0), axis=0)
    base_recharge_october_sum_avg = np.nanmean(np.stack(list_base_recharge_october_sum, axis=0), axis=0)
    base_recharge_november_sum_avg = np.nanmean(np.stack(list_base_recharge_november_sum, axis=0), axis=0)
    base_recharge_december_sum_avg = np.nanmean(np.stack(list_base_recharge_december_sum, axis=0), axis=0)

    base_recharge_annual_sum_avg = np.where(base_recharge_annual_sum_avg < 0, np.nan, base_recharge_annual_sum_avg)
    base_recharge_winter_sum_avg = np.where(base_recharge_winter_sum_avg < 0, np.nan, base_recharge_winter_sum_avg)
    base_recharge_spring_sum_avg = np.where(base_recharge_spring_sum_avg < 0, np.nan, base_recharge_spring_sum_avg)
    base_recharge_summer_sum_avg = np.where(base_recharge_summer_sum_avg < 0, np.nan, base_recharge_summer_sum_avg)
    base_recharge_autumn_sum_avg = np.where(base_recharge_autumn_sum_avg < 0, np.nan, base_recharge_autumn_sum_avg)
    base_recharge_january_sum_avg = np.where(base_recharge_january_sum_avg < 0, np.nan, base_recharge_january_sum_avg)
    base_recharge_february_sum_avg = np.where(base_recharge_february_sum_avg < 0, np.nan, base_recharge_february_sum_avg)
    base_recharge_march_sum_avg = np.where(base_recharge_march_sum_avg < 0, np.nan, base_recharge_march_sum_avg)
    base_recharge_april_sum_avg = np.where(base_recharge_april_sum_avg < 0, np.nan, base_recharge_april_sum_avg)
    base_recharge_may_sum_avg = np.where(base_recharge_may_sum_avg < 0, np.nan, base_recharge_may_sum_avg)
    base_recharge_june_sum_avg = np.where(base_recharge_june_sum_avg < 0, np.nan, base_recharge_june_sum_avg)
    base_recharge_july_sum_avg = np.where(base_recharge_july_sum_avg < 0, np.nan, base_recharge_july_sum_avg)
    base_recharge_august_sum_avg = np.where(base_recharge_august_sum_avg < 0, np.nan, base_recharge_august_sum_avg)
    base_recharge_september_sum_avg = np.where(base_recharge_september_sum_avg < 0, np.nan, base_recharge_september_sum_avg)
    base_recharge_october_sum_avg = np.where(base_recharge_october_sum_avg < 0, np.nan, base_recharge_october_sum_avg)
    base_recharge_november_sum_avg = np.where(base_recharge_november_sum_avg < 0, np.nan, base_recharge_november_sum_avg)
    base_recharge_december_sum_avg = np.where(base_recharge_december_sum_avg < 0, np.nan, base_recharge_december_sum_avg)

    scenarios = ["base-magnitude0-duration0_no-irrigation_no-yellow-mustard_soil-compaction",
                 "summer-drought-magnitude2-duration3_no-irrigation_no-yellow-mustard_soil-compaction"]

    for scenario in scenarios:
        dict_sustainability_of_extractions_m3 = {}
        dict_sustainability_of_extractions_percent = {}
        for year in [2016, 2017, 2018]:
                date_time_year = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")
                for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                    file = str(base_path_output / f"recharge_{scenario}_year{year}_month{month}.nc")
                    ds_recharge = xr.open_dataset(file)
                    recharge = ds_recharge["recharge"].values
                    ds_recharge.close()
                    recharge = np.where(mask, recharge, np.nan)
                    recharge_month = recharge[date_time_year.month == month, :, :]
                    recharge_monthly_sum = np.sum(recharge_month) * ((25 * 25) / 1000)  # convert mm/month to m3/month by multiplying with cell area (25m x 25m = 625 m2)
                    gw_extraction_monthly_sum = df_drinking_water_well_extraction_monthly.loc[f"{year}-{month:02d}", "well_extraction"]  # assume that 30% of the recharge can be sustainably extracted without causing groundwater depletion
                    dict_sustainability_of_extractions_m3[f"{year}-{month}"] = gw_extraction_monthly_sum - recharge_monthly_sum * 0.3
                    dict_sustainability_of_extractions_percent[f"{year}-{month}"] = ((gw_extraction_monthly_sum - recharge_monthly_sum * 0.3) / (recharge_monthly_sum * 0.3)) * 100

        # plot recharge anomalies as barplots and use dictionary keys as x-ticks
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.bar(list(dict_sustainability_of_extractions_m3.keys()), list(dict_sustainability_of_extractions_m3.values()), color="blue")
        ax.set_xlabel("Zeit", fontsize=12)
        ax.set_ylabel("GW-Entnahme Nachhaltigkeit [m³]", fontsize=12)
        # ax.set_ylim(-100, 100)
        ax.axhline(0, color="gray", linestyle="-")
        fig.tight_layout()
        output_file = figures_folder / f"sustainability_of_extractions_m3_{scenario}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.bar(list(dict_sustainability_of_extractions_percent.keys()), list(dict_sustainability_of_extractions_percent.values()), color="blue")
        ax.set_xlabel("Zeit", fontsize=12)
        ax.set_ylabel("GW-Entnahme Nachhaltigkeit [%]", fontsize=12)
        # ax.set_ylim(-100, 100)
        ax.axhline(0, color="gray", linestyle="-")
        fig.tight_layout()
        output_file = figures_folder / f"sustainability_of_extractions_percent_{scenario}.png"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return


if __name__ == "__main__":
    main()