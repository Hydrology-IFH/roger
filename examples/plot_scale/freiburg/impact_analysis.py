from pathlib import Path
import os
import numpy as onp
import pandas as pd
import click
import matplotlib as mpl
import seaborn as sns
from sklearn.linear_model import LinearRegression

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 10
mpl.rcParams["legend.title_fontsize"] = 11
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 10.0,
        "axes.labelsize": 11.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 10.0,
        "legend.title_fontsize": 11.0,
    },
)


@click.command("main")
def main():
    base_path = Path(__file__).parent

    dict_data_daily = {}
    dict_data_daily["irrigation"] = {}
    dict_data_daily["no-irrigation"] = {}
    dict_data_daily["irrigation_soil-compaction"] = {}
    dict_data_daily["no-irrigation_soil-compaction"] = {}

    years = [2003, 2018, 2021]

    # identifiers of simulations
    scenarios = ["irrigation", "no-irrigation", "irrigation_soil-compaction", "no-irrigation_soil-compaction"]
    scenarios = ["no-irrigation", "no-irrigation_soil-compaction"]
    irrigation_scenarios = ["crop-specific",
                            ]
    crop_rotation_scenarios = ["grain-corn",
                               "winter-wheat",
                               "summer-barley",
                               "potato"
                               ]
    soil_types = ["10", "12", "3", "4", "16", "5", "11"]
    for irrigation_scenario in irrigation_scenarios:
        dict_data_daily["no-irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no-irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation1 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation1.index = pd.to_datetime(df_simulation1.index)
                dir_csv_file = base_path / "output" / "nitrate" / "no-irrigation" / crop_rotation_scenario / soil_type
                df_simulation2 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation2.index = pd.to_datetime(df_simulation2.index)
                df_simulation = df_simulation1.join(df_simulation2.loc[:, :"NO3_soil_conc"])
                dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no-irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation1 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation1.index = pd.to_datetime(df_simulation1.index)
                dir_csv_file = base_path / "output" / "nitrate" / "no-irrigation_soil-compaction" / crop_rotation_scenario / soil_type
                df_simulation2 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation2.index = pd.to_datetime(df_simulation2.index)
                df_simulation = df_simulation1.join(df_simulation2.loc[:, :"NO3_soil_conc"])
                dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data_daily["irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data_daily["irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation1 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation1.index = pd.to_datetime(df_simulation1.index)
                dir_csv_file = base_path / "output" / "nitrate" / "irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation2 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation2.index = pd.to_datetime(df_simulation2.index)
                df_simulation = df_simulation1.join(df_simulation2.loc[:, :"NO3_soil_conc"])
                dict_data_daily["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data_daily["irrigation_soil-compaction"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data_daily["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation1 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation1.index = pd.to_datetime(df_simulation1.index)
                dir_csv_file = base_path / "output" / "nitrate" / "irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation2 = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation2.index = pd.to_datetime(df_simulation2.index)
                df_simulation = df_simulation1.join(df_simulation2.loc[:, :"NO3_soil_conc"])
                dict_data_daily["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    dict_data_monthly = {}
    dict_data_monthly["irrigation"] = {}
    dict_data_monthly["no-irrigation"] = {}
    dict_data_monthly["irrigation_soil-compaction"] = {}
    dict_data_monthly["no-irrigation_soil-compaction"] = {}

    variables_aggregate_by_sum = ["precip", "pet", "pt", "transp", "evap_soil", "perc", "N_fert", "N_uptake", "NO3-N_leach"]
    variables_aggregate_by_mean = ["canopy_cover", "z_root", "theta_rz", "ta_max", "heat_stress", "NO3_leach_conc", "NO3_soil_conc"]

    soil_types = ["10", "12", "3", "4", "16", "5", "11", "area_weighted"]
    ll_data = []
    for scenario in scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                data_daily = dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type]
                data_monthly_sum = data_daily.loc[:, variables_aggregate_by_sum].resample("ME").sum()
                data_monthly_mean = data_daily.loc[:, variables_aggregate_by_mean].resample("ME").mean()
                data_monthly = data_monthly_sum.join(data_monthly_mean)
                data_monthly["pt_precip_ratio"] = data_monthly["pt"] / data_monthly["precip"]
                data_monthly["perc_precip_ratio"] = data_monthly["perc"] / data_monthly["precip"]
                data_monthly["Nleach_Nfert_ratio"] = data_monthly["NO3-N_leach"] / data_monthly["N_fert"]
                data_monthly["Nup_Nfert_ratio"] = data_monthly["N_uptake"] / data_monthly["N_fert"]
                data_monthly["year"] = data_monthly.index.year
                data_monthly["month"] = data_monthly.index.month
                data_monthly["crop_rotation_scenario"] = crop_rotation_scenario
                data_monthly["soil_type"] = soil_type
                data_monthly["scenario"] = scenario
                ll_data.append(data_monthly)
    data_monthly = pd.concat(ll_data, axis=0)

    ll_data = []
    for scenario in scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                data_daily = dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type]
                data_annual_sum = data_daily.loc[:, variables_aggregate_by_sum].resample("YE").sum()
                data_annual_mean = data_daily.loc[:, variables_aggregate_by_mean].resample("YE").mean()
                data_annual = data_annual_sum.join(data_annual_mean)
                data_annual["pt_precip_ratio"] = data_annual["pt"] / data_annual["precip"]
                data_annual["perc_precip_ratio"] = data_annual["perc"] / data_annual["precip"]
                data_annual["Nleach_Nfert_ratio"] = data_annual["NO3-N_leach"] / data_annual["N_fert"]
                data_annual["Nup_Nfert_ratio"] = data_annual["N_uptake"] / data_annual["N_fert"]
                data_annual["year"] = data_annual.index.year
                data_annual["crop_rotation_scenario"] = crop_rotation_scenario
                data_annual["soil_type"] = soil_type
                data_annual["scenario"] = scenario
                ll_data.append(data_annual)
    data_annual = pd.concat(ll_data, axis=0)

    ll_data = []
    for scenario in scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                data_daily = dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type]
                data_total_sum = data_daily.loc[:, variables_aggregate_by_sum].sum()
                data_total_mean = data_daily.loc[:, variables_aggregate_by_mean].mean()
                data_total = data_total_sum.join(data_total_mean)
                data_total["pt_precip_ratio"] = data_total["pt"] / data_total["precip"]
                data_total["perc_precip_ratio"] = data_total["perc"] / data_total["precip"]
                data_total["Nleach_Nfert_ratio"] = data_total["NO3-N_leach"] / data_total["N_fert"]
                data_total["Nup_Nfert_ratio"] = data_total["N_uptake"] / data_total["N_fert"]
                data_total["crop_rotation_scenario"] = crop_rotation_scenario
                data_total["soil_type"] = soil_type
                data_total["scenario"] = scenario
                ll_data.append(data_total)
    data_total = pd.concat(ll_data, axis=0)

    return


if __name__ == "__main__":
    main()
