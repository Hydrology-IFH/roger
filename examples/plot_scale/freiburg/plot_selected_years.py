from pathlib import Path
import os
import numpy as onp
import pandas as pd
import click
import matplotlib as mpl
import seaborn as sns

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

    dict_data = {}
    dict_data["irrigation"] = {}
    dict_data["no-irrigation"] = {}
    dict_data["irrigation_soil-compaction"] = {}
    dict_data["no-irrigation_soil-compaction"] = {}

    years = [2003, 2018, 2021]

    # identifiers of simulations
    scenarios = ["irrigation", "no-irrigation", "irrigation_soil-compaction", "no-irrigation_soil-compaction"]
    irrigation_scenarios = ["crop-specific",
                            ]
    crop_rotation_scenarios = ["grain-corn",
                               "grain-corn_yellow-mustard",
                               "grain-corn_winter-wheat_winter-rape",
                               "grain-corn_winter-wheat_winter-rape_yellow-mustard",
                               "winter-wheat"
                               ]
    soil_types = ["sandy_soil_type", "silty_soil_type", "clayey_soil_type"]
    for irrigation_scenario in irrigation_scenarios:
        dict_data["no-irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no-irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data["no-irrigation_soil-compaction"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no-irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data["irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        dict_data["irrigation_soil-compaction"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    for irrigation_scenario in irrigation_scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                for year in years:
                    # fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
                    # for i, soil_type in enumerate(soil_types):
                    #     data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-30" , :]
                    #     axs[i].plot(data.index, data["canopy_cover"], color="orange")
                    #     axs[i].set_ylim(0, )
                    # axs[1].set_ylabel("Bodenbedeckung [-]")
                    # axs[-1].set_xlabel("[Jahr-Monat]")
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"canopy_cover_no-irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
                    # fig.savefig(file, dpi=300)

                    # for i, soil_type in enumerate(soil_types):
                    #     data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-30" , :]
                    #     axs[i].plot(data.index, data["canopy_cover"], color="orange")
                    #     axs[i].set_ylim(0, )
                    # axs[1].set_ylabel("Bodenbedeckung [-]")
                    # axs[-1].set_xlabel("[Jahr-Monat]")
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"canopy_cover_irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
                    # fig.savefig(file, dpi=300)

                    # fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
                    # for i, soil_type in enumerate(soil_types):
                    #     data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-30" , :]
                    #     axs[i].plot(data.index, data["canopy_cover"], color="orange")
                    #     data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-30" , :]
                    #     axs[i].plot(data.index, data["canopy_cover"], color="blue")
                    #     axs[i].set_ylim(0, )
                    # axs[1].set_ylabel("Bodenbedeckung [-]")
                    # axs[-1].set_xlabel("[Jahr-Monat]")
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"canopy_cover_no-irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
                    # fig.savefig(file, dpi=300)

                    # fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
                    # data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, data["canopy_cover"], color="orange")
                    # data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, data["canopy_cover"], color="blue")
                    # axs.set_ylim(0, )
                    # axs.set_xlim(data.index[0], data.index[-1])
                    # axs.set_ylabel("Bodenbedeckung [-]")
                    # axs.set_xlabel("[Jahr-Monat]")
                    # # rotate ticklabels of x-axis
                    # axs.set_xticklabels(axs.get_xticklabels(), rotation=20)
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"canopy_cover_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # fig.savefig(file, dpi=300)

                    # fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
                    # data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, (data["z_root"]/1000), color="orange")
                    # data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-04-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, data["z_root"]/1000, color="blue")
                    # axs.set_ylim(0, )
                    # # reverse y-axis
                    # axs.invert_yaxis()
                    # axs.set_xlim(data.index[0], data.index[-1])
                    # axs.set_ylabel("Wurzeltiefe [m]")
                    # axs.set_xlabel("[Jahr-Monat]")
                    # # rotate ticklabels of x-axis
                    # axs.set_xticklabels(axs.get_xticklabels(), rotation=20)
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"root_depth_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # fig.savefig(file, dpi=300)

                    # fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
                    # data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, data["irrigation_demand"], color="orange")
                    # axs.set_ylim(0, )
                    # axs.set_xlim(data.index[0], data.index[-1])
                    # axs.set_ylabel("potentieller\n Bewaesserungsbedarf\n [mm]")
                    # axs.set_xlabel("[Jahr-Monat]")
                    # # rotate ticklabels of x-axis
                    # axs.set_xticklabels(axs.get_xticklabels(), rotation=20)
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # fig.savefig(file, dpi=300)

                    # fig, axs = plt.subplots(1, 1, figsize=(6, 2.5))
                    # data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # axs.plot(data.index, data["irrig"], color="blue")
                    # axs.set_ylim(0, )
                    # axs.set_xlim(data.index[0], data.index[-1])
                    # axs.set_ylabel("Bewaesserung\n [mm/Tag]")
                    # axs.set_xlabel("[Jahr-Monat]")
                    # # rotate ticklabels of x-axis
                    # axs.set_xticklabels(axs.get_xticklabels(), rotation=20)
                    # fig.tight_layout()
                    # file = base_path / "figures" / f"{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # fig.savefig(file, dpi=300)


                    # fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
                    # data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # ax1 = ax.twinx()
                    # ax1.bar(data.index, data["precip"], color="blue", width=1)
                    # ax1.set_ylabel('[mm/day]')
                    # ax1.set_xlabel('')
                    # ax1.invert_yaxis()
                    # ax.plot(data.index, data["pet"], color="green", lw=2)
                    # ax.plot(data.index, data["transp"], color="green", lw=1.5, ls="--")
                    # ax.plot(data.index, data["evap_soil"], color="green", lw=1, ls=":")
                    # ax.plot(data.index, -data["perc"], color="purple", lw=2)
                    # ax.set_ylabel('[mm/day]')
                    # ax.set_xlabel('')
                    # ax.set_xlabel('[Year-Month]')
                    # fig.tight_layout()
                    # file_str = f"precip_ET_perc_no_irrigation_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # path_fig = base_path / "figures" / file_str
                    # fig.savefig(path_fig, dpi=300)
                    # plt.close("all")


                    # fig, ax = plt.subplots(1, 1, sharex=True, figsize=(6, 4))
                    # data = dict_data["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                    # ax1 = ax.twinx()
                    # ax1.bar(data.index, data["precip"], color="blue", width=1)
                    # ax1.set_ylabel('[mm/day]')
                    # ax1.set_xlabel('')
                    # ax1.invert_yaxis()
                    # ax1.set_ylim(0, 100)
                    # ax.plot(data.index, data["pet"], color="green", lw=2)
                    # ax.plot(data.index, data["transp"], color="green", lw=1.5, ls="--")
                    # ax.plot(data.index, data["evap_soil"], color="green", lw=1, ls=":")
                    # ax.plot(data.index, -data["perc"], color="purple", lw=2)
                    # ax.set_ylabel('[mm/day]')
                    # ax.set_xlabel('')
                    # ax.set_xlabel('[Year-Month]')
                    # fig.tight_layout()
                    # file_str = f"precip_ET_perc_no_irrigation_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                    # path_fig = base_path / "figures" / file_str
                    # fig.savefig(path_fig, dpi=300)
                    # plt.close("all")

                    for scenario in scenarios:
                        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
                        data = dict_data[scenario][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                        axs[0].bar(data.index, data["precip"], color="blue", width=1)
                        if scenario in ["irrigation", "irrigation_soil-compaction"]:
                            axs[0].bar(data.index, data["irrig"], color="cyan", width=1)
                        axs[0].set_ylabel('[mm/day]')
                        axs[0].set_xlabel('')
                        axs[0].set_ylim(0, 100)
                        axs[1].plot(data.index, data["pet"], color="green", lw=2)
                        axs[1].plot(data.index, data["transp"], color="green", lw=1.5, ls="--")
                        axs[1].plot(data.index, data["evap_soil"], color="green", lw=1, ls=":")
                        axs[1].set_ylabel('[mm/day]')
                        axs[1].set_xlabel('')
                        axs[1].set_ylim(0, 10)
                        axs[2].plot(data.index, data["theta_rz"], color="black", lw=2)
                        axs[2].plot(data.index, data["theta_irrig"], color="black", lw=1, ls="--")
                        axs[2].set_ylabel('[-]')
                        axs[2].set_xlabel('')
                        axs[2].set_ylim(0.1, 0.4)
                        axs[3].plot(data.index, data["perc"], color="magenta", lw=2)
                        axs[3].set_ylabel('[mm/day]')
                        axs[3].set_xlabel('')
                        axs[3].set_ylim(0, 20)
                        axs[3].set_xlabel('[Year-Month]')
                        fig.tight_layout()
                        file_str = f"precip_ET_theta_perc_{scenario}_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                        path_fig = base_path / "figures" / file_str
                        fig.savefig(path_fig, dpi=300)
                        plt.close("all")

                        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
                        data = dict_data[scenario][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                        if scenario in ["irrigation", "irrigation_soil-compaction"]:
                            axs[0].bar(data.index, data["irrig"], color="cyan", width=1)
                            axs[0].set_ylabel('[mm/day]')
                            axs[0].set_xlabel('')
                            axs[0].set_ylim(0, 30)
                        else:
                            axs[0].set_axis_off()
                        axs[1].plot(data.index, data["irrigation_demand"], color="orange", lw=2)
                        axs[1].set_ylabel('[mm]')
                        axs[1].set_xlabel('')
                        axs[1].set_ylim(0, 100)
                        axs[2].plot(data.index, data["root_ventilation"], color="black", lw=2)
                        axs[2].set_ylabel('[%]')
                        axs[2].set_xlabel('')
                        axs[2].set_ylim(0, 100)
                        axs[3].plot(data.index, data["heat_stress"].cumsum(), color="magenta", lw=2)
                        axs[3].set_ylabel('[days]')
                        axs[3].set_xlabel('')
                        axs[3].set_ylim(0, 50)
                        axs[3].set_xlabel('[Year-Month]')
                        fig.tight_layout()
                        file_str = f"irrig_irrig_demand_root_ventilation_heat_stress_{scenario}_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                        path_fig = base_path / "figures" / file_str
                        fig.savefig(path_fig, dpi=300)

                        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(6, 4))
                        cond_bare = (data["canopy_cover"] <= 0.03)
                        data = dict_data[scenario][irrigation_scenario][crop_rotation_scenario][soil_type].loc[f"{year}-03-01":f"{year}-09-15" , :]
                        axs[0].plot(data.index, data["canopy_cover"], color="green", lw=2)
                        axs[0].set_ylabel('[-]')
                        axs[0].set_xlabel('')
                        axs[0].set_ylim(0, 1)
                        axs[1].plot(data.index, data["irrigation_demand"], color="orange", lw=2)
                        axs[1].set_ylabel('[mm]')
                        axs[1].set_xlabel('')
                        axs[1].set_ylim(0, 100)
                        data["root_ventilation"][cond_bare] = onp.nan
                        axs[2].plot(data.index, data["root_ventilation"], color="black", lw=2)
                        axs[2].set_ylabel('[%]')
                        axs[2].set_xlabel('')
                        axs[2].set_ylim(0, 100)
                        data["heat_stress"][cond_bare] = 0
                        axs[3].plot(data.index, data["heat_stress"].cumsum(), color="magenta", lw=2)
                        axs[3].set_ylabel('[days]')
                        axs[3].set_xlabel('')
                        axs[3].set_ylim(0, 100)
                        axs[3].set_xlabel('[Year-Month]')
                        fig.tight_layout()
                        file_str = f"canopy_cover_irrig_demand_root_ventilation_heat_stress_{scenario}_{irrigation_scenario}_{crop_rotation_scenario}_{soil_type}_{year}.png"
                        path_fig = base_path / "figures" / file_str
                        fig.savefig(path_fig, dpi=300)

                        plt.close("all")




    return


if __name__ == "__main__":
    main()
