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
    dict_data["no_irrigation"] = {}

    # identifiers of simulations
    irrigation_scenarios = ["crop-specific",
                            ]
    crop_rotation_scenarios = ["grain-corn_winter-wheat_winter-rape"
                               ]
    soil_types = ["sandy_soil", "silty_soil", "clayey_soil"]
    for irrigation_scenario in irrigation_scenarios:
        dict_data["no_irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data["no_irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no_irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data["no_irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

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
        for crop_rotation_scenario in crop_rotation_scenarios:
            fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
            for i, soil_type in enumerate(soil_types):
                data = dict_data["no_irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc["2003-04-01":"2003-09-30" , :]
                axs[i].plot(data.index, data["canopy_cover"], color="red")
                axs[i].set_ylim(0, )
            axs[1].set_ylabel("Bodenbedeckung [-]")
            axs[-1].set_xlabel("[Jahr-Monat]")
            fig.tight_layout()
            file = base_path / "figures" / f"canopy_cover_no_irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=300)

            for i, soil_type in enumerate(soil_types):
                data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc["2003-04-01":"2003-09-30" , :]
                axs[i].plot(data.index, data["canopy_cover"], color="red")
                axs[i].set_ylim(0, )
            axs[1].set_ylabel("Bodenbedeckung [-]")
            axs[-1].set_xlabel("[Jahr-Monat]")
            fig.tight_layout()
            file = base_path / "figures" / f"canopy_cover_irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=300)

            fig, axs = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
            for i, soil_type in enumerate(soil_types):
                data = dict_data["no_irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc["2003-04-01":"2003-09-30" , :]
                axs[i].plot(data.index, data["canopy_cover"], color="red")
                data = dict_data["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type].loc["2003-04-01":"2003-09-30" , :]
                axs[i].plot(data.index, data["canopy_cover"], color="blue")
                axs[i].set_ylim(0, )
            axs[1].set_ylabel("Bodenbedeckung [-]")
            axs[-1].set_xlabel("[Jahr-Monat]")
            fig.tight_layout()
            file = base_path / "figures" / f"canopy_cover_no_irrigation_{irrigation_scenario}_irr_demand_{crop_rotation_scenario}.png"
            fig.savefig(file, dpi=300)


    return


if __name__ == "__main__":
    main()
