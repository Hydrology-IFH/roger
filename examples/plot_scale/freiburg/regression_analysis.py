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
    soil_types = ["sandy_soil_type", "silty_soil_type", "clayey_soil_type", "10", "12", "3", "4", "16", "5", "11"]
    for irrigation_scenario in irrigation_scenarios:
        dict_data_daily["no-irrigation"][irrigation_scenario] = {}
        for crop_rotation_scenario in crop_rotation_scenarios:
            dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
            for soil_type in soil_types:
                dir_csv_file = base_path / "output" / "no-irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
                df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
                df_simulation.index = pd.to_datetime(df_simulation.index)
                dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    # for irrigation_scenario in irrigation_scenarios:
    #     dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario] = {}
    #     for crop_rotation_scenario in crop_rotation_scenarios:
    #         dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
    #         for soil_type in soil_types:
    #             dir_csv_file = base_path / "output" / "no-irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
    #             df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
    #             df_simulation.index = pd.to_datetime(df_simulation.index)
    #             dict_data_daily["no-irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    # for irrigation_scenario in irrigation_scenarios:
    #     dict_data_daily["irrigation"][irrigation_scenario] = {}
    #     for crop_rotation_scenario in crop_rotation_scenarios:
    #         dict_data_daily["irrigation"][irrigation_scenario][crop_rotation_scenario] = {}
    #         for soil_type in soil_types:
    #             dir_csv_file = base_path / "output" / "irrigation" / irrigation_scenario / crop_rotation_scenario / soil_type
    #             df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
    #             df_simulation.index = pd.to_datetime(df_simulation.index)
    #             dict_data_daily["irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    # for irrigation_scenario in irrigation_scenarios:
    #     dict_data_daily["irrigation_soil-compaction"][irrigation_scenario] = {}
    #     for crop_rotation_scenario in crop_rotation_scenarios:
    #         dict_data_daily["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario] = {}
    #         for soil_type in soil_types:
    #             dir_csv_file = base_path / "output" / "irrigation_soil-compaction" / irrigation_scenario / crop_rotation_scenario / soil_type
    #             df_simulation = pd.read_csv(dir_csv_file / "simulation.csv", sep=";", skiprows=1, index_col=0)
    #             df_simulation.index = pd.to_datetime(df_simulation.index)
    #             dict_data_daily["irrigation_soil-compaction"][irrigation_scenario][crop_rotation_scenario][soil_type] = df_simulation

    dict_data_monthly = {}
    dict_data_monthly["irrigation"] = {}
    dict_data_monthly["no-irrigation"] = {}
    dict_data_monthly["irrigation_soil-compaction"] = {}
    dict_data_monthly["no-irrigation_soil-compaction"] = {}


    variables_aggregate_by_sum = ["precip", "pet", "pt", "transp", "evap_soil", "perc"]
    variables_aggregate_by_mean = ["canopy_cover", "z_root", "theta_rz", "ta_max", "heat_stress"]


    soil_types = ["10", "12", "3", "4", "16", "5", "11"]
    for scenario in scenarios:
        ll_data = []
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                data_daily = dict_data_daily["no-irrigation"][irrigation_scenario][crop_rotation_scenario][soil_type]
                data_monthly_sum = data_daily.loc[:, variables_aggregate_by_sum].resample("ME").sum()
                data_monthly_mean = data_daily.loc[:, variables_aggregate_by_mean].resample("ME").mean()
                data_monthly = data_monthly_sum.join(data_monthly_mean)
                data_monthly["pt_precip_ratio"] = data_monthly["pt"] / data_monthly["precip"]
                data_monthly["perc_precip_ratio"] = data_monthly["perc"] / data_monthly["precip"]
                data_monthly["year"] = data_monthly.index.year
                data_monthly["month"] = data_monthly.index.month
                data_monthly["crop_rotation_scenario"] = crop_rotation_scenario
                data_monthly["soil_type"] = soil_type
                ll_data.append(data_monthly)
        dict_data_monthly[scenario] = pd.concat(ll_data, axis=0)

    cmap = mpl.colormaps["Oranges"]  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]

    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize
    bounds = onp.linspace(2000, 2025, 26)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)


    scenarios = ["no-irrigation"]
    for scenario in scenarios:
        for month in range(3, 7):
            data_monthly = dict_data_monthly[scenario]
            data_monthly = data_monthly[data_monthly["month"] == month]

            # fit linear regression
            lm = LinearRegression()
            x = data_monthly["precip"].values.reshape((-1, 1))
            y = data_monthly["perc"].values
            lm.fit(x, y)
            r_sq = lm.score(x, y)
            x_ = onp.linspace(0, 1.1 * onp.max(x), 1000)
            y_pred = lm.intercept_ + lm.coef_ * x_

            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            im = ax.scatter(data_monthly["precip"], data_monthly["perc"], s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
            ax.plot(x_, y_pred, color="k", lw=1.5)
            fig.colorbar(im, ax=ax, label="Year")
            ax.set_xlabel("Precipitation [mm]")
            ax.set_ylabel("Percolation [mm]")
            ax.set_xlim(0, )
            ax.set_ylim(0, )
            ax.text(
            0.15, 0.93,
            f"$R^2$: {r_sq:.2f}",
            fontsize=11,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            )
            fig.tight_layout()
            file = base_path / "figures" / f"precip_vs_perc_{scenario}_month{month}.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

            # make data relative to the average
            precip_avg = data_monthly["precip"].mean()
            perc_avg = data_monthly["perc"].mean()
            perc_rel = (data_monthly["perc"] - perc_avg) / perc_avg
            precip_rel = (data_monthly["precip"] - precip_avg) / precip_avg

            # fit linear regression
            lm = LinearRegression()
            x = precip_rel.values.reshape((-1, 1)) * 100
            y = perc_rel.values * 100
            lm.fit(x, y)
            r_sq = lm.score(x, y)
            x_ = onp.linspace(-100, 1.1 * onp.max(x), 1000)
            y_pred = lm.intercept_ + lm.coef_ * x_

            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
            im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
            ax.plot(x_, y_pred, color="k", lw=1.5)
            fig.colorbar(im, ax=ax, label="Year")
            ax.set_xlabel("$\Delta$ Precipitation [%]")
            ax.set_ylabel("$\Delta$ Percolation [%]")
            ax.set_xlim(-100, 200)
            ax.set_ylim(-100, 200)
            ax.text(
            0.15, 0.93,
            f"$R^2$: {r_sq:.2f}",
            fontsize=11,
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            )
            fig.tight_layout()
            file = base_path / "figures" / f"precip_vs_perc_{scenario}_month{month}_relative.png"
            fig.savefig(file, dpi=300)
            plt.close(fig)

    for scenario in scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for month in range(3, 7):
                data_monthly = dict_data_monthly[scenario]
                data_monthly = data_monthly[(data_monthly["month"] == month) & (data_monthly["crop_rotation_scenario"] == crop_rotation_scenario)]

                # fit linear regression
                lm = LinearRegression()
                x = data_monthly["precip"].values.reshape((-1, 1))
                y = data_monthly["perc"].values
                lm.fit(x, y)
                r_sq = lm.score(x, y)
                x_ = onp.linspace(0, 1.1 * onp.max(x), 1000)
                y_pred = lm.intercept_ + lm.coef_ * x_

                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                im = ax.scatter(data_monthly["precip"], data_monthly["perc"], s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                ax.plot(x_, y_pred, color="k", lw=1.5)
                fig.colorbar(im, ax=ax, label="Year")
                ax.set_xlabel("Precipitation [mm]")
                ax.set_ylabel("Percolation [mm]")
                ax.set_xlim(0, )
                ax.set_ylim(0, )
                ax.text(
                0.15, 0.93,
                f"$R^2$: {r_sq:.2f}",
                fontsize=11,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                fig.tight_layout()
                file = base_path / "figures" / f"precip_vs_perc_{scenario}_{crop_rotation_scenario}_month{month}.png"
                fig.savefig(file, dpi=300)
                plt.close(fig)

                # make data relative to the average
                precip_avg = data_monthly["precip"].mean()
                perc_avg = data_monthly["perc"].mean()
                perc_rel = (data_monthly["perc"] - perc_avg) / perc_avg
                precip_rel = (data_monthly["precip"] - precip_avg) / precip_avg

                # fit linear regression
                lm = LinearRegression()
                x = precip_rel.values.reshape((-1, 1)) * 100
                y = perc_rel.values * 100
                lm.fit(x, y)
                r_sq = lm.score(x, y)
                x_ = onp.linspace(-100, 1.1 * onp.max(x), 1000)
                y_pred = lm.intercept_ + lm.coef_ * x_

                fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                ax.plot(x_, y_pred, color="k", lw=1.5)
                fig.colorbar(im, ax=ax, label="Year")
                ax.set_xlabel("$\Delta$ Precipitation [%]")
                ax.set_ylabel("$\Delta$ Percolation [%]")
                ax.set_xlim(-100, 200)
                ax.set_ylim(-100, 200)
                ax.text(
                0.15, 0.93,
                f"$R^2$: {r_sq:.2f}",
                fontsize=11,
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
                )
                fig.tight_layout()
                file = base_path / "figures" / f"precip_vs_perc_{scenario}_{crop_rotation_scenario}_month{month}_relative.png"
                fig.savefig(file, dpi=300)
                plt.close(fig)

    for scenario in scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for soil_type in soil_types:
                for month in range(3, 7):
                    data_monthly = dict_data_monthly[scenario]
                    data_monthly = data_monthly[(data_monthly["month"] == month) & (data_monthly["crop_rotation_scenario"] == crop_rotation_scenario) & (data_monthly["soil_type"] == soil_type)]

                    # fit linear regression
                    lm = LinearRegression()
                    x = data_monthly["precip"].values.reshape((-1, 1))
                    y = data_monthly["perc"].values
                    lm.fit(x, y)
                    r_sq = lm.score(x, y)
                    x_ = onp.linspace(0, 1.1 * onp.max(x), 1000)
                    y_pred = lm.intercept_ + lm.coef_ * x_

                    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                    im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                    ax.plot(x_, y_pred, color="k", lw=1.5)
                    fig.colorbar(im, ax=ax, label="Year")
                    ax.set_xlabel("Precipitation [mm]")
                    ax.set_ylabel("Percolation [mm]")
                    ax.set_xlim(0, )
                    ax.set_ylim(0, )
                    ax.text(
                    0.15, 0.93,
                    f"$R^2$: {r_sq:.2f}",
                    fontsize=11,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    )
                    fig.tight_layout()
                    file = base_path / "figures" / f"precip_vs_perc_{scenario}_{crop_rotation_scenario}_soil_type{soil_type}_month{month}.png"
                    fig.savefig(file, dpi=300)
                    plt.close(fig)

                    # fit linear regression
                    lm = LinearRegression()
                    x = data_monthly["pt_precip_ratio"].values.reshape((-1, 1))
                    y = data_monthly["perc_precip_ratio"].values
                    cond = (data_monthly["precip"] > 0) & (data_monthly["pt"] > 0)
                    if (data_monthly["pt"] > 0).all():
                        x = x[cond]
                        y = y[cond]
                        lm.fit(x, y)
                        r_sq = lm.score(x, y)
                        x_ = onp.linspace(0, 1.1 * onp.max(x), 1000)
                        y_pred = lm.intercept_ + lm.coef_ * x_

                        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                        im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                        ax.plot(x_, y_pred, color="k", lw=1.5)
                        fig.colorbar(im, ax=ax, label="Year")
                        ax.set_ylabel("PERC/PRECIP [-]")
                        ax.set_xlabel("PT/PRECIP [-]")
                        ax.set_xlim(0, )
                        ax.set_ylim(0, )
                        ax.text(
                        0.15, 0.93,
                        f"$R^2$: {r_sq:.2f}",
                        fontsize=11,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        )
                        fig.tight_layout()
                        file = base_path / "figures" / f"perc_precip_vs_pt_precip_{scenario}_{crop_rotation_scenario}_soil_type{soil_type}_month{month}.png"
                        fig.savefig(file, dpi=300)
                        plt.close(fig)

                
                    # make data relative to the average
                    perc_precip_ratio_avg = data_monthly["perc_precip_ratio"].mean()
                    pt_precip_ratio_avg = data_monthly["pt_precip_ratio"].mean()
                    perc_precip_ratio_rel = (data_monthly["perc_precip_ratio"] - perc_precip_ratio_avg) / perc_precip_ratio_avg
                    pt_precip_ratio_rel = (data_monthly["pt_precip_ratio"] - pt_precip_ratio_avg) / pt_precip_ratio_avg
                    # fit linear regression
                    lm = LinearRegression()
                    x = pt_precip_ratio_rel.values.reshape((-1, 1)) * 100
                    y = perc_precip_ratio_rel.values * 100
                    cond = (data_monthly["precip"] > 0) & (data_monthly["pt"] > 0)
                    if (data_monthly["pt"] > 0).all():
                        x = x[cond]
                        y = y[cond]
                        lm.fit(x, y)
                        r_sq = lm.score(x, y)
                        x_ = onp.linspace(-100, 1.1 * onp.max(x), 1000)
                        y_pred = lm.intercept_ + lm.coef_ * x_

                        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                        im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                        ax.plot(x_, y_pred, color="k", lw=1.5)
                        fig.colorbar(im, ax=ax, label="Year")
                        ax.set_xlabel("$\Delta$ PT/PRECIP [%]")
                        ax.set_ylabel("$\Delta$ PERC/PRECIP [%]")
                        ax.set_xlim(-100, 200)
                        ax.set_ylim(-100, 200)
                        ax.text(
                        0.15, 0.93,
                        f"$R^2$: {r_sq:.2f}",
                        fontsize=11,
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                        )
                        fig.tight_layout()
                        file = base_path / "figures" / f"perc_precip_vs_pt_precip_{scenario}_{crop_rotation_scenario}_soil_type{soil_type}_month{month}_relative.png"
                        fig.savefig(file, dpi=300)
                        plt.close(fig)

                    # make data relative to the average
                    precip_avg = data_monthly["precip"].mean()
                    perc_avg = data_monthly["perc"].mean()
                    perc_rel = (data_monthly["perc"] - perc_avg) / perc_avg
                    precip_rel = (data_monthly["precip"] - precip_avg) / precip_avg

                    # fit linear regression
                    lm = LinearRegression()
                    x = precip_rel.values.reshape((-1, 1)) * 100
                    y = perc_rel.values * 100
                    cond = (data_monthly["precip"] > 0)
                    x = x[cond]
                    y = y[cond]
                    lm.fit(x, y)
                    r_sq = lm.score(x, y)
                    x_ = onp.linspace(-100, 1.1 * onp.max(x), 1000)
                    y_pred = lm.intercept_ + lm.coef_ * x_

                    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
                    im = ax.scatter(x, y, s=4, c=data_monthly["year"], cmap=cmap, norm=norm)
                    ax.plot(x_, y_pred, color="k", lw=1.5)
                    fig.colorbar(im, ax=ax, label="Year")
                    ax.set_xlabel("$\Delta$ Precipitation [%]")
                    ax.set_ylabel("$\Delta$ Percolation [%]")
                    ax.set_xlim(-100, 200)
                    ax.set_ylim(-100, 200)
                    ax.text(
                    0.15, 0.93,
                    f"$R^2$: {r_sq:.2f}",
                    fontsize=11,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                    )
                    fig.tight_layout()
                    file = base_path / "figures" / f"precip_vs_perc_{scenario}_{crop_rotation_scenario}_soil_type{soil_type}_month{month}_relative.png"
                    fig.savefig(file, dpi=300)
                    plt.close(fig)
    return


if __name__ == "__main__":
    main()
