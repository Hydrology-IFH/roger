import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as onp
import click
import roger.tools.labels as labs

onp.random.seed(42)

mpl.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

mpl.rcParams["font.size"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["legend.title_fontsize"] = 9
sns.set_style("ticks")
sns.plotting_context(
    "paper",
    font_scale=1,
    rc={
        "font.size": 8.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.title_fontsize": 9.0,
    },
)

crops_lys2_lys3_lys8 = {2010: "winter barley",
                        2011: "sugar beet",
                        2012: "winter wheat",
                        2013: "winter rape",
                        2014: "winter triticale",
                        2015: "silage corn",
                        2016: "winter barley",
                        2017: "sugar beet"
}

fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                        "lys3": "100% N-fertilized",
                        "lys8": "70% N-fertilized"
}


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

    # directory of results
    base_path_output = base_path / "output" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    crops_lys2_lys3_lys8 = {2010: "winter barley",
                            2011: "sugar beet",
                            2012: "winter wheat",
                            2013: "winter rape",
                            2014: "winter triticale",
                            2015: "silage corn",
                            2016: "winter barley",
                            2017: "sugar beet"
    }

    crops_lys4_lys9 = {2010: "winter wheat",
                       2011: "winter rape\n& phacelia",
                       2012: "phacelia\n& silage corn",
                       2013: "beet root",
                       2014: "winter barley",
                       2015: "grass",
                       2016: "grass",
                       2017: "winter wheat"
    }

    fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                           "lys3": "100% N-fertilized",
                           "lys8": "70% N-fertilized"
    }

    fert_lys4_lys9 = {"lys4": "Organic",
                      "lys9": "PEP-intensive"
    }

    _lys = {"lys2": "Lys 2",
            "lys3": "Lys 3",
            "lys4": "Lys 4",
            "lys8": "Lys 8",
            "lys9": "Lys 9",
    }

    metrics = ["MAE_NO3_perc_mass_bs"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    lys_experiments = ["lys8", "lys3", "lys2"]

    for metric in metrics:
        nrow = len(years)
        ncol = len(lys_experiments)
        fig, ax = plt.subplots(nrow, 3, sharey=True, sharex=True, figsize=(6, 6))
        for jj, year in enumerate(years):
            for ii, lys_experiment in enumerate(lys_experiments):
                df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_{year}"]]
                df = df.iloc[:100, :]
                # g = sns.kdeplot(data=df, x=f"{metric}_all_{year}", ax=ax[jj, ii], fill=True, color="black", alpha=0.5)
                g = sns.kdeplot(data=df, x=f"{metric}_{year}", ax=ax[jj, ii], fill=True, color="grey", alpha=0.5)
                ax[jj, ii].set_xlabel("")
                ax[jj, ii].set_ylabel("")
                ax[jj, ii].set_title("")
                ax[0, ii].set_title("%s\n%s" % (_lys[lys_experiment], fert_lys2_lys3_lys8[lys_experiment]))
                ax[jj, 0].set_ylabel("%s\n%s" % (year, crops_lys2_lys3_lys8[year]))
                ax[jj, ii].set_xlim(-1, 0.8)

        ax[-1, 0].set_xlabel("$MAE_{PERC}$ [-]")
        ax[-1, 1].set_xlabel("$MAE_{PERC}$ [-]")
        ax[-1, 2].set_xlabel("$MAE_{PERC}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"kde_years_{metric}_lys2_lys3_lys8.png"
        fig.savefig(file, dpi=300)

    # metrics = ["KGE_q_ss"]
    # years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    # lys_experiments = ["lys4", "lys9"]

    # for metric in metrics:
    #     nrow = len(years)
    #     ncol = len(lys_experiments)
    #     fig, ax = plt.subplots(nrow, ncol, sharey=True, sharex=True, figsize=(4, 6))
    #     for jj, year in enumerate(years):
    #         metric_year = f"{metric}_{year}"
    #         for ii, lys_experiment in enumerate(lys_experiments):
    #             df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
    #             df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}"], ascending=False)
    #             df = df_params_metrics.loc[:, [f"{metric}_{year}"]]
    #             df = df.iloc[:100, :]
    #             g = sns.kdeplot(data=df, x=f"{metric}_{year}", ax=ax[jj, ii], fill=True, color="grey", alpha=0.5)
    #             df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_perc_dom"], ascending=False)
    #             df = df_params_metrics.loc[:, [f"{metric}_perc_dom_{year}"]]
    #             df = df.iloc[:100, :]
    #             g = sns.kdeplot(data=df, x=f"{metric}_perc_dom_{year}", ax=ax[jj, ii], fill=True, color="blue", alpha=1.0)
    #             df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_pet_dom"], ascending=False)
    #             df = df_params_metrics.loc[:, [f"{metric}_pet_dom_{year}"]]
    #             df = df.iloc[:100, :]
    #             g = sns.kdeplot(data=df, x=f"{metric}_pet_dom_{year}", ax=ax[jj, ii], fill=True, color="green", alpha=1.0)
    #             ax[jj, ii].set_xlabel("")
    #             ax[jj, ii].set_ylabel("")
    #             ax[jj, ii].set_title("")
    #             ax[0, ii].set_title("%s\n%s" % (_lys[lys_experiment], fert_lys4_lys9[lys_experiment]))
    #             ax[jj, 0].set_ylabel("%s\n%s" % (year, crops_lys4_lys9[year]))
    #             ax[jj, ii].set_xlim(-1, 0.8)

    #     if metric in ["KGE_q_ss"]:
    #         ax[-1, 0].set_xlabel("$KGE_{PERC}$ [-]")
    #         ax[-1, 1].set_xlabel("$KGE_{PERC}$ [-]")
    #     elif metric in ["r_theta"]:
    #         ax[-1, 0].set_xlabel("$r_{\theta}$ [-]")
    #         ax[-1, 1].set_xlabel("$r_{\theta}$ [-]")

    #     fig.tight_layout()
    #     file = base_path_figs / f"kde_years_{metric}_lys4_lys9.png"
    #     fig.savefig(file, dpi=300)

    param_names = ["alpha_transp", "alpha_q", "km_denit", "km_nit", "kmin", "kfix", "kngl", "dmax_denit", "dmax_nit", "phi_soil_temp", "clay", "soil_fertility", "z_soil"]



    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    params = ["km_denit", "km_nit", "dmax_denit", "dmax_nit"]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["MAE_NO3_perc_mass_bs"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_params = df_params_metrics_100.loc[:, params]
        df_params["lys"] = lys_experiment
        ll_df.append(df_params)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=params, id_vars=["lys"])

    fig, ax = plt.subplots(1, 4, figsize=(6, 2))
    for i, var in enumerate(params):
        data = df_long.loc[df_long["variable"] == var, :]
        sns.boxplot(data=data, x="variable", y="value", hue="lys", ax=ax[i], palette=colors, showfliers=False)
        ax[i].set_xlabel("")
        ax[i].set_xticklabels([])
        ax[i].set_ylabel(f"{labs._LABS[var]}")
        ax[i].legend_.remove()
    fig.subplots_adjust(wspace=0.8)
    ax[-1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0), labels=[fert_lys2_lys3_lys8[lys] for lys in lys_experiments])
    file = base_path_figs / f"boxplot_params_lys2_lys3_lys8_nit_denit.png"
    fig.savefig(file, dpi=300)

    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    params = ["alpha_transp", "alpha_q", "k_transp", "k_q"]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["MAE_NO3_perc_mass_bs"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_params = df_params_metrics_100.loc[:, params]
        df_params["lys"] = lys_experiment
        ll_df.append(df_params)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=params, id_vars=["lys"])

    fig, ax = plt.subplots(1, 4, figsize=(6, 2))
    for i, var in enumerate(params):
        data = df_long.loc[df_long["variable"] == var, :]
        sns.boxplot(data=data, x="variable", y="value", hue="lys", ax=ax[i], palette=colors, showfliers=False)
        ax[i].set_xlabel("")
        ax[i].set_xticklabels([])
        ax[i].set_ylabel(f"{labs._LABS[var]}")
        ax[i].legend_.remove()
    fig.subplots_adjust(wspace=0.8)
    ax[-1].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0), labels=[fert_lys2_lys3_lys8[lys] for lys in lys_experiments])
    file = base_path_figs / f"boxplot_params_lys2_lys3_lys8_sas.png"
    fig.savefig(file, dpi=300)


    fig, ax = plt.subplots(7, 4, figsize=(6, 6), sharey='col')
    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    params = ["km_denit", "km_nit", "dmax_denit", "dmax_nit"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    for j, year in enumerate(years):
        ll_df = []
        for lys_experiment in lys_experiments:
            df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
            df_params_metrics = df_params_metrics.sort_values(by=[f"MAE_NO3_perc_mass_bs_{year}"], ascending=False)
            df_params_metrics_100 = df_params_metrics.iloc[:100, :]
            df_params = df_params_metrics_100.loc[:, params]
            df_params["lys"] = lys_experiment
            ll_df.append(df_params)
        df = pd.concat(ll_df)
        df_long = pd.melt(df, value_vars=params, id_vars=["lys"])

        for i, var in enumerate(params):
            data = df_long.loc[df_long["variable"] == var, :]
            sns.boxplot(data=data, x="variable", y="value", hue="lys", ax=ax[j, i], palette=colors, showfliers=False)
            ax[j, i].set_xlabel("")
            ax[j, i].set_xticklabels([])
            ax[j, i].set_ylabel(f"{labs._LABS[var]}")
            ax[j, i].legend_.remove()
        ax[j, 0].set_title(f"{year}: {crops_lys2_lys3_lys8[year]}")
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    ax[-1, -1].legend(frameon=False, loc="lower left", bbox_to_anchor=(-5.0, -0.8), labels=[fert_lys2_lys3_lys8[lys] for lys in lys_experiments], ncol=3)
    file = base_path_figs / f"boxplot_params_lys2_lys3_lys8_nit_denit_2011_2017.png"
    fig.savefig(file, dpi=300)


    fig, ax = plt.subplots(7, 4, figsize=(6, 6), sharey='col')
    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    params = ["alpha_transp", "alpha_q", "k_transp", "k_q"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    for j, year in enumerate(years):
        ll_df = []
        for lys_experiment in lys_experiments:
            df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
            df_params_metrics[f"MAE_NO3_perc_mass_bs_{year}"] = 0.7 * df_params_metrics[f"KGE_q_ss_{year}"] + 0.3 * (1 - ((1 - df_params_metrics[f"r_S_{year}"])**2 + (1 - df_params_metrics[f"KGE_alpha_S_{year}"])**2)**(0.5))
            df_params_metrics = df_params_metrics.sort_values(by=[f"MAE_NO3_perc_mass_bs_{year}"], ascending=False)
            df_params_metrics_100 = df_params_metrics.iloc[:100, :]
            df_params = df_params_metrics_100.loc[:, params]
            df_params["lys"] = lys_experiment
            ll_df.append(df_params)
        df = pd.concat(ll_df)
        df_long = pd.melt(df, value_vars=params, id_vars=["lys"])

        for i, var in enumerate(params):
            data = df_long.loc[df_long["variable"] == var, :]
            sns.boxplot(data=data, x="variable", y="value", hue="lys", ax=ax[j, i], palette=colors, showfliers=False)
            ax[j, i].set_xlabel("")
            ax[j, i].set_xticklabels([])
            ax[j, i].set_ylabel(f"{labs._LABS[var]}")
            ax[j, i].legend_.remove()
        ax[j, 0].set_title(f"{year}: {crops_lys2_lys3_lys8[year]}")
    fig.subplots_adjust(wspace=0.8, hspace=0.8)
    ax[-1, -1].legend(frameon=False, loc="lower left", bbox_to_anchor=(-5.0, -0.8), labels=[fert_lys2_lys3_lys8[lys] for lys in lys_experiments], ncol=3)
    file = base_path_figs / f"boxplot_params_lys2_lys3_lys8_sas_2011_2017.png"
    fig.savefig(file, dpi=300)


    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["MAE_NO3_perc_mass_bs"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        metrics = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
        df_metrics = df_params_metrics_100.loc[:, metrics]
        df_metrics.columns = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
        df_metrics["lys"] = lys_experiment
        df_metrics["year"] = ""
        ll_df.append(df_metrics)
        for year in years:
            metrics = [f"MAE_NO3_perc_mass_bs_{year}", f"MAE_NO3_perc_bs_{year}"]
            df_metrics = df_params_metrics_100.loc[:, metrics]
            df_metrics.columns = ["MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
            df_metrics["lys"] = lys_experiment
            df_metrics["year"] = f"{year}"
            ll_df.append(df_metrics)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"], id_vars=["lys", "year"])

    metrics = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
    fig, ax = plt.subplots(3, 1, figsize=(3, 5), sharex=True)
    for i, var in enumerate(metrics):
        data = df_long.loc[df_long["variable"] == var, :]
        g = sns.boxplot(data=data, x="year", y="value", hue="lys", ax=ax[i], palette=colors, showfliers=False)
        ax[i].set_xlabel("")
    ax[0].set_ylabel("$MAE_{N-UPTAKE}$ [m]")
    ax[1].set_ylabel("$MAE_{NO_3-PERC}$ [mg]")
    ax[2].set_ylabel("$MAE_{NO_3-PERC}$ [mg/l]")
    ax[0].legend_.remove()
    ax[1].legend_.remove()
    handels,labels = g.get_legend_handles_labels()
    labels = [fert_lys2_lys3_lys8[lys] for lys in lys_experiments]
    g.legend(handels, labels, frameon=False, title="")     
    fig.tight_layout()
    file = base_path_figs / "boxplot_metrics_lys2_lys3_lys8_2011-2017.png"
    fig.savefig(file, dpi=300)

    colors = ["#fbb4b9", "#f768a1", "#c51b8a"]
    lys_experiments = ["lys8", "lys3", "lys2"]
    years = [2011, 2012, 2013, 2014, 2015, 2016, 2017]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["MAE_NO3_perc_mass_bs"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        metrics = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
        df_metrics = df_params_metrics_100.loc[:, metrics]
        metrics = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
        df_metrics["lys"] = lys_experiment
        df_metrics["year"] = ""
        ll_df.append(df_metrics)
        for year in years:
            metrics = ["MAE_N_uptake", f"MAE_NO3_perc_mass_bs_{year}", f"MAE_NO3_perc_bs_{year}"]
            df_params_metrics = df_params_metrics.sort_values(by=[f"MAE_NO3_perc_mass_bs_{year}"], ascending=False)
            df_params_metrics_100 = df_params_metrics.iloc[:100, :]
            df_metrics = df_params_metrics_100.loc[:, metrics]
            df_metrics.columns = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
            df_metrics["lys"] = lys_experiment
            df_metrics["year"] = f"{year}"
            ll_df.append(df_metrics)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"], id_vars=["lys", "year"])

    metrics = ["MAE_N_uptake", "MAE_NO3_perc_mass_bs", "MAE_NO3_perc_bs"]
    fig, ax = plt.subplots(3, 1, figsize=(3, 5), sharex=True)
    for i, var in enumerate(metrics):
        data = df_long.loc[df_long["variable"] == var, :]
        g = sns.boxplot(data=data, x="year", y="value", hue="lys", ax=ax[i], palette=colors, showfliers=False)
        ax[i].set_xlabel("")
    ax[0].set_ylabel("$MAE_{N-UPTAKE}$ [m]")
    ax[1].set_ylabel("$MAE_{NO_3-PERC}$ [mg]")
    ax[2].set_ylabel("$MAE_{NO_3-PERC}$ [mg/l]")
    ax[0].legend_.remove()
    ax[1].legend_.remove()
    handels,labels = g.get_legend_handles_labels()
    labels = [fert_lys2_lys3_lys8[lys] for lys in lys_experiments]
    g.legend(handels, labels, frameon=False, title="")     
    fig.tight_layout()
    file = base_path_figs / "boxplot_metrics_lys2_lys3_lys8_Emulti_2011-2017_.png"
    fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
