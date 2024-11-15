import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as onp
import click
import roger.tools.labels as labs

onp.random.seed(42)


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

    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        # calculate multi-objective efficiency
        df_params_metrics["E_dS"] = (1 - ((1 - df_params_metrics["r_S_all"])**2 + (1 - df_params_metrics["KGE_alpha_S_all"])**2)**(0.5))
        df_params_metrics["E_multi"] = 0.7 * df_params_metrics["KGE_q_ss_perc_pet"] + 0.3 * (1 - ((1 - df_params_metrics["r_S_perc_pet"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet"])**2)**(0.5))

        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()

        df_metrics = df_params_metrics.loc[:, ["KGE_q_ss_perc_pet", "E_dS", "E_multi"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, alpha=0.2, s=4, color="grey")
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                if metric_var in ["KGE_q_ss_perc_pet"]:
                    ax[i, j].set_ylim((0.3, 0.8))
                elif metric_var in ["E_dS"]:
                    ax[i, j].set_ylim((0.5, 1.0))
                elif metric_var in ["E_multi"]:
                    ax[i, j].set_ylim((0.4, 0.7))

                # best parameter set for individual evaluation metric at specific storage conditions
                df_params_metrics_sc1 = df_params_metrics.copy()
                df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                for idx_best_sc in idx_best_sc1:
                    y_best_sc = df_metrics.iloc[idx_best_sc, i]
                    x_best_sc = df_params.iloc[idx_best_sc, j]
                    ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8, zorder=1)
                if metric_var in ["KGE_q_ss_perc_pet", "E_dS"]:
                    # best parameter sets for multi-objective criteria
                    ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1, zorder=2)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$E_{\Delta S}$ [-]")
        ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_swr_{lys_experiment}.png"
        fig.savefig(file, dpi=300)

        df_params = df_params_metrics.loc[:, ["dmpv", "lmpv", "c_canopy", "c_root"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, color="grey", alpha=0.2, s=4)
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                if metric_var in ["KGE_q_ss_perc_pet"]:
                    ax[i, j].set_ylim((0.3, 0.8))
                elif metric_var in ["E_dS"]:
                    ax[i, j].set_ylim((0.5, 1.0))
                elif metric_var in ["E_multi"]:
                    ax[i, j].set_ylim((0.4, 0.7))

                # best parameter set for individual evaluation metric at specific storage conditions
                df_params_metrics_sc1 = df_params_metrics.copy()
                df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                for idx_best_sc in idx_best_sc1:
                    y_best_sc = df_metrics.iloc[idx_best_sc, i]
                    x_best_sc = df_params.iloc[idx_best_sc, j]
                    ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8, zorder=1)
                if metric_var in ["KGE_q_ss_perc_pet", "E_dS"]:
                    # best parameter sets for multi-objective criteria
                    ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1, zorder=2)
        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$E_{dS}$ [-]")
        ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_mp_crops_{lys_experiment}.png"
        fig.savefig(file, dpi=300)


    lys_experiments = ["lys2", "lys3", "lys8"]
    for lys_experiment in lys_experiments:
        for year in range(2011, 2017):
            df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
            # calculate multi-objective efficiency
            df_params_metrics["E_dS"] = (1 - ((1 - df_params_metrics[f"r_S_all_{year}"])**2 + (1 - df_params_metrics[f"KGE_alpha_S_all_{year}"])**2)**(0.5))
            df_params_metrics["E_multi"] = 0.7 * df_params_metrics["KGE_q_ss_perc_pet"] + 0.3 * (1 - ((1 - df_params_metrics["r_S_perc_pet"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_pet"])**2)**(0.5))

            df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
            df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
            idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()

            df_metrics = df_params_metrics.loc[:, [f"KGE_q_ss_perc_pet_{year}", "E_dS", "E_multi"]]
            df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
            nrow = len(df_metrics.columns)
            ncol = len(df_params.columns)
            fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, alpha=0.2, s=4, color="grey")
                    ax[i, j].set_xlabel("")
                    ax[i, j].set_ylabel("")
                    if metric_var in [f"KGE_q_ss_perc_pet_{year}"]:
                        ax[i, j].set_ylim((0.3, 0.7))
                    elif metric_var in ["E_dS"]:
                        ax[i, j].set_ylim((0.4, 1.0))
                    elif metric_var in ["E_multi"]:
                        ax[i, j].set_ylim((0.2, 0.6))

                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                    idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8, zorder=1)
                    if metric_var in [f"KGE_q_ss_perc_pet_{year}", "E_dS"]:
                        # best parameter sets for multi-objective criteria
                        ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1, zorder=2)

            for j in range(ncol):
                xlabel = labs._LABS[df_params.columns[j]]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
            ax[1, 0].set_ylabel(r"$E_{\Delta S}$ [-]")
            ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

            fig.tight_layout()
            file = base_path_figs / f"dotty_plots_swr_{lys_experiment}_{year}.png"
            fig.savefig(file, dpi=300)

            df_params = df_params_metrics.loc[:, ["dmpv", "lmpv", "c_canopy", "c_root"]]
            nrow = len(df_metrics.columns)
            ncol = len(df_params.columns)
            fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, color="grey", alpha=0.2, s=4)
                    ax[i, j].set_xlabel("")
                    ax[i, j].set_ylabel("")
                    if metric_var in [f"KGE_q_ss_perc_pet_{year}"]:
                        ax[i, j].set_ylim((0.3, 0.7))
                    elif metric_var in ["E_dS"]:
                        ax[i, j].set_ylim((0.4, 1.0))
                    elif metric_var in ["E_multi"]:
                        ax[i, j].set_ylim((0.2, 0.6))

                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                    idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8, zorder=1)
                    if metric_var in ["KGE_q_ss_perc_pet", "E_dS"]:
                        # best parameter sets for multi-objective criteria
                        ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1, zorder=2)
            for j in range(ncol):
                xlabel = labs._LABS[df_params.columns[j]]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
            ax[1, 0].set_ylabel(r"$E_{dS}$ [-]")
            ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

            fig.tight_layout()
            file = base_path_figs / f"dotty_plots_mp_crops_{lys_experiment}_{year}.png"
            fig.savefig(file, dpi=300)


        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        df_params_metrics["E_multi"] = 0.5 * df_params_metrics["KGE_q_ss_perc_dom"] + 0.5 * (1 - ((1 - df_params_metrics["r_S_all"])**2 + (1 - df_params_metrics["KGE_alpha_S_all"])**2)**(0.5))
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()
        df_metrics = df_params_metrics.loc[:, ["KGE_q_ss_perc_dom", "KGE_alpha_S_all", "r_S_all"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, alpha=0.2, s=4, color="grey")
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                if metric_var in ["KGE_q_ss_perc_dom"]:
                    ax[i, j].set_ylim((0.2, 0.6))
                elif metric_var in ["KGE_alpha_S_all"]:
                    ax[i, j].set_ylim((0.5, 1.5))
                elif metric_var in ["r_S_all"]:
                    ax[i, j].set_ylim((0.5, 1.0))

                # best parameter set for individual evaluation metric at specific storage conditions
                if metric_var in ["KGE_q_ss_perc_dom", "r_S_all"]:
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                    idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8)
                if metric_var in ["KGE_q_ss_perc_dom", "KGE_alpha_S_all", "r_S_all"]:
                    # best parameter sets for multi-objective criteria
                    ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$\alpha_{S}$ [-]")
        ax[2, 0].set_ylabel(r"$r_{S}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_swr_{lys_experiment}_Emulti.png"
        fig.savefig(file, dpi=300)


        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        df_params_metrics["E_multi"] = 0.5 * df_params_metrics["KGE_q_ss_perc_pet"] + 0.5 * (1 - ((1 - df_params_metrics["r_S_all"])**2 + (1 - df_params_metrics["KGE_alpha_S_all"])**2)**(0.5))
        df_params_metrics["E_multi_perc_dom"] = 0.5 * df_params_metrics["KGE_q_ss_perc_dom"] + 0.5 * (1 - ((1 - df_params_metrics["r_S_perc_dom"])**2 + (1 - df_params_metrics["KGE_alpha_S_perc_dom"])**2)**(0.5))
        for year in range(2011, 2017):
            df_params_metrics[f"E_multi_{year}"] = 0.5 * df_params_metrics[f"KGE_q_ss_perc_pet_{year}"] + 0.5 * (1 - ((1 - df_params_metrics[f"r_S_all_{year}"])**2 + (1 - df_params_metrics[f"KGE_alpha_S_all_{year}"])**2)**(0.5))
        df_params_metrics["E_multi_year"] = df_params_metrics.loc[:, ["E_multi_2011", "E_multi_2012", "E_multi_2013", "E_multi_2014", "E_multi_2015", "E_multi_2016"]].mean(axis=1) 
        df_metrics = df_params_metrics.loc[:, ["E_multi", "E_multi_perc_dom", "E_multi_year"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, alpha=0.2, s=4, color="grey")
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                ax[i, j].set_ylim((0.1, 0.7))

                # best parameter set for individual evaluation metric at specific storage conditions
                df_params_metrics_sc1 = df_params_metrics.copy()
                df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                df_metrics_sc1 = df_params_metrics_sc1.loc[:, ["E_multi", "E_multi_perc_dom", "E_multi_year"]]
                df_params_sc1 = df_params_metrics_sc1.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
                y_best_sc = df_metrics_sc1.iloc[:100, i]
                x_best_sc = df_params_sc1.iloc[:100, j]
                ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$E_{multi}$ [-]")
        ax[1, 0].set_ylabel("perc. dom.\n $E_{multi}$ [-]")
        ax[2, 0].set_ylabel("split\n $E_{multi}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_swr_{lys_experiment}_Emulti_comparison.png"
        fig.savefig(file, dpi=300)


        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        df_params_metrics["KGE_q_ss_year"] = df_params_metrics.loc[:, ["KGE_q_ss_perc_pet_2011", "KGE_q_ss_perc_pet_2012", "KGE_q_ss_perc_pet_2013", "KGE_q_ss_perc_pet_2014", "KGE_q_ss_perc_pet_2015", "KGE_q_ss_perc_pet_2016"]].mean(axis=1) 
        df_metrics = df_params_metrics.loc[:, ["KGE_q_ss_perc_pet", "KGE_q_ss_perc_dom", "KGE_q_ss_year"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 4.5))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, alpha=0.2, s=4, color="grey")
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                ax[i, j].set_ylim((0.1, 0.7))

                # best parameter set for individual evaluation metric at specific storage conditions
                df_params_metrics_sc1 = df_params_metrics.copy()
                df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                df_metrics_sc1 = df_params_metrics_sc1.loc[:, ["KGE_q_ss_perc_pet", "KGE_q_ss_perc_dom", "KGE_q_ss_year"]]
                df_params_sc1 = df_params_metrics_sc1.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
                y_best_sc = df_metrics_sc1.iloc[:100, i]
                x_best_sc = df_params_sc1.iloc[:100, j]
                ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8)

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel("perc. dom.\n $KGE_{PERC}$ [-]")
        ax[2, 0].set_ylabel("split\n $KGE_{PERC}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_swr_{lys_experiment}_KGE_comparison.png"
        fig.savefig(file, dpi=300)


    return


if __name__ == "__main__":
    main()
