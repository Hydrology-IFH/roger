from pathlib import Path
import os
import pandas as pd
import click
import roger.tools.labels as labs
import matplotlib as mpl
import seaborn as sns

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

_LABS_TM = {
    "complete-mixing": "CM",
    "piston": "PI",
    "advection-dispersion-power": "AD",
    "time-variant advection-dispersion-power": "AD-TV",
    "preferential-power": "PF",
    "older-preference-power": "OP",
    "advection-dispersion-kumaraswamy": "ADK",
    "time-variant advection-dispersion-kumaraswamy": "ADK-TV",
}


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]

    # dotty plots
    ll_param_sets = [["c1_mak", "c2_mak"], ["dmpv", "lmpv", "ks"], ["theta_ac", "theta_ufc", "theta_pwp"]]
    file = base_path / "svat_monte_carlo" / "figures" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")
    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by="KGE_multi", ascending=False)
    idx_best100 = df_params_metrics100.loc[: df_params_metrics100.index[99], "id"].values.tolist()
    dict_metrics_best = {}
    for sc in ["", "dry", "normal", "wet"]:
        dict_metrics_best[sc] = pd.DataFrame(index=range(len(idx_best100)))
    for sc, sc1 in enumerate(["", "dry", "normal", "wet"]):
        for ps, ps1 in enumerate(ll_param_sets):
            df_metrics = df_params_metrics.loc[
                :, [f"KGE_aet{sc1}", f"KGE_dS{sc1}", f"KGE_q_ss{sc1}", f"KGE_multi{sc1}"]
            ]
            df_params = df_params_metrics.loc[:, ps1]
            nrow = len(df_metrics.columns)
            ncol = len(df_params.columns)
            fig, ax = plt.subplots(nrow, ncol, sharey="row", sharex="col", figsize=(6, 5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param_var in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, s=2, c="grey", alpha=0.5)
                    ax[i, j].set_xlabel("")
                    ax[i, j].set_ylabel("")
                    ax[i, j].set_ylim((0, 1))
                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1.loc[:, "id"] = range(len(df_params_metrics100.index))
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(
                        by=[df_metrics.columns[i]], ascending=False
                    )
                    idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, c="blue", alpha=0.8)
                    # best parameter sets for multi-objective criteria
                    for ii, idx_best in enumerate(idx_best100):
                        y_best = df_metrics.iloc[idx_best, i]
                        x_best = df_params.iloc[idx_best, j]
                        ax[i, j].scatter(x_best, y_best, s=2, c="red", alpha=1)
                        dict_metrics_best[sc1].loc[
                            dict_metrics_best[sc1].index[ii], df_metrics.columns[i]
                        ] = df_params_metrics.loc[idx_best, df_metrics.columns[i]]

            for j, param_var in enumerate(df_params.columns):
                xlabel = labs._LABS[param_var]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel("$KGE_{ET}$ [-]")
            ax[1, 0].set_ylabel(r"$KGE_{\Delta S}$ [-]")
            ax[2, 0].set_ylabel("$KGE_{PERC}$ [-]")
            ax[3, 0].set_ylabel("$KGE_{multi}$ [-]")

            fig.subplots_adjust(bottom=0.1, top=0.98, right=0.98, wspace=0.1, hspace=0.2)
            file = base_path_figs / f"dotty_plots{ps}_{sc1}_optimized_with_KGE_multi.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

            # dotty plots for different antecedent moisture conditions
            fig, ax = plt.subplots(nrow, ncol, sharey="row", sharex="col", figsize=(6, 5))
            for i, metric_var in enumerate(df_metrics.columns):
                for j, param_var in enumerate(df_params.columns):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, s=2, c="grey", alpha=0.5)
                    ax[i, j].set_xlabel("")
                    ax[i, j].set_ylabel("")
                    if metric_var in ["KGE_aet", "KGE_aetwet", "KGE_aetnormal", "KGE_aetdry"]:
                        ax[i, j].set_ylim((0.6, 1.0))
                    elif metric_var in ["KGE_dS", "KGE_dSwet", "KGE_dSnormal", "KGE_dSdry"]:
                        ax[i, j].set_ylim((0.6, 1.0))
                    elif metric_var in ["KGE_q_ss", "KGE_q_sswet", "KGE_q_ssnormal", "KGE_q_ssdry"]:
                        ax[i, j].set_ylim((0.3, 0.6))
                    elif metric_var in ["KGE_multi", "KGE_multiwet", "KGE_multinormal", "KGE_multidry"]:
                        ax[i, j].set_ylim((0.5, 0.8))
                    # best parameter set for individual evaluation metric at specific storage conditions
                    df_params_metrics_sc1 = df_params_metrics.copy()
                    df_params_metrics_sc1.loc[:, "id"] = range(len(df_params_metrics100.index))
                    df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(
                        by=[df_metrics.columns[i]], ascending=False
                    )
                    idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                    for idx_best_sc in idx_best_sc1:
                        y_best_sc = df_metrics.iloc[idx_best_sc, i]
                        x_best_sc = df_params.iloc[idx_best_sc, j]
                        ax[i, j].scatter(x_best_sc, y_best_sc, s=2, c="blue", alpha=0.8)
                    # best parameter sets for multi-objective criteria
                    for ii, idx_best in enumerate(idx_best100):
                        y_best = df_metrics.iloc[idx_best, i]
                        x_best = df_params.iloc[idx_best, j]
                        ax[i, j].scatter(x_best, y_best, s=2, c="red", alpha=1)
                        dict_metrics_best[sc1].loc[
                            dict_metrics_best[sc1].index[ii], df_metrics.columns[i]
                        ] = df_params_metrics.loc[idx_best, df_metrics.columns[i]]

            for j, param_var in enumerate(df_params.columns):
                xlabel = labs._LABS[param_var]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel("$KGE_{ET}$ [-]")
            ax[1, 0].set_ylabel(r"$KGE_{\Delta S}$ [-]")
            ax[2, 0].set_ylabel("$KGE_{PERC}$ [-]")
            ax[3, 0].set_ylabel("$KGE_{multi}$ [-]")

            fig.subplots_adjust(bottom=0.1, top=0.98, right=0.98, wspace=0.1, hspace=0.2)
            file = base_path_figs / f"dotty_plots{ps}_{sc1}inset_optimized_with_KGE_multi.png"
            fig.savefig(file, dpi=300)
            plt.close("all")

    # load metrics of transport simulations
    dict_params_metrics_tm_mc = {}
    for tm_structure in [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
        "preferential-power",
        "older-preference-power",
        "advection-dispersion-kumaraswamy",
        "time-variant advection-dispersion-kumaraswamy",
    ]:
        tms = tm_structure.replace(" ", "_")
        file = base_path / "svat_oxygen18_monte_carlo" / "output" / f"params_metrics_{tms}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_mc[tm_structure] = {}
        dict_params_metrics_tm_mc[tm_structure]["params_metrics"] = df_params_metrics

    # dotty plots of transport simulations
    fig, axes = plt.subplots(8, 4, sharey=True, figsize=(6, 8))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[
            :, ["c1_mak", "c2_mak", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
        ]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=1, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(8):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # dotty plots for PET parameters
    fig, axes = plt.subplots(2, 4, sharey=True, figsize=(6, 3))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["c1_mak", "c2_mak"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(2):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots1_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # dotty plots for macropore parameters and saturated hydraulic conductivity
    fig, axes = plt.subplots(3, 4, sharey=True, figsize=(6, 5))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["dmpv", "lmpv", "ks"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots2_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # dotty plots for soil water retention parameters
    fig, axes = plt.subplots(3, 4, sharey=True, figsize=(6, 5))
    for ncol, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        df_params = df_params_metrics.loc[:, ["theta_ac", "theta_ufc", "theta_pwp"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((-1, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots3_hm_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # # dotty plots for power law parameters
    fig, axes = plt.subplots(6, 2, figsize=(4, 8))
    for ncol, tm_structure in enumerate(tm_structures[2:]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_mc[tm_structure]["params_metrics"]
        df_metrics = df_params_metrics.loc[:, ["KGE_C_iso_q_ss"]]
        if tm_structure == "advection-dispersion-power":
            df_params = df_params_metrics.loc[:, ["k_transp", "k_q_rz", "k_q_ss"]]
        elif tm_structure == "time-variant advection-dispersion-power":
            df_params = df_params_metrics.loc[:, ["k1_transp", "k2_transp", "k1_q_rz", "k2_q_rz", "k1_q_ss", "k2_q_ss"]]
        # select best model run
        idx_best = df_params_metrics["KGE_C_iso_q_ss"].idxmax()
        for nrow, param_name in enumerate(df_params.columns):
            y = df_metrics.loc[:, "KGE_C_iso_q_ss"]
            x = df_params.loc[:, param_name]
            axes[nrow, ncol].scatter(x, y, s=2, c="grey", alpha=0.5)
            xlabel = labs._LABS[param_name]
            axes[nrow, ncol].set_xlabel(xlabel)
            axes[nrow, ncol].set_ylabel("")
            axes[nrow, ncol].set_ylim((0.2, 0.8))
            # best model run
            y_best = df_metrics.iloc[idx_best, 0]
            x_best = df_params.iloc[idx_best, nrow]
            axes[nrow, ncol].scatter(x_best, y_best, s=2, c="red", alpha=1)

        for nrow in range(6):
            if not axes[nrow, ncol].has_data():
                axes[nrow, ncol].set_axis_off()

        axes[0, ncol].set_title(_LABS_TM[tm_structure])

    for j in range(3):
        axes[j, 0].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")
    for j in range(3, 6):
        axes[j, 1].set_ylabel(r"$KGE_{\delta^{18}O}$ [-]")

    fig.tight_layout()
    file = base_path_figs / f"dotty_plots_sas_params_kge_d18O_perc_optimized_with_KGE_multi.png"
    fig.savefig(file, dpi=300)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
