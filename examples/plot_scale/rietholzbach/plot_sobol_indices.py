from pathlib import Path
import os
from SALib.analyze import sobol
from matplotlib.colors import Normalize
import pandas as pd
import numpy as onp
import matplotlib.cm as cm
import yaml
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

    # perform sensitivity analysis
    dict_params_metrics_tm_sa = {}
    for tm_structure in tm_structures:
        tms = tm_structure.replace(" ", "_")
        file = base_path / "svat_oxygen18_sensitivity" / "figures" / f"params_metrics_{tms}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_sa[tm_structure] = {}
        dict_params_metrics_tm_sa[tm_structure]["params_metrics"] = df_params_metrics

    # sampled parameter space
    file_path = base_path / "svat_oxygen18_sensitivity" / "param_bounds.yml"
    with open(file_path, "r") as file:
        bounds = yaml.safe_load(file)

    _LABS_TITLE = {
        "KGE_C_iso_q_ss": "$KGE_{\delta^{18}O}$",
        "ttavg_transp": r"$\overline{TT_{transp}}$",
        "tt10_transp": r"$TT_{10-transp}$",
        "tt50_transp": r"$TT_{50-transp}$",
        "tt90_transp": r"$TT_{90-transp}$",
        "ttavg_q_ss": r"$\overline{TT_{perc_{ss}}}$",
        "tt10_q_ss": r"$TT_{10-perc_{ss}}$",
        "tt50_q_ss": r"$TT_{50-perc_{ss}}$",
        "tt90_q_ss": r"$TT_{90-perc_{ss}}$",
        "rtavg_s": r"$\overline{RT}$",
        "rt10_s": r"$RT_{10}$",
        "rt50_s": r"$RT_{50}$",
        "rt90_s": r"$RT_{90}$",
    }
    metrics_tt = ["ttavg", "tt10", "tt50", "tt90"]
    for metric_tt in metrics_tt:
        metrics_sa = [f"{metric_tt}_transp", f"{metric_tt}_q_ss", "KGE_C_iso_q_ss"]
        ncol = len(metrics_sa)
        nrow = len(tm_structures)
        cmap = cm.get_cmap("Reds")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 6))
        for j, tm_structure in enumerate(tm_structures):
            tms = tm_structure.replace(" ", "_")
            df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
            dict_si = {}
            for name in metrics_sa:
                Y = df_params_metrics.loc[:, name].values
                Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
                Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
                Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

            # plot sobol indices
            xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
            colors = cmap(norm([0.5, 1.5]))
            for i, name in enumerate(metrics_sa):
                indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
                err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
                indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors, width=1.0)
                ax[j, i].set_xticklabels(xaxis_labels)
                ax[0, i].set_title(_LABS_TITLE[name])
                ax[j, i].legend(["First-order", "Total"], frameon=False)
                ax[j, i].legend().set_visible(False)
                if tm_structure in [
                    "complete-mixing",
                    "advection-dispersion-power",
                    "time-variant advection-dispersion-power",
                ]:
                    ax[j, i].set_ylim(0, 1.2)
                else:
                    ax[j, i].set_ylim(0, 3.0)
        ax[-1, -1].legend().set_visible(True)
        ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
        ax[0, 0].set_ylabel("CM\nSobol index [-]")
        ax[1, 0].set_ylabel("PI\nSobol index [-]")
        ax[2, 0].set_ylabel("AD\nSobol index [-]")
        ax[3, 0].set_ylabel("AD-TV\nSobol index [-]")
        # fig.subplots_adjust(bottom=0.1, right=0.95, hspace=0.65)
        fig.tight_layout()
        file = base_path_figs / f"sobol_indices_{metric_tt}_hm.png"
        fig.savefig(file, dpi=300)

        metrics_sa = [f"{metric_tt}_transp", f"{metric_tt}_q_ss", "KGE_C_iso_q_ss"]
        ncol = len(metrics_sa)
        nrow = 2
        cmap = cm.get_cmap("Reds")
        norm = Normalize(vmin=0, vmax=2)
        colors = cmap(norm([0.5, 1.5]))
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 3))
        for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
            tms = tm_structure.replace(" ", "_")
            df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
            dict_si = {}
            for name in metrics_sa:
                Y = df_params_metrics.loc[:, name].values
                Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
                Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

            # plot sobol indices
            xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
            colors = cmap(norm([0.5, 1.5]))
            for i, name in enumerate(metrics_sa):
                indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
                err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
                indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors)
                ax[j, i].set_xticklabels(xaxis_labels)
                ax[j, i].tick_params(axis="x", rotation=33)
                ax[0, i].set_title(_LABS_TITLE[name])
                ax[j, i].legend(["First-order", "Total"], frameon=False)
                ax[j, i].legend().set_visible(False)
                ax[j, i].set_ylim(0, 1.0)
        ax[-1, -2].legend().set_visible(True)
        ax[-1, -2].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
        ax[0, 0].set_ylabel("AD\nSobol index [-]")
        ax[1, 0].set_ylabel("AD-TV\nSobol index [-]")
        fig.tight_layout()
        file = base_path_figs / f"sobol_indices_{metric_tt}_tm.png"
        fig.savefig(file, dpi=300)

    metrics_sa = ["rtavg_s", "rt10_s", "rt50_s", "rt90_s"]
    ncol = len(metrics_sa)
    nrow = len(tm_structures)
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 6))
    for j, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors, width=1)
            ax[j, i].set_xticklabels(xaxis_labels)
            ax[0, i].set_title(_LABS_TITLE[name])
            ax[j, i].legend(["First-order", "Total"], frameon=False)
            ax[j, i].legend().set_visible(False)
    ax[-1, -1].legend().set_visible(True)
    ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper right", fontsize=8, bbox_to_anchor=(1.8, 1.1))
    ax[0, 0].set_ylabel("CM\nSobol index [-]")
    ax[1, 0].set_ylabel("PI\nSobol index [-]")
    ax[2, 0].set_ylabel("AD\nSobol index [-]")
    ax[3, 0].set_ylabel("AD-TV\nSobol index [-]")
    fig.subplots_adjust(bottom=0.2, right=0.85, hspace=0.65)
    file = base_path_figs / "sobol_indices_rt_hm.png"
    fig.savefig(file, dpi=300)

    ncol = len(metrics_sa)
    nrow = 2
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 3))
    for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j, i], color=colors)
            ax[j, i].set_xticklabels(xaxis_labels)
            ax[j, i].tick_params(axis="x", rotation=33)
            ax[0, i].set_title(_LABS_TITLE[name])
            ax[j, i].legend(["First-order", "Total"], frameon=False)
            ax[j, i].legend().set_visible(False)
    ax[-1, -1].legend().set_visible(True)
    ax[-1, -1].legend(["First-order", "Total"], frameon=False, loc="upper right", fontsize=8, bbox_to_anchor=(1.8, 1.1))
    ax[0, 0].set_ylabel("AD\nSobol index [-]")
    ax[1, 0].set_ylabel("AD-TV\nSobol index [-]")
    fig.subplots_adjust(bottom=0.2, right=0.85, hspace=0.65)
    file = base_path_figs / "sobol_indices_rt_tm.png"
    fig.savefig(file, dpi=300)

    metrics_sa = ["KGE_C_iso_q_ss"]
    nrow = len(metrics_sa)
    ncol = len(tm_structures)
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 1.2))
    for j, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Y = onp.where(onp.isnan(Y), onp.nanmean(Y), Y)
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][:8]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[:8, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[:8, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j], color=colors, width=1)
            ax[j].set_xticklabels(xaxis_labels)
            ax[j].legend(["First-order", "Total"], frameon=False)
            ax[j].legend().set_visible(False)
    ax[-1].legend().set_visible(True)
    ax[-1].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
    ax[0].set_title("CM")
    ax[1].set_title("PI")
    ax[2].set_title("AD")
    ax[3].set_title("AD-TV")
    ax[0].set_ylabel("Sobol index [-]")
    fig.tight_layout()
    file = base_path_figs / "sobol_indices_kge_18O_perc_hm.png"
    fig.savefig(file, dpi=300)

    nrow = len(metrics_sa)
    ncol = 2
    cmap = cm.get_cmap("Reds")
    norm = Normalize(vmin=0, vmax=2)
    colors = cmap(norm([0.5, 1.5]))
    fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(4, 1.2))
    for j, tm_structure in enumerate(["advection-dispersion-power", "time-variant advection-dispersion-power"]):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics = dict_params_metrics_tm_sa[tm_structure]["params_metrics"]
        dict_si = {}
        for name in metrics_sa:
            Y = df_params_metrics.loc[:, name].values
            Si = sobol.analyze(bounds[tm_structure], Y, calc_second_order=False)
            Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
            dict_si[name] = pd.DataFrame(Si_filter, index=bounds[tm_structure]["names"])

        # plot sobol indices
        xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds[tm_structure]["names"]][8:]
        colors = cmap(norm([0.5, 1.5]))
        for i, name in enumerate(metrics_sa):
            indices = dict_si[name][["S1", "ST"]].iloc[8:, :]
            err = dict_si[name][["S1_conf", "ST_conf"]].iloc[8:, :]
            indices.plot.bar(yerr=err.values.T, ax=ax[j], color=colors, width=1)
            ax[j].set_xticklabels(xaxis_labels)
            ax[j].tick_params(axis="x", rotation=33)
            ax[j].legend(["First-order", "Total"], frameon=False)
            ax[j].legend().set_visible(False)
    ax[0].legend().set_visible(True)
    ax[0].legend(["First-order", "Total"], frameon=False, loc="upper left", fontsize=8)
    ax[0].set_title("AD")
    ax[1].set_title("AD-TV")
    ax[0].set_ylabel("Sobol index [-]")
    fig.tight_layout()
    file = base_path_figs / "sobol_indices_kge_18O_perc_tm.png"
    fig.savefig(file, dpi=300)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
