from pathlib import Path
import os
import pandas as pd
import numpy as onp
import click
from de import de
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

    # diagnostic polar plots
    file = base_path / "svat_monte_carlo" / "figures" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")

    df_params_metrics100 = df_params_metrics.copy()
    df_params_metrics100.loc[:, "id"] = range(len(df_params_metrics100.index))
    df_params_metrics100 = df_params_metrics100.sort_values(by="KGE_multi", ascending=False)
    df_for_diag100 = df_params_metrics100.loc[: df_params_metrics100.index[99], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag100.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag100.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag100.loc[:, f"DE_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag100.loc[:, f"phi_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag100.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag100.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag100.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag100.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag100.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_100_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag100.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag100.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag100.loc[:, f"DE_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag100.loc[:, f"phi_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag100.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag100.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag100.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_100_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

    df_params_metrics10 = df_params_metrics.copy()
    df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
    df_params_metrics10 = df_params_metrics10.sort_values(by="KGE_multi", ascending=False)
    df_for_diag10 = df_params_metrics10.loc[: df_params_metrics10.index[9], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag10.loc[:, f"DE_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag10.loc[:, f"phi_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag10.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag10.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.pdf"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag10.loc[:, f"DE_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag10.loc[:, f"phi_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)
    file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi.pdf"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

    df_for_diag1 = df_params_metrics10.loc[: df_params_metrics10.index[0], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag1.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag1.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag1.loc[:, f"DE_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag1.loc[:, f"phi_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag1.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag1.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag1.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag1.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag1.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_1_optimized_with_KGE_multi.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "dS"
    fig = de.diag_polar_plot_multi(
        df_for_diag1.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag1.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag1.loc[:, f"DE_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag1.loc[:, f"phi_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag1.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag1.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag1.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_1_optimized_with_KGE_multi.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)

    # diagnostic polar plots for transport models
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
        file = base_path / "svat_oxygen18_monte_carlo" / "figures" / f"params_metrics_{tms}.txt"
        df_params_metrics = pd.read_csv(file, sep="\t")
        dict_params_metrics_tm_mc[tm_structure] = {}
        dict_params_metrics_tm_mc[tm_structure]["params_metrics"] = df_params_metrics

    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]
    for i, tm_structure in enumerate(tm_structures):
        tms = tm_structure.replace(" ", "_")
        df_params_metrics10 = dict_params_metrics_tm_mc[tm_structure]["params_metrics"].copy()
        df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
        df_params_metrics10 = df_params_metrics10.sort_values(by=["KGE_C_iso_q_ss"], ascending=False)
        df_for_diag10 = df_params_metrics10.loc[: df_params_metrics10.index[9], :]

        var_sim = "C_iso_q_ss"
        fig = de.diag_polar_plot_multi(
            df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag10.loc[:, f"DE_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag10.loc[:, f"phi_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
        )
        file = f"diag_polar_plot_{var_sim}_{tms}.pdf"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)


    file = base_path / "hydrus_benchmark" / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, sep="\t")

    df_params_metrics10 = df_params_metrics.copy()
    df_params_metrics10.loc[:, "id"] = range(len(df_params_metrics10.index))
    df_params_metrics10 = df_params_metrics10.sort_values(by="KGE_multi", ascending=False)
    df_for_diag10 = df_params_metrics10.loc[: df_params_metrics10.index[9], :]
    vars_sim = ["aet", "q_ss"]
    for var_sim in vars_sim:
        fig = de.diag_polar_plot_multi(
            df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
            df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
            df_for_diag10.loc[:, f"DE_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
            df_for_diag10.loc[:, f"phi_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
            df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
            df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
            a0=df_for_diag10.loc[:, f"ioa0_{var_sim}"].values,
            share0=onp.round(onp.max(df_for_diag10.loc[:, f"p0_{var_sim}"]), 2),
        )
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi_hydrus.png"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)
        file = f"diag_polar_plot_{var_sim}_10_optimized_with_KGE_multi_hydrus.pdf"
        path = base_path_figs / file
        # fig.tight_layout()
        fig.savefig(path, dpi=300)

    var_sim = "d18O_perc_bs"
    fig = de.diag_polar_plot_multi(
        df_for_diag10.loc[:, f"brel_mean_{var_sim}"].values,
        df_for_diag10.loc[:, f"temp_cor_{var_sim}"].values,
        df_for_diag10.loc[:, f"DE_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_dir_{var_sim}"].values,
        df_for_diag10.loc[:, f"phi_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_lf_{var_sim}"].values,
        df_for_diag10.loc[:, f"b_tot_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_hf_{var_sim}"].values,
        df_for_diag10.loc[:, f"err_lf_{var_sim}"].values,
    )
    file = f"diag_polar_plot_{var_sim}_{tms}_10_optimized_with_KGE_multi_hydrus.png"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)
    file = f"diag_polar_plot_{var_sim}_{tms}_10_optimized_with_KGE_multi_hydrus.pdf"
    path = base_path_figs / file
    # fig.tight_layout()
    fig.savefig(path, dpi=300)


    plt.close("all")
    return


if __name__ == "__main__":
    main()
