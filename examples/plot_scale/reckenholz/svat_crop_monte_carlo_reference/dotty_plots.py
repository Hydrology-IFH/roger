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
        base_path = Path(__file__).parent

    # directory of results
    base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz") / "output" / "svat_crop_monte_carlo_reference"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_crop_monte_carlo_reference"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["E_multi"], ascending=False)
        df_params_metrics.loc[:, "id"] = range(len(df_params_metrics.index))
        idx_best100 = df_params_metrics.loc[: df_params_metrics.index[99], "id"].values.tolist()

        df_metrics = df_params_metrics.loc[:, ["KGE_q_ss", "r_theta", "E_multi"]]
        df_params = df_params_metrics.loc[:, ["dmpv", "lmpv", "c_canopy_growth", "c_root_growth", "c_basal_crop_coeff"]]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 6))
        for i, metric_var in enumerate(df_metrics.columns):
            for j, param in enumerate(df_params.columns):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, color="grey", alpha=0.2, s=4)
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")
                if metric_var in ["KGE_q_ss"]:
                    ax[i, j].set_ylim((0.0, 0.5))
                elif metric_var in ["r_theta"]:
                    ax[i, j].set_ylim((0.5, 0.8))
                elif metric_var in ["E_multi"]:
                    ax[i, j].set_ylim((0.4, 0.6))

                # best parameter set for individual evaluation metric at specific storage conditions
                df_params_metrics_sc1 = df_params_metrics.copy()
                df_params_metrics_sc1 = df_params_metrics_sc1.sort_values(by=[df_metrics.columns[i]], ascending=False)
                idx_best_sc1 = df_params_metrics_sc1.loc[: df_params_metrics_sc1.index[99], "id"].values.tolist()
                for idx_best_sc in idx_best_sc1:
                    y_best_sc = df_metrics.iloc[idx_best_sc, i]
                    x_best_sc = df_params.iloc[idx_best_sc, j]
                    ax[i, j].scatter(x_best_sc, y_best_sc, s=2, color="blue", alpha=0.8)
                if metric_var in ["KGE_q_ss", "theta_r"]:
                    # best parameter sets for multi-objective criteria
                    ax[i, j].scatter(x[:100], y[:100], s=2, color="red", alpha=1)
        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$r_{\theta}$ [-]")
        ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_mp_crops_{lys_experiment}.png"
        fig.savefig(file, dpi=250)
    return


if __name__ == "__main__":
    main()
