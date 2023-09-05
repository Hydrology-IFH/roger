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
    base_path_output = base_path.parent / "output" / "svat_monte_carlo"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path.parent / "figures" / "svat_monte_carlo"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    lys_experiments = ["lys2", "lys3", "lys4", "lys8", "lys9"]
    for lys_experiment in lys_experiments:
        df_params_eff = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")

        df_eff = df_params_eff.loc[:, ["KGE_q_ss", "r_theta", "E_multi"]]
        df_params = df_params_eff.loc[:, ["theta_ac", "theta_ufc", "theta_pwp", "ks"]]
        nrow = len(df_eff.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(6, 4.5))
        for i in range(nrow):
            for j in range(ncol):
                y = df_eff.iloc[:, i]
                x = df_params.iloc[:, j]
                sns.regplot(
                    x=x, y=y, ax=ax[i, j], ci=None, color="k", scatter_kws={"alpha": 0.2, "s": 4, "color": "grey"}
                )
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$r_{\theta}$ [-]")
        ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_swr_{lys_experiment}.png"
        fig.savefig(file, dpi=250)

        df_params = df_params_eff.loc[:, ["dmpv", "lmpv"]]
        nrow = len(df_eff.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey=True, figsize=(3, 4.5))
        for i in range(nrow):
            for j in range(ncol):
                y = df_eff.iloc[:, i]
                x = df_params.iloc[:, j]
                sns.regplot(
                    x=x, y=y, ax=ax[i, j], ci=None, color="k", scatter_kws={"alpha": 0.2, "s": 4, "color": "grey"}
                )
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[1, 0].set_ylabel(r"$r_{\theta}$ [-]")
        ax[2, 0].set_ylabel("$E_{multi}$\n [-]")

        fig.tight_layout()
        file = base_path_figs / f"dotty_plots_mp_{lys_experiment}.png"
        fig.savefig(file, dpi=250)
    return


if __name__ == "__main__":
    main()
