import os
from pathlib import Path
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


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent
    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    file = base_path_figs / "params_metrics.txt"
    df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")
    # dotty plots for different antecedent moisture conditions
    click.echo("Dotty plots ...")
    for sc, sc1 in zip([0, 1, 2, 3], ["", "dry", "normal", "wet"]):
        df_metrics = df_params_metrics.loc[:, [f"KGE_aet{sc1}", f"KGE_dS{sc1}", f"KGE_q_ss{sc1}", f"KGE_multi{sc1}"]]
        df_params = df_params_metrics.loc[
            :, ["c1_mak", "c2_mak", "dmpv", "lmpv", "theta_ac", "theta_ufc", "theta_pwp", "ks"]
        ]
        nrow = len(df_metrics.columns)
        ncol = len(df_params.columns)
        fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(14, 7))
        for i in range(nrow):
            for j in range(ncol):
                y = df_metrics.iloc[:, i]
                x = df_params.iloc[:, j]
                ax[i, j].scatter(x, y, s=4, c="grey", alpha=0.5)
                ax[i, j].set_xlabel("")
                ax[i, j].set_ylabel("")

        for j in range(ncol):
            xlabel = labs._LABS[df_params.columns[j]]
            ax[-1, j].set_xlabel(xlabel)

        ax[0, 0].set_ylabel("$KGE_{ET}$ [-]")
        ax[1, 0].set_ylabel(r"$KGE_{\Delta S}$ [-]")
        ax[2, 0].set_ylabel("$KGE_{PERC}$ [-]")
        ax[3, 0].set_ylabel("$KGE_{multi}$\n [-]")

        fig.subplots_adjust(wspace=0.2, hspace=0.3)
        file = base_path_figs / f"dotty_plots_{sc1}.png"
        fig.savefig(file, dpi=250)
        plt.close("all")
    return


if __name__ == "__main__":
    main()
