import os
from pathlib import Path
import pandas as pd
from SALib.analyze import sobol
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import yaml
import numpy as onp
import roger.tools.labels as labs
import click
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

onp.random.seed(42)


@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path(__file__).parent

    # sampled parameter space
    file_path = base_path / "param_bounds.yml"
    with open(file_path, "r") as file:
        bounds = yaml.safe_load(file)

    # directory of results
    base_path_output = base_path / "output"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = base_path / "figures"
    if not os.path.exists(base_path_figs):
        os.mkdir(base_path_figs)

    # load simulation
    tm_structures = [
        "complete-mixing",
        "piston",
        "advection-dispersion-power",
        "time-variant advection-dispersion-power",
    ]
    for tm_structure in tm_structures:
        # perform sensitivity analysis
        click.echo(f"Perform sensitivity analysis ({tm_structure})...")
        tms = tm_structure.replace(" ", "_")
        file = base_path_figs / f"params_metrics_{tms}.txt"
        df_params_metrics = pd.read_csv(file, header=0, index_col=False, sep="\t")
        bounds_sobol = {}
        bounds_sobol["num_vars"] = 8
        bounds_sobol["names"] = bounds[tm_structure]["names"][:8]
        bounds_sobol["bounds"] = bounds[tm_structure]["bounds"][:8]
        for sc, sc1 in zip([0, 1, 2, 3], ["", "dry", "normal", "wet"]):
            df_params = df_params_metrics.loc[:, bounds_sobol["names"]]
            df_metrics = df_params_metrics.loc[
                :, [f"KGE_aet{sc1}", f"KGE_dS{sc1}", f"KGE_q_ss{sc1}", f"KGE_multi{sc1}"]
            ]
            df_metrics.columns = ["KGE_aet", "KGE_dS", "KGE_q_ss", "KGE_multi"]

            dict_si = {}
            for name in df_metrics.columns:
                Y = df_metrics[name].values
                Si = sobol.analyze(bounds_sobol, Y, calc_second_order=False)
                Si_filter = {k: Si[k] for k in ["ST", "ST_conf", "S1", "S1_conf"]}
                dict_si[name] = pd.DataFrame(Si_filter, index=bounds_sobol["names"])

            # plot sobol indices
            _LABS = {
                "KGE_aet": r"$KGE_{AET}$",
                "KGE_q_ss": r"$KGE_{PERC}$",
                "KGE_dS": r"$KGE_{\Delta S}$",
                "r_dS": r"$r_{\Delta S}$",
                "E_multi": r"$E_{multi}$",
                "KGE_multi": r"$E_{multi}$",
            }
            ncol = len(df_metrics.columns)
            xaxis_labels = [labs._LABS[k].split(" ")[0] for k in bounds_sobol["names"]]
            cmap = cm.get_cmap("Reds")
            norm = Normalize(vmin=0, vmax=2)
            colors = cmap(norm([0.5, 1.5]))
            fig, ax = plt.subplots(1, ncol, sharey=True, figsize=(6, 1.5))
            for i, name in enumerate(df_metrics.columns):
                indices = dict_si[name][["S1", "ST"]]
                err = dict_si[name][["S1_conf", "ST_conf"]]
                indices.plot.bar(yerr=err.values.T, ax=ax[i], color=colors, width=1.0)
                ax[i].set_xticklabels(xaxis_labels)
                ax[i].set_title(_LABS[name])
                ax[i].legend(["First-order", "Total"], frameon=False, loc="upper left")
            ax[-1].legend().set_visible(False)
            ax[-2].legend().set_visible(False)
            ax[-3].legend().set_visible(False)
            ax[0].set_ylabel("Sobol index [-]")
            fig.tight_layout()
            file = base_path_figs / f"sobol_indices_{sc1}_for_{tms}.png"
            fig.savefig(file, dpi=250)

            # make dotty plots
            nrow = len(df_metrics.columns)
            ncol = 8
            fig, ax = plt.subplots(nrow, ncol, sharey="row", figsize=(6, 5))
            for i in range(nrow):
                for j in range(ncol):
                    y = df_metrics.iloc[:, i]
                    x = df_params.iloc[:, j]
                    ax[i, j].scatter(x, y, s=4, c="grey", alpha=0.5)
                    ax[i, j].set_xlabel("")
                    ax[i, j].set_ylabel("")

            for j in range(ncol):
                xlabel = labs._LABS[bounds_sobol["names"][j]]
                ax[-1, j].set_xlabel(xlabel)

            ax[0, 0].set_ylabel("$KGE_{ET}$ [-]")
            ax[1, 0].set_ylabel(r"$KGE_{\Delta S}$ [-]")
            ax[2, 0].set_ylabel("$KGE_{PERC}$ [-]")
            ax[3, 0].set_ylabel("$KGE_{multi}$\n [-]")

            fig.subplots_adjust(wspace=0.2, hspace=0.3)
            file = base_path_figs / f"dotty_plots_{sc1}_for_{tms}.png"
            fig.savefig(file, dpi=250)

    return


if __name__ == "__main__":
    main()
