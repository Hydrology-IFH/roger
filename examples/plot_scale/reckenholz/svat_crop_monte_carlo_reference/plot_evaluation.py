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

@click.option("-td", "--tmp-dir", type=str, default=None)
@click.command("main")
def main(tmp_dir):
    if tmp_dir:
        base_path = Path(tmp_dir)
    else:
        base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/reckenholz")

    # directory of results
    base_path_output = base_path / "output" / "svat_crop_monte_carlo_reference"
    if not os.path.exists(base_path_output):
        os.mkdir(base_path_output)
    # directory of figures
    base_path_figs = Path(__file__).parent.parent / "figures" / "svat_crop_monte_carlo_reference"
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

    metrics = ["KGE_q_ss"]
    years = [2011, 2012, 2013, 2014, 2015, 2016]
    lys_experiments = ["lys8", "lys3", "lys2"]

    for metric in metrics:
        nrow = len(years)
        ncol = len(lys_experiments)
        fig, ax = plt.subplots(nrow, ncol, sharey=True, sharex=True, figsize=(6, 6))
        for jj, year in enumerate(years):
            for ii, lys_experiment in enumerate(lys_experiments):
                df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_all"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_all_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_all_{year}", ax=ax[jj, ii], fill=True, color="grey", alpha=0.5)
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_perc_dom"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_perc_dom_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_perc_dom_{year}", ax=ax[jj, ii], fill=True, color="blue", alpha=1.0)
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_pet_dom"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_pet_dom_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_pet_dom_{year}", ax=ax[jj, ii], fill=True, color="green", alpha=1.0)
                ax[jj, ii].set_xlabel("")
                ax[jj, ii].set_ylabel("")
                ax[jj, ii].set_title("")
                ax[0, ii].set_title("%s\n%s" % (_lys[lys_experiment], fert_lys2_lys3_lys8[lys_experiment]))
                ax[jj, 0].set_ylabel("%s\n%s" % (year, crops_lys2_lys3_lys8[year]))
                ax[jj, ii].set_xlim(-1, 0.8)

        if metric in ["KGE_q_ss"]:
            ax[-1, 0].set_xlabel("$KGE_{PERC}$ [-]")
            ax[-1, 1].set_xlabel("$KGE_{PERC}$ [-]")
            ax[-1, 2].set_xlabel("$KGE_{PERC}$ [-]")
        elif metric in ["r_theta"]:
            ax[-1, 0].set_xlabel("$r_{\theta}$ [-]")
            ax[-1, 1].set_xlabel("$r_{\theta}$ [-]")
            ax[-1, 2].set_xlabel("$r_{\theta}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"kde_years_{metric}_lys2_lys3_lys8.png"
        fig.savefig(file, dpi=300)


    metrics = ["KGE_q_ss"]
    years = [2011, 2012, 2013, 2014, 2015, 2016]
    lys_experiments = ["lys4", "lys9"]

    for metric in metrics:
        nrow = len(years)
        ncol = len(lys_experiments)
        fig, ax = plt.subplots(nrow, ncol, sharey=True, sharex=True, figsize=(4, 6))
        for jj, year in enumerate(years):
            metric_year = f"{metric}_{year}"
            for ii, lys_experiment in enumerate(lys_experiments):
                df_params_metrics = pd.read_csv(base_path_output / f"params_eff_{lys_experiment}.txt", sep="\t")
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_all"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_all_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_all_{year}", ax=ax[jj, ii], fill=True, color="grey", alpha=0.5)
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_perc_dom"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_perc_dom_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_perc_dom_{year}", ax=ax[jj, ii], fill=True, color="blue", alpha=1.0)
                df_params_metrics = df_params_metrics.sort_values(by=[f"{metric}_pet_dom"], ascending=False)
                df = df_params_metrics.loc[:, [f"{metric}_pet_dom_{year}"]]
                df = df.iloc[:100, :]
                g = sns.kdeplot(data=df, x=f"{metric}_pet_dom_{year}", ax=ax[jj, ii], fill=True, color="green", alpha=1.0)
                ax[jj, ii].set_xlabel("")
                ax[jj, ii].set_ylabel("")
                ax[jj, ii].set_title("")
                ax[0, ii].set_title("%s\n%s" % (_lys[lys_experiment], fert_lys4_lys9[lys_experiment]))
                ax[jj, 0].set_ylabel("%s\n%s" % (year, crops_lys4_lys9[year]))
                ax[jj, ii].set_xlim(-1, 0.8)

        if metric in ["KGE_q_ss"]:
            ax[-1, 0].set_xlabel("$KGE_{PERC}$ [-]")
            ax[-1, 1].set_xlabel("$KGE_{PERC}$ [-]")
        elif metric in ["r_theta"]:
            ax[-1, 0].set_xlabel("$r_{\theta}$ [-]")
            ax[-1, 1].set_xlabel("$r_{\theta}$ [-]")

        fig.tight_layout()
        file = base_path_figs / f"kde_years_{metric}_lys4_lys9.png"
        fig.savefig(file, dpi=300)
    return


if __name__ == "__main__":
    main()
