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

    # directory of figures
    base_path_figs = Path(__file__).parent / "figures" 
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

    fert_lys2_lys3_lys8 = {"lys2": "130% N-fertilized",
                           "lys3": "100% N-fertilized",
                           "lys8": "70% N-fertilized"
    }

    _lys = {"lys2": "Lys 2",
            "lys3": "Lys 3",
            "lys8": "Lys 8",
    }

    lys_experiments = ["lys2", "lys3", "lys8"]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_monte_carlo" / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_q_ss_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_q_ss_2011-2015"].to_frame()
        df_metrics.columns = ["KGE_perc"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "non-explicit"
        ll_df.append(df_metrics)
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_crop_monte_carlo_crop-specific" / f"params_eff_{lys_experiment}_bulk_samples.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_q_ss_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_q_ss_2011-2015"].to_frame()
        df_metrics.columns = ["KGE_perc"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "explicit"
        ll_df.append(df_metrics)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=["KGE_perc"], id_vars=["lys", "model"])
    colors = ["#bdbdbd", "#31a354"]
    fig, axs = plt.subplots(3, 1, figsize=(2, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        data = df_long.loc[df_long["lys"] == _lys[lys_experiment], :]
        sns.stripplot(data=data, x="lys", y="value", hue="model", ax=axs[i], palette=colors, size=3)
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])
        axs[i].set_ylabel("KGE [-]")
        axs[i].legend_.remove()
        axs[i].set_ylim(0.0, 0.9)
    fig.tight_layout()
    file = base_path_figs / f"stripplot_KGE_percolation.png"
    fig.savefig(file, dpi=300)


    lys_experiments = ["lys2", "lys3", "lys8"]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_nitrate_monte_carlo" / f"params_metrics_{lys_experiment}_complete-mixing.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_NO3_perc_mass_bs_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_NO3_perc_mass_bs_2011-2015"].to_frame()
        df_metrics.columns = ["KGE_NO3"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "non-explicit"
        ll_df.append(df_metrics)
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_crop_nitrate_monte_carlo" / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_NO3_perc_mass_bs_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_NO3_perc_mass_bs_2011-2015"].to_frame()
        df_metrics.loc[:, :] = onp.nan
        df_metrics.columns = ["KGE_NO3"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "explicit"
        ll_df.append(df_metrics)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=["KGE_NO3"], id_vars=["lys", "model"])
    colors = ["#bdbdbd", "#31a354"]
    fig, axs = plt.subplots(3, 1, figsize=(2, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        data = df_long.loc[df_long["lys"] == _lys[lys_experiment], :]
        sns.stripplot(data=data, x="lys", y="value", hue="model", ax=axs[i], palette=colors, size=3)
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])
        axs[i].set_ylabel("KGE [-]")
        axs[i].legend_.remove()
        axs[i].set_ylim(0.0, 0.9)
    fig.tight_layout()
    file = base_path_figs / f"stripplot_KGE_nitrate_leaching_1.png"
    fig.savefig(file, dpi=300)

    lys_experiments = ["lys2", "lys3", "lys8"]
    ll_df = []
    for lys_experiment in lys_experiments:
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_nitrate_monte_carlo" / f"params_metrics_{lys_experiment}_complete-mixing.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_NO3_perc_mass_bs_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_NO3_perc_mass_bs_2011-2015"].to_frame()
        df_metrics.columns = ["KGE_NO3"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "non-explicit"
        ll_df.append(df_metrics)
        df_params_metrics = pd.read_csv(base_path / "output" / "svat_crop_nitrate_monte_carlo" / f"params_metrics_{lys_experiment}_advection-dispersion-power.txt", sep="\t")
        df_params_metrics = df_params_metrics.sort_values(by=["KGE_NO3_perc_mass_bs_2011-2015"], ascending=False)
        df_params_metrics_100 = df_params_metrics.iloc[:100, :]
        df_metrics = df_params_metrics_100.loc[:, "KGE_NO3_perc_mass_bs_2011-2015"].to_frame()
        df_metrics.columns = ["KGE_NO3"]
        df_metrics["lys"] = _lys[lys_experiment]
        df_metrics["model"] = "explicit"
        ll_df.append(df_metrics)
    df = pd.concat(ll_df)
    df_long = pd.melt(df, value_vars=["KGE_NO3"], id_vars=["lys", "model"])
    colors = ["#bdbdbd", "#31a354"]
    fig, axs = plt.subplots(3, 1, figsize=(2, 4))
    for i, lys_experiment in enumerate(lys_experiments):
        data = df_long.loc[df_long["lys"] == _lys[lys_experiment], :]
        sns.stripplot(data=data, x="lys", y="value", hue="model", ax=axs[i], palette=colors, size=3)
        axs[i].set_xlabel("")
        axs[i].set_xticklabels([])
        axs[i].set_ylabel("KGE [-]")
        axs[i].legend_.remove()
        axs[i].set_ylim(0.0, 0.9)
    fig.tight_layout()
    file = base_path_figs / f"stripplot_KGE_nitrate_leaching_2.png"
    fig.savefig(file, dpi=300)


    return


if __name__ == "__main__":
    main()
