from pathlib import Path
import os
import numpy as onp
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


def kumaraswamy_cdf(x, a, b):
    return 1 - (1 - (x) ** a) ** b


def kumaraswamy_pdf(x, a, b):
    return a * b * x ** (a - 1) * (1 - x**a) ** (b - 1)


def power_cdf(x, k):
    return x**k


def power_pdf(x, k):
    return k**k * x ** (k - 1)


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

    # plot SAS function
    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, kumaraswamy_cdf(x, 1, 20), color="#034e7b", lw=1, label="a=1, b=20")
    axs.plot(x, kumaraswamy_cdf(x, 1.5, 20), color="#0570b0", lw=1, label="a=1.5, b=20")
    axs.plot(x, kumaraswamy_cdf(x, 1, 10), color="#3690c0", lw=1, label="a=1, b=10")
    axs.plot(x, kumaraswamy_cdf(x, 3, 1), color="#74a9cf", lw=1, label="a=3, b=1")
    axs.plot(x, kumaraswamy_cdf(x, 5, 1), color="#a6bddb", lw=1, label="a=5, b=1")
    axs.plot(x, kumaraswamy_cdf(x, 5, 1.5), color="#d0d1e6", lw=1, label="a=5, b=1.5")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 1))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel("$P_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.57, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "kumaraswamy_cdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "kumaraswamy_cdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, kumaraswamy_pdf(x, 1, 20), color="#034e7b", lw=1, label="a=1, b=20")
    axs.plot(x, kumaraswamy_pdf(x, 1.5, 20), color="#0570b0", lw=1, label="a=1.5, b=20")
    axs.plot(x, kumaraswamy_pdf(x, 1, 10), color="#3690c0", lw=1, label="a=1, b=10")
    axs.plot(x, kumaraswamy_pdf(x, 3, 1), color="#74a9cf", lw=1, label="a=3, b=1")
    axs.plot(x, kumaraswamy_pdf(x, 5, 1), color="#a6bddb", lw=1, label="a=5, b=1")
    axs.plot(x, kumaraswamy_pdf(x, 5, 1.5), color="#d0d1e6", lw=1, label="a=5, b=1.5")
    axs.set_xlim((0, 1))
    axs.set_ylim(
        0,
    )
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel(r"$\omega_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.57, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "kumaraswamy_pdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "kumaraswamy_pdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, power_cdf(x, 0.3), color="#034e7b", lw=1, label="k=0.3")
    axs.plot(x, power_cdf(x, 0.5), color="#0570b0", lw=1, label="k=0.5")
    axs.plot(x, power_cdf(x, 0.7), color="#3690c0", lw=1, label="k=0.7")
    axs.plot(x, power_cdf(x, 1.5), color="#74a9cf", lw=1, label="k=1.5")
    axs.plot(x, power_cdf(x, 2), color="#a6bddb", lw=1, label="k=2")
    axs.plot(x, power_cdf(x, 3), color="#d0d1e6", lw=1, label="k=3")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 1))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel("$P_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.43, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "power_cdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "power_cdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)

    fig, axs = plt.subplots(1, 1, figsize=(3, 2))
    x = onp.linspace(0, 1, num=1000)
    axs.plot(x, power_pdf(x, 0.3), color="#034e7b", lw=1, label="k=0.3")
    axs.plot(x, power_pdf(x, 0.5), color="#0570b0", lw=1, label="k=0.5")
    axs.plot(x, power_pdf(x, 0.7), color="#3690c0", lw=1, label="k=0.7")
    axs.plot(x, power_pdf(x, 1.5), color="#74a9cf", lw=1, label="k=1.5")
    axs.plot(x, power_pdf(x, 2), color="#a6bddb", lw=1, label="k=2")
    axs.plot(x, power_pdf(x, 3), color="#d0d1e6", lw=1, label="k=3")
    axs.set_xlim((0, 1))
    axs.set_ylim((0, 30))
    axs.set_xlabel("$P_S$ [-]")
    axs.set_ylabel(r"$\omega_Q$ [-]")
    axs.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.43, 1.05))
    fig.subplots_adjust(left=0.15, bottom=0.2, right=0.7)
    file = base_path_figs / "power_pdf.png"
    fig.savefig(file, dpi=300)
    file = base_path_figs / "power_pdf.pdf"
    fig.savefig(file, dpi=300)
    plt.close(fig=fig)
    plt.close("all")
    return


if __name__ == "__main__":
    main()
