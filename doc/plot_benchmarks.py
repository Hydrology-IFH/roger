import json
import click
import numpy as np

import matplotlib as mpl
import seaborn as sns

mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.titlesize'] = 8
mpl.rcParams['axes.labelsize'] = 9
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['legend.title_fontsize'] = 9
mpl.use("agg")

import matplotlib.pyplot as plt  # noqa: E402

sns.set_style("ticks")
sns.plotting_context("paper", font_scale=1, rc={'font.size': 8.0,
                                                'axes.labelsize': 9.0,
                                                'axes.titlesize': 9.0,
                                                'xtick.labelsize': 8.0,
                                                'ytick.labelsize': 8.0,
                                                'legend.fontsize': 8.0,
                                                'legend.title_fontsize': 9.0})

COMPONENT_COLORS = {
    "numpy": "orangered",
    "numpy-mpi": "coral",
    "jax": "#6baed6",
    "jax-mpi": "#2171b5",
    "jax-gpu": "#08306b",
}


@click.argument("INFILES", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.option("--xaxis", type=click.Choice(["nproc", "size"]), required=True)
@click.option("--norm-component", default=None)
@click.command()
def plot_benchmarks(infiles, xaxis, norm_component):
    benchmarks = set()
    components = set()
    sizes = set()
    nprocs = set()

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        meta = data["settings"]
        benchmarks |= set(meta["only"])
        components |= set(meta["backends"])
        sizes |= set(meta["sizes"])
        nprocs.add(meta["nproc"])

    if xaxis == "nproc":
        assert len(sizes) == 1
        xvals = np.array(sorted(nprocs))
    elif xaxis == "size":
        assert len(nprocs) == 1
        xvals = np.array(sorted(sizes))
    else:
        assert False

    if norm_component is not None and norm_component not in components:
        raise ValueError(f"Did not find norm component {norm_component} in data")

    component_data = {benchmark: {comp: np.full(len(xvals), np.nan) for comp in components} for benchmark in benchmarks}

    for infile in infiles:
        with open(infile) as f:
            data = json.load(f)

        for benchmark, bench_res in data["benchmarks"].items():
            for res in bench_res:
                if xaxis == "size":
                    # sizes are approximate, take the closest one
                    x_idx = np.argmin(np.abs(np.array(xvals) - res["size"]))
                else:
                    x_idx = xvals.tolist().index(data["settings"]["nproc"])

                time = float(res["per_iteration"]["mean"])
                component_data[benchmark][res["backend"]][x_idx] = time

    for benchmark in benchmarks:
        fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), dpi=250)

        last_coords = {}
        for component in components:
            if norm_component:
                # compute rel. speedup
                yvals = component_data[benchmark][norm_component] / component_data[benchmark][component]
            else:
                yvals = component_data[benchmark][component]

            plt.plot(xvals, yvals, ".--", color=COMPONENT_COLORS[component], lw=1)

            finite_mask = np.isfinite(yvals)
            if finite_mask.any():
                last_coords[component] = (xvals[finite_mask][-1], yvals[finite_mask][-1])
            else:
                last_coords[component] = (xvals[0], 1)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        title_kwargs = dict(fontdict=dict(weight="bold", size=11), ha="left", x=0.05, y=1.05)
        if xaxis == "nproc":
            plt.xlabel("Number of MPI processes [# CPU cores]")
            mantissa, exponent = f"{list(sizes)[0]:.1e}".split("e")
            exponent = exponent.lstrip("+0")
            plt.title(f"Benchmark '{benchmark}' for size {mantissa} $\\times$ 10$^{{{exponent}}}$", **title_kwargs)

        elif xaxis == "size":
            nproc = list(nprocs)[0]
            plt.xlabel("Grid size [# grid cells]")
            plt.title("Benchmark %s \non %s processes" % (benchmark, nproc), **title_kwargs)

        if norm_component:
            plt.ylabel(f"Relative speedup to {norm_component} [-]")
            plt.text(0.05, 1.0, "Speedup (higher is better)", transform=ax.transAxes, va="top", color="0.4")
        else:
            plt.ylabel("Average time per iteration [s]")
            plt.text(0.05, 1.0, "Wall time (lower is better)", transform=ax.transAxes, va="top", color="0.4")

        plt.xscale("log")
        plt.yscale("log")

        fig.canvas.draw()

        # add annotations, make sure they don"t overlap
        last_text_pos = 0
        for component, (x, y) in sorted(last_coords.items(), key=lambda k: k[1][1]):
            trans = ax.transData
            _, tp = trans.transform((0, y))
            tp = max(tp, last_text_pos + 20)
            _, y = trans.inverted().transform((0, tp))

            if norm_component:
                speedup = np.round(y, 2)
                if component == 'numpy':
                    component_label = component
                else:
                    component_label = component + f' ({speedup}x)'
            else:
                time_last = np.round(y, 2)
                component_label = component + f' ({time_last} s)'

            plt.annotate(
                component_label,
                (x, y),
                xytext=(10, 0),
                textcoords="offset points",
                annotation_clip=False,
                color=COMPONENT_COLORS[component],
                va="center",
                weight="bold",
            )

            last_text_pos = tp

        fig.tight_layout()

        suffix = ""
        if norm_component:
            suffix = f"_speedup_{norm_component}"
        else:
            suffix = "_scaling"

        fig.savefig(f"{benchmark.split('_')[0]}_{xaxis}{suffix}.png")
        fig.savefig(f"{benchmark.split('_')[0]}_{xaxis}{suffix}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    plot_benchmarks()
