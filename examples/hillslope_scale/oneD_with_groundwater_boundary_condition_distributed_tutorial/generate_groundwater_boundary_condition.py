from pathlib import Path
import numpy as onp
import pandas as pd
import click
import matplotlib.pyplot as plt


def sin_func(t, amp, phase, off):
    return amp * onp.sin(2 * onp.pi * t - phase) + off


@click.option("--head-type", type=click.Choice(["shallow", "deep"]), default="shallow")
@click.option("--time-variant", is_flag=True)
@click.command("main")
def main(head_type, time_variant):
    base_path = Path(__file__).parent

    df_ta = pd.read_csv(
        base_path / "input" / "TA.txt",
        sep=r"\s+",
        skiprows=0,
        header=0,
        parse_dates=[[0, 1, 2, 3, 4]],
        index_col=0,
        na_values=-9999,
        date_format="%Y %m %d %H %M",
    )
    df_ta.index = pd.to_datetime(df_ta.index, format="%Y %m %d %H %M")
    df_ta.index = df_ta.index.rename("Index")

    # write groundwater head to txt
    df_zgw = pd.DataFrame(index=df_ta.index)
    df_zgw["YYYY"] = df_zgw.index.year.values
    df_zgw["MM"] = df_zgw.index.month.values
    df_zgw["DD"] = df_zgw.index.day.values
    df_zgw["hh"] = df_zgw.index.hour.values
    df_zgw["mm"] = df_zgw.index.minute.values

    if time_variant:
        if head_type == "shallow":
            z = 1.25
        elif head_type == "deep":
            z = 9.9
        ndays = len(df_zgw.index)
        rng = onp.random.default_rng(42)
        offset = z + rng.uniform(-0.01, 0.01, ndays)
        amp = 0.2 + rng.uniform(-0.01, 0.01, ndays)
        t = (onp.arange(0, ndays) % 365) / 365
        df_zgw.loc[:, "Z_GW"] = sin_func(t, amp, 3, offset)
    else:
        if head_type == "shallow":
            z = 2.5
        elif head_type == "deep":
            z = 9.9
        df_zgw.loc[:, "Z_GW"] = z

    fig, axes = plt.subplots(1, 1, sharex="row", sharey=True, figsize=(6, 2.5))
    axes.plot(df_zgw.index, df_zgw["Z_GW"], ls="-", color="black", lw=1)
    axes.invert_yaxis()
    axes.set_ylabel(r"$z_{gw}$ [m]")
    axes.set_xlabel("Time [year]")
    axes.set_xlim(df_zgw.index.min(), df_zgw.index.max())
    fig.autofmt_xdate()
    fig.tight_layout()
    file = base_path / "figures" / "zgw.png"
    fig.savefig(file, dpi=300)
    plt.close("all")

    # write parameters to csv
    df_zgw.columns = [
        ["", "", "", "", "", "[m]"],
        ["YYYY", "MM", "DD", "hh", "mm", "Z_GW"],
    ]
    df_zgw.to_csv(base_path / "input" / "ZGW.txt", index=False, sep=";")

    return


if __name__ == "__main__":
    main()
