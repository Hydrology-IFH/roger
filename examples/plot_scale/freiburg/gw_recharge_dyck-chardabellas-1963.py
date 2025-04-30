from pathlib import Path
import numpy as np
import pandas as pd
import click


def load_precpitation_data(path):
    df_precip_10mins = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=0,
        header=0,
        na_values=-9999,
    )
    df_precip_10mins.index = pd.to_datetime(dict(year=df_precip_10mins.YYYY, month=df_precip_10mins.MM, day=df_precip_10mins.DD, hour=df_precip_10mins.hh, minute=df_precip_10mins.mm))
    df_precip_10mins = df_precip_10mins.loc[:, ["PREC"]]
    df_precip_10mins.index = df_precip_10mins.index.rename("Index")
    df_precip_10mins.columns = ["PRECIP"]

    return df_precip_10mins

def calculate_recharge(precip, line):
    if line == 1:
        # 1 sand, no or little vegetation vover
        recharge = 1.1 * precip - 306

    elif line == 2:
        # 2 loamy sand, no vegetation cover
        recharge = 1.1 * precip - 405

    elif line == 3:
        # 3 sand, vegetation cover
        recharge = 1.1 * precip - 433

    elif line == 4:
        # 4 loamy sand, vegetation cover
        recharge = 1.1 * precip - 474

    elif line == 5:
        # 5, sandy loam
        recharge = 1.1 * precip - 519

    elif line == 6:
        # 6 loam
        recharge = 1.1 * precip - 597    

    return recharge


def calculate_recharge_6lines_weighted(precip, weights):
    if not np.sum(weights) == 1:
        raise ValueError("Weights must sum to 1!")
    
    if not len(weights) == 6:
        raise ValueError("6 weighting factors are required!")

    # 1 sand, no or little vegetation vover
    recharge1 = 1.1 * precip - 306

    # 2 loamy sand, no vegetation cover
    recharge2 = 1.1 * precip - 405

    # 3 sand, vegetation cover
    recharge3 = 1.1 * precip - 433

    # 4 loamy sand, vegetation cover
    recharge4 = 1.1 * precip - 474

    # 5, sandy loam
    recharge5 = 1.1 * precip - 519

    # 6 loam
    recharge6 = 1.1 * precip - 597

    recharge = np.sum(recharge1 * weights[0] + recharge2 * weights[1] + recharge3 * weights[2] + recharge4 * weights[3] + recharge5 * weights[4] + recharge6 * weights[5], axis=0)

    return recharge

def calculate_recharge_6lines(precip):
    # 1 sand, no or little vegetation vover
    recharge1 = 1.1 * precip - 306

    # 2 loamy sand, no vegetation cover
    recharge2 = 1.1 * precip - 405

    # 3 sand, vegetation cover
    recharge3 = 1.1 * precip - 433

    # 4 loamy sand, vegetation cover
    recharge4 = 1.1 * precip - 474

    # 5, sandy loam
    recharge5 = 1.1 * precip - 519

    # 6 loam
    recharge6 = 1.1 * precip - 597

    recharge = np.array([recharge1, recharge2, recharge3, recharge4, recharge5, recharge6]).T

    return recharge


@click.command("main")
def main():
    base_path = Path(__file__).parent
    precip_path = base_path / "input" / "PREC.txt"
    df_precip_10mins = load_precpitation_data(precip_path)
    # Resample the data to annual frequency
    df_precip_annual = df_precip_10mins.resample("YE").sum()
    precip = df_precip_annual["PRECIP"].values

    # calculate recharge according to DYCK & CHARDABELLAS (1963)
    recharge = calculate_recharge_6lines(precip)
    recharge_avg = np.mean(recharge, axis=0)

    df_recharge = pd.DataFrame(recharge, columns=["PERC1", "PERC2", "PERC3", "PERC4", "PERC5", "PERC6"], index=df_precip_annual.index)
    df_recharge.columns = [["[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]"],
                           ["PERC1", "PERC2", "PERC3", "PERC4", "PERC5", "PERC6"]]
    df_recharge.index = df_recharge.index.rename("")

    df_recharge_avg = pd.DataFrame(columns=["PERC1", "PERC2", "PERC3", "PERC4", "PERC5", "PERC6"], index=["avg"])
    df_recharge_avg.loc["avg", :] = recharge_avg
    df_recharge_avg.columns = [["[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]", "[mm/year]"],
                               ["PERC1", "PERC2", "PERC3", "PERC4", "PERC5", "PERC6"]]
    df_recharge_avg.index = df_recharge_avg.index.rename("")

    # write to csv
    output_path = base_path / "output" / f"gw_recharge_dyck-chardabellas.csv"
    df_recharge.to_csv(output_path, sep=";", index=True)
    output_path_avg = base_path / "output" / f"gw_recharge_avg_dyck-chardabellas.csv"
    df_recharge_avg.to_csv(output_path_avg, sep=";", index=True)
    return


if __name__ == "__main__":
    main()
