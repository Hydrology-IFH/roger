import re
import json
import click
import numpy as np
import time
from pathlib import Path

TESTDIR = Path(__file__).parent
TIME_PATTERN = r"Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s"


@click.option("--benchmark-type", type=click.Choice(['var_size', 'var_proc']), default='var_size')
@click.argument("INFILES", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.command()
def get_timings(benchmark_type, infiles):
    for infile in infiles:
        with open(TESTDIR / benchmark_type / infile) as f:
            data = json.load(f)

        for benchmark, bench_res in data["benchmarks"].items():
            for i, res in enumerate(bench_res):
                with open(TESTDIR / benchmark_type / res["timing_file"]) as file:
                    lines = file.read().splitlines()
                output = "\n".join(lines)

                iteration_times = list(map(float, re.findall(TIME_PATTERN, output)))[data["settings"]["burnin"] :]
                if not iteration_times:
                    file_name = res["timing_file"]
                    raise RuntimeError(f"could not extract iteration times from {file_name}")

                total_elapsed = sum(iteration_times)

                data["benchmarks"][benchmark][i]["wall_time"] = total_elapsed
                data["benchmarks"][benchmark][i]["per_iteration"] = {
                    "best": float(np.min(iteration_times)),
                    "worst": float(np.max(iteration_times)),
                    "mean": float(np.mean(iteration_times)),
                    "stdev": float(np.std(iteration_times)),
                }

        with open(TESTDIR / benchmark_type / "benchmark_{}.json".format(time.time()), "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    get_timings()
