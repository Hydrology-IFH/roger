import re
import json
import click
import numpy as np
import time
from pathlib import Path

TESTDIR = Path(__file__).parent.parent / "benchmarks"
TIME_PATTERN = r"Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s"


@click.argument("INFILES", nargs=-1, type=click.Path(dir_okay=False, exists=True))
@click.command()
def get_timings(infiles):
    for infile in infiles:
        with open(TESTDIR / infile) as f:
            data = json.load(f)

        for benchmark, bench_res in data["benchmarks"].items():
            for res in bench_res:
                with open(TESTDIR / res["timing_file"]) as file:
                    lines = file.read().splitlines()
                output = "\n".join(lines)

                iteration_times = list(map(float, re.findall(TIME_PATTERN, output)))[data["settings"]["burnin"] :]
                if not iteration_times:
                    raise RuntimeError("could not extract iteration times from output")

                total_elapsed = sum(iteration_times)
                data["benchmarks"][benchmark][res].append(
                       {"wall_time": total_elapsed,
                        "per_iteration": {
                            "best": float(np.min(iteration_times)),
                            "worst": float(np.max(iteration_times)),
                            "mean": float(np.mean(iteration_times)),
                            "stdev": float(np.std(iteration_times)),
                        },
                        }
                )

        with open("benchmark_{}.json".format(time.time()), "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)


if __name__ == "__main__":
    get_timings()
