#! /usr/bin/env python

import sys
import os
import subprocess
import multiprocessing
import re
import time
import math
import itertools
import json

import click
import numpy as np

"""
Runs selected Roger benchmarks back to back and writes timing results to a JSON file.
"""

TESTDIR = os.path.join(os.path.dirname(__file__), os.path.relpath("benchmarks"))

BACKENDS = ["numpy", "numpy-mpi", "jax", "jax-gpu", "jax-mpi", "jax-gpu-mpi"]

STATIC_SETTINGS = " --size {nx} {ny} {nz} --timesteps {timesteps} --float-type {float_type}"

BENCHMARK_COMMANDS = {
    "numpy": "{python} {filename}" + STATIC_SETTINGS,
    "numpy-mpi": "{python} {filename} --n {decomp}" + STATIC_SETTINGS,
    "jax": "{python} {filename}" + STATIC_SETTINGS,
    "jax-gpu": "{python} {filename} --device gpu" + STATIC_SETTINGS,
    "jax-mpi": "{python} {filename} -n {decomp}" + STATIC_SETTINGS,
    "jax-gpu-mpi": "{python} {filename} -n {decomp} --device gpu" + STATIC_SETTINGS,
}
SLURM_COMMANDS = {
    "numpy": "{mpiexec} --nodes 1 --ntasks 1 --mem=180000 --partition single --time=02:00:00 --job-name={backend} {python} {filename}" + STATIC_SETTINGS,
    "numpy-mpi": "{mpiexec} --nodes 5 --ntasks-per-node {nproc} --mem=90000 --partition multiple --time=02:00:00 --job-name={backend} {python} {filename} --nproc {decomp}"
    + STATIC_SETTINGS,
    "jax": "{mpiexec} --nodes 1 --ntasks 1 --mem=180000 --partition single --time=02:00:00 --job-name={backend} {python} {filename}" + STATIC_SETTINGS,
    "jax-gpu": "{mpiexec} --gres=gpu:1 --ntasks-per-node 1 --mem=752000 --partition gpu_8 --time=02:00:00 --job-name={backend} {python} {filename} --device gpu"
    + STATIC_SETTINGS,
    "jax-mpi": "{mpiexec} --nodes 5 --ntasks-per-node {nproc} --mem=90000 --partition multiple --time=02:00:00 --job-name={backend} {python} {filename} --nproc {decomp}"
    + STATIC_SETTINGS,
    "jax-gpu-mpi": "{mpiexec} --gres=gpu:5 --ntasks-per-node {nproc} --mem=752000 --partition gpu_8 --time=02:00:00 --job-name={backend} {python} {filename} --device gpu --nproc {decomp}"
    + STATIC_SETTINGS,
}

AVAILABLE_BENCHMARKS = [f for f in os.listdir(TESTDIR) if f.endswith("_benchmark.py")]

TIME_PATTERN = r"Time step took ([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)s"


def _factorize(num):
    j = 2
    while num > 1:
        for i in range(j, int(math.sqrt(num + 0.05)) + 1):
            if num % i == 0:
                num /= i
                j = i
                yield i
                break
        else:
            if num > 1:
                yield num
                break


def _decompose_num(num, into=2):
    out = [1] * into
    for fac, i in zip(_factorize(num), itertools.cycle(range(into))):
        out[i] *= fac

    return tuple(map(int, out))


def _round_to_multiple(num, divisor):
    return int(round(num / divisor) * divisor)


@click.command("roger-benchmarks", help="Run Roger benchmarks")
@click.option(
    "-s", "--sizes", multiple=True, type=float, required=True, help="Problem sizes to test (total number of elements)"
)
@click.option(
    "--backends",
    multiple=True,
    type=click.Choice(BACKENDS),
    default=["numpy"],
    metavar="BACKEND",
    help="Numerical backend components to benchmark (possible values: {})".format(", ".join(BACKENDS)),
)
@click.option(
    "-n",
    "--nproc",
    type=int,
    default=multiprocessing.cpu_count(),
    help="Number of processes / threads for parallel execution",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default="benchmark_{}.json".format(time.time()),
    help="JSON file to write timings to",
)
@click.option("-t", "--timesteps", default=30, type=int, help="Number of time steps that each benchmark is run for")
@click.option(
    "--only",
    multiple=True,
    default=AVAILABLE_BENCHMARKS,
    help="Run only these benchmarks (possible values: {})".format(", ".join(AVAILABLE_BENCHMARKS)),
    type=click.Choice(AVAILABLE_BENCHMARKS),
    required=False,
    metavar="BENCHMARK",
)
@click.option("--mpiexec", default="mpirun", help="Executable used for calling MPI (e.g. mpirun, mpiexec)")
@click.option("--slurm", is_flag=False, help="Run benchmarks using SLURM scheduling command (srun)")
@click.option("--debug", is_flag=True, help="Additionally print each command that is executed")
@click.option("--float-type", default="float64", help="Data type for floating point arrays in Roger components")
@click.option("--burnin", default=3, type=int, help="Number of iterations to exclude in timings")
def run(**kwargs):
    proc_decom = _decompose_num(kwargs["nproc"], 2)
    nproc = kwargs["nproc"]

    settings = kwargs.copy()
    settings["decomp"] = f"{proc_decom[0]} {proc_decom[1]}"

    out_data = {}
    all_passed = True
    try:
        for f in kwargs["only"]:
            out_data[f] = []
            click.echo(f"running benchmark {f}")

            for size in kwargs["sizes"]:
                nz = 1
                n = math.ceil((size / nz) ** (1 / 2))
                nx = _round_to_multiple(n, proc_decom[0])
                ny = _round_to_multiple(n, proc_decom[1])
                real_size = nx * ny * nz

                click.echo(f" current size: {real_size}")

                cmd_args = settings.copy()
                cmd_args.update(
                    {
                        "python": sys.executable,
                        "filename": os.path.realpath(os.path.join(TESTDIR, f)),
                        "nx": nx,
                        "ny": ny,
                        "nz": nz,
                    }
                )

                if cmd_args["mpiexec"] is None:
                    if kwargs["slurm"]:
                        cmd_args["mpiexec"] = "srun"
                    else:
                        cmd_args["mpiexec"] = "mpirun"

                for backend in kwargs["backends"]:
                    if kwargs["slurm"]:
                        cmd = SLURM_COMMANDS[backend]
                    else:
                        cmd_sh = BENCHMARK_COMMANDS[backend].format(**cmd_args)
                        # write shell shell script to submit job
                        lines = []
                        lines.append('#!/bin/bash\n')
                        lines.append('#\n')
                        lines.append('# load module dependencies\n')
                        lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
                        lines.append(' \n')
                        lines.append('# adapt command to your available scheduler / MPI implementation\n')
                        if backend in ['numpy']:
                            lines.append('conda activate roger\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax']:
                            lines.append('conda activate roger-jax\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax-gpu']:
                            lines.append('conda activate roger-jax\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['numpy-mpi']:
                            lines.append('conda activate roger-mpi\n')
                            lines.append(f'mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                        elif backend in ['jax-mpi']:
                            lines.append('conda activate roger-jax-mpi\n')
                            lines.append(f'mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                        elif backend in ['jax-gpu-mpi']:
                            lines.append('conda activate roger-jax-gpu-mpi\n')
                            lines.append(f'mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                        file_path = TESTDIR / 'job.sh'
                        file = open(file_path, "w")
                        file.writelines(lines)
                        file.close()
                        # make shell-script executable
                        subprocess.run(f"chmod +x {file_path}")
                        # submit job to queue
                        if backend in ['numpy', 'jax']:
                            cmd = f"sbatch -p single -N 1 -n 1 --mem=180000 --time=02:00:00 --job-name={backend} ./{file_path}"
                        elif backend in ['numpy-mpi', 'jax-mpi']:
                            cmd = f"sbatch -p multiple -N 5 -n {nproc} --mem=90000 --time=02:00:00 --job-name={backend} ./{file_path}"
                        elif backend in ['jax-gpu']:
                            cmd = f"sbatch -p gpu_8 --gres=gpu:1 --ntasks-per-node 1 --mem=752000 --time=02:00:00 --job-name={backend} ./{file_path}"
                        elif backend in ['jax-gpu-mpi']:
                            cmd = f"sbatch -p gpu_8 --gres=gpu:5 --ntasks-per-node {nproc} --mem=752000 --time=02:00:00 --job-name={backend} ./{file_path}"

                    if kwargs["debug"]:
                        click.echo(f"  $ {cmd}")

                    sys.stdout.write(f"  {backend:<15} ... ")
                    sys.stdout.flush()

                    try:
                        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                    except subprocess.CalledProcessError as e:
                        click.echo("failed")
                        click.echo(e.output.decode("utf-8"))
                        all_passed = False
                        continue

                    output = output.decode("utf-8")
                    iteration_times = list(map(float, re.findall(TIME_PATTERN, output)))[kwargs["burnin"] :]
                    if not iteration_times:
                        raise RuntimeError("could not extract iteration times from output")

                    total_elapsed = sum(iteration_times)
                    click.echo(f"{total_elapsed:>6.2f}s")

                    out_data[f].append(
                        {
                            "backend": backend,
                            "size": real_size,
                            "wall_time": total_elapsed,
                            "per_iteration": {
                                "best": float(np.min(iteration_times)),
                                "worst": float(np.max(iteration_times)),
                                "mean": float(np.mean(iteration_times)),
                                "stdev": float(np.std(iteration_times)),
                            },
                        }
                    )
    finally:
        with open(kwargs["outfile"], "w") as f:
            json.dump({"benchmarks": out_data, "settings": settings}, f, indent=4, sort_keys=True)

    raise SystemExit(int(not all_passed))


if __name__ == "__main__":
    run()
    # python run_benchmarks.py --sizes 1000 --nproc 200
    # python run_benchmarks.py --sizes [200, 1000, 10000] --nproc 200
