#! /usr/bin/env python

import sys
import os
import subprocess
import re
import time
import math
import itertools
import json
from pathlib import Path

import click
import numpy as np

"""
Runs selected Roger benchmarks on bwForCluster BinAC back to back and writes timing results to a JSON file.
"""

TESTDIR = Path(os.path.join(os.path.dirname(__file__), os.path.relpath("benchmarks")))

BACKENDS = ["numpy", "numpy-mpi", "jax", "jax-gpu", "jax-mpi", "jax-gpu-mpi"]

STATIC_SETTINGS = " --size {nx} {ny} --timesteps {timesteps} --float-type {float_type}"

BENCHMARK_COMMANDS = {
    "numpy": "{python} {filename} --backend numpy --device cpu" + STATIC_SETTINGS,
    "numpy-mpi": "{python} {filename} --backend numpy --device cpu -n {decomp}" + STATIC_SETTINGS,
    "jax": "{python} {filename} --backend jax --device cpu" + STATIC_SETTINGS,
    "jax-gpu": "{python} {filename} --backend jax --device gpu" + STATIC_SETTINGS,
    "jax-mpi": "{python} {filename} --backend jax --device cpu -n {decomp}" + STATIC_SETTINGS,
    "jax-gpu-mpi": "{python} {filename} --backend jax --device gpu -n {decomp}" + STATIC_SETTINGS,
}

AVAILABLE_BENCHMARKS = [f for f in os.listdir(TESTDIR.as_posix()) if f.endswith("_benchmark.py")]

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
    if into > 1:
        out = [1] * into
        for fac, i in zip(_factorize(num), itertools.cycle(range(into))):
            out[i] *= fac
            result = tuple(map(int, out))
    else:
        result = tuple(int(num), 1)

    return result


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
    default=25,
    help="Number of processes / threads for parallel execution",
)
@click.option(
    "-pm",
    "--pmem",
    type=int,
    default=4000,
    help="Process memory in mb",
)
@click.option(
    "-o",
    "--outfile",
    type=click.Path(exists=False),
    default="benchmark_{}.json".format(time.time()),
    help="JSON file to write timings to",
)
@click.option("-t", "--timesteps", default=5, type=int, help="Number of days that each benchmark is run for")
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
@click.option("--debug", is_flag=True, help="Prints each command that is executed")
@click.option("--local", is_flag=True, help="Run benchmark on local computer")
@click.option("--float-type", default="float64", help="Data type for floating point arrays in Roger components")
@click.option("--burnin", default=3, type=int, help="Number of iterations to exclude in performance statistic")
def run(**kwargs):
    proc_decom = _decompose_num(kwargs["nproc"], 1)
    nproc = kwargs["nproc"]
    pmem = kwargs["pmem"]

    settings = kwargs.copy()
    settings["decomp"] = f"{proc_decom[0]} {proc_decom[1]}"

    out_data = {}
    all_passed = True
    try:
        for f in kwargs["only"]:
            out_data[f] = []
            click.echo(f"running benchmark {f}")

            for size in kwargs["sizes"]:
                # n = math.ceil((size) ** (1 / 2))
                # nx = _round_to_multiple(n, proc_decom[0])
                # ny = _round_to_multiple(n, proc_decom[1])
                nx = _round_to_multiple(size, proc_decom[0])
                ny = 1
                real_size = nx * ny

                click.echo(f" current size: {real_size}")

                cmd_args = settings.copy()
                cmd_args.update(
                    {
                        "python": "python",
                        "filename": f,
                        "dir": TESTDIR.as_posix(),
                        "nx": nx,
                        "ny": ny,
                    }
                )

                if cmd_args["mpiexec"] is None:
                    cmd_args["mpiexec"] = "mpirun"

                for backend in kwargs["backends"]:
                    cmd_sh = BENCHMARK_COMMANDS[backend].format(**cmd_args)
                    if kwargs["local"]:
                        # write shell script to run job on local computer
                        lines = []
                        lines.append('#!/bin/bash\n')
                        lines.append(' \n')
                        if backend in ['numpy']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax-gpu']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-gpu\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['numpy-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-mpi\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'mpirun -n {nproc} {cmd_sh}\n')
                        elif backend in ['jax-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-mpi\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'mpirun -n {nproc} {cmd_sh}\n')
                        elif backend in ['jax-gpu-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-gpu\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'MPI4JAX_USE_CUDA_MPI=1 mpirun -n {nproc} {cmd_sh}\n')
                    else:
                        # write shell script to submit job to cluster
                        lines = []
                        lines.append('#!/bin/bash\n')
                        lines.append(' \n')
                        if backend in ['numpy-mpi', 'jax-mpi']:
                            lines.append('# load module dependencies\n')
                            lines.append('module purge\n')
                            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                            lines.append('export OMP_NUM_THREADS=1\n')
                            lines.append(' \n')
                        elif backend in ['jax-gpu', 'jax-gpu-mpi']:
                            lines.append('# load module dependencies\n')
                            lines.append('module purge\n')
                            lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                            lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
                            lines.append('export OMP_NUM_THREADS=1\n')
                            lines.append(' \n')
                        if backend in ['numpy']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['jax-gpu']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-gpu\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'{cmd_sh}\n')
                        elif backend in ['numpy-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-mpi\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                        elif backend in ['jax-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-mpi\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                        elif backend in ['jax-gpu-mpi']:
                            lines.append('eval "$(conda shell.bash hook)"\n')
                            lines.append('conda activate roger-gpu\n')
                            lines.append(f'cd {cmd_args["dir"]}\n')
                            lines.append(f'MPI4JAX_USE_CUDA_MPI=1 mpirun --bind-to core --map-by core -report-bindings {cmd_sh}\n')
                    file_path = TESTDIR / 'job.sh'
                    file = open(file_path, "w")
                    file.writelines(lines)
                    file.close()
                    # make shell-script executable
                    subprocess.Popen(f"chmod +x {file_path.as_posix()}", shell=True)
                    os.chdir(TESTDIR)
                    if kwargs["local"]:
                        cmd = "./job.sh"
                    else:
                        # submit job to queue
                        if backend in ['numpy', 'jax']:
                            pmem_serial = int(4000 * (size / 10000))
                            if pmem_serial > 128000:
                                pmem_serial = 128000
                            cmd = f"qsub -q short -N {f.split('.')[0]}_{backend}_{real_size} -l nodes=1:ppn=1,walltime=2:00:00,pmem={pmem_serial}mb job.sh"
                        elif backend in ['numpy-mpi', 'jax-mpi']:
                            if nproc <= 28:
                                nnodes = 1
                                ppn = nproc
                            else:
                                nnodes = int(nproc/28)
                                ppn = 28
                            cmd = f"qsub -q short -N {f.split('.')[0]}_{backend}_{real_size} -l nodes={nnodes}:ppn={ppn},walltime=2:00:00,pmem={pmem}mb job.sh"
                        elif backend in ['jax-gpu']:
                            cmd = f"qsub -q gpu -N {f.split('.')[0]}_{backend}_{real_size} -l nodes=1:ppn=1:gpus=1:default,walltime=2:00:00,pmem=24000mb job.sh"
                        elif backend in ['jax-gpu-mpi']:
                            cmd = f"qsub -q gpu -N {f.split('.')[0]}_{backend}_{real_size} -l nodes=1:ppn=2:gpus=2:default,walltime=2:00:00,pmem=24000mb job.sh"

                    if kwargs["debug"]:
                        click.echo(f"  $ {cmd} {nproc} {proc_decom[0]} {proc_decom[1]}")

                    sys.stdout.write(f"  {backend:<15} ... ")
                    sys.stdout.flush()

                    if kwargs["local"]:
                        try:
                            # read output stream
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
                    else:
                        try:
                            # submit job
                            job_id1 = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
                            job_id1 = job_id1.decode("utf-8")
                            job_id = re.sub(r'[\n]', '', job_id1)
                        except subprocess.CalledProcessError as e:
                            click.echo("failed")
                            click.echo(e.output.decode("utf-8"))
                            all_passed = False
                            continue

                        click.echo(f"submitted {f.split('.')[0]}_{backend}_{real_size}")
                        out_data[f].append(
                            {
                                "backend": backend,
                                "size": real_size,
                                "timing_file": f"{f.split('.')[0]}_{backend}_{real_size}.o{job_id}",
                            }
                        )

    finally:
        if kwargs["local"]:
            with open(kwargs["outfile"], "w") as f:
                json.dump({"benchmarks": out_data, "settings": settings}, f, indent=4, sort_keys=True)
        else:
            kwargs["outfile"] = "timing_files_{}.json".format(time.time())
            with open(kwargs["outfile"], "w") as f:
                json.dump({"benchmarks": out_data, "settings": settings}, f, indent=4, sort_keys=True)

    raise SystemExit(int(not all_passed))


if __name__ == "__main__":
    run()
