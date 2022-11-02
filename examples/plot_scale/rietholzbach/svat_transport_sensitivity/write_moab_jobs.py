from pathlib import Path
import subprocess
import click
import yaml
import numpy as onp


@click.option("-ns", "--nsamples", type=int, default=1024)
@click.option("--job-type", type=click.Choice(['single-node', 'multi-node', 'gpu']), default='gpu')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.option("--split-size", type=int, default=1000)
@click.command("main")
def main(nsamples, job_type, sas_solver, split_size):
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
    transport_models_abrev = {'complete-mixing': 'cm',
                              'piston': 'pi',
                              'advection-dispersion': 'ad',
                              'time-variant advection-dispersion': 'adt',
                              }

    file_path = base_path / "param_bounds.yml"
    with open(file_path, 'r') as file:
        bounds = yaml.safe_load(file)

    tracer = 'oxygen18'
    transport_models = list(bounds.keys())
    for tm in transport_models:
        nruns = nsamples * (bounds[tm]['num_vars'] + 2)
        if job_type == 'single-node':
            x1 = 0
            x2 = nruns
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_sa'
            data_dir = base_path_ws / 'rietholzbach' / 'svat_transport_sensitivity'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=16\n')
            lines.append('#PBS -l walltime=96:00:00\n')
            lines.append('#PBS -l pmem=8000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-mpi\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes --x1 %s --x2 %s -b jax -d cpu -n 16 1 -tms %s -td "${TMPDIR}" -ss %s\n' % (x1, x2, tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {data_dir.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (data_dir.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (data_dir.as_posix()))
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

        elif job_type == 'multi-node':
            x1 = 0
            x2 = nruns
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_sa'
            data_dir = base_path_ws / 'rietholzbach' / 'svat_transport_sensitivity'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=2:ppn=16\n')
            lines.append('#PBS -l walltime=120:00:00\n')
            lines.append('#PBS -l pmem=4000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-mpi\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes --x1 %s --x2 %s -b jax -d cpu -n 32 1 -ns 1024 -tms %s -td %s -ss %s\n' % (x1, x2, tms, data_dir.as_posix(), sas_solver))
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

        elif job_type == 'gpu':
            x1x2 = onp.arange(0, nruns, split_size).tolist()
            if nruns not in x1x2:
                x1x2.append(nruns)

            for i, x in enumerate(x1x2[:-1]):
                x1 = x1x2[i]
                x2 = x1x2[i+1]
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'{tracer}_{sas_solver}_svat_{tm1}_sa_{x1}_{x2}'
                data_dir = base_path_ws / 'rietholzbach' / 'svat_transport_sensitivity'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1:gpus=1:default\n')
                lines.append('#PBS -l walltime=48:00:00\n')
                lines.append('#PBS -l pmem=24000mb\n')
                lines.append(f'#PBS -N {script_name}\n')
                lines.append('#PBS -m bea\n')
                lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
                lines.append(' \n')
                lines.append('# load module dependencies\n')
                lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
                lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
                lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append('conda activate roger-gpu\n')
                lines.append(f'cd {base_path_binac}\n')
                lines.append(' \n')
                lines.append('python svat_transport.py --log-all-processes --x1 %s --x2 %s -b jax -d gpu -tms %s -td "${TMPDIR}" -ss %s\n' % (x1, x2, tms, sas_solver))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {data_dir.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (data_dir.as_posix()))
                lines.append('mv "${TMPDIR}"/*.nc %s\n' % (data_dir.as_posix()))
                file_path = base_path / f'{script_name}_gpu.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {script_name}_gpu.sh", shell=True)

    return


if __name__ == "__main__":
    main()
