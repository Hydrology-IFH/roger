from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(['serial', 'single-node', 'multi-node', 'single-node-gpu', 'single-node-multi-gpu', 'multi-node-multi-gpu']), default='serial')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.command("main")
def main(job_type, sas_solver):
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
    transport_models_abrev = {'complete-mixing': 'cm',
                              'piston': 'pi',
                              'preferential': 'pf',
                              'preferential1': 'pf1',
                              'preferential2': 'pf2',
                              'advection-dispersion': 'ad',
                              'advection-dispersion1': 'ad1',
                              'advection-dispersion2': 'ad2',
                              'time-variant preferential': 'pft',
                              'time-variant preferential1': 'pft1',
                              'time-variant preferential2': 'pft2',
                              'time-variant advection-dispersion': 'adt',
                              'time-variant advection-dispersion1': 'adt1',
                              'time-variant advection-dispersion2': 'adt2',
                              'time-variant': 'tv',
                              'time-variant1': 'tv1',
                              'time-variant2': 'tv2',
                              'preferential + advection-dispersion': 'pfad',
                              'time-variant preferential + advection-dispersion': 'pfadt',
                              'power': 'pow',
                              'time-variant power': 'powt',
                              'time-variant power reverse': 'powtr'}

    tracer = 'oxygen18'
    transport_models = ['preferential', 'preferential1', 'preferential2',
                        'advection-dispersion', 'advection-dispersion1', 'advection-dispersion2',
                        'time-variant preferential', 'time-variant preferential1', 'time-variant preferential2',
                        'time-variant advection-dispersion', 'time-variant advection-dispersion1', 'time-variant advection-dispersion2',
                        'time-variant', 'time-variant1', 'time-variant2',
                        'preferential + advection-dispersion', 'time-variant preferential + advection-dispersion',
                        'power', 'time-variant power', 'time-variant power reverse']
    for tm in transport_models:
        if job_type == 'serial':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=4000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('python svat_transport.py -b numpy -d cpu -ns 250 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

        elif job_type == 'single-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=20\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=6000mb\n')
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
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b numpy -d cpu -n 20 1 -ns 2000 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

        elif job_type == 'multi-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=5:ppn=20\n')
            lines.append('#PBS -l walltime=168:00:00\n')
            lines.append('#PBS -l pmem=2000mb\n')
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
            lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b numpy -d cpu -n 100 1 -ns 1000 -tms {tms} -td {output_path_ws.as_posix()} -ss {sas_solver}\n')
            file_path = base_path / f'{script_name}_moab.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

        elif job_type == 'single-node-gpu':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1:gpus=1:default\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=12000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('export XLA_PYTHON_CLIENT_PREALLOCATE=false\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-gpu\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('python svat_transport.py -b jax -d gpu -ns 100 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab_gpu.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab_gpu.sh", shell=True)

        elif job_type == 'single-node-multi-gpu':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1:gpus=2:default\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=12000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('export XLA_PYTHON_CLIENT_PREALLOCATE=false\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-gpu\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('python svat_transport.py -b jax -d gpu -ns 200 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_moab_gpu.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab_gpu.sh", shell=True)

        elif job_type == 'multi-node-multi-gpu':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=10:ppn=1:gpus=1\n')
            lines.append('#PBS -l walltime=48:00:00\n')
            lines.append('#PBS -l pmem=12000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('export XLA_PYTHON_CLIENT_PREALLOCATE=false\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-gpu\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d gpu -n 10 1 -ns 1000 -tms {tms} -td {output_path_ws.as_posix()} -ss {sas_solver}\n')
            file_path = base_path / f'{script_name}_moab_gpu.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_moab_gpu.sh", shell=True)

    # serial job
    transport_models = ['complete-mixing', 'piston']
    for tm in transport_models:
        tm1 = transport_models_abrev[tm]
        tms = tm.replace(" ", "_")
        script_name = f'{tracer}_{tm1}_mc'
        output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
        tms = tm.replace(" ", "_")
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=1:ppn=1\n')
        lines.append('#PBS -l walltime=01:00:00\n')
        lines.append('#PBS -l pmem=4000mb\n')
        lines.append(f'#PBS -N {script_name}\n')
        lines.append('#PBS -m bea\n')
        lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
        lines.append(' \n')
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append('conda activate roger\n')
        lines.append(f'cd {base_path_binac}\n')
        lines.append(' \n')
        lines.append('# adapt command to your available scheduler / MPI implementation\n')
        lines.append('python svat_transport.py -b numpy -d cpu -ns 1 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
        lines.append('# Write output to temporary SSD of computing node\n')
        lines.append('echo "Write output to $TMPDIR"\n')
        lines.append('# Move output from temporary SSD to workspace\n')
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / f'{script_name}_moab.sh'
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)

    return


if __name__ == "__main__":
    main()
