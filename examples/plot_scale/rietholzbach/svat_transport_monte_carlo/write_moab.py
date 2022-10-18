from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(['serial', 'single-node', 'multi-node', 'gpu', 'multi-gpu']), default='serial')
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
                              'time-variant power': 'powt'}

    tracer = 'oxygen18'
    transport_models = ['advection-dispersion', 'time-variant advection-dispersion']
    for tm in transport_models:
        if job_type == 'serial':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=1\n')
            lines.append('#PBS -l walltime=120:00:00\n')
            lines.append('#PBS -l pmem=80000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('python svat_transport.py -b jax -d cpu -ns 10000 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

        elif job_type == 'single-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=20\n')
            lines.append('#PBS -l walltime=120:00:00\n')
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
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b jax -d cpu -n 20 1 -ns 5000 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

        elif job_type == 'multi-node':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=5:ppn=20\n')
            lines.append('#PBS -l walltime=120:00:00\n')
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
            lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b jax -d cpu -n 100 1 -ns 10000 -tms {tms} -td {output_path_ws.as_posix()} -ss {sas_solver}\n')
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

        elif job_type == 'gpu':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            for i in range(20):
                script_name = f'{tracer}_{sas_solver}_svat_{tm1}_mc_{i}'
                output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
                tms = tm.replace(" ", "_")
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=1:gpus=1:default\n')
                lines.append('#PBS -l walltime=10:00:00\n')
                lines.append('#PBS -l pmem=4000mb\n')
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
                lines.append('python svat_transport.py --log-all-processes --id %s -b jax -d gpu -ns 500 -tms %s -td "${TMPDIR}" -ss %s\n' % (i, tms, sas_solver))
                lines.append('# Move output from local SSD to global workspace\n')
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / f'{script_name}_gpu.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {script_name}_gpu.sh", shell=True)

        elif job_type == 'multi-gpu':
            tm1 = transport_models_abrev[tm]
            tms = tm.replace(" ", "_")
            script_name = f'{tracer}_{sas_solver}_svat_{tm1}_mc'
            output_path_ws = base_path_ws / 'rietholzbach' / 'svat_transport_monte_carlo'
            tms = tm.replace(" ", "_")
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#PBS -l nodes=1:ppn=2:gpus=2:default\n')
            lines.append('#PBS -l walltime=24:00:00\n')
            lines.append('#PBS -l pmem=4000mb\n')
            lines.append(f'#PBS -N {script_name}\n')
            lines.append('#PBS -m bea\n')
            lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load mpi/openmpi/4.1-gnu-9.2-cuda-11.4\n')
            lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
            lines.append('module load lib/cudnn/8.2-cuda-11.4\n')
            lines.append('export OMP_NUM_THREADS=1\n')
            lines.append('export MPI4JAX_USE_CUDA_MPI=1\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append('conda activate roger-gpu\n')
            lines.append(f'cd {base_path_binac}\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py --log-all-processes -b jax -d gpu -n 2 1 -ns 400 -tms %s -td "${TMPDIR}" -ss %s\n' % (tms, sas_solver))
            lines.append('# Move output from local SSD to global workspace\n')
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append('mkdir -p %s\n' % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f'{script_name}_multi_gpu.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}_multi_gpu.sh", shell=True)

    return


if __name__ == "__main__":
    main()
