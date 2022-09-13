from pathlib import Path
import subprocess
import click


@click.option("--job-type", type=click.Choice(['single-node', 'multi-node']), default='single-node')
@click.option("--sas-solver", type=click.Choice(['RK4', 'Euler', 'deterministic']), default='deterministic')
@click.command("main")
def main(job_type, sas_solver):
    base_path = Path(__file__).parent
    base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_transport_sensitivity'
    base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
    transport_models_abrev = {'complete-mixing': 'cm',
                              'preferential': 'pf',
                              'advection-dispersion': 'ad',
                              'time-variant preferential': 'pft',
                              'time-variant advection-dispersion': 'adt',
                              'time-variant': 'tv',
                              'power': 'pow',
                              'time-variant power': 'powt'}

    tracer = 'nitrate'
    lysimeters = ['lys2', 'lys3', 'lys4', 'lys8', 'lys9']
    transport_models = ['complete-mixing', 'power',
                        'time-variant power']
    for lys in lysimeters:
        for tm in transport_models:
            if job_type == 'single-node':
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'{tracer}_{lys}_svat_crop_{tm1}_sa'
                output_path_ws = base_path_ws / 'reckenholz' / 'svat_transport_sensitivity'
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=1:ppn=20\n')
                lines.append('#PBS -l walltime=48:00:00\n')
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
                lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport_%s.py --log-all-processes -b numpy -d cpu -ns 2000 -n 20 1 -lys %s -tms %s -td "${TMPDIR}" -ss %s\n' % (tracer, lys, tms, sas_solver))
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
            elif job_type == 'multi-node':
                tm1 = transport_models_abrev[tm]
                tms = tm.replace(" ", "_")
                script_name = f'{tracer}_{lys}_svat_crop_{tm1}_sa'
                output_path_ws = base_path_ws / 'reckenholz' / 'svat_transport_sensitivity'
                lines = []
                lines.append('#!/bin/bash\n')
                lines.append('#PBS -l nodes=2:ppn=20\n')
                lines.append('#PBS -l walltime=48:00:00\n')
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
                lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_transport_%s.py --log-all-processes -b numpy -d cpu -ns 4000 -n 40 1 -lys %s -tms %s -td %s -ss %s\n' % (tracer, lys, tms, output_path_ws.as_posix(), sas_solver))
                file_path = base_path / f'{script_name}_moab.sh'
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)
    return


if __name__ == "__main__":
    main()
