from pathlib import Path
import subprocess


base_path = Path(__file__).parent
base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_sensitivity_reverse'
transport_models_abrev = {'complete-mixing': 'cm',
                          'piston': 'pi',
                          'preferential': 'pf',
                          'advection-dispersion': 'ad',
                          'time-variant preferential': 'pft',
                          'time-variant advection-dispersion': 'adt',
                          'time-variant': 'tv'}

tracer = 'oxygen18'
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'advection-dispersion',
                    'time-variant preferential',
                    'time-variant advection-dispersion',
                    'time-variant']
for tm in transport_models:
    tm1 = transport_models_abrev[tm]
    tms = tm.replace(" ", "_")
    script_name = f'{tracer}_{tm1}_sar'
    tms = tm.replace(" ", "_")
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#PBS -l nodes=2:ppn=16\n')
    lines.append('#PBS -l walltime=30:00:00\n')
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
    lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b jax -d cpu -n 32 1 -tms {tms}\n')
    file_path = base_path / f'{script_name}_moab.sh'
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {script_name}_moab.sh", shell=True)
