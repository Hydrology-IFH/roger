from pathlib import Path
import subprocess


base_path = Path(__file__).parent
base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/rietholzbach/svat_transport_monte_carlo'
base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
transport_models_abrev = {'complete-mixing': 'cm',
                          'piston': 'pi',
                          'preferential': 'pf',
                          'advection-dispersion': 'ad',
                          'time-variant preferential': 'pft',
                          'time-variant advection-dispersion': 'adt',
                          'time-variant': 'tv'}

tracer = 'oxygen18'
transport_models = ['preferential', 'advection-dispersion',
                    'time-variant preferential',
                    'time-variant advection-dispersion',
                    'time-variant']
for tm in transport_models:
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
    lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport.py -b numpy -d cpu -n 50 1 -tms %s -td "${TMPDIR}"\n' % (tms))
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
    lines.append('#PBS -l pmem=16000mb\n')
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
    lines.append('python svat_transport.py -b numpy -d cpu -tms %s -td "${TMPDIR}"\n' % (tms))
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
