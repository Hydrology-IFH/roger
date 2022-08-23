from pathlib import Path
import subprocess

base_path = Path(__file__).parent
base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_transport_sensitivity'
base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
transport_models_abrev = {'complete-mixing': 'cm',
                          'piston': 'pi',
                          'preferential': 'pf',
                          'complete-mixing + advection-dispersion': 'ad',
                          'time-variant preferential': 'pft',
                          'time-variant complete-mixing + advection-dispersion': 'adt'}

tracer = 'bromide'
lysimeters = ['lys2_bromide', 'lys8_bromide', 'lys9_bromide']
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'complete-mixing + advection-dispersion',
                    'time-variant preferential',
                    'time-variant complete-mixing + advection-dispersion']
for lys in lysimeters:
    for tm in transport_models:
        tm1 = transport_models_abrev[tm]
        tms = tm.replace(" ", "_")
        script_name = f'{tracer}_{lys}_{tm1}_mc'
        output_path_ws = base_path_ws / 'reckenholz' / 'svat_transport_sensitivity'
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=2:ppn=16\n')
        lines.append('#PBS -l walltime=30:00:00\n')
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
        lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport_%s.py -b numpy -d cpu -n 32 1 -lys %s -tms %s -td "${TMPDIR}"\n' % (tracer, lys, tms))
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

tracer = 'nitrate2'
lysimeters = ['lys2', 'lys3', 'lys4', 'lys8', 'lys9']
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'complete-mixing + advection-dispersion',
                    'time-variant preferential',
                    'time-variant complete-mixing + advection-dispersion']
for lys in lysimeters:
    for tm in transport_models:
        tm1 = transport_models_abrev[tm]
        tms = tm.replace(" ", "_")
        script_name = f'{tracer}_{lys}_{tm1}_mc'
        output_path_ws = base_path_ws / 'reckenholz' / 'svat_transport_sensitivity'
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=2:ppn=16\n')
        lines.append('#PBS -l walltime=30:00:00\n')
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
        lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport_%s.py -b numpy -d cpu -n 32 1 -lys %s -tms %s -td "${TMPDIR}"\n' % (tracer, lys, tms))
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

tracer = 'nitrate1'
lysimeters = ['lys2', 'lys3', 'lys4', 'lys8', 'lys9']
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'complete-mixing + advection-dispersion',
                    'time-variant preferential',
                    'time-variant complete-mixing + advection-dispersion']
for lys in lysimeters:
    for tm in transport_models:
        tm1 = transport_models_abrev[tm]
        tms = tm.replace(" ", "_")
        script_name = f'{tracer}_{lys}_{tm1}_mc'
        output_path_ws = base_path_ws / 'reckenholz' / 'svat_transport_sensitivity'
        lines = []
        lines.append('#!/bin/bash\n')
        lines.append('#PBS -l nodes=2:ppn=16\n')
        lines.append('#PBS -l walltime=30:00:00\n')
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
        lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_transport_%s.py -b numpy -d cpu -n 32 1 -lys %s -tms %s -td "${TMPDIR}"\n' % (tracer, lys, tms))
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
