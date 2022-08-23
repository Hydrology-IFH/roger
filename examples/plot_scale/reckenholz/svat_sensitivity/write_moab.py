from pathlib import Path
import subprocess

base_path = Path(__file__).parent
base_path_binac = '/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_sensitivity'
base_path_ws = Path('/beegfs/work/workspace/ws/fr_rs1092-workspace-0')
lysimeters = ['lys1', 'lys2', 'lys3', 'lys4', 'lys8', 'lys9', 'lys2_bromide',
              'lys8_bromide', 'lys9_bromide']
for lys in lysimeters:
    script_name = f'{lys}_mc'
    output_path_ws = base_path_ws / 'reckenholz' / 'svat_sensitivity'
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#PBS -l nodes=1:ppn=1\n')
    lines.append('#PBS -l walltime=40:00:00\n')
    lines.append('#PBS -l pmem=8000mb\n')
    lines.append(f'#PBS -N {script_name}\n')
    lines.append('#PBS -m bea\n')
    lines.append('#PBS -M robin.schwemmle@hydrology.uni-freiburg.de\n')
    lines.append(' \n')
    lines.append('# load module dependencies\n')
    lines.append('module load lib/hdf5/1.12.0-openmpi-4.1-gnu-9.2\n')
    lines.append('export OMP_NUM_THREADS=1\n')
    lines.append('eval "$(conda shell.bash hook)"\n')
    lines.append('conda activate roger\n')
    lines.append(f'cd {base_path_binac}\n')
    lines.append(' \n')
    lines.append('# adapt command to your available scheduler / MPI implementation\n')
    lines.append('python svat_transport.py -b numpy -d cpu -td "${TMPDIR}"\n')
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
