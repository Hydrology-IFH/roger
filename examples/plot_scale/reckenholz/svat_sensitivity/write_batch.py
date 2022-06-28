from pathlib import Path
import os

base_path = Path(__file__).parent

lysimeters = ['lys1', 'lys2', 'lys3', 'lys4', 'lys8', 'lys9', 'lys2_bromide', 'lys8_bromide', 'lys9_bromide']
for lys in lysimeters:
    script_name = f'{lys}_sa'
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#\n')
    lines.append('#SBATCH --partition=single\n')
    lines.append(f'#SBATCH --job-name={script_name}\n')
    lines.append('#SBATCH --nodes=1\n')
    lines.append('#SBATCH --ntasks=32\n')
    lines.append('#SBATCH --mem=180000mb\n')
    lines.append('#SBATCH --mail-type=ALL\n')
    lines.append('#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n')
    lines.append('#SBATCH --export=ALL\n')
    lines.append('#SBATCH --time=24:00:00\n')
    lines.append(' \n')
    lines.append('# load module dependencies\n')
    lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
    lines.append(' \n')
    lines.append('# adapt command to your available scheduler / MPI implementation\n')
    lines.append('conda activate roger-mpi\n')
    lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_crop.py {lys}\n')
    file_path = base_path / f'{script_name}.sh'
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    os.system(f"chmod +x {script_name}.sh")

os.system("chmod +x submit.sh")
