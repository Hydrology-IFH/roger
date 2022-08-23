from pathlib import Path
import subprocess

base_path = Path(__file__).parent
lysimeters = ['lys1', 'lys2', 'lys3', 'lys4', 'lys8', 'lys9', 'lys2_bromide',
              'lys8_bromide', 'lys9_bromide']
for lys in lysimeters:
    script_name = f'{lys}_mc'
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#\n')
    lines.append('#SBATCH --partition=single\n')
    lines.append(f'#SBATCH --job-name={script_name}\n')
    lines.append('#SBATCH --nodes=1\n')
    lines.append('#SBATCH --ntasks=40\n')
    lines.append('#SBATCH --mem=180000\n')
    lines.append('#SBATCH --mail-type=ALL\n')
    lines.append('#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n')
    lines.append('#SBATCH --export=ALL\n')
    lines.append('#SBATCH --time=24:00:00\n')
    lines.append(' \n')
    lines.append('# load module dependencies\n')
    lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
    lines.append(' \n')
    lines.append('# adapt command to your available scheduler / MPI implementation\n')
    lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_crop.py -b numpy -d cpu -n 40 1 -lys {lys}\n')
    file_path = base_path / f'{script_name}.sh'
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

subprocess.Popen("chmod +x submit.sh", shell=True)
