from pathlib import Path
import subprocess

base_path = Path(__file__).parent
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
        script_name = f'{tracer}_{lys}_{tm1}_sa'
        tms = tm.replace(" ", "_")
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
        lines.append('#SBATCH --time=72:00:00\n')
        lines.append(' \n')
        lines.append('# load module dependencies\n')
        lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
        lines.append(' \n')
        lines.append('# adapt command to your available scheduler / MPI implementation\n')
        lines.append('conda activate roger-mpi\n')
        lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport_{tracer}.py {lys} {tms}\n')
        file_path = base_path / f'{script_name}.sh'
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

subprocess.Popen("chmod +x submit.sh", shell=True)

tracer = 'nitrate'
lysimeters = ['lys2', 'lys3', 'lys4', 'lys8', 'lys9']
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'complete-mixing + advection-dispersion',
                    'time-variant preferential',
                    'time-variant complete-mixing + advection-dispersion']
for lys in lysimeters:
    for tm in transport_models:
        tm1 = transport_models_abrev[tm]
        tms = tm.replace(" ", "_")
        script_name = f'{tracer}_{lys}_{tm1}_sa'
        tms = tm.replace(" ", "_")
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
        lines.append('#SBATCH --time=72:00:00\n')
        lines.append(' \n')
        lines.append('# load module dependencies\n')
        lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
        lines.append(' \n')
        lines.append('# adapt command to your available scheduler / MPI implementation\n')
        lines.append('conda activate roger-mpi\n')
        lines.append(f'mpirun --bind-to core --map-by core -report-bindings python svat_transport_{tracer}.py {lys} {tms}\n')
        file_path = base_path / f'{script_name}.sh'
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

subprocess.Popen("chmod +x submit.sh", shell=True)
