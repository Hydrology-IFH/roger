from pathlib import Path
import subprocess

base_path = Path(__file__).parent
transport_models_abrev = {'complete-mixing': 'cm',
                          'piston': 'pi',
                          'preferential': 'pf',
                          'advection-dispersion': 'ad',
                          'time-variant preferential': 'pft',
                          'time-variant advection-dispersion': 'adt'}

tracer = 'oxygen18'
transport_models = ['complete-mixing', 'piston',
                    'preferential', 'advection-dispersion',
                    'time-variant preferential',
                    'time-variant advection-dispersion']
for tm in transport_models:
    tm1 = transport_models_abrev[tm]
    tms = tm.replace(" ", "_")
    script_name = f'{tracer}_{tm1}_sa'
    tms = tm.replace(" ", "_")
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('#\n')
    lines.append('#SBATCH --partition=gpu_8\n')
    lines.append(f'#SBATCH --job-name={script_name}\n')
    lines.append('#SBATCH --gres=gpu:1\n')
    lines.append('#SBATCH --ntasks=1\n')
    lines.append('#SBATCH --cpus-per-task=1\n')
    lines.append('#SBATCH --mem=752000\n')
    lines.append('#SBATCH --mail-type=ALL\n')
    lines.append('#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n')
    lines.append('#SBATCH --export=ALL\n')
    lines.append('#SBATCH --time=6:00:00\n')
    lines.append(' \n')
    lines.append('# load module dependencies\n')
    lines.append('module load devel/cudnn/9.2\n')
    lines.append('module load devel/cuda/11.4\n')
    lines.append(' \n')
    lines.append(f'python svat_transport.py -b jax -d gpu -ns 10000 -tms {tms}\n')
    file_path = base_path / f'{script_name}_gpu.sh'
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {script_name}_gpu.sh", shell=True)

subprocess.Popen("chmod +x submit.sh", shell=True)
