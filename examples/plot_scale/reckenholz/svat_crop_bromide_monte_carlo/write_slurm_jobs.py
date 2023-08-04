from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    lysimeters = ['lys2_bromide', 'lys8_bromide', 'lys9_bromide']
    transport_models = ['advection-dispersion-power', 'time-variant_advection-dispersion-power']
    for lys in lysimeters:
        for tm in transport_models:
            script_name = f'SVATCROPBR_{tm}_{lys}_mc'
            lines = []
            lines.append('#!/bin/bash\n')
            lines.append('#\n')
            lines.append('#SBATCH --partition=single\n')
            lines.append(f'#SBATCH --job-name={script_name}\n')
            lines.append('#SBATCH --nodes=1\n')
            lines.append('#SBATCH --ntasks=20\n')
            lines.append('#SBATCH --mem=64000mb\n')
            lines.append('#SBATCH --mail-type=ALL\n')
            lines.append('#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n')
            lines.append('#SBATCH --export=ALL\n')
            lines.append('#SBATCH --time=72:00:00\n')
            lines.append(' \n')
            lines.append('# load module dependencies\n')
            lines.append('module load lib/hdf5/1.12.1-gnu-11.2-openmpi-4.1\n')
            lines.append(' \n')
            lines.append('# adapt command to your available scheduler / MPI implementation\n')
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_bromide.py -lys %s -tms %s -ecp False -td "${TMPDIR}"\n' % (lys, tm))
            file_path = base_path / f'{script_name}.sh'
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

    subprocess.Popen("chmod +x submit_slurm_jobs.sh", shell=True)
    return


if __name__ == "__main__":
    main()