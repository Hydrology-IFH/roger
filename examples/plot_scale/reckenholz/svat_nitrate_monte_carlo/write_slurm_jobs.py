from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_bwuc = "/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_crop_monte_carlo"
    output_path_ws = Path("/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/output/svat_crop_monte_carlo")
    lysimeters = ['lys2', 'lys3', 'lys4', 'lys8', 'lys9']    
    transport_models = ['advection-dispersion-power', 'time-variant_advection-dispersion-power']
    for lys in lysimeters:
        for tm in transport_models:
            script_name = f'SVATN_{tm}_{lys}_mc'
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
            lines.append('mpirun --bind-to core --map-by core -report-bindings python svat_crop_nitrate.py -lys %s -tms %s -ecp False -td "${TMPDIR}"\n' % (lys, tm))
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATN_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f"{script_name}_slurm.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {script_name}.sh", shell=True)

    subprocess.Popen("chmod +x submit_slurm_jobs.sh", shell=True)
    return


if __name__ == "__main__":
    main()
