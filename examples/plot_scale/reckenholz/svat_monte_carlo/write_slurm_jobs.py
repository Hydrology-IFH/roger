from pathlib import Path
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    base_path_bwuc = "/home/fr/fr_fr/fr_rs1092/roger/examples/plot_scale/reckenholz/svat_monte_carlo"
    output_path_ws = Path("/pfs/work7/workspace/scratch/fr_rs1092-workspace/roger/examples/plot_scale/reckenholz/output/svat_monte_carlo")

    lysimeters = ['lys1', 'lys2', 'lys3', 'lys4', 'lys8', 'lys9', 'lys2_bromide',
                'lys8_bromide', 'lys9_bromide']
    for lys in lysimeters:
        script_name = f'svat_{lys}_mc'
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=4:00:00\n")
        lines.append("#SBATCH --nodes=1\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=8000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append(" \n")
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append("conda activate roger\n")
        lines.append(f"cd {base_path_bwuc}\n")
        lines.append(" \n")
        lines.append(
            'python svat.py -b numpy -d cpu --lys-experiment %s -td "${TMPDIR}"\n'
            % (lys)
        )
        lines.append("# Move output from local SSD to global workspace\n")
        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
        lines.append('mv "${TMPDIR}"/SVAT_*.nc %s\n' % (output_path_ws.as_posix()))
        file_path = base_path / f"{script_name}_slurm.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
    return


if __name__ == "__main__":
    main()