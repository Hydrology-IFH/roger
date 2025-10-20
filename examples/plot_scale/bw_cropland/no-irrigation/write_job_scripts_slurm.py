from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc = "/pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland"
    base_path_ws = Path("/pfs/10/work/fr_rs1092-workspace/roger/examples/plot_scale/bw_cropland")

    # load the configuration file
    with open(base_path.parent / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    # identifiers of the simulations
    locations = config["locations"]
    crop_rotation_scenarios = config["crop_rotation_scenarios"]
    jobs = []
    for location in locations:# --- jobs to calculate fluxes and states for each irrigation scenario -------------------------
        for crop_rotation_scenario in crop_rotation_scenarios:
            script_name = f"svat_crop_{location}_{crop_rotation_scenario}_no-irrigation"
            output_path_ws = base_path_ws / "output" / dir_name
            lines = []
            lines.append("#!/bin/bash\n")
            lines.append("#SBATCH --time=4:00:00\n")
            lines.append("#SBATCH --nodes=1\n")
            lines.append("#SBATCH --ntasks=1\n")
            lines.append("#SBATCH --cpus-per-task=1\n")
            lines.append("#SBATCH --mem=1000\n")
            lines.append("#SBATCH --mail-type=FAIL\n")
            lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
            lines.append(f"#SBATCH --job-name={script_name}\n")
            lines.append(f"#SBATCH --output={script_name}.out\n")
            lines.append(f"#SBATCH --error={script_name}_err.out\n")
            lines.append("#SBATCH --export=ALL\n")
            lines.append(" \n")
            lines.append('module load devel/miniforge\n')
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger\n")
            lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
            lines.append(" \n")
            lines.append(
                'python svat_crop.py -b numpy -d cpu --location %s --crop-rotation-scenario %s -td "${TMPDIR}"\n' % (location, crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROP_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f"{script_name}.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)
            jobs.append(f"{script_name}.sh")

    file_path = base_path / "submit_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
            job_file.write(f"sbatch -p compute {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
