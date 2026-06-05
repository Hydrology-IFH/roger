from pathlib import Path
import yaml
import subprocess
import os
import pandas as pd
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc = str(base_path.parent)
    base_path_ws = base_path.parent

    # load the configuration file
    with open(base_path.parent / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    # load the configuration file
    with open(base_path.parent / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    # load the subregions and crop rotations
    df_subregions_crop_rotations = pd.read_csv(base_path.parent / "subregions_crop_rotations.csv", sep=";")

    # identifiers of the simulations
    locations = df_subregions_crop_rotations.loc[:, "subregion"].values.astype(str).tolist()
    crop_rotation_scenarios = df_subregions_crop_rotations.loc[:, "crop_rotation_type"].values.astype(str).tolist()
    stress_tests_meteo = config["climate_scenarios"]

    jobs = []
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            durations_magnitudes = [(0, 0)]
        elif stress_test_meteo in ["spring-drought", "summer-drought"]:
            durations_magnitudes = [(3, 0), (3, 2)]
        elif stress_test_meteo == "long-term":
            durations_magnitudes = [(0, 2)]
        for duration_magnitude in durations_magnitudes:
            duration = duration_magnitude[0]
            magnitude = duration_magnitude[1]
            for location, crop_rotation_scenario in zip(locations, crop_rotation_scenarios):
                script_name = f"svat_crop_nitrate_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}_irrigation"
                input_path_ws = base_path_ws / "output" / "irrigation"
                output_path_ws = base_path_ws / "output" / dir_name / "irrigation"
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
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append("conda activate roger\n")
                lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
                lines.append(' \n')
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.nc"
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {input_path_ws.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/%s "${TMPDIR}"\n' % (input_path_ws.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("sleep 60\n")
                lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append(
                    'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario %s --crop-rotation-scenario %s -td "${TMPDIR}"\n' % (irrigation_scenario, crop_rotation_scenario)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / f"{script_name}.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)
                jobs.append(f"{script_name}.sh")

                script_name = f"svat_crop_nitrate_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}_irrigation_soil-compaction"
                input_path_ws = base_path_ws / "output" / "irrigation_soil-compaction"
                output_path_ws = base_path_ws / "output" / dir_name / "irrigation_soil-compaction"
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
                lines.append('eval "$(conda shell.bash hook)"\n')
                lines.append("conda activate roger\n")
                lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
                lines.append(' \n')
                lines.append("# Copy fluxes and states from global workspace to local SSD\n")
                lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
                lines.append("# Compares hashes\n")
                file_nc = f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.nc"
                lines.append(
                    f'checksum_gws=$(shasum -a 256 {input_path_ws.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
                )
                lines.append("checksum_ssd=0a\n")
                lines.append('cp %s/%s "${TMPDIR}"\n' % (input_path_ws.as_posix(), file_nc))
                lines.append("# Wait for termination of moving files\n")
                lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
                lines.append("sleep 60\n")
                lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
                lines.append("done\n")
                lines.append('echo "Copying was successful"\n')
                lines.append(" \n")
                lines.append(
                    'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario %s --crop-rotation-scenario %s -td "${TMPDIR}"\n' % (irrigation_scenario, crop_rotation_scenario)
                )
                lines.append("# Move output from local SSD to global workspace\n")
                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
                file_path = base_path / f"{location}_{crop_rotation_scenario}.sh"
                file = open(file_path, "w")
                file.writelines(lines)
                file.close()
                subprocess.Popen(f"chmod +x {file_path}", shell=True)
                jobs.append(f"{script_name}.sh")

            script_name = f"svat_crop_nitrate_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}"
            input_path_ws = base_path_ws / "output" / "no-irrigation"
            output_path_ws = base_path_ws / "output" / dir_name / "no-irrigation"
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
            lines.append('eval "$(conda shell.bash hook)"\n')
            lines.append("conda activate roger\n")
            lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
            lines.append(' \n')
            lines.append("# Copy fluxes and states from global workspace to local SSD\n")
            lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
            lines.append("# Compares hashes\n")
            file_nc = f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.nc"
            lines.append(
                f'checksum_gws=$(shasum -a 256 {input_path_ws.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
            )
            lines.append("checksum_ssd=0a\n")
            lines.append('cp %s/%s "${TMPDIR}"\n' % (input_path_ws.as_posix(), file_nc))
            lines.append("# Wait for termination of moving files\n")
            lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
            lines.append("sleep 60\n")
            lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
            lines.append("done\n")
            lines.append('echo "Copying was successful"\n')
            lines.append(" \n")
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario no-irrigation --crop-rotation-scenario %s -td "${TMPDIR}"\n' % (crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
            file_path = base_path / f"{script_name}.sh"
            file = open(file_path, "w")
            file.writelines(lines)
            file.close()
            subprocess.Popen(f"chmod +x {file_path}", shell=True)
            jobs.append(f"{script_name}.sh")

            script_name = f"svat_crop_nitrate_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}_soil-compaction"
            input_path_ws = base_path_ws / "output" / "no-irrigation_soil-compaction"
            output_path_ws = base_path_ws / "output" / dir_name / "no-irrigation_soil-compaction"
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
            lines.append(' \n')
            lines.append("# Copy fluxes and states from global workspace to local SSD\n")
            lines.append('echo "Copy fluxes and states from global workspace to local SSD"\n')
            lines.append("# Compares hashes\n")
            file_nc = f"SVATCROP_{stress_test_meteo}-duration{duration}-magnitude{magnitude}_{location}_{crop_rotation_scenario}.nc"
            lines.append(
                f'checksum_gws=$(shasum -a 256 {input_path_ws.as_posix()}/{file_nc} | cut -f 1 -d " ")\n'
            )
            lines.append("checksum_ssd=0a\n")
            lines.append('cp %s/%s "${TMPDIR}"\n' % (input_path_ws.as_posix(), file_nc))
            lines.append("# Wait for termination of moving files\n")
            lines.append('while [ "${checksum_gws}" != "${checksum_ssd}" ]; do\n')
            lines.append("sleep 60\n")
            lines.append('checksum_ssd=$(shasum -a 256 "${TMPDIR}"/%s | cut -f 1 -d " ")\n' % (file_nc))
            lines.append("done\n")
            lines.append('echo "Copying was successful"\n')
            lines.append(" \n")
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario no-irrigation --crop-rotation-scenario %s -td "${TMPDIR}"\n' % (crop_rotation_scenario)
            )
            lines.append("# Move output from local SSD to global workspace\n")
            lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
            lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
            lines.append('mv "${TMPDIR}"/SVATCROPNITRATE_*.nc %s\n' % (output_path_ws.as_posix()))
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
