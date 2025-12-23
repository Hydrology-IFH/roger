from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))
    base_path_bwhpc = "/pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed"
    base_path_ws = Path("/pfs/10/work/fr_rs1092-workspace/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed")

    # identifiers of the simulations
    stress_tests_meteo = ["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 1, 2]
    stress_test_meteo_durations = [0, 2, 3]
    irrigation_scenarios = ["no-irrigation", "irrigation"]
    soil_compaction_scenarios = ["no-soil-compaction", "soil-compaction"]
    catch_crop_scenarios = ["no-yellow-mustard", "yellow-mustard"]
    scenario_flags = {
        "no-irrigation": "",
        "irrigation": "--irrigation",
        "no-soil-compaction": "",
        "soil-compaction": "--soil-compaction",
        "no-yellow-mustard": "",
        "yellow-mustard": "--yellow-mustard",
    }

    jobs = []
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            for irrigation_scenario in irrigation_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    for catch_crop_scenario in catch_crop_scenarios:
                        script_name = f"oneD_crop_{stress_test_meteo}_{irrigation_scenario}_{soil_compaction_scenario}_{catch_crop_scenario}"
                        output_path_ws = base_path_ws / "output" / dir_name
                        lines = []
                        lines.append("#!/bin/bash\n")
                        lines.append("#SBATCH --time=7-00:00:00\n")
                        lines.append("#SBATCH --nodes=1\n")
                        lines.append("#SBATCH --ntasks=1\n")
                        lines.append("#SBATCH --cpus-per-task=1\n")
                        lines.append("#SBATCH --mem=32000\n")
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
                        lines.append("mkdir ${TMPDIR}/roger\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/output\n")
                        lines.append("cp -r %s/oneD_crop.py ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc)) 
                        lines.append("cp -r %s/parameters_roger.nc ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append("cp -r %s/config.yml ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append("cp -r %s/input ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append('sleep 90\n')
                        lines.append('python oneD_crop.py -b jax -d cpu --stress-test-meteo %s %s %s %s -td\n' % (stress_test_meteo, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))
                        lines.append("# Move output from local SSD to global workspace\n")
                        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                        lines.append('mv "${TMPDIR}"/ONEDCROP_*.nc %s\n' % (output_path_ws.as_posix()))
                        file_path = base_path / f"{script_name}.sh"
                        file = open(file_path, "w")
                        file.writelines(lines)
                        file.close()
                        subprocess.Popen(f"chmod +x {file_path}", shell=True)
                        jobs.append(f"{script_name}.sh")

        elif stress_test_meteo == "base_2000-2024":
            for irrigation_scenario in irrigation_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    for catch_crop_scenario in catch_crop_scenarios:
                        script_name = f"oneD_crop_{stress_test_meteo}_{irrigation_scenario}_{soil_compaction_scenario}_{catch_crop_scenario}"
                        output_path_ws = base_path_ws / "output" / dir_name
                        lines = []
                        lines.append("#!/bin/bash\n")
                        lines.append("#SBATCH --time=7-00:00:00\n")
                        lines.append("#SBATCH --nodes=1\n")
                        lines.append("#SBATCH --ntasks=1\n")
                        lines.append("#SBATCH --cpus-per-task=1\n")
                        lines.append("#SBATCH --mem=32000\n")
                        lines.append("#SBATCH --mail-type=FAIL\n")
                        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                        lines.append(f"#SBATCH --job-name={script_name}\n")
                        lines.append(f"#SBATCH --output={script_name}.out\n")
                        lines.append(f"#SBATCH --error={script_name}_err.out\n")
                        lines.append("#SBATCH --export=ALL\n")
                        lines.append(" \n")
                        lines.append('module load devel/miniforge\n')
                        lines.append("conda activate roger\n")
                        lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
                        lines.append(" \n")
                        lines.append("mkdir ${TMPDIR}/roger\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen\n")
                        lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/output\n")
                        lines.append("cp -r %s/oneD_crop.py ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc)) 
                        lines.append("cp -r %s/parameters_roger.nc ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append("cp -r %s/config.yml ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append("cp -r %s/input ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                        lines.append('sleep 90\n')
                        lines.append('python oneD_crop.py -b jax -d cpu --stress-test-meteo %s %s %s %s -td\n' % (stress_test_meteo, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))
                        lines.append("# Move output from local SSD to global workspace\n")
                        lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                        lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                        lines.append('mv "${TMPDIR}"/ONEDCROP_*.nc %s\n' % (output_path_ws.as_posix()))
                        file_path = base_path / f"{script_name}.sh"
                        file = open(file_path, "w")
                        file.writelines(lines)
                        file.close()
                        subprocess.Popen(f"chmod +x {file_path}", shell=True)
                        jobs.append(f"{script_name}.sh")


        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    for irrigation_scenario in irrigation_scenarios:
                        for soil_compaction_scenario in soil_compaction_scenarios:
                            for catch_crop_scenario in catch_crop_scenarios:
                                script_name = f"oneD_crop_{stress_test_meteo}_magnitude{magnitude}_duration{duration}_{irrigation_scenario}_{soil_compaction_scenario}_{catch_crop_scenario}"
                                output_path_ws = base_path_ws / "output" / dir_name
                                lines = []
                                lines.append("#!/bin/bash\n")
                                lines.append("#SBATCH --time=7-00:00:00\n")
                                lines.append("#SBATCH --nodes=1\n")
                                lines.append("#SBATCH --ntasks=1\n")
                                lines.append("#SBATCH --cpus-per-task=1\n")
                                lines.append("#SBATCH --mem=32000\n")
                                lines.append("#SBATCH --mail-type=FAIL\n")
                                lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
                                lines.append(f"#SBATCH --job-name={script_name}\n")
                                lines.append(f"#SBATCH --output={script_name}.out\n")
                                lines.append(f"#SBATCH --error={script_name}_err.out\n")
                                lines.append("#SBATCH --export=ALL\n")
                                lines.append(" \n")
                                lines.append('module load devel/miniforge\n')
                                lines.append("conda activate roger\n")
                                lines.append(f"cd {base_path_bwhpc}/{dir_name}\n")
                                lines.append(" \n")
                                lines.append("mkdir ${TMPDIR}/roger\n")
                                lines.append("mkdir ${TMPDIR}/roger/examples\n")
                                lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale\n")
                                lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen\n")
                                lines.append("mkdir ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/output\n")
                                lines.append("cp -r %s/oneD_crop.py ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc)) 
                                lines.append("cp -r %s/parameters_roger.nc ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                                lines.append("cp -r %s/config.yml ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                                lines.append("cp -r %s/input ${TMPDIR}/roger/examples/catchment_scale/dreisam_moehlin_neumagen/oneD_crop_distributed\n" % (base_path_bwhpc))
                                lines.append('sleep 90\n')
                                lines.append('python oneD_crop.py -b jax -d cpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s %s %s %s -td\n' % (stress_test_meteo, magnitude, duration, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))
                                lines.append("# Move output from local SSD to global workspace\n")
                                lines.append(f'echo "Move output to {output_path_ws.as_posix()}"\n')
                                lines.append("mkdir -p %s\n" % (output_path_ws.as_posix()))
                                lines.append('mv "${TMPDIR}"/ONEDCROP_*.nc %s\n' % (output_path_ws.as_posix()))
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
