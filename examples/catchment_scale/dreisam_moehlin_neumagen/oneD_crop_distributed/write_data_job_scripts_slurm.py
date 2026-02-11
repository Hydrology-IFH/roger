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

    # identifiers of the simulations
    stress_tests_meteo = ["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    stress_test_meteo_magnitudes = [0, 2]
    stress_test_meteo_durations = [0, 3]
    scenario_flags = []
    script_names = []
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo))

            script_names.append('write_data_%s_soil-compaction' % (stress_test_meteo))
            script_names.append('write_data_%s' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_irrigation' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_yellow-mustard' % (stress_test_meteo))
            script_names.append('write_data_%s_irrigation_yellow-mustard' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo))

        elif stress_test_meteo == "base_2000-2024":
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo))
            scenario_flags.append('--stress-test-meteo %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo))

            script_names.append('write_data_%s_soil-compaction' % (stress_test_meteo))
            script_names.append('write_data_%s' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_irrigation' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_yellow-mustard' % (stress_test_meteo))
            script_names.append('write_data_%s_irrigation_yellow-mustard' % (stress_test_meteo))
            script_names.append('write_data_%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo))

        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    if magnitude == 0 and duration == 0:
                        pass
                    else:
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction\n' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s\n' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --irrigation irrigation\n' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --yellow-mustard yellow-mustard\n' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo, magnitude, duration))
                        scenario_flags.append('--stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s --soil-compaction soil-compaction --irrigation irrigation --yellow-mustard yellow-mustard\n' % (stress_test_meteo, magnitude, duration))

                        script_names.append('write_data_%s_magnitude%s_duration%s_soil-compaction' % (stress_test_meteo, magnitude, duration))
                        script_names.append('write_data_%s_magnitude%s_duration%s' % (stress_test_meteo, magnitude, duration))
                        script_names.append('write_data_%s_magnitude%s_duration%s_soil-compaction_irrigation' % (stress_test_meteo, magnitude, duration))
                        script_names.append('write_data_%s_magnitude%s_duration%s_soil-compaction_yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        script_names.append('write_data_%s_magnitude%s_duration%s_irrigation_yellow-mustard' % (stress_test_meteo, magnitude, duration))
                        script_names.append('write_data_%s_magnitude%s_duration%s_soil-compaction_irrigation_yellow-mustard' % (stress_test_meteo, magnitude, duration))

    jobs = []
    for scenario_flag, script_name in zip(scenario_flags, script_names):
        lines = []
        lines.append("#!/bin/bash\n")
        lines.append("#SBATCH --time=02:00:00\n")
        lines.append("#SBATCH --nodes=1\n")
        lines.append("#SBATCH --ntasks=1\n")
        lines.append("#SBATCH --cpus-per-task=1\n")
        lines.append("#SBATCH --mem=64000\n")
        lines.append("#SBATCH --mail-type=FAIL\n")
        lines.append("#SBATCH --mail-user=robin.schwemmle@hydrology.uni-freiburg.de\n")
        lines.append(f"#SBATCH --job-name={script_name}\n")
        lines.append(f"#SBATCH --output={script_name}.out\n")
        lines.append(f"#SBATCH --error={script_name}_err.out\n")
        lines.append("#SBATCH --export=ALL\n")
        lines.append("\n")
        lines.append('module load devel/miniforge\n')
        lines.append('eval "$(conda shell.bash hook)"\n')
        lines.append("conda activate roger\n")
        lines.append(f"cd {base_path_bwhpc}\n")
        lines.append("\n")
        lines.append('python write_roger_data_for_modflow.py %s\n' % (scenario_flag))
        file_path = base_path / f"{script_name}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)
        jobs.append(f"{script_name}.sh")


    file_path = base_path / "submit_data_jobs.sh"
    with open(file_path, "w") as job_file:
        job_file.write("#!/bin/bash\n")
        job_file.write("\n")
        for job in jobs:
            job_file.write(f"sbatch -p compute {job}\n")
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
