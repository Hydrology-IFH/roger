from pathlib import Path
import yaml
import subprocess
import os
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent
    dir_name = os.path.basename(str(Path(__file__).parent))

    # identifiers of the simulations
    stress_tests_meteo = ["base",  "spring-drought"]
    stress_test_meteo_magnitudes = [2]
    stress_test_meteo_durations = [0]
    irrigation_scenarios = ["no-irrigation"]
    soil_compaction_scenarios = ["no-soil-compaction"]
    catch_crop_scenarios = ["no-yellow-mustard"]

    # stress_tests_meteo = ["base", "base_2000-2024", "spring-drought", "summer-drought", "spring-summer-drought", "spring-summer-wet"]
    # stress_test_meteo_magnitudes = [0, 1, 2]
    # stress_test_meteo_durations = [0, 2, 3]
    # irrigation_scenarios = ["no-irrigation", "irrigation"]
    # soil_compaction_scenarios = ["no-soil-compaction", "soil-compaction"]
    # catch_crop_scenarios = ["no-yellow-mustard", "yellow-mustard"]
    scenario_flags = {
        "no-irrigation": "",
        "irrigation": "--irrigation",
        "no-soil-compaction": "",
        "soil-compaction": "--soil-compaction",
        "no-yellow-mustard": "",
        "yellow-mustard": "--yellow-mustard",
    }

    jobs = []
    jobs.append("#!/bin/bash\n")
    jobs.append("\n")
    for stress_test_meteo in stress_tests_meteo:
        if stress_test_meteo == "base":
            for irrigation_scenario in irrigation_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    for catch_crop_scenario in catch_crop_scenarios:
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s %s %s %s&\n' % (stress_test_meteo, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))
        elif stress_test_meteo == "base_2000-2024":
            for irrigation_scenario in irrigation_scenarios:
                for soil_compaction_scenario in soil_compaction_scenarios:
                    for catch_crop_scenario in catch_crop_scenarios:
                        jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s %s %s %s&\n' % (stress_test_meteo, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))


        else:
            for magnitude in stress_test_meteo_magnitudes:
                for duration in stress_test_meteo_durations:
                    for irrigation_scenario in irrigation_scenarios:
                        for soil_compaction_scenario in soil_compaction_scenarios:
                            for catch_crop_scenario in catch_crop_scenarios:
                                jobs.append('python oneD_crop.py -b jax -d gpu --stress-test-meteo %s --stress-test-meteo-magnitude %s --stress-test-meteo-duration %s %s %s %s&\n' % (stress_test_meteo, magnitude, duration, scenario_flags[irrigation_scenario], scenario_flags[soil_compaction_scenario], scenario_flags[catch_crop_scenario]))

    file_path = base_path / "gpu_jobs.sh"
    file = open(file_path, "w")
    file.writelines(jobs)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
