from pathlib import Path
import subprocess
import yaml
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent

    # load the configuration file
    with open(base_path.parent / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    irrigation_scenarios = config["irrigation_scenarios"]
    irrigation_scenarios.append("no_irrigation")
    crop_rotation_scenarios = config["crop_rotation_scenarios"]

    # --- jobs to calculate nitrate transport --------------------------------------------------------
    file_path = base_path / "run_roger-sas-nitrate.sh"
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('eval "$(conda shell.bash hook)"\n')
    lines.append("conda activate roger\n")
    for irrigation_scenario in irrigation_scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario %s --crop-rotation-scenario %s\n' % (irrigation_scenario, crop_rotation_scenario)
            )
    lines.append('python merge_output.py\n')
    lines.append('python write_simulations_to_csv.py')
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    file_path = base_path / "run_roger-sas-nitrate_compaction.sh"
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('eval "$(conda shell.bash hook)"\n')
    lines.append("conda activate roger\n")
    for irrigation_scenario in irrigation_scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario %s --crop-rotation-scenario %s\n' % (irrigation_scenario, crop_rotation_scenario)
            )
    lines.append('python merge_output.py\n')
    lines.append('python write_simulations_to_csv.py')
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for irrigation_scenario in irrigation_scenarios:
        lines = []
        lines.append('#!/bin/bash\n')
        for crop_rotation_scenario in crop_rotation_scenarios:
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario no_compaction --irrigation-scenario %s --crop-rotation-scenario %s\n' % (irrigation_scenario, crop_rotation_scenario)
            )
        file_path = base_path / f"run_roger-sas-nitrate_{irrigation_scenario}.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    for irrigation_scenario in irrigation_scenarios:
        lines = []
        lines.append('#!/bin/bash\n')
        for crop_rotation_scenario in crop_rotation_scenarios:
            lines.append(
                'python svat_crop_nitrate.py -b jax -d cpu --soil-compaction-scenario compaction --irrigation-scenario %s --crop-rotation-scenario %s\n' % (irrigation_scenario, crop_rotation_scenario)
            )
        file_path = base_path / f"run_roger-sas-nitrate_{irrigation_scenario}_compaction.sh"
        file = open(file_path, "w")
        file.writelines(lines)
        file.close()
        subprocess.Popen(f"chmod +x {file_path}", shell=True)

    return


if __name__ == "__main__":
    main()
