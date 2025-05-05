from pathlib import Path
import subprocess
import click
import yaml


@click.command("main")
def main():
    base_path = Path(__file__).parent

    # load the configuration file
    with open(base_path.parent / "config.yml", "r") as file:
        config = yaml.safe_load(file)

    crop_rotation_scenarios = config["crop_rotation_scenarios"]

    # --- jobs to calculate fluxes and states --------------------------------------------------------
    file_path = base_path / "run_roger.sh"
    lines = []
    lines.append('#!/bin/bash\n')
    # lines.append('eval "$(conda shell.bash hook)"\n')
    # lines.append("conda activate roger\n")
    for crop_rotation_scenario in crop_rotation_scenarios:
        lines.append(
            'python svat_crop.py -b jax -d cpu --crop-rotation-scenario %s\n' % (crop_rotation_scenario)
        )
    # lines.append('python merge_output.py\n')
    # lines.append('python write_simulations_to_csv.py')
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)
    return


if __name__ == "__main__":
    main()
