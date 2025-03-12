from pathlib import Path
import pandas as pd
import subprocess
import click


@click.command("main")
def main():
    base_path = Path(__file__).parent

    irrigation_scenarios = ["35-ufc",
                            "45-ufc",
                            "50-ufc",
                            "80-ufc",
                            "crop-specific",
                            ]
    crop_rotation_scenarios = ["winter-wheat_clover",
                               "winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat",
                               "summer-wheat_clover_winter-wheat",
                               "winter-wheat_clover_silage-corn",
                               "winter-wheat_sugar-beet_silage-corn",
                               "summer-wheat_winter-wheat_silage-corn",
                               "summer-wheat_winter-wheat_winter-rape",
                               "winter-wheat_winter-rape",
                               "winter-wheat_soybean_winter-rape",
                               "sugar-beet_winter-wheat_winter-barley", 
                               "grain-corn_winter-wheat_winter-rape", 
                               "grain-corn_winter-wheat_winter-barley",
                               "grain-corn_winter-wheat_clover",
                               "winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_yellow-mustard",
                               "winter-wheat_sugar-beet_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_silage-corn_yellow-mustard",
                               "summer-wheat_winter-wheat_winter-rape_yellow-mustard",
                               "sugar-beet_winter-wheat_winter-barley_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-rape_yellow-mustard", 
                               "grain-corn_winter-wheat_winter-barley_yellow-mustard",
                               "miscanthus",
                               "bare-grass"]

    # --- jobs to calculate fluxes and states --------------------------------------------------------
    file_path = base_path / "run_roger.sh"
    lines = []
    lines.append('#!/bin/bash\n')
    lines.append('eval "$(conda shell.bash hook)"\n')
    lines.append("conda activate roger\n")
    lines.append('python write_parameters_to_netcdf.py\n')
    for irrigation_scenario in irrigation_scenarios:
        for crop_rotation_scenario in crop_rotation_scenarios:
            lines.append(
                'python svat_crop.py -b numpy -d cpu --irrigation-scenario %s --crop-rotation-scenario %s\n' % (irrigation_scenario, crop_rotation_scenario)
            )
    lines.append('python merge_output.py\n')
    lines.append('python simulations_to_csv.py')
    file = open(file_path, "w")
    file.writelines(lines)
    file.close()
    subprocess.Popen(f"chmod +x {file_path}", shell=True)
    return


if __name__ == "__main__":
    main()
