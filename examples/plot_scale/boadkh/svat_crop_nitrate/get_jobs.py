from pathlib import Path
import os
import click


@click.command("main")
def main():
    # base_path = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh")
    base_path = Path("/pfs/work7/workspace/scratch/fr_rs1092-workspace1/roger/examples/plot_scale/boadkh")

    # identifiers for simulations
    locations = ["freiburg", "lahr", "muellheim", 
                "stockach", "gottmadingen", "weingarten",
                "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                "ehingen-kirchen", "merklingen", "hayingen",
                "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
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

    fertilization_intensities = ["low", "medium", "high"]

    # merge model output into single file
    for location in locations:
        for crop_rotation_scenario in crop_rotation_scenarios:
            for fertilization_intensity in fertilization_intensities:
                output_tm_file = (
                    base_path
                    / "output"
                    / "svat_crop_nitrate"
                    / f"SVATCROPNITRATE_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert.nc"
                )
                if not os.path.exists(output_tm_file):
                        print(f"sbatch --partition=single svat_crop_nitrate_{location}_{crop_rotation_scenario}_{fertilization_intensity}_Nfert_slurm.sh")
    return


if __name__ == "__main__":
    main()
