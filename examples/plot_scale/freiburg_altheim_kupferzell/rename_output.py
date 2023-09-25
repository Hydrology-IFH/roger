from pathlib import Path
import os

base_path = Path(__file__).parent

# identifiers for simulations
locations = ["freiburg", "altheim", "kupferzell"]
land_cover_scenarios = ["grass", "corn", "corn_catch_crop", "crop_rotation"]
climate_scenarios = ["CCCma-CanESM2_CCLM4-8-17", "MPI-M-MPI-ESM-LR_RCA4"]
periods = ["1985-2014", "2030-2059", "2070-2099"]
periods1 = ["1985-2015", "2030-2060", "2070-2100"]

for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period, period1 in zip(periods, periods1):
                path = str(
                    base_path
                    / "output"
                    / "svat"
                    / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.nc"
                )
                path1 = str(
                    base_path
                    / "output"
                    / "svat"
                    / f"SVAT_{location}_{land_cover_scenario}_{climate_scenario}_{period1}.nc"
                )
                if not os.path.exists(path):
                    os.rename(path1, path)

for location in locations:
    for land_cover_scenario in land_cover_scenarios:
        for climate_scenario in climate_scenarios:
            for period, period1 in zip(periods, periods1):
                for agg_type in ["average", "rate", "maximum", "collect"]:
                    path = str(
                        base_path
                        / "output"
                        / "svat_transport"
                        / f"SVATTRANSPORT_{location}_{land_cover_scenario}_{climate_scenario}_{period}.{agg_type}.nc"
                    )
                    path1 = str(
                        base_path
                        / "output"
                        / "svat_transport"
                        / f"SVATTRANSPORT_{location}_{land_cover_scenario}_{climate_scenario}_{period1}.{agg_type}.nc"
                    )
                    if not os.path.exists(path):
                        os.rename(path1, path)
