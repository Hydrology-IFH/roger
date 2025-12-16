from pathlib import Path
import numpy as np
import pandas as pd
import click
import roger.lookuptables as lut

_dict_crop_id = {557: "winter wheat",
                 567: "clover",
                 580: "clover",
                 581: "clover",
                 582: "clover",
                 583: "clover",
                 584: "clover",
                 585: "clover",
                 539: "silage corn",
                 543: "summer wheat",
                 563: "sugar beet",
                 559: "winter rape",
                 541: "soybean",
                 525: "grain corn",
                 556: "winter barley",
                 523: "potato",
                 510: "potato early",
                 521: "vegetables",
                 542: "summer barley",
                 576: "faba bean",
                 586: "yellow mustard",
                 587: "yellow mustard",
                 588: "yellow mustard",
                 589: "miscanthus",
                 590: "misacanthus",
                 591: "miscanthus",
                 571: "grass",
                 572: "grass",
                 573: "grass",
                 574: "grass",
                 599: "bare"
                }

crop_rotation_scenarios = ["grain-corn",
                           "grain-corn_yellow-mustard",
                           "silage-corn",
                           "silage-corn_yellow-mustard",
                           "summer-barley",
                           "summer-barley_yellow-mustard",
                           "clover",
                           "winter-wheat",
                           "winter-barley",
                           "winter-rape",
                           "faba-bean",
                           "potato-early",
                           "potato",
                           "sugar-beet",
                           "sugar-beet_yellow-mustard",
                           "vegetables",
                           "strawberry",
                           "asparagus",
                           "winter-wheat_clover",
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
                           "grain-corn",
                           "grain-corn_yellow-mustard",
                           "winter-wheat",
                           "yellow-mustard",
                           "miscanthus",
                           "bare-grass"]
@click.command("main")
def main():
    base_path = Path(__file__).parent
    for crop_rotation_scenario in crop_rotation_scenarios:
        file = base_path / "input" / "crop_rotation_scenarios" / f"{crop_rotation_scenario}" / "crop_rotation.csv"
        df = pd.read_csv(file, sep=";", skiprows=1).iloc[:, 3:]

        # seclect every second column
        nn = int(len(df.columns))
        summer_ids = list(range(0, nn, 2))
        winter_ids = list(range(1, nn, 2))

        df_summer = df.iloc[:, summer_ids]
        df_winter = df.iloc[:, winter_ids]

        index = pd.date_range(start="2000-01-01", end="2024-12-31", freq="YE")
        df_crops = pd.DataFrame(index=index)
        df_crops["summer_crop"] = df_summer.values.flatten()
        df_crops["winter_crop"] = df_winter.values.flatten()
        df_crops["lu_id"] = 599
        for i in range(len(index)-1):
            summer_crop = df_crops["summer_crop"].iloc[i]
            winter_crop = df_crops["winter_crop"].iloc[i]
            if summer_crop != 599:
                df_crops["lu_id"].iloc[i] = summer_crop
            if winter_crop != 599:
                df_crops["lu_id"].iloc[i+1] = winter_crop
        summer_crop = df_crops["summer_crop"].iloc[-1]
        if summer_crop != 599:
            df_crops["lu_id"].iloc[-1] = summer_crop

        df_nitrate_leaching = pd.DataFrame(index=index)
        df_nitrate_leaching["low_NFert"] = 0
        df_nitrate_leaching["medium_NFert"] = 0
        df_nitrate_leaching["high_NFert"] = 0
        df_nitrate_leaching["lu_id"] = df_crops["lu_id"].values
        df_nitrate_leaching["crop_type"] = df_crops["lu_id"].map(_dict_crop_id)
        for i, lu_id in enumerate(df_nitrate_leaching["lu_id"]):
            row_id = np.where(lut.ARR_FERT1[:, 0] == lu_id)[0][0]
            Nin1 = lut.ARR_FERT1[row_id, 4] + lut.ARR_FERT1[row_id, 5] + lut.ARR_FERT1[row_id, 6] + lut.ARR_FERT1[row_id, 10] + lut.ARR_FERT1[row_id, 11] + lut.ARR_FERT1[row_id, 12]
            Nin2 = lut.ARR_FERT2[row_id, 4] + lut.ARR_FERT2[row_id, 5] + lut.ARR_FERT2[row_id, 6] + lut.ARR_FERT2[row_id, 10] + lut.ARR_FERT2[row_id, 11] + lut.ARR_FERT2[row_id, 12]
            Nin3 = lut.ARR_FERT3[row_id, 4] + lut.ARR_FERT3[row_id, 5] + lut.ARR_FERT3[row_id, 6] + lut.ARR_FERT3[row_id, 10] + lut.ARR_FERT3[row_id, 11] + lut.ARR_FERT3[row_id, 12]
            if "yellow-mustard" in crop_rotation_scenario:
                if lu_id in [525, 542, 543, 544, 545, 539, 563]:
                    if Nin1 > 40:
                        Nin1 = Nin1 - 40
                    if Nin2 > 40:
                        Nin2 = Nin2 - 40
                    if Nin3 > 40:
                        Nin3 = Nin3 - 40
            df_nitrate_leaching["low_NFert"].iloc[i] = Nin1 * 0.3
            df_nitrate_leaching["medium_NFert"].iloc[i] = Nin2 * 0.3
            df_nitrate_leaching["high_NFert"].iloc[i] = Nin3 * 0.3

        df_nitrate_leaching_avg = df_nitrate_leaching.iloc[:, 0:3].mean().to_frame().T

        df_nitrate_leaching = df_nitrate_leaching[["low_NFert", "medium_NFert", "high_NFert", "crop_type"]]
        df_nitrate_leaching.columns = [["[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]", ""],
                                    ["low_NFert", "medium_NFert", "high_NFert", "crop_type"]]
        df_nitrate_leaching.index = df_nitrate_leaching.index.rename("")

        df_nitrate_leaching_avg.columns = [["[kg N/ha/year]", "[kg N/ha/year]", "[kg N/ha/year]"],
                                    ["low_NFert", "medium_NFert", "high_NFert"]]
        df_nitrate_leaching_avg.index = df_nitrate_leaching_avg.index.rename("")

        file = base_path / "output" / "nitrate" / "thuenen" / f"nitrate_leaching_thuenen_{crop_rotation_scenario}.csv"   
        df_nitrate_leaching.to_csv(file, sep=";", index=True)

        file = base_path / "output" / "nitrate" / "thuenen" / f"nitrate_leaching_avg_thuenen_{crop_rotation_scenario}.csv"   
        df_nitrate_leaching_avg.to_csv(file, sep=";", index=False)
    return

if __name__ == "__main__":
    main()