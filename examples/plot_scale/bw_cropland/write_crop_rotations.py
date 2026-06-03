from pathlib import Path
import os
import numpy as np
import pandas as pd

_dict_crop_to_luid = {
    "GE": 513,
    "Ka": 523,
    "KG": 580,
    "KM": 525,
    "SB": 547,
    "SJ": 541,
    "SM": 539,
    "WR": 559,
    "WW": 557,
    "ZR": 563,
    "Som": 542,
    "Win": 556,
    "ZW": 586,
    "nan": 599
}

_dict_crop_to_crop = {
    "GE": "vegetables",
    "Ka": "potato",
    "KG": "clover",
    "KM": "grain-corn",
    "SB": "sunflower",
    "SJ": "soybean",
    "SM": "silage-corn",
    "WR": "winter-rape",
    "WW": "winter-wheat",
    "ZR": "sugar-beet",
    "Som": "summer-barley",
    "Win": "winter-barley",
}

dict_crop_to_luid = {
    "vegetables": 513,
    "potato": 523,
    "clover": 580,
    "grain-corn": 525,
    "sunflower": 547,
    "soybean": 541,
    "silage-corn": 539,
    "winter-rape": 559,
    "winter-wheat": 557,
    "sugar-beet": 563,
    "summer-barley": 542,
    "winter-barley": 556,
}

summer_crops = [513, 523, 525, 580, 547, 541, 539, 563, 542]
winter_crops = [559, 557, 556]

dict_station_to_id = {
    "altheim": 4189,
    "bruchsal-heidelsheim": 731,
    "ellwangen-rindelbach": 1197,
    "elztal-rittersbach": 1216,
    "freiburg": 1443,
    "grosserlach-mannenweiler": 6259,
    "hechingen": 2074,
    "heidelberg": 2080,
    "klippeneck": 2638,
    "lahr": 2812,
    "mergentheim": 3257,
    "muellheim": 259,
    "nagold": 3432,
    "notzingen": 6275,
    "pfullendorf": 3927,
    "rheinau-memprechtshofen": 4169,
    "rheinstetten": 4177,
    "rot-buch": 5724,
    "sachsenheim": 4349,
    "singen": 6263,
    "stoetten": 4887,
    "ulm": 5155,
    "weingarten": 4094,
    "wutoeschingen-ofteringen": 5731
}

# reverse dict_station_to_id
dict_id_to_station = {v: k for k, v in dict_station_to_id.items()}

years = np.arange(2000, 2025).astype(int).tolist()

base_path = Path(__file__).parent

file = base_path / "input" / "crop_rotations.csv"
df_crop_rotations_ = pd.read_csv(file, sep=";")
cond = np.isfinite(df_crop_rotations_.loc[:, "station_id"])
df_crop_rotations_ = df_crop_rotations_.loc[cond]
df_crop_rotations_.loc[:, "station_id"] = df_crop_rotations_.loc[:, "station_id"].astype(int)
df_crop_rotations_["Crop1"] = df_crop_rotations_["Crop1"].astype(str)
df_crop_rotations_["Crop2"] = df_crop_rotations_["Crop2"].astype(str)
df_crop_rotations_["Crop3"] = df_crop_rotations_["Crop3"].astype(str)
df_crop_rotations_["Crop4"] = df_crop_rotations_["Crop4"].astype(str)
df_crop_rotations_["Crop5"] = df_crop_rotations_["Crop5"].astype(object).astype(str)
df_crop_rotations_.loc[:, "Crop5"] = "nan"

df_crop_rotations_all_combinations = pd.DataFrame(index=[0], columns=df_crop_rotations_.columns)
df_crop_rotations_all_combinations["group_flag"] = False
for i, row in df_crop_rotations_.iterrows():
    if len(df_crop_rotations_all_combinations.index) == 1:
        ii = 0
    else:
        ii = len(df_crop_rotations_all_combinations.index) + 1
    df_crop_rotations_all_combinations.loc[ii, :"Sum_station"] = row.values
    df_crop_rotations_all_combinations.loc[ii, "group_flag"] = True
    crops = row[["Crop1", "Crop2", "Crop3", "Crop4"]].values
    cond_crops = (crops != "nan")
    n_crops = np.sum(cond_crops)
    if n_crops == 2:
        crop1 = crops[0]
        crop2 = crops[1]
        df_crop_rotations_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_all_combinations.loc[ii+1, "Crop2"] = crop1
    elif n_crops == 3:
        crop1 = crops[0]
        crop2 = crops[1]
        crop3 = crops[2]
        df_crop_rotations_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_all_combinations.loc[ii+1, "Crop2"] = crop3
        df_crop_rotations_all_combinations.loc[ii+1, "Crop3"] = crop1
        df_crop_rotations_all_combinations.loc[ii+2, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+2, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+2, "Crop1"] = crop3
        df_crop_rotations_all_combinations.loc[ii+2, "Crop2"] = crop1
        df_crop_rotations_all_combinations.loc[ii+2, "Crop3"] = crop2
    elif n_crops == 4:
        crop1 = crops[0]
        crop2 = crops[1]
        crop3 = crops[2]
        crop4 = crops[3]
        df_crop_rotations_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_all_combinations.loc[ii+1, "Crop2"] = crop3
        df_crop_rotations_all_combinations.loc[ii+1, "Crop3"] = crop4
        df_crop_rotations_all_combinations.loc[ii+1, "Crop4"] = crop1
        df_crop_rotations_all_combinations.loc[ii+2, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+2, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+2, "Crop1"] = crop3
        df_crop_rotations_all_combinations.loc[ii+2, "Crop2"] = crop4
        df_crop_rotations_all_combinations.loc[ii+2, "Crop3"] = crop1
        df_crop_rotations_all_combinations.loc[ii+2, "Crop4"] = crop2
        df_crop_rotations_all_combinations.loc[ii+3, :"Sum_station"] = row.values
        df_crop_rotations_all_combinations.loc[ii+3, "group_flag"] = False
        df_crop_rotations_all_combinations.loc[ii+3, "Crop1"] = crop4
        df_crop_rotations_all_combinations.loc[ii+3, "Crop2"] = crop1
        df_crop_rotations_all_combinations.loc[ii+3, "Crop3"] = crop2
        df_crop_rotations_all_combinations.loc[ii+3, "Crop4"] = crop3

cr_columns = []
cr_columns_summer = []
cr_columns_winter = []
cr_columns.append("No")
for i, year in enumerate(years):
    cr_columns.append(f"{i}_summer")
    cr_columns.append(f"{i}_winter")
for i, year in enumerate(years):
    cr_columns_summer.append(f"{i+1}_summer")
    cr_columns_winter.append(f"{i+1}_winter")

df_crop_rotations_unique = df_crop_rotations_all_combinations.loc[:, "Crop1":"Crop5"].drop_duplicates().loc[:, "Crop1":"Crop5"].reset_index(drop=True)

# loop over unique crop rotations
for i, row in df_crop_rotations_unique.iterrows():
    crops = row.values.tolist()
    df_crop_rotation = pd.DataFrame(index=[0], columns=cr_columns)
    df_crop_rotation.loc[0, "No"] = 1
    df_crop_rotation.loc[0, "0_summer"] = 599
    df_crop_rotation.loc[0, "0_winter"] = 599
    df_crops_summer = pd.DataFrame(index=years, columns=["crop"])
    df_crops_winter = pd.DataFrame(index=years, columns=["crop"])
    df_crops_summer.loc[:, "crop"] = 599
    df_crops_winter.loc[:, "crop"] = 599
    n_summer_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in summer_crops])
    n_winter_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in winter_crops])
    if crops[2] == "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 1 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops:
                crop_summer = crops[0]
                crop_winter = crops[1]
            else:
                crop_summer = crops[1]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year+2, "crop"] = 599
        elif n_summer_crops == 2 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"
            for year in range(2000, 2024, 2):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_winter.loc[year, "crop"] = 599
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_winter.loc[year+1, "crop"] = 599

        elif n_summer_crops == 0 and n_winter_crops == 2:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"
            for year in range(2000, 2024, 2):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
        
    elif crops[2] != "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 2 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter = crops[1]
            else:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter = crops[0]

            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer2]
                df_crops_winter.loc[year+2, "crop"] = 599

        elif n_summer_crops == 1 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_summer = crops[1]
            else:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter2]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crop_winter2]

        elif n_summer_crops == 3 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_winter.loc[year, "crop"] = 599
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                df_crops_winter.loc[year+2, "crop"] = 599

        elif n_summer_crops == 0 and n_winter_crops == 3:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]

    elif crops[2] != "nan" and crops[3] != "nan" and crops[4] == "nan":
        if n_summer_crops == 3 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[2]
                crop_winter = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[3]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[1]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}_{_dict_crop_to_crop[crop_summer3]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer2]
                df_crops_winter.loc[year+2, "crop"] = 599
                df_crops_summer.loc[year+3, "crop"] = _dict_crop_to_luid[crop_summer3]
                df_crops_winter.loc[year+3, "crop"] = 599

        elif n_summer_crops == 1 and n_winter_crops == 3:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[2]
                crop_summer = crops[3]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[3]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[1]
            elif _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_winter3]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crop_winter2]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crop_winter3]
                df_crops_summer.loc[year+3 , "crop"] = 599
                df_crops_winter.loc[year+3 , "crop"] = 599

        elif n_summer_crops == 2 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter1 = crops[2]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter1 = crops[1]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[3]
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter1 = crops[0]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[2]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_summer2]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crop_winter2]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = 599
                df_crops_summer.loc[year+3 , "crop"] = _dict_crop_to_luid[crop_summer2]
                df_crops_winter.loc[year+3 , "crop"] = 599

        elif n_summer_crops == 4 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_winter.loc[year, "crop"] = 599
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                df_crops_winter.loc[year+2, "crop"] = 599
                df_crops_summer.loc[year+3, "crop"] = _dict_crop_to_luid[crops[3]]
                df_crops_winter.loc[year+3, "crop"] = 599

        elif n_summer_crops == 0 and n_winter_crops == 4:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                df_crops_summer.loc[year+3 , "crop"] = 599
                df_crops_winter.loc[year+3 , "crop"] = _dict_crop_to_luid[crops[3]]

    df_crop_rotation.loc[0, cr_columns_summer] = df_crops_summer.loc[:, "crop"].values
    df_crop_rotation.loc[0, cr_columns_winter] = df_crops_winter.loc[:, "crop"].values
    unit_header = ["[year_season]" for col in df_crop_rotation.columns]
    header = df_crop_rotation.columns
    df_crop_rotation.columns = [unit_header, header]

    file_dir = base_path / "input" / "crop_rotation_scenarios" / crop_rotation_type
    os.makedirs(file_dir, exist_ok=True)
    file = file_dir / "crop_rotation.csv"
    df_crop_rotation.to_csv(file, sep=";", index=False)

df_crop_rotations_all_combinations["crop_rotation_group"] = ""
df_crop_rotations_all_combinations["crop_rotation_type"] = ""
df_crop_rotations_all_combinations["subregion"] = ""
for i, row in df_crop_rotations_all_combinations.iterrows():
    crops = row[["Crop1", "Crop2", "Crop3", "Crop4", "Crop5"]].values.tolist()
    n_summer_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in summer_crops])
    n_winter_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in winter_crops])
    if crops[2] == "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 1 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops:
                crop_summer = crops[0]
                crop_winter = crops[1]
            else:
                crop_summer = crops[1]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer]}"
        elif n_summer_crops == 2 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"

        elif n_summer_crops == 0 and n_winter_crops == 2:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"        

    elif crops[2] != "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 2 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter = crops[1]
            else:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter = crops[0]

            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}"

        elif n_summer_crops == 1 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_summer = crops[1]
            else:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter2]}"

        elif n_summer_crops == 3 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"

        elif n_summer_crops == 0 and n_winter_crops == 3:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"

    elif crops[2] != "nan" and crops[3] != "nan" and crops[4] == "nan":
        if n_summer_crops == 3 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[2]
                crop_winter = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[3]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[1]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}_{_dict_crop_to_crop[crop_summer3]}"

        elif n_summer_crops == 1 and n_winter_crops == 3:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[2]
                crop_summer = crops[3]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[3]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[1]
            elif _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_winter3]}"

        elif n_summer_crops == 2 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter1 = crops[2]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter1 = crops[1]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[3]
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter1 = crops[0]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[2]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_summer2]}"

        elif n_summer_crops == 4 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"

        elif n_summer_crops == 0 and n_winter_crops == 4:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"

    df_crop_rotations_all_combinations.loc[i, "crop_rotation_type"] = crop_rotation_type
    station_id = df_crop_rotations_all_combinations.loc[i, "station_id"]
    df_crop_rotations_all_combinations.loc[i, "subregion"] = dict_id_to_station[station_id]

cond_group = (df_crop_rotations_all_combinations["group_flag"] == True)
df_crop_rotations_all_combinations.loc[cond_group, "crop_rotation_group"] = df_crop_rotations_all_combinations.loc[cond_group, "crop_rotation_type"]
# forward fill the crop_rotation_group column
df_crop_rotations_all_combinations["crop_rotation_group"] = df_crop_rotations_all_combinations["crop_rotation_group"].ffill()
df_crop_rotations_all_combinations.to_csv(base_path / "input" / "crop_rotations_all_combinations.csv", sep=";", index=False)

file = base_path / "input" / "crop_rotations_catchcrop.csv"
df_crop_rotations_catchcrop_ = pd.read_csv(file, sep=";", na_values="nan")
cond = np.isfinite(df_crop_rotations_catchcrop_.loc[:, "station_id"])
df_crop_rotations_catchcrop_ = df_crop_rotations_catchcrop_.loc[cond]
df_crop_rotations_catchcrop_["station_id"] = df_crop_rotations_catchcrop_["station_id"].astype(int)
df_crop_rotations_catchcrop_["Crop1"] = df_crop_rotations_catchcrop_["Crop1"].astype(str)
df_crop_rotations_catchcrop_["Crop2"] = df_crop_rotations_catchcrop_["Crop2"].astype(str)
df_crop_rotations_catchcrop_["Crop3"] = df_crop_rotations_catchcrop_["Crop3"].astype(str)
df_crop_rotations_catchcrop_["Crop4"] = df_crop_rotations_catchcrop_["Crop4"].astype(str)
df_crop_rotations_catchcrop_["Crop5"] = df_crop_rotations_catchcrop_["Crop5"].astype(object)
df_crop_rotations_catchcrop_["Crop5"] = "nan"
df_crop_rotations_catchcrop_["Zwischen_1_2"] = df_crop_rotations_catchcrop_["Zwischen_1_2"].astype(str)
df_crop_rotations_catchcrop_["Zwischen_2_3"] = df_crop_rotations_catchcrop_["Zwischen_2_3"].astype(str)
df_crop_rotations_catchcrop_["Zwischen_3_4"] = df_crop_rotations_catchcrop_["Zwischen_3_4"].astype(str)
df_crop_rotations_catchcrop_["Zwischen_4_5"] = df_crop_rotations_catchcrop_["Zwischen_4_5"].astype(object)
df_crop_rotations_catchcrop_["Zwischen_4_5"] = "nan"

df_crop_rotations_catchcrop_all_combinations = pd.DataFrame(index=[0], columns=df_crop_rotations_catchcrop_.columns)
df_crop_rotations_catchcrop_all_combinations["group_flag"] = False

for i, row in df_crop_rotations_catchcrop_.iterrows():
    if len(df_crop_rotations_catchcrop_all_combinations.index) == 1:
        ii = 0
    else:
        ii = len(df_crop_rotations_catchcrop_all_combinations.index) + 1
    df_crop_rotations_catchcrop_all_combinations.loc[ii, :"Sum_station"] = row.values
    df_crop_rotations_catchcrop_all_combinations.loc[ii, "group_flag"] = True
    crops = row[["Crop1", "Crop2", "Crop3", "Crop4"]].values
    catchcrops = row[["Zwischen_1_2", "Zwischen_2_3", "Zwischen_3_4"]].values
    cond_crops = (crops != "nan")
    n_crops = np.sum(cond_crops)
    if n_crops == 2:
        crop1 = crops[0]
        crop2 = crops[1]
        catchcrop1 = catchcrops[0]
        catchcrop2 = catchcrops[1]
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop2"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_1_2"] = catchcrop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_2_3"] = catchcrop1
    elif n_crops == 3:
        crop1 = crops[0]
        crop2 = crops[1]
        crop3 = crops[2]
        catchcrop1 = catchcrops[0]
        catchcrop2 = catchcrops[1]
        catchcrop3 = catchcrops[2]
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop2"] = crop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop3"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_1_2"] = catchcrop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_2_3"] = catchcrop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_3_4"] = catchcrop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop1"] = crop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop2"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop3"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_1_2"] = catchcrop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_2_3"] = catchcrop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_3_4"] = catchcrop2
    elif n_crops == 4:
        crop1 = crops[0]
        crop2 = crops[1]
        crop3 = crops[2]
        crop4 = crops[3]
        catchcrop1 = catchcrops[0]
        catchcrop2 = catchcrops[1]
        catchcrop3 = catchcrops[2]
        catchcrop4 = "nan"
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop1"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop2"] = crop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop3"] = crop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Crop4"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_1_2"] = catchcrop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_2_3"] = catchcrop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_3_4"] = catchcrop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+1, "Zwischen_4_5"] = catchcrop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop1"] = crop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop2"] = crop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop3"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Crop4"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_1_2"] = catchcrop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_2_3"] = catchcrop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_3_4"] = catchcrop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+2, "Zwischen_4_5"] = catchcrop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, :"Sum_station"] = row.values
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "group_flag"] = False
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Crop1"] = crop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Crop2"] = crop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Crop3"] = crop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Crop4"] = crop3
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Zwischen_1_2"] = catchcrop4
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Zwischen_2_3"] = catchcrop1
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Zwischen_3_4"] = catchcrop2
        df_crop_rotations_catchcrop_all_combinations.loc[ii+3, "Zwischen_4_5"] = catchcrop3

df_crop_rotations_catchcrop_unique = df_crop_rotations_catchcrop_all_combinations.loc[:, "Crop1":"Crop5"].drop_duplicates().reset_index(drop=True)
# loop over unique crop rotations
for i, row in df_crop_rotations_catchcrop_unique.iterrows():
    crops = row[["Crop1", "Crop2", "Crop3", "Crop4", "Crop5"]].values.tolist()
    df_crop_rotation = pd.DataFrame(index=[0], columns=cr_columns)
    df_crop_rotation.loc[0, "No"] = 1
    df_crop_rotation.loc[0, "0_summer"] = 599
    df_crop_rotation.loc[0, "0_winter"] = 599
    df_crops_summer = pd.DataFrame(index=years, columns=["crop"])
    df_crops_winter = pd.DataFrame(index=years, columns=["crop"])
    df_crops_summer.loc[:, "crop"] = 599
    df_crops_winter.loc[:, "crop"] = 599
    n_summer_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in summer_crops])
    n_winter_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in winter_crops])
    if crops[2] == "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 1 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops:
                crop_summer = crops[0]
                crop_winter = crops[1]
            else:
                crop_summer = crops[1]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter]}_yellow-mustard_{_dict_crop_to_crop[crop_summer]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 586
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year+2, "crop"] = 599
        elif n_summer_crops == 2 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard"
            for year in range(2000, 2024, 2):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                if _dict_crop_to_luid[crops[0]] in [525, 539]:
                    df_crops_winter.loc[year, "crop"] = 587
                else:
                    df_crops_winter.loc[year, "crop"] = 586
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                if _dict_crop_to_luid[crops[1]] in [525, 539]:
                    df_crops_winter.loc[year+1, "crop"] = 587
                else:
                    df_crops_winter.loc[year+1, "crop"] = 586

        elif n_summer_crops == 0 and n_winter_crops == 2:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"
            for year in range(2000, 2024, 2):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]

    elif crops[2] != "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 2 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter = crops[1]
            else:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter = crops[0]

            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_yellow-mustard_{_dict_crop_to_crop[crop_summer2]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 587
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer2]
                df_crops_winter.loc[year+2, "crop"] = 599

        elif n_summer_crops == 1 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_summer = crops[1]
            else:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter1]}_yellow-mustard_{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter2]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 587
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crop_winter2]

        elif n_summer_crops == 3 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard_{_dict_crop_to_crop[crops[2]]}_yellow-mustard"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                if _dict_crop_to_luid[crops[0]] in [525, 539]:
                    df_crops_winter.loc[year, "crop"] = 587
                else:
                    df_crops_winter.loc[year, "crop"] = 586
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                if _dict_crop_to_luid[crops[1]] in [525, 539]:
                    df_crops_winter.loc[year+1, "crop"] = 587
                else:
                    df_crops_winter.loc[year+1, "crop"] = 586
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                if _dict_crop_to_luid[crops[2]] in [525, 539]:
                    df_crops_winter.loc[year+2, "crop"] = 587
                else:
                    df_crops_winter.loc[year+2, "crop"] = 586

        elif n_summer_crops == 0 and n_winter_crops == 3:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"
            for year in range(2000, 2024, 3):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]

    elif crops[2] != "nan" and crops[3] != "nan" and crops[4] == "nan":
        if n_summer_crops == 3 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[2]
                crop_winter = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[3]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[1]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}_yellow-mustard_{_dict_crop_to_crop[crop_summer3]}_yellow-mustard"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = 599
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crop_summer2]
                if _dict_crop_to_luid[crop_summer2] in [525, 539]:
                    df_crops_winter.loc[year+2, "crop"] = 587
                else:
                    df_crops_winter.loc[year+2, "crop"] = 586
                df_crops_summer.loc[year+3, "crop"] = _dict_crop_to_luid[crop_summer3]
                if _dict_crop_to_luid[crop_summer3] in [525, 539]:
                    df_crops_winter.loc[year+3, "crop"] = 587
                else:
                    df_crops_winter.loc[year+3, "crop"] = 586

        elif n_summer_crops == 1 and n_winter_crops == 3:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[2]
                crop_summer = crops[3]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[3]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[1]
            elif _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_winter3]}_yellow-mustard"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crop_winter2]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crop_winter3]
                df_crops_summer.loc[year+3 , "crop"] = 599
                df_crops_winter.loc[year+3 , "crop"] = 586

        elif n_summer_crops == 2 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter1 = crops[2]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter1 = crops[1]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[3]
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter1 = crops[0]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[2]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_yellow-mustard_{_dict_crop_to_crop[crop_summer2]}_yellow-mustard"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crop_summer1]
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crop_winter1]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crop_winter2]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = 586
                df_crops_summer.loc[year+3 , "crop"] = _dict_crop_to_luid[crop_summer2]
                df_crops_winter.loc[year+3 , "crop"] = 587

        elif n_summer_crops == 4 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                if _dict_crop_to_luid[crops[0]] in [525, 539]:
                    df_crops_winter.loc[year, "crop"] = 587
                else:
                    df_crops_winter.loc[year, "crop"] = 586
                df_crops_summer.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                if _dict_crop_to_luid[crops[1]] in [525, 539]:
                    df_crops_winter.loc[year+1, "crop"] = 587
                else:
                    df_crops_winter.loc[year+1, "crop"] = 586
                df_crops_summer.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                if _dict_crop_to_luid[crops[2]] in [525, 539]:
                    df_crops_winter.loc[year+2, "crop"] = 587
                else:
                    df_crops_winter.loc[year+2, "crop"] = 586
                df_crops_summer.loc[year+3, "crop"] = _dict_crop_to_luid[crops[3]]
                if _dict_crop_to_luid[crops[0]] in [525, 539]:
                    df_crops_winter.loc[year+3, "crop"] = 587
                else:
                    df_crops_winter.loc[year+3, "crop"] = 586

        elif n_summer_crops == 0 and n_winter_crops == 4:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"
            for year in range(2000, 2024, 4):
                df_crops_summer.loc[year, "crop"] = 599
                df_crops_winter.loc[year, "crop"] = _dict_crop_to_luid[crops[0]]
                df_crops_summer.loc[year+1, "crop"] = 599
                df_crops_winter.loc[year+1, "crop"] = _dict_crop_to_luid[crops[1]]
                df_crops_summer.loc[year+2, "crop"] = 599
                df_crops_winter.loc[year+2, "crop"] = _dict_crop_to_luid[crops[2]]
                df_crops_summer.loc[year+3 , "crop"] = 599
                df_crops_winter.loc[year+3 , "crop"] = _dict_crop_to_luid[crops[3]]

    df_crop_rotation.loc[0, cr_columns_summer] = df_crops_summer.loc[:, "crop"].values
    df_crop_rotation.loc[0, cr_columns_winter] = df_crops_winter.loc[:, "crop"].values
    unit_header = ["[year_season]" for col in df_crop_rotation.columns]
    header = df_crop_rotation.columns
    df_crop_rotation.columns = [unit_header, header]

    file_dir = base_path / "input" / "crop_rotation_scenarios" / crop_rotation_type
    os.makedirs(file_dir, exist_ok=True)
    file = file_dir / "crop_rotation.csv"
    df_crop_rotation.to_csv(file, sep=";", index=False)

df_crop_rotations_catchcrop_all_combinations["crop_rotation_group"] = ""
df_crop_rotations_catchcrop_all_combinations["crop_rotation_type"] = ""
df_crop_rotations_catchcrop_all_combinations["subregion"] = ""
for i, row in df_crop_rotations_catchcrop_all_combinations.iterrows():
    crops = row[["Crop1", "Crop2", "Crop3", "Crop4", "Crop5"]].values.tolist()
    n_summer_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in summer_crops])
    n_winter_crops = np.sum([1 for crop in crops if _dict_crop_to_luid[crop] in winter_crops])
    if crops[2] == "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 1 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops:
                crop_summer = crops[0]
                crop_winter = crops[1]
            else:
                crop_summer = crops[1]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter]}_yellow-mustard_{_dict_crop_to_crop[crop_summer]}"
        elif n_summer_crops == 2 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard"

        elif n_summer_crops == 0 and n_winter_crops == 2:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}"

    elif crops[2] != "nan" and crops[3] == "nan" and crops[4] == "nan":
        if n_summer_crops == 2 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter = crops[1]
            else:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter = crops[0]

            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_yellow-mustard_{_dict_crop_to_crop[crop_summer2]}"

        elif n_summer_crops == 1 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_summer = crops[1]
            else:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_winter1]}_yellow-mustard_{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter2]}"

        elif n_summer_crops == 3 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard_{_dict_crop_to_crop[crops[2]]}_yellow-mustard"

        elif n_summer_crops == 0 and n_winter_crops == 3:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}"

    elif crops[2] != "nan" and crops[3] != "nan" and crops[4] == "nan":
        if n_summer_crops == 3 and n_winter_crops == 1:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[2]
                crop_winter = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_summer3 = crops[3]
                crop_winter = crops[2]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[1]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_summer3 = crops[3]
                crop_winter = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter]}_{_dict_crop_to_crop[crop_summer2]}_yellow-mustard_{_dict_crop_to_crop[crop_summer3]}_yellow-mustard"

        elif n_summer_crops == 1 and n_winter_crops == 3:
            if _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[2]
                crop_summer = crops[3]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
                crop_winter3 = crops[3]
                crop_summer = crops[2]
            elif _dict_crop_to_luid[crops[0]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[1]
            elif _dict_crop_to_luid[crops[1]] in winter_crops and _dict_crop_to_luid[crops[2]] in winter_crops and _dict_crop_to_luid[crops[3]] in winter_crops:
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
                crop_winter3 = crops[3]
                crop_summer = crops[0]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_{_dict_crop_to_crop[crop_winter3]}_yellow-mustard"

        elif n_summer_crops == 2 and n_winter_crops == 2:
            if _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[1]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[1]
                crop_winter1 = crops[2]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[2]
                crop_winter1 = crops[1]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[0]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[0]
                crop_summer2 = crops[3]
                crop_winter1 = crops[1]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[2]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[2]
                crop_winter1 = crops[0]
                crop_winter2 = crops[3]
            elif _dict_crop_to_luid[crops[1]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[1]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[2]
            elif _dict_crop_to_luid[crops[2]] in summer_crops and _dict_crop_to_luid[crops[3]] in summer_crops:
                crop_summer1 = crops[2]
                crop_summer2 = crops[3]
                crop_winter1 = crops[0]
                crop_winter2 = crops[1]
            crop_rotation_type = f"{_dict_crop_to_crop[crop_summer1]}_{_dict_crop_to_crop[crop_winter1]}_{_dict_crop_to_crop[crop_winter2]}_yellow-mustard_{_dict_crop_to_crop[crop_summer2]}_yellow-mustard"

        elif n_summer_crops == 4 and n_winter_crops == 0:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_yellow-mustard_{_dict_crop_to_crop[crops[1]]}_yellow-mustard_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"

        elif n_summer_crops == 0 and n_winter_crops == 4:
            crop_rotation_type = f"{_dict_crop_to_crop[crops[0]]}_{_dict_crop_to_crop[crops[1]]}_{_dict_crop_to_crop[crops[2]]}_{_dict_crop_to_crop[crops[3]]}"

    df_crop_rotations_catchcrop_all_combinations.loc[i, "crop_rotation_type"] = crop_rotation_type
    station_id = df_crop_rotations_catchcrop_all_combinations.loc[i, "station_id"]
    df_crop_rotations_catchcrop_all_combinations.loc[i, "subregion"] = dict_id_to_station[station_id]

cond_group = (df_crop_rotations_catchcrop_all_combinations["group_flag"] == True)
df_crop_rotations_catchcrop_all_combinations.loc[cond_group, "crop_rotation_group"] = df_crop_rotations_catchcrop_all_combinations.loc[cond_group, "crop_rotation_type"]
df_crop_rotations_catchcrop_all_combinations["crop_rotation_group"] = df_crop_rotations_catchcrop_all_combinations["crop_rotation_group"].ffill()
df_crop_rotations_catchcrop_all_combinations.to_csv(base_path / "input" / "crop_rotations_catchcrop_all_combinations.csv", sep=";", index=False)

df1 = df_crop_rotations_all_combinations.loc[:, ["station_id", "crop_rotation_type", "subregion"]]
df2 = df_crop_rotations_catchcrop_all_combinations.loc[:, ["station_id", "crop_rotation_type", "subregion"]]
df_subregions_crop_rotations = pd.concat([df1, df2], axis=0)
df_subregions_crop_rotations = df_subregions_crop_rotations[["station_id", "subregion", "crop_rotation_type"]]
# remove duplicate rows
df_subregions_crop_rotations = df_subregions_crop_rotations.drop_duplicates()
df_subregions_crop_rotations.to_csv(base_path / "subregions_crop_rotations.csv", sep=";", index=False)
