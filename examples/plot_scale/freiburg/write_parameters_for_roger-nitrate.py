from pathlib import Path
import pandas as pd

base_path = Path(__file__).parent

# load the parameters
file = base_path / "parameters.csv"
df_parameters = pd.read_csv(file, sep=";", skiprows=1, index_col=0)

# write soil depth, clay content and soil fertility to csv
df_zsoil = df_parameters[["z_soil"]]   
df_clay = df_parameters[["clay"]]   
df_soil_fertility = df_parameters[["soil_fertility"]]

file = base_path / "clay.csv"
df_clay.columns = [
    ["[-]"],
    ["clay"],
]
df_clay.to_csv(file, index=True, sep=";")

file = base_path / "soil_fertility.csv"
df_soil_fertility.columns = [
    [""],
    ["soil_fertility"],
]
df_soil_fertility.to_csv(file, index=True, sep=";")

file = base_path / "z_soil.csv"
df_zsoil.columns = [
    ["[mm]"],
    ["z_soil"],
]
df_zsoil.to_csv(file, index=True, sep=";")