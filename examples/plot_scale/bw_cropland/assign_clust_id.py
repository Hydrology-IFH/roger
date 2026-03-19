from pathlib import Path
import pandas as pd
import geopandas as gpd

base_path = Path(__file__).parent

# load linkage between BK50 and cropland clusters
file = base_path / "input" / "link_cluster_geometries_cropland.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)

# load model parameters
csv_file = base_path / "representative_agricultural_soil_types_parameters_.csv"
df_params = pd.read_csv(csv_file, sep=";", skiprows=1)
cond = (df_params["CLUST_flag"] == 1)
df_params = df_params.loc[cond, :]
clust_ids = pd.unique(df_params["CLUST_ID"].values).tolist()

columns1 = ["", "", "", "[mm]", "[1/m2]", "[mm]", "[-]", "[-]", "[-]", "[mm/hour]", "[mm/hour]", "", "[-]"]
columns2 = df_params.columns.tolist()
df_params.columns = [columns1, columns2]
csv_file = base_path / "representative_agricultural_soil_types_parameters.csv"
df_params.to_csv(csv_file, sep=";", index=False)

file = base_path / "input" / "BK50_cropland_areas_.gpkg"
gdf = gpd.read_file(file)
gdf["CLUST_ID"] = ""
# assign aggregated values to polygons
for clust_id in clust_ids:
    cond = (df_link_bk50_cluster_cropland["CLUST_ID"] == clust_id)
    shp_ids = df_link_bk50_cluster_cropland.loc[cond, :].index.tolist()
    cond2 = gdf["SHP_ID"].isin(shp_ids)
    gdf.loc[cond2, "CLUST_ID"] = str(clust_id)

# remove if clust_id is not assigned
gdf = gdf[gdf["CLUST_ID"] != ""]

# save the updated geodataframe
output_file = base_path / "input" / "BK50_cropland_areas.gpkg"
gdf.to_file(output_file, driver="GPKG")
