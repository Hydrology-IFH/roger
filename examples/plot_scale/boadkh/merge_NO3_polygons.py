from pathlib import Path
import pandas as pd
import geopandas as gpd
import click

@click.command("main")
def main():
    # directory of results
    # base_path_output = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "output" / "data_for_nitrate_leaching"
    base_path_output = Path(__file__).parent / "output" / "data_for_nitrate_leaching"
    locations = ["freiburg", "lahr", "muellheim", 
                 "stockach", "gottmadingen", "weingarten",
                 "eppingen-elsenz", "bruchsal-heidelsheim", "bretten",
                 "ehingen-kirchen", "merklingen", "hayingen",
                 "kupferzell", "oehringen", "vellberg-kleinaltdorf"]
    
    ll_df = [] 
    for location in locations:
        file = base_path_output / f"nitrate_leaching_{location}.gpkg"
        gdf = gpd.read_file(file)
        ll_df.append(gdf)

    gdf = pd.concat(ll_df, axis=0)
    gdf = gdf.to_crs("EPSG:25832")
    file = base_path_output / "nitrate_leaching.gpkg"
    gdf.to_file(file, driver="GPKG")
    file = base_path_output / "nitrate_leaching.shp"
    gdf.to_file(file)
    file = base_path_output / "nitrate_leaching.csv"
    df = pd.DataFrame(gdf)
    df.to_csv(file, index=False, sep=";")


if __name__ == "__main__":
    main()