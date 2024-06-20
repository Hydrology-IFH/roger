from pathlib import Path
import pandas as pd

# load linkage between BK50 and cropland clusters
file = Path("/Volumes/LaCie/roger/examples/plot_scale/boadkh") / "link_shp_clust_acker.h5"
# file = base_path_output / "link_shp_clust_acker.h5"
df_link_bk50_cluster_cropland = pd.read_hdf(file)
cond = (df_link_bk50_cluster_cropland.index == 1707568)

print(df_link_bk50_cluster_cropland.loc[cond, :])