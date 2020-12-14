
import geopandas as gpd
from .filepaths import filenames

zone_shp = gpd.read_file(
	filenames.zone_shapefile
    #"/Users/jeffnewman/Cambridge Systematics/PROJ CMAP Trip-Based - General/GIS/zones17.shp"
).set_index('zone17')


