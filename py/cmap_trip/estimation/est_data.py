import os
import cmap_trip

from .est_logging import L
from .est_config import EstimationDir

L("###### Set Directories and Prep Data ######")

from ..data_handlers import DataHandler


dh = DataHandler(
	emme_database_dir=os.path.join(cmap_trip.__path__[0], "../tests/data"),
	omx_skims_dir=EstimationDir/"SkimsForEstimation",
	cache_dir=EstimationDir/"cache",
	zone_shapefile=EstimationDir/"../GIS/From CMAP/zones17.shp",
	emmemat_archive=EstimationDir/"SkimsForEstimation/emmemat.zip",
)

dh.SURVEY_DATA_DIR = EstimationDir/"../HTS/preproc"
dh.AE_DATA_DIR = EstimationDir/"../HTS/cmap-proc"

L("Data Handlers Ready")
