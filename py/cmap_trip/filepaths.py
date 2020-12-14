from pathlib import Path
import os
import pandas as pd

_here = os.path.abspath(os.path.dirname(__file__))
emme_database_dir = os.path.normpath(os.path.join(_here, "../tests/data"))

class FileNames:

	def __init__(self, emme_database_dir, omx_skims_dir=None, cache_dir=None):
		self.emme_database_dir = Path(emme_database_dir)
		if omx_skims_dir is not None:
			omx_skims_dir = Path(omx_skims_dir)
		self.omx_skims_dir = omx_skims_dir
		if cache_dir is not None:
			cache_dir = Path(cache_dir)
			if not os.path.exists(cache_dir):
				os.makedirs(cache_dir)
		self.cache_dir = cache_dir
		self.zone_shapefile = None

	def __getattr__(self, item):
		if item[-6:] == "_DISTR":
			return self.emme_database_dir / f"{item}.TXT"
		if item[-4:] == "_M01":
			return self.emme_database_dir / f"{item}.TXT"
		if item[-5:] == "_M023":
			return self.emme_database_dir / f"{item}.TXT"
		if item[-5:] == "_M023":
			return self.emme_database_dir / f"{item}.TXT"
		if item[-6:] == "_skims":
			if self.omx_skims_dir:
				return self.omx_skims_dir / f"{item}.omx"
			else:
				return self.emme_database_dir / f"{item}.omx"
		raise AttributeError(item)

	def save(self, name, data):
		try:
			if isinstance(data, pd.DataFrame):
				if self.cache_dir:
					pth = self.cache_dir / f"{name}.pq"
				else:
					pth = f"{name}.pq"
				data.to_parquet(pth)
			else:
				raise TypeError(str(type(data)))
		except Exception as err:
			import warnings
			warnings.warn(f"failed to save {name}: {err}")

	def load(self, name):
		if self.cache_dir:
			pth = self.cache_dir / f"{name}.pq"
		else:
			pth = f"{name}.pq"
		if os.path.exists(pth):
			return pd.read_parquet(pth)
		# else file does not exist
		return None

filenames = FileNames(emme_database_dir)

def set_database_dir(emme_database_dir):
	global filenames
	if emme_database_dir is None:
		raise ValueError("emme_database_dir cannot be None")
	filenames.emme_database_dir = Path(emme_database_dir)

def set_skims_dir(omx_skims_dir):
	global filenames
	if omx_skims_dir:
		filenames.omx_skims_dir = Path(omx_skims_dir)
	else:
		filenames.omx_skims_dir = None

def set_cache_dir(cache_dir):
	global filenames
	if cache_dir:
		filenames.cache_dir = Path(cache_dir)
		if not os.path.exists(filenames.cache_dir):
			os.makedirs(filenames.cache_dir)
	else:
		filenames.cache_dir = None

def set_zone_shapefile(shpfilename):
	global filenames
	filenames.zone_shapefile = Path(shpfilename)

def save(*args):
	global filenames
	return filenames.save(*args)

def load(*args):
	global filenames
	return filenames.load(*args)