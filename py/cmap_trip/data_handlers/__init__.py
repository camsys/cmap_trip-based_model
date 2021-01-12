import cloudpickle
import numpy as np
import pandas as pd
import tempfile
import larch
from addict import Dict
from pathlib import Path
try:
	from geopandas import GeoDataFrame
except ImportError:
	GeoDataFrame = ()

from .filepaths import FileNames, PathAttr
from logging import getLogger
log = getLogger("CMAP")

class DataHandler:

	serial_dir = PathAttr()

	def __init__(self, filenames=None, serial_dir=None, **kwargs):
		
		self.artifacts = {}
		# artifacts tell how to reload things if they are not cached
		# this makes it easier to pass this DataHandler to subprocesses,
		# as we don't need to serialize data that is already serialized.

		self._cache = {}
		# the _cache is where we store things that we have already loaded
		# in this process, so they are available in RAM already.

		log.debug("load filenames")
		if filenames is None:
			filenames = FileNames(**kwargs)

		if serial_dir is None:
			log.debug("create serial temp dir")
			self._temporary_dir = tempfile.TemporaryDirectory()
			serial_dir = self._temporary_dir.name

		log.debug("serial_dir init")
		self.serial_dir = serial_dir

		log.debug("filenames init")
		self.filenames = filenames

		log.debug("load distr")
		from .distr_handler import load_distr
		self['distr'] = load_distr(filenames)

		log.debug("load m01")
		from .m01_handler import load_m01
		self['m01'] = load_m01(filenames)

		log.debug("load m023")
		from .m023_handler import load_m023
		self['m023'] = load_m023(filenames)

		log.debug("load shp")
		from .shp_handler import load_zone_shapes
		self['zone_shp'] = load_zone_shapes(filenames)

		log.debug("load skims")
		from .skims_handler import load_skims
		self['skims'] = load_skims(filenames)

		log.debug("load tg")
		from .tg_handler import load_tg
		tg = load_tg(filenames)
		self['trip_attractions'] = tg.trip_attractions
		self['trip_productions'] = tg.trip_productions
		self['zone_productions'] = tg.zone_productions

		log.debug("load parking")
		from .parking_handler import load_cbd_parking
		parking = load_cbd_parking(filenames)
		self['cbd_parking_prices'] = parking.cbd_parking_prices
		self['cbd_parking_price_prob'] = parking.cbd_parking_price_prob
		self['cbd_parking2'] = parking.cbd_parking2
		self['CBD_PARKING_ZONES'] = parking.CBD_PARKING_ZONES

	@property
	def cfg(self):
		return self.filenames.cfg

	@property
	def choice_model_params(self):
		return self.filenames.choice_model_params

	def __getstate__(self):
		return {
			'artifacts': self.artifacts,
			'filenames': self.filenames,
			'_serial_dir': self._serial_dir,
		}

	def __setstate__(self, state):
		self._cache = {}
		self.artifacts = state['artifacts']
		self.filenames = state['filenames']
		self._serial_dir = state['_serial_dir']

	def __setitem__(self, key, value):
		self._cache[key] = value
		if not isinstance(key, str):
			raise ValueError("keys must be str")
		if isinstance(value, GeoDataFrame):
			filename = self.serial_dir / f"{key}.pkl"
			with open(filename, "wb") as f:
				cloudpickle.dump(value, f)
			self.artifacts[key] = ('pickle', filename)
		elif isinstance(value, pd.DataFrame):
			filename = self.serial_dir / f"{key}.pq"
			try:
				value.to_parquet(filename)
			except ValueError:
				filename = self.serial_dir / f"{key}.h5"
				value.to_hdf(filename, key)
				self.artifacts[key] = ('DataFrame.h5', filename)
			else:
				self.artifacts[key] = ('DataFrame', filename)
		elif isinstance(value, pd.Series):
			filename = self.serial_dir / f"{key}.spq"
			pd.DataFrame(value).to_parquet(filename)
			self.artifacts[key] = ('Series', filename)
		elif isinstance(value, np.ndarray):
			filename = self.serial_dir / f"{key}.nm"
			mm = np.memmap(
				filename,
				dtype=value.dtype,
				mode='w+',
				shape=value.shape,
			)
			mm[:] = value[:]
			self.artifacts[key] = ('array', filename, value.dtype, value.shape)
		elif isinstance(value, larch.OMX):
			filename = value.filename
			self.artifacts[key] = ('OMX', filename)
		elif isinstance(value, Dict):
			self.artifacts[key] = ('Dict', set(value.keys()))
			for k1, v1 in value.items():
				self[f"{key}.{k1}"] = v1
		else:
			filename = self.serial_dir / f"{key}.pkl"
			with open(filename, "wb") as f:
				cloudpickle.dump(value, f)
			self.artifacts[key] = ('pickle', filename)

	def __getitem__(self, key):
		if key in self._cache:
			return self._cache[key]
		art = self.artifacts[key]
		if art[0] == 'DataFrame':
			result = pd.read_parquet(art[1])
		elif art[0] == 'DataFrame.h5':
			result = pd.read_hdf(art[1], key)
		elif art[0] == 'Series':
			result = pd.read_parquet(art[1]).iloc[:,0]
		elif art[0] == 'array':
			result = np.memmap(
				art[1],
				dtype=art[2],
				mode='r+',
				shape=art[3],
			)
		elif art[0] == 'OMX':
			result = larch.OMX(art[1], 'r')
		elif art[0] == 'Dict':
			result = Dict()
			for k1 in art[1]:
				result[k1] = self[f"{key}.{k1}"]
		elif art[0] == 'pickle':
			with open(art[1], "rb") as f:
				result = cloudpickle.load(f)
		else:
			raise TypeError(art[0])
		self._cache[key] = result
		return result

	def __getattr__(self, item):
		try:
			return self[item]
		except:
			raise AttributeError(item)

	@property
	def n_internal_zones(self):
		return len(self.m01)

