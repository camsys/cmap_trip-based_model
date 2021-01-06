import numpy as np
import logging
log = logging.getLogger('CMAP')


"""
# Mode Choice Model Impedances

## Highway Costs

### Times and distances

Skims:
HBW trips
	Times – mf44 (SOV AM peak), mf76 (HOV AM peak)
	Distances - mf45 (SOV AM peak), mf77 (HOV AM peak)

HBO/NHB trips
Times – mf46 (SOV midday)
Distances - mf47 (SOV midday)

### Parking Costs

HBW trips to the Central Area – Monte Carlo simulation using
costs from MCHW_CBDPARK.TXT

All other trips – zonal Park and Ride cost from MCxx_M01.TXT

Hours of parking: HBW=10, HBO=6, NHB=3

### Auto Operating Costs

Supplied by MCxx_M023.TXT in 5 MPH increments

## Transit Costs

Times and fares	Skims:
HBW trips
In-vehicle time – mf822 (AM peak)
Out of vehicle time – mf823 (AM peak) [walk transfer but not access/egress]
Headway – mf838 (AM peak)
Fare – mf828 (AM peak)

HBO/NHB trips
In-vehicle time – mf922 (midday)
Out of vehicle time – mf923 (midday) [walk transfer but not access/egress]
Headway – mf938 (midday)
Fare – mf928 (midday)

Access/egress are simulated for each traveler.
	First mode and last mode are obtained from transit skims
	(mf829|mf929 and mf831|mf931) – these are used to determine
	which of the 5 access/egress modes (walk, bus, feeder bus, P&R, K&R)
	are simulated.  That information is used to pull the appropriate data
	from MCxx_DISTR: average zonal distance and standard deviation of
	the modes to simulate actual distance.

"""

from addict import Dict
import larch
import os
from os.path import join as pj

def load_skims(filenames):

	skims = Dict()

	skims.auto.raw = larch.OMX(filenames.auto_skims)
	log.debug(repr(skims.auto.raw))
	skims.auto.col_mapping = dict(
		am_time='mf44_amtime',
		am_dist='mf45_amdist',
		md_time='mf46_mdtime',
		md_dist='mf47_mddist',
		am_time_hov='mf76_amhovt',
		am_dist_hov='mf77_amhovd',
	)

	def pick_in(x, *arg):
		for a in arg:
			if a in x.data:
				return a
		raise KeyError(arg)

	skims.transit_pk.raw = larch.OMX(filenames.pk_transit_skims)
	log.debug(repr(skims.transit_pk.raw))
	skims.transit_pk.col_mapping = Dict(
		ivtt=        pick_in(skims.transit_pk.raw, 'mf822_min'   , 'mf822'),
		ovtt=        pick_in(skims.transit_pk.raw, 'mf823_min'   , 'mf823'),
		headway=     pick_in(skims.transit_pk.raw, 'mf838_phdway', 'mf838'),
		fare=        pick_in(skims.transit_pk.raw, 'mf828_$'     , 'mf828'),
		firstmode=   pick_in(skims.transit_pk.raw, 'mf829_$'     , 'mf829'),
		prioritymode=pick_in(skims.transit_pk.raw, 'mf830_$'     , 'mf830'),
		lastmode=    pick_in(skims.transit_pk.raw, 'mf831_$'     , 'mf831'),
	)

	skims.transit_op.raw = larch.OMX(filenames.op_transit_skims)
	log.debug(repr(skims.transit_op.raw))
	skims.transit_op.col_mapping = Dict(
		ivtt=        pick_in(skims.transit_op.raw, 'mf922_min'   , 'mf922'),
		ovtt=        pick_in(skims.transit_op.raw, 'mf923_min'   , 'mf923'),
		headway=     pick_in(skims.transit_op.raw, 'mf938_phdway', 'mf938'),
		fare=        pick_in(skims.transit_op.raw, 'mf928_$'     , 'mf928'),
		firstmode=   pick_in(skims.transit_op.raw, 'mf929_$'     , 'mf929'),
		prioritymode=pick_in(skims.transit_op.raw, 'mf930_$'     , 'mf930'),
		lastmode=    pick_in(skims.transit_op.raw, 'mf931_$'     , 'mf931'),
	)


	skims.first_mode_peak       = skims.transit_pk.raw[pick_in(skims.transit_pk.raw, 'mf829_$', 'mf829')][:].astype(np.int8)+1
	skims.priority_mode_peak    = skims.transit_pk.raw[pick_in(skims.transit_pk.raw, 'mf830_$', 'mf830')][:].astype(np.int8)+1
	skims.last_mode_peak        = skims.transit_pk.raw[pick_in(skims.transit_pk.raw, 'mf831_$', 'mf831')][:].astype(np.int8)+1
	skims.first_mode_offpeak    = skims.transit_op.raw[pick_in(skims.transit_op.raw, 'mf929_$', 'mf929')][:].astype(np.int8)+1
	skims.priority_mode_offpeak = skims.transit_op.raw[pick_in(skims.transit_op.raw, 'mf930_$', 'mf930')][:].astype(np.int8)+1
	skims.last_mode_offpeak     = skims.transit_op.raw[pick_in(skims.transit_op.raw, 'mf931_$', 'mf931')][:].astype(np.int8)+1

	return skims
