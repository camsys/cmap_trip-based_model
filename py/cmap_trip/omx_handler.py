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
from .filepaths import filenames
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
# skims.auto.am.time = skims.auto.raw.mf44_amtime
# skims.auto.am.dist = skims.auto.raw.mf45_amdist
# skims.auto.md.time = skims.auto.raw.mf46_mdtime
# skims.auto.md.dist = skims.auto.raw.mf47_mddist
# skims.auto.am.time_hov = skims.auto.raw.mf76_amhovt
# skims.auto.am.dist_hov = skims.auto.raw.mf77_amhovd
# skims.auto.md.time_hov = skims.auto.raw.mf46_mdtime
# skims.auto.md.dist_hov = skims.auto.raw.mf47_mddist

skims.transit_pk.raw = larch.OMX(filenames.pk_transit_skims)
log.debug(repr(skims.transit_pk.raw))
skims.transit_pk.col_mapping = dict(
	ivtt='mf822_min',
	ovtt='mf823_min',
	headway='mf838_phdway',
	fare='mf828_$',
	firstmode='mf829_$',
	prioritymode='mf830_$',
	lastmode='mf831_$',
)
# skims.transit.am.ivtt = skims.transit_pk.raw.mf822_min
# skims.transit.am.ovtt = skims.transit_pk.raw.mf823_min
# skims.transit.am.headway = skims.transit_pk.raw.mf838_phdway
# skims.transit.am.fare = skims.transit_pk.raw['mf828_$']
# skims.transit.am.firstmode = skims.transit_pk.raw['mf829_$']
# skims.transit.am.lastmode = skims.transit_pk.raw['mf831_$']

skims.transit_op.raw = larch.OMX(filenames.op_transit_skims)
log.debug(repr(skims.transit_op.raw))
skims.transit_op.col_mapping = dict(
	ivtt='mf922_min',
	ovtt='mf923_min',
	headway='mf938_phdway',
	fare='mf928_$',
	firstmode='mf929_$',
	prioritymode='mf930_$',
	lastmode='mf931_$',
)
# skims.transit.md.ivtt = skims.transit_op.raw.mf922_min
# skims.transit.md.ovtt = skims.transit_op.raw.mf923_min
# skims.transit.md.headway = skims.transit_op.raw.mf938_phdway
# skims.transit.md.fare = skims.transit_op.raw['mf928_$']
# skims.transit.md.firstmode = skims.transit_op.raw['mf929_$']
# skims.transit.md.lastmode = skims.transit_op.raw['mf931_$']


first_mode_peak = skims.transit_pk.raw['mf829_$'][:].astype(np.int8)+1
priority_mode_peak = skims.transit_pk.raw['mf830_$'][:].astype(np.int8)+1
last_mode_peak = skims.transit_pk.raw['mf831_$'][:].astype(np.int8)+1
first_mode_offpeak = skims.transit_op.raw['mf929_$'][:].astype(np.int8)+1
priority_mode_offpeak = skims.transit_op.raw['mf930_$'][:].astype(np.int8)+1
last_mode_offpeak = skims.transit_op.raw['mf931_$'][:].astype(np.int8)+1

