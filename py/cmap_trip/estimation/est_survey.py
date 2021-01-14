import logging
log = logging.getLogger('CMAP')
import re
from addict import Dict

import pandas as pd
import numpy as np
from os.path import join as pj
from .est_config import mode_modeled, mode_modeled5
from ..incomes import income_levels_1, income_levels_2
from ..timeperiods import hours_by_timeperiod

import cmap_trip
from .est_data import dh
from cmap_trip.modecodes import (
    N_DIST_TO_TYPES,
    DIST_TO_BUS,
    DIST_TO_CTA_RAIL,
    DIST_TO_METRA,
    DIST_TO_FEEDER_BUS,
    DIST_TO_PARK_N_RIDE_STATION,
)
from cmap_trip.transit_approach import N_TRIP_ENDS, transit_approach
cfg = dh.cfg
skims = dh.skims


TRIPS_CACHE_FILE = "trips_with_ae_v12"

hh = pd.read_csv(dh.SURVEY_DATA_DIR / 'household.csv')
ae = pd.read_csv(dh.AE_DATA_DIR / 'access_egress.csv')

core_cbd_zones = [
	1, 2, 3, 4, 5, 6,
	8, 9, 10, 11, 12, 13,
	14, 15, 16, 17, 18,
	20, 21, 22, 23, 24,
	26, 27, 28, 29, 30,
	32, 33, 34, 35, 36,
	37, 38, 39, 40, 41,
	43, 44, 45, 46, 47,
]

trips = dh.filenames.load(TRIPS_CACHE_FILE)
if trips is None:

	trips = pd.read_csv(dh.SURVEY_DATA_DIR / 'trips.csv')

	# Convert trips to 3 modes only (auto, transit, tnc), and drop all other trips
	trips['mode3'] = trips['mode'].map(mode_modeled).astype('category')
	trips['mode5'] = trips['mode'].map(mode_modeled5).astype('category')
	trips_filter = ~trips['mode3'].isna()

	trips = trips.merge(ae, how='left', on=['sampno', 'perno', 'placeno'])

	oz = 'o_zone'
	dz = 'd_zone'
	pz = 'p_zone'
	az = 'a_zone'


	# develop production & attraction zones
	trips[pz] = trips[oz]
	trips[az] = trips[dz]

	flip_pa = (trips['PA'] == 0)
	# trips.loc[flip_pa, [pz,az]] = trips.loc[flip_pa, [az,pz]].values
	trips_filter &= ~trips[pz].isna()
	trips_filter &= ~trips[az].isna()

	# Filter trips based on rules
	trips = trips[trips_filter].copy()

	# Convert zone codes to integers
	trips[[pz,az,oz,dz]] = trips[[pz,az,oz,dz]].astype(int)

	# Convert tripCat to category
	trips['tripCat'] = trips['tripCat'].astype('category')

	trips['tripPurpose'] = trips['tripCat'].map({
		'HBWL': 'HW',
		'HBWH': 'HW',
		'HBO': 'HO',
		'NHB': 'NH',
	})

	# hour_re = re.compile(r'.*\s([0-9]+):[0-9]+:[0-9]+')
	# def get_hour(y):
	# 	match = hour_re.match(y)
	# 	if match:
	# 		return int(match.group(1))
	# 	else:
	# 		return -1
	trips['arrHalfHour'] = np.floor(trips['arrHour']*2)/2
	trips['depHalfHour'] = np.floor(trips['depHour']*2)/2
	trips['in_am_peak'] = trips['depHour'].between(6.5,9.5) | trips['arrHour'].between(6.5,9.5)
	trips['in_pm_peak'] = trips['depHour'].between(14.5,19) | trips['arrHour'].between(14.5,19)
	trips['in_peak'] = trips['in_am_peak'] | trips['in_pm_peak']

	trips['timeperiod'] = np.floor(trips.depHour).astype(int).map(hours_by_timeperiod)

	hhinc_dollars = trips.hhinc.map(income_levels_1)
	hhinc_dollars = hhinc_dollars.fillna(trips.hhinc2.map(income_levels_2))
	hhinc_dollars = hhinc_dollars.fillna(57_999)
	trips['hhinc_dollars'] = hhinc_dollars

	# Append skims to trips
	for k, s in skims.items():
		if k not in ('auto', 'transit_pk', 'transit_op'): continue
		log.debug(f"adding skims for {k}")
		trips_1 = s.raw.get_rc_dataframe(trips[pz]-1, trips[az]-1)
		renames = dict([(value, f"{k}_{key}") for key, value in s.col_mapping.items()])
		trips_2 = trips_1[s.col_mapping.values()].rename(columns=renames)
		trips = pd.concat([trips, trips_2], axis=1)

	# Append reverse transit skims to trips
	for k, s in skims.items():
		if k not in ('transit_pk', 'transit_op'): continue
		log.debug(f"adding reverse skims for {k}")
		trips_1 = s.raw.get_rc_dataframe(trips[pz]-1, trips[az]-1)
		renames = dict([(value, f"{k}_{key}_reverse") for key, value in s.col_mapping.items()])
		trips_2 = trips_1[s.col_mapping.values()].rename(columns=renames)
		trips = pd.concat([trips, trips_2], axis=1)

	# Match time period skims
	peak = trips['in_peak']
	hw = trips['tripPurpose'] == 'HW'
	auto_cols = ['time','dist']
	for a in auto_cols:
		trips[f'auto_{a}'] = trips[f'auto_md_{a}']
		trips.loc[peak,f'auto_{a}'] = trips.loc[peak,f'auto_am_{a}']

	transit_cols = ['ivtt','ovtt','headway','fare','firstmode','lastmode','prioritymode']
	for a in transit_cols:
		trips[f'transit_{a}'] = trips[f'transit_op_{a}']
		trips.loc[peak,f'transit_{a}'] = trips.loc[peak,f'transit_pk_{a}']
		# convert invalid to nan
		to_nan = trips[f'transit_{a}'] >= 9999
		trips.loc[to_nan,f'transit_{a}'] = np.nan

	# trips = trips.drop([f'auto_am_{a}' for a in auto_cols], axis=1)
	# trips = trips.drop([f'auto_md_{a}' for a in auto_cols], axis=1)
	# trips = trips.drop([f'transit_pk_{a}' for a in transit_cols], axis=1)
	# trips = trips.drop([f'transit_op_{a}' for a in transit_cols], axis=1)


	# attach transit_approach model data
	for t in ['single', 'max', 'mean']:
		trips[f'ae_drivetime_{t}'] = 0
		trips[f'ae_walktime_{t}'] = 0
		trips[f'ae_cost_{t}'] = 0
		trips[f'ae_waittime_{t}'] = 0

	trips[f'auto_parking_cost'] = 0
	from ..transit_approach import transit_approach
	from ..parking_costs import parking_cost_v2

	for purpose in ['HW','HO','NH']:
		q = (trips.tripPurpose == purpose)
		_trips_by_purpose = trips[q]
		result_purpose = transit_approach(
			dh,
			_trips_by_purpose.o_zone,
			_trips_by_purpose.d_zone,
			purpose,
			replication=50,
			approach_distances=None,
			trace=False,
			random_state=hash(purpose),
		)

		for key in ['drivetime', 'walktime', 'cost', 'waittime']:
			trips.loc[q, f'ae_{key}_max'] = result_purpose[key].max(1)
			trips.loc[q, f'ae_{key}_mean'] = result_purpose[key].mean(1)
			trips.loc[q, f'ae_{key}_single'] = result_purpose[key][:,0]

		# Attach parking costs
		_parking_cost, _free_parking = parking_cost_v2(
			dh,
			_trips_by_purpose.d_zone,
			_trips_by_purpose.hhinc_dollars,
			cfg.default_activity_durations[purpose],
			purpose,
			random_state=hash(purpose)+1,
		)
		trips.loc[q, f'auto_parking_cost'] = _parking_cost
		trips.loc[q, f'auto_parking_free'] = _free_parking

	# for n in trips.index:
	# 	out = transit_approach(
	# 		trips.o_zone[n], trips.d_zone[n], 'HW' if trips.in_peak[n] else 'HO', replication=50
	# 	)
	# 	trips.loc[n, 'ae_drivetime_max'] = out.drivetime.max()
	# 	trips.loc[n, 'ae_walktime_max'] = out.walktime.max()
	# 	trips.loc[n, 'ae_cost_max'] = out.cost.max()
	# 	trips.loc[n, 'ae_waittime_max'] = out.waittime.max()
	# 	trips.loc[n, 'ae_drivetime_mean'] = out.drivetime.mean()
	# 	trips.loc[n, 'ae_walktime_mean'] = out.walktime.mean()
	# 	trips.loc[n, 'ae_cost_mean'] = out.cost.mean()
	# 	trips.loc[n, 'ae_waittime_mean'] = out.waittime.mean()
	# 	trips.loc[n, 'ae_drivetime_single'] = out.drivetime[0]
	# 	trips.loc[n, 'ae_walktime_single'] = out.walktime[0]
	# 	trips.loc[n, 'ae_cost_single'] = out.cost[0]
	# 	trips.loc[n, 'ae_waittime_single'] = out.waittime[0]

	trips['touches_core_cbd'] = trips['o_zone'].isin(core_cbd_zones) | trips['d_zone'].isin(core_cbd_zones)
	trips['transit_trip_not_in_skim'] = trips.eval("~(0 < transit_ivtt < 999) and mode3=='TRANSIT'")
	trips['crow_distance'] = np.sqrt((trips.X_coord1 - trips.X_coord2)**2 + (trips.Y_coord1 - trips.Y_coord2)**2) / 5280

	dh.filenames.save(TRIPS_CACHE_FILE, trips)

# columns not compatible with Parquet
trips['crow_distance_q5'] = pd.qcut(trips.crow_distance, 5)



def attach_selected_skims(
		od_df,
		o_col,
		d_col,
		omx,
		skim_cols,
):
	"""
	Attach selected columns from an OMX file to a DataFrame.

	Parameters
	----------
	od_df : pandas.DataFrame
	o_col, d_col : str
		Names of the O and D columns in `od_df`.
	omx : OMX
	skim_cols : Dict[str, Dict[str, str]]
		Top level keys of this dict give filters on `od_df` that
		ideally give a mutually exclusive and collectively exhaustive
		partition.  Top level values are dicts, from which the keys
		are the skim columns to pull for this partition group and the
		values are the names of the ultimate columns to attach.
		Final column value namess in all partition groups should be
		the same to populate a common set of columns in the output.

	Returns
	-------
	pandas.DataFrame
	"""
	cols_to_add = {}
	for filter_qry, use_cols in skim_cols.items():
		for c in use_cols.values():
			if c not in od_df.columns:
				cols_to_add[c] = np.nan
	df = od_df.assign(**cols_to_add)
	for filter_qry, use_cols in skim_cols.items():
		if filter_qry:
			group = od_df.query(filter_qry)
		else:
			group = od_df
		s = omx.get_rc_dataframe(
			group[o_col] - 1,
			group[d_col] - 1,
			use_cols,
		)
		df.update(s)
	return df



# pMode == 3
#    walk -> DIST_TO_BUS
#
# pMode == 6
#    bus -> DIST_TO_BUS
#    walk -> DIST_TO_CTA_RAIL
#    pnr -> DIST_TO_PARK_N_RIDE_STATION
#
# pMode == 7
#    bus -> DIST_TO_BUS
#    walk -> DIST_TO_METRA
#    pnr -> DIST_TO_PARK_N_RIDE_STATION
#    feeder -> DIST_TO_FEEDER_BUS

def ae_approach_los(trips):
	approach_distances = np.full([len(trips), N_DIST_TO_TYPES, N_TRIP_ENDS], 999.0)
	fm3 = trips.transit_firstmode == 3
	approach_distances[fm3, DIST_TO_BUS, 0] = trips[fm3].walkAccDistance
	fm6 = trips.transit_firstmode == 6
	approach_distances[fm6, DIST_TO_BUS, 0] = trips[fm6].busAccDistance
	approach_distances[fm6, DIST_TO_CTA_RAIL, 0] = trips[fm6].walkAccDistance
	approach_distances[fm6, DIST_TO_PARK_N_RIDE_STATION, 0] = trips[fm6].pnrAccDistance
	fm7 = trips.transit_firstmode == 7
	approach_distances[fm7, DIST_TO_BUS, 0] = trips[fm7].busAccDistance
	approach_distances[fm7, DIST_TO_METRA, 0] = trips[fm7].walkAccDistance
	approach_distances[fm7, DIST_TO_PARK_N_RIDE_STATION, 0] = trips[fm7].pnrAccDistance
	approach_distances[fm7, DIST_TO_FEEDER_BUS, 0] = trips[fm7].feederAccDistance

	lm3 = trips.transit_lastmode == 3
	approach_distances[lm3, DIST_TO_BUS, 1] = trips[lm3].walkEgrDistance
	lm6 = trips.transit_lastmode == 6
	approach_distances[lm6, DIST_TO_BUS, 1] = trips[lm6].busEgrDistance
	approach_distances[lm6, DIST_TO_CTA_RAIL, 1] = trips[lm6].walkEgrDistance
	lm7 = trips.transit_lastmode == 7
	approach_distances[lm7, DIST_TO_BUS, 1] = trips[lm7].busEgrDistance
	approach_distances[lm7, DIST_TO_METRA, 1] = trips[lm7].walkEgrDistance

	result = Dict()
	for purpose in ['HW','HO','NH']:
		q = (trips.tripPurpose == purpose)
		_trips_by_purpose = trips[q]
		result[purpose] = transit_approach(
			dh,
			_trips_by_purpose.o_zone,
			_trips_by_purpose.d_zone,
			purpose,
			replication=1,
			approach_distances=approach_distances[q],
			trace=False,
			random_state=123,
		)

		for key in ['drivetime', 'walktime', 'cost', 'waittime']:
			trips.loc[q,f'transit_approach_{key}'] = result[purpose][key].reshape(-1)
		trips.loc[q,f'transit_approach_acc_mode'] = result[purpose]['approach_mode'][...,0].reshape(-1)
		trips.loc[q,f'transit_approach_egr_mode'] = result[purpose]['approach_mode'][...,1].reshape(-1)

	ad = pd.DataFrame(
		approach_distances.reshape([-1, 10]),
		columns=[
			'acc_DIST_TO_BUS',
			'egr_DIST_TO_BUS',
			'acc_DIST_TO_CTA_RAIL',
			'egr_DIST_TO_CTA_RAIL',
			'acc_DIST_TO_METRA',
			'egr_DIST_TO_METRA',
			'acc_DIST_TO_FEEDER_BUS',
			'egr_DIST_TO_FEEDER_BUS',
			'acc_DIST_TO_PARK_N_RIDE_STATION',
			'egr_DIST_TO_PARK_N_RIDE_STATION',
		],
		index=trips.index,
	)
	for c in ad.columns:
		trips[c] = ad[c]

	return approach_distances, result
