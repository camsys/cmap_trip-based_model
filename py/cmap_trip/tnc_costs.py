import numpy as np
import pandas as pd


def taxi_cost(dh, auto_time, auto_dist, o_zone, d_zone):
	"""
	Compute taxi fare.

	A single set of rates (Chicago medallion rates for in-city trips)
	is used; fares for taxi trips outside Chicago are close to this
	rate and rare enough that more precision is unneeded.

	Parameters
	----------
	auto_time, auto_dist : array-like

	Returns
	-------
	fare : array-like
	"""
	return (
		dh.cfg.taxi.cost.flag_pull
		+ auto_time * dh.cfg.taxi.cost.per_minute
		+ auto_dist * dh.cfg.taxi.cost.per_mile
	)


def tnc_solo_cost(dh, auto_time, auto_dist, o_zone, d_zone):
	"""
	Compute the solo rider TNC cost.

	Parameters
	----------
	auto_time, auto_dist : array-like of float
		The auto travel time and distance for a set of trips/
		Shapes must match.
	o_zone, d_zone : pd.Series of int
		Zone numbers for origin and destination
		Shapes must match `auto_time`.

	Returns
	-------
	ndarray
		Same shape as inputs
	"""

	fare = (
		dh.cfg.tnc.cost.per_minute * auto_time
		+ dh.cfg.tnc.cost.per_mile * auto_dist
		+ dh.cfg.tnc.cost.base_fare
	)
	fare = np.fmax(fare, dh.cfg.tnc.cost.min_fare)
	cost = fare + dh.cfg.tnc.cost.booking_fee
	for bucket_name, bucket_price in dh.cfg.tnc.surcharge_rates.items():
		if bucket_price:
			bucket_applies = (
					o_zone.isin(dh.cfg.tnc.surcharge_zones[bucket_name])
					| d_zone.isin(dh.cfg.tnc.surcharge_zones[bucket_name])
			).astype(float)
			cost += bucket_applies * bucket_price
	return cost


def tnc_pool_cost(dh, auto_time, auto_dist, o_zone, d_zone):
	"""
	Compute the pooled rider TNC cost.

	Parameters
	----------
	auto_time, auto_dist : array-like of float
		The auto travel time and distance for a set of trips/
		Shapes must match.
	o_zone, d_zone : pd.Series of int
		Zone numbers for origin and destination
		Shapes must match `auto_time`.

	Returns
	-------
	ndarray
		Same shape as inputs
	"""

	fare = (
		dh.cfg.tnc_pooled.cost.per_minute * auto_time
		+ dh.cfg.tnc_pooled.cost.per_mile * auto_dist
		+ dh.cfg.tnc_pooled.cost.base_fare
	)
	fare = np.fmax(fare, dh.cfg.tnc_pooled.cost.min_fare)
	cost = fare + dh.cfg.tnc_pooled.cost.booking_fee
	for bucket_name, bucket_price in dh.cfg.tnc_pooled.surcharge_rates.items():
		if bucket_price:
			bucket_applies = (
					o_zone.isin(dh.cfg.tnc_pooled.surcharge_zones[bucket_name])
					| d_zone.isin(dh.cfg.tnc_pooled.surcharge_zones[bucket_name])
			).astype(float)
			cost += bucket_applies * bucket_price
	return cost
