import numpy as np
import pandas as pd
import larch
from larch import P,X
from larch.util.data_expansion import piecewise_linear
from addict import Dict
from .cmap_logging import getLogger
from .timeperiods import timeperiod_names

log = getLogger()

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})

mode5names = ['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT']
mode5codes = Dict(zip(
	mode5names,
	np.arange(len(mode5names)) + 1,
))


def alt_codes_and_names(
		n_timeperiods=None,
		n_sampled_dests=5,
		modenames=None,
		include_actual_dest=True,
):
	if n_timeperiods is None:
		n_timeperiods = len(timeperiod_names)
	if modenames is None:
		modenames = mode5names
	n_modes = len(modenames)
	alt_codes = np.arange(n_timeperiods * n_modes * (n_sampled_dests + 1)) + 1
	if n_timeperiods > 1:
		alt_names = []
		for t in range(n_timeperiods):
			alt_names.extend([f"{i}_{timeperiod_names[t]}" for i in mode5names])
		for i in range(n_sampled_dests):
			for t in range(n_timeperiods):
				alt_names.extend([(j + f"d{i + 1:04d}_{timeperiod_names[t]}") for j in mode5names])
		if not include_actual_dest:
			alt_codes = alt_codes[n_modes*n_timeperiods:]
			alt_names = alt_names[n_modes*n_timeperiods:]
	else:
		alt_names = [i for i in mode5names]
		for i in range(n_sampled_dests):
			alt_names.extend([(j + f"d{i + 1:04d}") for j in mode5names])
		if not include_actual_dest:
			alt_codes = alt_codes[n_modes:]
			alt_names = alt_names[n_modes:]
	return alt_codes, alt_names


def model_utility_for_dest(
		m,
		dest_number,
		n_timeperiods=8,
		n_modes=5,
		auto_cost_per_mile=15,
):
	"""

	Parameters
	----------
	m
	dest_number : int
		The number of the destination.  In application, this is
		the TAZ index (TAZ ID minus 1).  In estimation, this is
		the sampling slot, or for actual destination, give -1.
	n_timeperiods
	n_modes

	Returns
	-------

	"""
	if dest_number == -1:
		dest_label = "actualdest"
	else:
		dest_label = f'altdest{dest_number + 1:04d}'

	alts_per_dest = n_modes * n_timeperiods
	utility_destination = (
			+ P("samp_af") * X(f"log(1/{dest_label}_samp_wgt)")
			+ P("log_attraction") * X(f"log({dest_label}_attractions)")
			+ P("intrazonal") * X(f"o_zone == {dest_label}")
			+ piecewise_linear(f"{dest_label}_auto_dist", "distance", breaks=[5, 10])
	)
	for t in range(n_timeperiods):
		tname = timeperiod_names[t]
		shift = (t * n_modes) + ((dest_number+1) * alts_per_dest)
		jAUTO = mode5codes.AUTO + shift
		jTNC1 = mode5codes.TNC1 + shift
		jTNC2 = mode5codes.TNC2 + shift
		jTAXI = mode5codes.TAXI + shift
		jTRANSIT = mode5codes.TRANSIT + shift
		utility_timeperiod = P(f"Time-{tname}") if t != 4 else ()
		m.utility_co[jAUTO] = (
				+ P("cost") * X(f"{dest_label}_auto_dist_{tname}") * auto_cost_per_mile / 100.0
				+ P("auto_time") * X(f"{dest_label}_auto_time_{tname}")
				+ P("cost") * X(f"{dest_label}_auto_parking_cost")
				# TODO add walk terminal time cost
			) + utility_destination + utility_timeperiod
		m.utility_co[jTNC1] = (
				P.Const_TNC1
				+ P("cost") * X(f"{dest_label}_tnc_solo_fare_{tname}") / 100.0
				+ P("ovtt") * X(f"{dest_label}_tnc_solo_wait_time")
				+ P("tnc_time") * X(f"{dest_label}_auto_time_{tname}")
			) + utility_destination + utility_timeperiod
		m.utility_co[jTNC2] = (
				P.Const_TNC2
				+ P("cost") * X(f"{dest_label}_tnc_pool_fare_{tname}") / 100.0
				+ P("ovtt") * X(f"{dest_label}_tnc_pool_wait_time")
				+ P("tnc_time") * X(f"{dest_label}_auto_time_{tname}")
			) + utility_destination + utility_timeperiod
		m.utility_co[jTAXI] = (
				P.Const_TAXI
				+ P("cost") * X(f"{dest_label}_taxi_fare_{tname}") / 100.0
				+ P("ovtt") * X(f"{dest_label}_taxi_wait_time")
				+ P("tnc_time") * X(f"{dest_label}_auto_time_{tname}")
			) + utility_destination + utility_timeperiod
		m.utility_co[jTRANSIT] = (
				P.Const_Transit
				+ P("cost") * X(f"{dest_label}_transit_fare_{tname}") / 100.0
				+ P("transit_ivtt") * X(f"{dest_label}_transit_ivtt_{tname}")
				+ P("ovtt") * X(f"{dest_label}_transit_ovtt_{tname}")
				+ P("cost") * X(f"{dest_label}_transit_approach_cost") / 100.0
				+ P("transit_ivtt") * X(f"{dest_label}_transit_approach_drivetime")
				+ P("ovtt") * X(f"{dest_label}_transit_approach_walktime")
				+ P("ovtt") * X(f"{dest_label}_transit_approach_waittime")
			) + utility_destination + utility_timeperiod

	## IMPORTANT be sure to change `nests_per_dest` elswwhere (including estimation code)
	#            if/when the number of nests per destination is altered here
	time_nests = []
	for mode_j in range(5):
		time_nests.append(m.graph.new_node(
			parameter="Mu-Timeperiod",
			children=[
				mode_j + (t * n_modes) + ((dest_number + 1) * alts_per_dest) + 1
				for t in range(n_timeperiods)
			],
			name=f"{mode5names[mode_j]}-{dest_label}",
		))

	hired_car = m.graph.new_node(
		parameter="Mu-HiredCar",
		children=[time_nests[1], time_nests[2], time_nests[3]],
		name=f"hiredcar-{dest_label}",
	)
	m.graph.new_node(
		parameter="Mu-Dest",
		children=[time_nests[0], hired_car, time_nests[4]],
		name=f"{dest_label}",
	)


def model_builder(
		purpose,
		include_actual_dest=True,
		n_sampled_dests=5,
		parameter_values=None,
		auto_cost_per_mile=30, # cents
):
	log.debug(f"model_builder({purpose}, n_sampled_dests={n_sampled_dests})")

	n_timeperiods = 8
	n_modes = 5 # len(mode5names)

	alt_codes, alt_names = alt_codes_and_names(
		n_timeperiods=n_timeperiods,
		n_sampled_dests=n_sampled_dests,
		modenames=mode5names,
		include_actual_dest=include_actual_dest,
	)

	dummy_dfs = larch.DataFrames(
		alt_codes=alt_codes,
		alt_names=alt_names,
	)

	# Define the alternative availability for each alternative in this model.
	av = {}
	dzone_has_nonzero_attractions = "actualdest_attractions > 1e-290"
	if include_actual_dest:
		for t in range(n_timeperiods):
			tshift = t*n_modes
			av[mode5codes.AUTO+tshift] = dzone_has_nonzero_attractions
			av[mode5codes.TNC1+tshift] = dzone_has_nonzero_attractions
			av[mode5codes.TNC2+tshift] = dzone_has_nonzero_attractions
			av[mode5codes.TAXI+tshift] = dzone_has_nonzero_attractions
			av[mode5codes.TRANSIT+tshift] = (
				f"(actualdest_transit_ivtt_{timeperiod_names[t]} < 999) "
				f"& (actualdest_transit_approach_walktime < 999) "
				f"& (actualdest_transit_approach_drivetime < 999) "
				f"& ({dzone_has_nonzero_attractions})"
			)
	num = n_modes*n_timeperiods
	for i in range(n_sampled_dests):
		for t in range(n_timeperiods):
			tshift = t*n_modes
			altdest_has_nonzero_attractions = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.AUTO+tshift] = altdest_has_nonzero_attractions
			av[num + mode5codes.TNC1+tshift] = altdest_has_nonzero_attractions
			av[num + mode5codes.TNC2+tshift] = altdest_has_nonzero_attractions
			av[num + mode5codes.TAXI+tshift] = altdest_has_nonzero_attractions
			av[num + mode5codes.TRANSIT+tshift] = f"altdest{i + 1:04d}_transit_avail_{timeperiod_names[t]}"
		num += n_modes*n_timeperiods


	m = larch.Model(
		dataservice=dummy_dfs
	)

	m.title = f"{purpose} Mode & Destination"

	m.availability_co_vars = av

	if include_actual_dest:
		model_utility_for_dest(
			m,
			dest_number=-1,
			n_timeperiods=n_timeperiods,
			n_modes=n_modes,
			auto_cost_per_mile=auto_cost_per_mile,
		)
		# utility_destination = (
		# 		+ P("samp_af") * X(f"log(1/obs_samp_wgt)")
		# 		+ P("log_attraction") * X(f"log(actualdest_attractions)")
		# 		+ P("intrazonal") * X(f"o_zone == d_zone")
		# 		+ piecewise_linear(f"auto_dist", "distance", breaks=[5, 10])
		# )
		# for t in range(n_timeperiods):
		# 	tshift = t * n_modes
		# 	utility_timeperiod = P(f"Time-{timeperiod_names[t]}") if t!=4 else ()
		# 	m.utility_co[mode5codes.AUTO+tshift] = (
		# 			+ P("cost") * X(f"auto_dist") * auto_cost_per_mile / 100.0
		# 			+ P("auto_time") * X(f"auto_time_{timeperiod_names[t]}")
		# 			+ P("cost") * X("auto_parking_cost")
		# 			# TODO add walk terminal time cost
		# 	) + utility_destination + utility_timeperiod
		# 	m.utility_co[mode5codes.TNC1+tshift] = (
		# 			P.Const_TNC1
		# 			+ P("cost") * X(f"tnc_solo_cost") / 100.0
		# 			+ P("tnc_time") * X(f"auto_time_{timeperiod_names[t]}")
		# 			+ P("ovtt") * X("tnc_solo_wait_time")
		# 	) + utility_destination + utility_timeperiod
		# 	m.utility_co[mode5codes.TNC2+tshift] = (
		# 			P.Const_TNC2
		# 			+ P("cost") * X(f"tnc_pool_cost") / 100.0
		# 			+ P("tnc_time") * X(f"auto_time_{timeperiod_names[t]}")
		# 			+ P("ovtt") * X("tnc_pool_wait_time")
		# 	) + utility_destination + utility_timeperiod
		# 	m.utility_co[mode5codes.TAXI+tshift] = (
		# 			P.Const_TAXI
		# 			+ P("cost") * X(f"taxi_fare") / 100.0
		# 			+ P("tnc_time") * X(f"auto_time_{timeperiod_names[t]}")
		# 			+ P("ovtt") * X("taxi_wait_time")
		# 	) + utility_destination + utility_timeperiod
		# 	m.utility_co[mode5codes.TRANSIT+tshift] = (
		# 			P.Const_Transit
		# 			+ P("cost") * X(f"transit_fare") / 100.0
		# 			+ P("cost") * X(f"transit_approach_cost") / 100.0
		# 			+ P("transit_ivtt") * X(f"transit_ivtt")
		# 			+ P("transit_ivtt") * X(f"transit_approach_drivetime")
		# 			+ P("ovtt") * X(f"transit_ovtt")
		# 			+ P("ovtt") * X(f"transit_approach_walktime")
		# 			+ P("ovtt") * X(f"transit_approach_waittime")
		# 	) + utility_destination + utility_timeperiod
		#
		# ## IMPORTANT be sure to change `nests_per_dest` elswwhere (including estimation code)
		# #            if/when the number of nests per destination is altered here
		# time_nests = []
		# for mode_j in range(5):
		# 	time_nests.append(m.graph.new_node(
		# 		parameter="Mu-Timeperiod",
		# 		children=[
		# 			(mode_j + 1 + t * n_modes)
		# 			for t in range(n_timeperiods)
		# 		],
		# 		name=f"mode{mode_j + 1}-actualdest",
		# 	))
		#
		# hired_car = m.graph.new_node(
		# 	parameter="Mu-HiredCar",
		# 	children=[time_nests[1], time_nests[2], time_nests[3]],
		# 	name=f"hiredcar-actualdest",
		# )
		# m.graph.new_node(
		# 	parameter="Mu-Dest",
		# 	children=[time_nests[0], hired_car, time_nests[4]],
		# 	name=f"actualdest",
		# )

	labeler = lambda i: f'altdest{i + 1:04d}'
	alts_per_dest = n_modes * n_timeperiods
	for i in range(n_sampled_dests):
		model_utility_for_dest(
			m,
			dest_number=i,
			n_timeperiods=n_timeperiods,
			n_modes=n_modes,
			auto_cost_per_mile=auto_cost_per_mile,
		)

	# 	utility_destination = (
	# 			+ P("samp_af") * X(f"log(1/altdest{i + 1:04d}_samp_wgt)")
	# 			+ P("log_attraction") * X(f"log(altdest{i + 1:04d}_attractions)")
	# 			+ P("intrazonal") * X(f"o_zone == altdest{i + 1:04d}")
	# 			+ piecewise_linear(f"altdest{i + 1:04d}_auto_dist", "distance", breaks=[5,10])
	# 	)
	# 	for t in range(n_timeperiods):
	# 		tshift = t * n_modes
	# 		jAUTO = i * alts_per_dest + alts_per_dest + mode5codes.AUTO + tshift
	# 		jTNC1 = i * alts_per_dest + alts_per_dest + mode5codes.TNC1 + tshift
	# 		jTNC2 = i * alts_per_dest + alts_per_dest + mode5codes.TNC2 + tshift
	# 		jTAXI = i * alts_per_dest + alts_per_dest + mode5codes.TAXI + tshift
	# 		jTRANSIT = i * alts_per_dest + alts_per_dest + mode5codes.TRANSIT + tshift
	# 		utility_timeperiod = P(f"Time-{timeperiod_names[t]}") if t!=4 else ()
	# 		m.utility_co[jAUTO] = (
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_auto_dist_{timeperiod_names[t]}") * auto_cost_per_mile / 100.0
	# 				+ P("auto_time") * X(f"altdest{i + 1:04d}_auto_time_{timeperiod_names[t]}")
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_auto_parking_cost")
	# 				# TODO add walk terminal time cost
	# 		) + utility_destination + utility_timeperiod
	# 		m.utility_co[jTNC1] = (
	# 				P.Const_TNC1
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_tnc_solo_fare") / 100.0
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_tnc_solo_wait_time")
	# 				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time_{timeperiod_names[t]}")
	# 		) + utility_destination + utility_timeperiod
	# 		m.utility_co[jTNC2] = (
	# 				P.Const_TNC2
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_tnc_pool_fare") / 100.0
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_tnc_pool_wait_time")
	# 				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time_{timeperiod_names[t]}")
	# 		) + utility_destination + utility_timeperiod
	# 		m.utility_co[jTAXI] = (
	# 				P.Const_TAXI
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_taxi_fare") / 100.0
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_taxi_wait_time")
	# 				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time_{timeperiod_names[t]}")
	# 		) + utility_destination + utility_timeperiod
	# 		m.utility_co[jTRANSIT] = (
	# 				P.Const_Transit
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_transit_fare") / 100.0
	# 				+ P("transit_ivtt") * X(f"altdest{i + 1:04d}_transit_ivtt")
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_ovtt")
	# 				+ P("cost") * X(f"altdest{i + 1:04d}_transit_approach_cost") / 100.0
	# 				+ P("transit_ivtt") * X(f"altdest{i + 1:04d}_transit_approach_drivetime")
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_approach_walktime")
	# 				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_approach_waittime")
	# 		) + utility_destination + utility_timeperiod
	# 		# hired_car = m.graph.new_node(
	# 		# 	parameter="Mu-HiredCar",
	# 		# 	children=[jTNC1, jTNC2, jTAXI],
	# 		# 	name=f"hiredcar-altdest{i + 1:04d}",
	# 		# )
	# 		# m.graph.new_node(
	# 		# 	parameter="Mu-Dest",
	# 		# 	children=[jAUTO, hired_car, jTRANSIT],
	# 		# 	name=f"altdest{i + 1:04d}",
	# 		# )
	# for i in range(n_sampled_dests):
	#
	# 	time_nests = []
	# 	for mode_j in range(5):
	# 		time_nests.append(m.graph.new_node(
	# 			parameter="Mu-Timeperiod",
	# 			children=[
	# 				(i * alts_per_dest + alts_per_dest + mode_j+1 + t * n_modes)
	# 				for t in range(n_timeperiods)
	# 			],
	# 			name=f"mode{mode_j+1}-altdest{i + 1:04d}",
	# 		))
	#
	# 	hired_car = m.graph.new_node(
	# 		parameter="Mu-HiredCar",
	# 		children=[time_nests[1], time_nests[2], time_nests[3]],
	# 		name=f"hiredcar-altdest{i + 1:04d}",
	# 	)
	# 	m.graph.new_node(
	# 		parameter="Mu-Dest",
	# 		children=[time_nests[0], hired_car, time_nests[4]],
	# 		name=f"altdest{i + 1:04d}",
	# 	)

	m.lock_value("samp_af", value=1.0)
	m.lock_value("log_attraction", value=1.0)

	m.set_value("cost", maximum=-0.00001)
	m.set_value("auto_time", maximum=-0.01, minimum=-0.03)
	m.set_value("tnc_time", maximum=-0.01, minimum=-0.03)
	m.set_value("transit_ivtt", maximum=-0.01, minimum=-0.03)
	m.set_value("ovtt", maximum=-0.01)
	if parameter_values is None:
		m.set_values(
			cost=-0.0001,
			auto_time=-0.01,
			tnc_time=-0.02,
			transit_ivtt=-0.01,
			ovtt=-0.01,
			Const_TNC1=-1.0,
			Const_TNC2=-1.0,
			Const_Transit=-1.0,
			intrazonal=-0.1,
		)
	else:
		m.set_values(**parameter_values)

	from larch.model.constraints import RatioBound
	m.constraints = [
		RatioBound(P("ovtt"), P("transit_ivtt"), min_ratio=1.5, max_ratio=3.0, scale=1),
		RatioBound(P("Mu-HiredCar"), P("Mu-Dest"), min_ratio=1e-5, max_ratio=1.0, scale=1),
		RatioBound(P("Mu-Timeperiod"), P("Mu-HiredCar"), min_ratio=1e-5, max_ratio=1.0, scale=1),
	]

	return m