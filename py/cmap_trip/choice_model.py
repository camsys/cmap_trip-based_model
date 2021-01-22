import numpy as np
import pandas as pd
import larch
from larch import P,X
from larch.util.data_expansion import piecewise_linear
from addict import Dict
from .cmap_logging import getLogger

log = getLogger()

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})

mode5names = ['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT']
mode5codes = Dict(zip(
	mode5names,
	np.arange(len(mode5names)) + 1,
))


def alt_codes_and_names(
		n_sampled_dests=5,
		modenames=None,
		include_actual_dest=True,
):
	if modenames is None:
		modenames = mode5names
	n_modes = len(modenames)
	alt_codes = np.arange(n_modes * (n_sampled_dests + 1)) + 1
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
		purpose,
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
	purpose : str
	n_modes : int

	Returns
	-------

	"""
	if dest_number == -1:
		dest_label = "actualdest"
	else:
		dest_label = f'altdest{dest_number + 1:04d}'

	alts_per_dest = n_modes
	utility_destination = (
			+ P("samp_af") * X(f"log(1/{dest_label}_samp_wgt)")
			+ P("log_attraction") * X(f"{dest_label}_log_attractions_{purpose}")
			+ P("intrazonal") * X(f"o_zone == {dest_label}")
			+ piecewise_linear(f"{dest_label}_auto_dist", "distance", breaks=[5, 10])
	)
	shift = (dest_number+1) * alts_per_dest
	jAUTO = mode5codes.AUTO + shift
	jTNC1 = mode5codes.TNC1 + shift
	jTNC2 = mode5codes.TNC2 + shift
	jTAXI = mode5codes.TAXI + shift
	jTRANSIT = mode5codes.TRANSIT + shift
	peaky = 'PEAK' if 'W' in purpose else 'OFFPEAK'
	purpose3 = {
		'HBWH': 'HW',
		'HBWL': 'HW',
		'HBO': 'HO',
		'NHB': 'NH',
	}
	m.utility_co[jAUTO] = (
			+ P("cost") * X(f"{dest_label}_auto_dist_{peaky}") * auto_cost_per_mile / 100.0
			+ P("auto_time") * X(f"{dest_label}_auto_time_{peaky}")
			+ P("cost") * X(f"{dest_label}_auto_parking_cost_{purpose}")
			# TODO add walk terminal time cost
		) + utility_destination
	m.utility_co[jTNC1] = (
			P.Const_TNC1
			+ P("cost") * X(f"{dest_label}_tnc_solo_fare_{peaky}") / 100.0
			+ P("ovtt") * X(f"{dest_label}_tnc_solo_wait_time")
			+ P("tnc_time") * X(f"{dest_label}_auto_time_{peaky}")
		) + utility_destination
	m.utility_co[jTNC2] = (
			P.Const_TNC2
			+ P("cost") * X(f"{dest_label}_tnc_pool_fare_{peaky}") / 100.0
			+ P("ovtt") * X(f"{dest_label}_tnc_pool_wait_time")
			+ P("tnc_time") * X(f"{dest_label}_auto_time_{peaky}")
		) + utility_destination
	m.utility_co[jTAXI] = (
			P.Const_TAXI
			+ P("cost") * X(f"{dest_label}_taxi_fare_{peaky}") / 100.0
			+ P("ovtt") * X(f"{dest_label}_taxi_wait_time")
			+ P("tnc_time") * X(f"{dest_label}_auto_time_{peaky}")
		) + utility_destination
	m.utility_co[jTRANSIT] = (
			P.Const_Transit
			+ P("cost") * X(f"{dest_label}_transit_fare_{peaky}") / 100.0
			+ P("transit_ivtt") * X(f"{dest_label}_transit_ivtt_{peaky}")
			+ P("ovtt") * X(f"{dest_label}_transit_ovtt_{peaky}")
			+ P("cost") * X(f"{dest_label}_transit_approach_cost_{purpose3[purpose]}") / 100.0
			+ P("transit_ivtt") * X(f"{dest_label}_transit_approach_drivetime_{purpose3[purpose]}")
			+ P("ovtt") * X(f"{dest_label}_transit_approach_walktime_{purpose3[purpose]}")
			+ P("ovtt") * X(f"{dest_label}_transit_approach_waittime_{purpose3[purpose]}")
		) + utility_destination

	## IMPORTANT be sure to change `nests_per_dest` elswwhere (including estimation code)
	#            if/when the number of nests per destination is altered here

	hired_car = m.graph.new_node(
		parameter="Mu-HiredCar",
		children=[jTAXI, jTNC1, jTNC2],
		name=f"hiredcar-{dest_label}",
	)
	m.graph.new_node(
		parameter="Mu-Dest",
		children=[jAUTO, hired_car, jTRANSIT],
		name=f"{dest_label}",
	)


def _lock_value(self, name, value, note=None, change_check=True):
	"""
	Set a fixed value for a model parameter.

	Parameters with a fixed value (i.e., with "holdfast" set to 1)
	will not be changed during estimation by the likelihood
	maximization algorithm.

	Parameters
	----------
	name : str
		The name of the parameter to set to a fixed value.
	value : float
		The numerical value to set for the parameter.
	note : str, optional
		A note as to why this parameter is set to a fixed value.
		This will not affect the mathematical treatment of the
		parameter in any way, but may be useful for reporting.
	change_check : bool, default True
		Whether to trigger a check to see if any parameter frame
		values have changed.  Can be set to false to skip this
		check if you know that the values have not changed or want
		to delay this check for later, but this may result in
		problems if the check is needed but not triggered before
		certain other modeling tasks are performed.

	"""
	name = str(name)
	if value == 'null':
		value = self.pf.loc[name, 'nullvalue']
	self.set_value(name, value, holdfast=1, initvalue=value, nullvalue=value, minimum=value, maximum=value)
	if note is not None:
		self._frame.loc[name, 'note'] = note
	if change_check:
		self._check_if_frame_values_changed()


def model_builder(
		purpose,
		include_actual_dest=True,
		n_sampled_dests=5,
		parameter_values=None,
		auto_cost_per_mile=30, # cents
		constraints=True,
		n_threads=-1,
		application_mode=False,
):
	log.debug(f"model_builder({purpose}, n_sampled_dests={n_sampled_dests})")

	n_modes = 5 # len(mode5names)

	alt_codes, alt_names = alt_codes_and_names(
		n_sampled_dests=n_sampled_dests,
		modenames=mode5names,
		include_actual_dest=include_actual_dest,
	)

	dummy_dfs = larch.DataFrames(
		alt_codes=alt_codes,
		alt_names=alt_names,
	)
	peaky = 'PEAK' if 'W' in purpose else 'OFFPEAK'
	purpose4to3 = {
		'HBWH': 'HW',
		'HBWL': 'HW',
		'HBO': 'HO',
		'NHB': 'NH',
	}

	# Define the alternative availability for each alternative in this model.
	av = {}
	dzone_has_nonzero_attractions = f"actualdest_log_attractions_{purpose} > -666"
	if include_actual_dest:
		av[mode5codes.AUTO] = dzone_has_nonzero_attractions
		av[mode5codes.TNC1] = dzone_has_nonzero_attractions
		av[mode5codes.TNC2] = dzone_has_nonzero_attractions
		av[mode5codes.TAXI] = dzone_has_nonzero_attractions
		av[mode5codes.TRANSIT] = (
			f"(actualdest_transit_ivtt_{peaky} < 999) "
			f"& (actualdest_transit_approach_walktime_{purpose4to3[purpose]} < 999) "
			f"& (actualdest_transit_approach_drivetime_{purpose4to3[purpose]} < 999) "
			f"& ({dzone_has_nonzero_attractions})"
		)
	num = n_modes
	for i in range(n_sampled_dests):
		altdest_has_nonzero_attractions = f"altdest{i + 1:04d}_auto_avail_{purpose}"
		av[num + mode5codes.AUTO] = altdest_has_nonzero_attractions
		av[num + mode5codes.TNC1] = altdest_has_nonzero_attractions
		av[num + mode5codes.TNC2] = altdest_has_nonzero_attractions
		av[num + mode5codes.TAXI] = altdest_has_nonzero_attractions
		av[num + mode5codes.TRANSIT] = f"altdest{i + 1:04d}_transit_avail_{purpose}"
		num += n_modes


	m = larch.Model(
		dataservice=dummy_dfs,
		n_threads=n_threads,
	)

	m.title = f"{purpose} Mode & Destination"

	m.availability_co_vars = av

	if include_actual_dest:
		model_utility_for_dest(
			m,
			dest_number=-1,
			purpose=purpose,
			n_modes=n_modes,
			auto_cost_per_mile=auto_cost_per_mile,
		)

	for i in range(n_sampled_dests):
		model_utility_for_dest(
			m,
			dest_number=i,
			purpose=purpose,
			n_modes=n_modes,
			auto_cost_per_mile=auto_cost_per_mile,
		)

	m.unmangle()
	set1(m)
	set2(m)

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
			transit_ivtt=-0.015,
			ovtt=-0.01,
			Const_TNC1=-1.0,
			Const_TNC2=-1.0,
			Const_Transit=-1.0,
			intrazonal=-0.1,
		)
	else:
		m.set_values(**parameter_values)

	if constraints:
		from larch.model.constraints import RatioBound
		m.constraints = [
			RatioBound(P("ovtt"), P("transit_ivtt"), min_ratio=1.5, max_ratio=3.0, scale=1),
			RatioBound(P("Mu-HiredCar"), P("Mu-Dest"), min_ratio=1e-5, max_ratio=1.0, scale=1),
		]

	if application_mode:
		m._preload_tree_structure()

	return m


def set1(m):
	_lock_value(m, "samp_af", value=1.0)


def set2(m):
	_lock_value(m, "log_attraction", value=1.0)
