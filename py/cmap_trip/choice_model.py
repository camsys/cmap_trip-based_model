import numpy as np
import pandas as pd
import larch
from larch import P,X
from larch.util.data_expansion import piecewise_linear
from addict import Dict

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})

mode5names = ['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT']
mode5codes = Dict(zip(
	mode5names,
	np.arange(len(mode5names)) + 1,
))


def model_builder(
		purpose,
		include_actual_dest=True,
		n_sampled_dests=5,
		parameter_values=None,
		auto_cost_per_mile=30, # cents
):
	alt_codes = np.arange(5 * (n_sampled_dests + 1)) + 1
	alt_names = [i for i in mode5names]
	for i in range(n_sampled_dests):
		alt_names.extend([(j + f"d{i + 1:04d}") for j in mode5names])
	if not include_actual_dest:
		alt_codes = alt_codes[5:]
		alt_names = alt_names[5:]

	dummy_dfs = larch.DataFrames(
	    alt_codes=alt_codes,
	    alt_names=alt_names,
	)

	# Define the alternative availability for each alternative in this model.
	av = {}
	dzone_has_nonzero_attractions = "DzoneAttractions > 1e-290"
	if include_actual_dest:
		av[mode5codes.AUTO] = dzone_has_nonzero_attractions
		av[mode5codes.TNC1] = dzone_has_nonzero_attractions
		av[mode5codes.TNC2] = dzone_has_nonzero_attractions
		av[mode5codes.TAXI] = dzone_has_nonzero_attractions
		av[mode5codes.TRANSIT] = (
			f"(transit_ivtt < 999) "
			f"& (transit_approach_walktime < 999) "
			f"& (transit_approach_drivetime < 999) "
			f"& ({dzone_has_nonzero_attractions})"
		)
	num = 5
	for i in range(n_sampled_dests):
		altdest_has_nonzero_attractions = f"altdest{i + 1:04d}_auto_avail"
		av[num + mode5codes.AUTO] = altdest_has_nonzero_attractions
		av[num + mode5codes.TNC1] = altdest_has_nonzero_attractions
		av[num + mode5codes.TNC2] = altdest_has_nonzero_attractions
		av[num + mode5codes.TAXI] = altdest_has_nonzero_attractions
		av[num + mode5codes.TRANSIT] = f"altdest{i + 1:04d}_transit_avail"
		num += 5


	m = larch.Model(
		dataservice=dummy_dfs
	)

	m.title = f"{purpose} Mode & Destination"

	m.availability_co_vars = av

	if include_actual_dest:
		utility_destination = (
				+ P("samp_af") * X(f"log(1/obs_samp_wgt)")
				+ P("log_attraction") * X(f"log(DzoneAttractions)")
				+ P("intrazonal") * X(f"o_zone == d_zone")
				+ piecewise_linear(f"auto_dist", "distance", breaks=[5, 10])
		)
		m.utility_co[mode5codes.AUTO] = (
				+ P("cost") * X(f"auto_dist") * auto_cost_per_mile / 100.0
				+ P("auto_time") * X(f"auto_time")
		) + utility_destination
		m.utility_co[mode5codes.TNC1] = (
				P.Const_TNC1
				+ P("cost") * X(f"tnc_solo_cost") / 100.0
				+ P("tnc_time") * X(f"auto_time")
		) + utility_destination
		m.utility_co[mode5codes.TNC2] = (
				P.Const_TNC2
				+ P("cost") * X(f"tnc_solo_cost") / 100.0  # TODO TNC Shared cost
				+ P("tnc_time") * X(f"auto_time")
		) + utility_destination
		m.utility_co[mode5codes.TAXI] = (
				P.Const_TAXI
				+ P("cost") * X(f"taxi_fare") / 100.0
				+ P("tnc_time") * X(f"auto_time")
				+ P("ovtt") * X("taxi_wait_time")
		) + utility_destination
		m.utility_co[mode5codes.TRANSIT] = (
				P.Const_Transit
				+ P("cost") * X(f"transit_fare") / 100.0
				+ P("cost") * X(f"transit_approach_cost") / 100.0
				+ P("transit_ivtt") * X(f"transit_ivtt")
				+ P("transit_ivtt") * X(f"transit_approach_drivetime")
				+ P("ovtt") * X(f"transit_ovtt")
				+ P("ovtt") * X(f"transit_approach_walktime")
				+ P("ovtt") * X(f"transit_approach_waittime")
		) + utility_destination

		m.graph.new_node(
			parameter="Mu",
			children=[
				mode5codes.AUTO,
				mode5codes.TNC1,
				mode5codes.TNC2,
				mode5codes.TAXI,
				mode5codes.TRANSIT,
			],
			name="actualdest",
		)

	for i in range(n_sampled_dests):
		utility_destination = (
				+ P("samp_af") * X(f"log(1/altdest{i + 1:04d}_samp_wgt)")
				+ P("log_attraction") * X(f"log(altdest{i + 1:04d}_attractions)")
				+ P("intrazonal") * X(f"o_zone == altdest{i + 1:04d}")
				+ piecewise_linear(f"altdest{i + 1:04d}_auto_dist", "distance", breaks=[5,10])
		)
		jAUTO = i * 5 + 5 + mode5codes.AUTO
		jTNC1 = i * 5 + 5 + mode5codes.TNC1
		jTNC2 = i * 5 + 5 + mode5codes.TNC2
		jTAXI = i * 5 + 5 + mode5codes.TAXI
		jTRANSIT = i * 5 + 5 + mode5codes.TRANSIT
		m.utility_co[jAUTO] = (
				+ P("cost") * X(f"altdest{i + 1:04d}_auto_dist") * auto_cost_per_mile / 100.0
				+ P("auto_time") * X(f"altdest{i + 1:04d}_auto_time")
		) + utility_destination
		m.utility_co[jTNC1] = (
				P.Const_TNC1
				+ P("cost") * X(f"altdest{i + 1:04d}_tnc_solo_fare") / 100.0
				+ P("ovtt") * X(f"altdest{i + 1:04d}_tnc_wait_time")
				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time")
		) + utility_destination
		m.utility_co[jTNC2] = (
				P.Const_TNC2
				+ P("cost") * X(f"altdest{i + 1:04d}_tnc_solo_fare") / 100.0 # TODO - shared trip fares
				+ P("ovtt") * X(f"altdest{i + 1:04d}_tnc_wait_time")
				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time")
		) + utility_destination
		m.utility_co[jTAXI] = (
				P.Const_TAXI
				+ P("cost") * X(f"altdest{i + 1:04d}_taxi_fare") / 100.0
				+ P("ovtt") * X(f"altdest{i + 1:04d}_taxi_wait_time")
				+ P("tnc_time") * X(f"altdest{i + 1:04d}_auto_time")
		) + utility_destination
		m.utility_co[jTRANSIT] = (
				P.Const_Transit
				+ P("cost") * X(f"altdest{i + 1:04d}_transit_fare") / 100.0
				+ P("transit_ivtt") * X(f"altdest{i + 1:04d}_transit_ivtt")
				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_ovtt")
				+ P("cost") * X(f"altdest{i + 1:04d}_transit_approach_cost") / 100.0
				+ P("transit_ivtt") * X(f"altdest{i + 1:04d}_transit_approach_drivetime")
				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_approach_walktime")
				+ P("ovtt") * X(f"altdest{i + 1:04d}_transit_approach_waittime")
		) + utility_destination
		m.graph.new_node(
			parameter="Mu",
			children=[jAUTO, jTNC1, jTNC2, jTAXI, jTRANSIT],
			name=f"altdest{i + 1:04d}",
		)

	m.lock_value("samp_af", value=1.0)
	m.lock_value("log_attraction", value=1.0)

	m.set_value("cost", maximum=-0.00001)
	m.set_value("auto_time", maximum=-0.01, minimum=-0.03)
	m.set_value("tnc_time", maximum=-0.01, minimum=-0.03)
	m.set_value("transit_ivtt", maximum=-0.01, minimum=-0.03)
	m.set_value("ovtt", maximum=-0.01)
	if parameter_values is None:
		m.set_values(
			cost=-0.001,
			auto_time=-0.01,
			tnc_time=-0.01,
			transit_ivtt=-0.01,
			ovtt=-0.01,
			Const_TNC1=-4.0,
			Const_TNC2=-4.0,
			Const_Transit=-2.0,
		)
	else:
		m.set_values(**parameter_values)

	return m