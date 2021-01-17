import numpy as np
import pandas as pd
import larch
from larch.util import piece
from larch.util.dataframe import columnize
import re
from addict import Dict

from .tnc_costs import taxi_cost, tnc_solo_cost, tnc_pool_cost
from .transit_approach import transit_approach
from .choice_model import model_builder, alt_codes_and_names
from .random_states import check_random_state
from .data_handlers import DataHandler
from .timeperiods import timeperiod_names

from .cmap_logging import getLogger
log = getLogger()

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})
n_timeperiods = len(timeperiod_names)
n_modes = len(mode5codes)

av = {}


def _data_for_application_1(dh, otaz=1, peak=True, purpose='HBWH', replication=None, debug1=False):
	"""

	Parameters
	----------
	dh : DataHandler
	otaz : int
	peak : bool
	purpose : str
	replication : int, optional

	Returns
	-------
	pd.DataFrame
	"""
	global av

	n_zones = dh.n_internal_zones
	if len(av) != n_zones * 5 * n_timeperiods:
		av = {}
		num = 5
		for i in range(n_zones):
			for t in range(n_timeperiods):
				av[num + mode5codes.AUTO] = f"altdest{i + 1:04d}_auto_avail"
				av[num + mode5codes.TNC1] = f"altdest{i + 1:04d}_auto_avail"
				av[num + mode5codes.TNC2] = f"altdest{i + 1:04d}_auto_avail"
				av[num + mode5codes.TAXI] = f"altdest{i + 1:04d}_auto_avail"
				av[num + mode5codes.TRANSIT] = f"altdest{i + 1:04d}_transit_avail_{timeperiod_names[t]}"
				num += 5

	if replication is None:
		replication = dh.cfg.get('n_replications', 50)

	df1 = pd.DataFrame({'otaz': otaz, 'dtaz': np.arange(n_zones) + 1})
	df1.index = df1['dtaz']

	attractions_mapping = {
		'HBWH': 'hwahi',
		'HBWL': 'hwalo',
		'HBO': 'hoa',
		'NHB': 'nha',
	}
	df1['attractions'] = dh.trip_attractions[attractions_mapping[purpose]]
	with np.errstate(divide='ignore'):
		df1['log(attractions)'] = np.log(df1['attractions'])
	df1['attractions > 1e-290'] = (df1['attractions'] > 1e-290)

	df1['o_zone == dtaz'] = (otaz == df1['dtaz'])

	df1['auto_time_NIGHT'  ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_time']][otaz - 1, :n_zones]
	df1['auto_time_AM_PRE' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_AM_PEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_AM_POST'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_MIDDAY' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_time']][otaz - 1, :n_zones]
	df1['auto_time_PM_PRE' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_PM_PEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_PM_POST'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]

	df1['auto_dist_NIGHT'  ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_dist']][otaz - 1, :n_zones]
	df1['auto_dist_AM_PRE' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_AM_PEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_AM_POST'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_MIDDAY' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_dist']][otaz - 1, :n_zones]
	df1['auto_dist_PM_PRE' ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_PM_PEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_PM_POST'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]

	tskims = {
		'NIGHT'  :dh.skims.transit_op,
		'AM_PRE' :dh.skims.transit_pk,
		'AM_PEAK':dh.skims.transit_pk,
		'AM_POST':dh.skims.transit_pk,
		'MIDDAY' :dh.skims.transit_op,
		'PM_PRE' :dh.skims.transit_pk,
		'PM_PEAK':dh.skims.transit_pk,
		'PM_POST':dh.skims.transit_pk,
	}

	peak_tnc_pricing = {
		'NIGHT'  : 0,
		'AM_PRE' : 1,
		'AM_PEAK': 1,
		'AM_POST': 1,
		'MIDDAY' : 0,
		'PM_PRE' : 1,
		'PM_PEAK': 1,
		'PM_POST': 1,
	}

	# if peak:
	# 	df1['auto_time'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	# 	df1['auto_dist'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	# 	tskims = dh.skims.transit_pk
	# else:
	# 	df1['auto_time'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_time']][otaz - 1, :n_zones]
	# 	df1['auto_dist'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_dist']][otaz - 1, :n_zones]
	# 	tskims = dh.skims.transit_op

	df1[f'piece(auto_dist,None,5)'] = piece(df1[f'auto_dist_MIDDAY'], None, 5)
	df1[f'piece(auto_dist,5,10)'] = piece(df1[f'auto_dist_MIDDAY'], 5, 10)
	df1[f'piece(auto_dist,10,None)'] = piece(df1[f'auto_dist_MIDDAY'], 10, None)

	for t in timeperiod_names:
		df1[f'piece(auto_dist_{t},None,5)'] = piece(df1[f'auto_dist_{t}'],None,5)
		df1[f'piece(auto_dist_{t},5,10)'] = piece(df1[f'auto_dist_{t}'],5,10)
		df1[f'piece(auto_dist_{t},10,None)'] = piece(df1[f'auto_dist_{t}'],10,None)

		df1[f'transit_ivtt_{t}'] = tskims[t].raw[tskims[t].col_mapping['ivtt']][otaz - 1, :n_zones]
		df1[f'transit_ovtt_{t}'] = tskims[t].raw[tskims[t].col_mapping['ovtt']][otaz - 1, :n_zones]
		df1[f'transit_fare_{t}'] = tskims[t].raw[tskims[t].col_mapping['fare']][otaz - 1, :n_zones]

		df1[f'taxi_fare_{t}'] = taxi_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], df1['otaz'], df1['dtaz'],
		)
		df1[f'tnc_solo_fare_{t}'] = tnc_solo_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], df1['otaz'], df1['dtaz'], peak_tnc_pricing[t],
		)
		df1[f'tnc_pool_fare_{t}'] = tnc_pool_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], df1['otaz'], df1['dtaz'], peak_tnc_pricing[t],
		)
		if peak_tnc_pricing[t]:
			df1[f'tnc_solo_wait_time_{t}'] = dh.m01['tnc_solo_wait_pk'][otaz]
			df1[f'tnc_pool_wait_time_{t}'] = dh.m01['tnc_pool_wait_pk'][otaz]
			df1[f'taxi_wait_time_{t}'] = dh.m01['taxi_wait_pk'][otaz]
		else:
			df1[f'tnc_solo_wait_time_{t}'] = dh.m01['tnc_solo_wait_op'][otaz]
			df1[f'tnc_pool_wait_time_{t}'] = dh.m01['tnc_pool_wait_op'][otaz]
			df1[f'taxi_wait_time_{t}'] = dh.m01['taxi_wait_op'][otaz]

	# TODO peak and offpeak wait times
	df1[f'taxi_wait_time'] = df1[f'taxi_wait_time_MIDDAY']
	df1[f'tnc_solo_wait_time'] = df1[f'tnc_solo_wait_time_MIDDAY']
	df1[f'tnc_pool_wait_time'] = df1[f'tnc_pool_wait_time_MIDDAY']

	df2 = pd.concat([df1.drop(columns=['otaz'])] * replication)

	# transit approach
	trapp = transit_approach(
		dh,
		ozone=np.full(3632, otaz),
		dzone=np.arange(3632) + 1,
		TPTYPE='HW' if peak else 'HO',
		replication=replication,
		approach_distances=None,
		trace=False,
		random_state=otaz,
	)

	df2['transit_approach_drivetime'] = trapp['drivetime'].T.reshape(-1)
	df2['transit_approach_waittime'] = trapp['waittime'].T.reshape(-1)
	df2['transit_approach_walktime'] = trapp['walktime'].T.reshape(-1)
	df2['transit_approach_cost'] = trapp['cost'].T.reshape(-1)

	# TODO fix auto parking cost
	# _parking_cost, _free_parking = parking_cost_v2(
	# 	dh,
	# 	_trips_by_purpose.d_zone,
	# 	_trips_by_purpose.hhinc_dollars,
	# 	cfg.default_activity_durations[purpose],
	# 	purpose,
	# 	random_state=hash(purpose) + 1,
	# )
	# trips.loc[q, f'actualdest_auto_parking_cost'] = _parking_cost
	# trips.loc[q, f'actualdest_auto_parking_free'] = _free_parking
	df2['auto_parking_cost'] = 0.0

	df2['samp_wgt'] = 1.0
	df2['log(1/samp_wgt)'] = 0.0

	for t in timeperiod_names:
		df2[f'transit_avail_{t}'] = (
			(df2[f'transit_ivtt_{t}'] < 999)
			& (df2['transit_approach_walktime'] < 999)
			& (df2['transit_approach_drivetime'] < 999)
			& (df2['attractions'] > 1e-290)
		)
	df2['auto_avail'] = (
		df2['attractions'] > 1e-290
	)

	# df2['rep'] = np.repeat(np.arange(replication)+1, len(df1))
	# df2['altdest'] = df2['dtaz'].apply(lambda x: f"altdest{x:04d}")
	# df3 = df2.set_index(['rep','altdest']).unstack()

	df3 = pd.DataFrame(
		df2.to_numpy(dtype=np.float64).reshape(replication, -1),
		columns=pd.MultiIndex.from_product([
			[f"altdest{x:04d}" for x in range(1,n_zones+1)],
			df2.columns,
		])
	)

	df3.columns = [f"{j[0]}_{j[1]}" for j in df3.columns]
	df3['o_zone'] = otaz

	return df3

def _data_for_application_2(dh, df2):

	#columns = [f"{j[1]}_{j[0]}" for j in df2.columns]
	columns = df2.columns

	if columns[-1] == '_o_zone':
		columns[-1] = 'o_zone'

	s0f = lambda c: c.replace("_dtaz", "")

	s1 = re.compile("(altdest[0-9]+)_o_zone == dtaz")
	s1f = lambda c: s1.sub("o_zone == \g<1>", c)

	s2 = re.compile("(altdest[0-9]+_)piece\((.*)\)")
	s2f = lambda c: s2.sub(r"piece(\g<1>\g<2>)", c)

	s3 = re.compile(r"(altdest[0-9]+_)log\(attractions\)")
	s3f = lambda c: s3.sub(r"log(\g<1>attractions)", c)

	s4 = re.compile(r"(altdest[0-9]+_)log\(1/samp_wgt\)")
	s4f = lambda c: s4.sub(r"log(1/\g<1>samp_wgt)", c)

	columns = [s4f(s3f(s2f(s1f(s0f(j))))) for j in columns]

	df2.columns = columns

	n_zones = dh.n_internal_zones

	alt_codes, alt_names = alt_codes_and_names(
		n_timeperiods=8,
		n_sampled_dests=n_zones,
		include_actual_dest=False,
	)

	dfas = larch.DataFrames(
		co=df2.astype(np.float64),
		alt_codes=alt_codes,
		av=columnize(df2, av, inplace=False, dtype=np.int8)
	)

	return dfas



def data_for_application(dh, otaz=1, peak=True, purpose='HBWH', replication=None):
	"""

	Parameters
	----------
	otaz : int or array-like
	peak : bool
	purpose : str
	replication : int, optional

	Returns
	-------

	"""
	if replication is None:
		replication = dh.cfg.get('n_replications', 50)

	if isinstance(otaz, int):
		df2 = _data_for_application_1(dh, otaz=otaz, peak=peak, purpose=purpose, replication=replication)
	else:
		df2_list = [
			_data_for_application_1(dh, otaz=z, peak=peak, purpose=purpose, replication=replication)
			for z in otaz
		]
		df2 = pd.concat(df2_list)
	return _data_for_application_2(dh, df2)



def blockwise_mean(a, blocksize):
	"""

	Parameters
	----------
	a : array-like
	blocksize : int

	Returns
	-------
	array
	"""
	n_blocks = a.shape[0]//blocksize + (1 if a.shape[0]%blocksize else 0)
	mean = np.zeros([n_blocks, *a.shape[1:]])
	for j in range(n_blocks):
		mean[j] = a[j*blocksize:(j+1)*blocksize].mean(0)
	return mean


choice_simulator_global = Dict()

def choice_simulator_initialize(dh, return_simulators=True):

	log.debug("initializing choice simulator")
	auto_cost_per_mile = dh.cfg.auto.cost.per_mile
	n_zones = dh.n_internal_zones
	choice_model_params = dh.choice_model_params

	if (auto_cost_per_mile, n_zones) in choice_simulator_global:
		choice_simulator = choice_simulator_global[(auto_cost_per_mile, n_zones)]
	else:
		choice_simulator = Dict()
		for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
			choice_simulator[purpose] = model_builder(
				purpose=purpose,
				include_actual_dest=False,
				n_sampled_dests=n_zones,  # 3632,
				parameter_values=choice_model_params[purpose],
				constraints=False,
				n_threads=1,
			)
		choice_simulator_global[(auto_cost_per_mile, n_zones)] = choice_simulator

	if return_simulators:
		return choice_simulator

def choice_simulator_prob(dh, otaz):
	"""

	Parameters
	----------
	otaz : int or array-like

	Returns
	-------

	"""
	log.debug("choice_simulator_prob data_for_application")
	dfa = data_for_application(dh, otaz=otaz, peak=True) # TODO no more peak

	log.debug("choice_simulator_prob settings")
	replication = dh.cfg.get('n_replications', 50)

	choice_simulator = choice_simulator_initialize(dh)
	simulated_probability = {}

	for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
		sim = choice_simulator[purpose]
		log.debug(f"choice_simulator_prob {purpose} attach dataframes")
		if sim.dataframes is None:
			sim.dataframes = dfa
		else:
			sim.set_dataframes(dfa, False)
		log.debug(f"choice_simulator_prob {purpose} simulate probability")
		sim_pr = sim.probability()
		log.debug(f"choice_simulator_prob {purpose} blockwise_mean")
		simulated_probability[purpose] = blockwise_mean(sim_pr, replication)

	log.debug("choice_simulator_prob complete")
	return simulated_probability #.reshape([sim_pr.shape[0],-1,5])



def choice_simulator_trips(dh, otaz, purposes=None, random_state=None):
	"""

	Parameters
	----------
	otaz : int or array-like
	purposes : Collection, optional

	Returns
	-------

	"""
	if purposes is None:
		purposes = ['HBWH', 'HBWL', 'HBO', 'NHB']

	if isinstance(otaz, int):
		otaz = [otaz]
	simulated_probability = choice_simulator_prob(
		dh,
		otaz=otaz,
	)
	simulated_choices = {}

	random_state = check_random_state(random_state or otaz[0])
	for purpose in purposes:
		choices_data = {}

		for n, _o in enumerate(otaz):
			p = simulated_probability[purpose][n]
			c = random_state.choice(p.size, size=dh.zone_productions.loc[_o,purpose], p=p)
			c_ = (c // n_modes)
			choices_data[_o] = pd.DataFrame(dict(
				mode=(c % n_modes) + 1,
				timeperiod=(c_ % n_timeperiods) + 1,
				zone=(c_ // n_timeperiods) + 1,
			)).value_counts().sort_index().rename(_o).astype(np.int16)

		full_index = pd.MultiIndex.from_product(
			[
				np.arange(n_modes) + 1,
				np.arange(n_timeperiods) + 1,
				np.arange(dh.n_internal_zones) + 1,
			],
			names=[
				'mode',
				'timeperiod',
				'a_zone',
			],
		)

		simulated_choices[purpose] = pd.concat(
			choices_data,
			axis=1,
		).reindex(full_index).fillna(0).astype(np.int16)
		simulated_choices[purpose].columns.name = 'p_zone'

	return pd.concat(simulated_choices)


def choice_simulator_trips_many(dh, otaz=None, purposes=None, max_chunk_size=20, n_jobs=5, init_step=True):

	if otaz is None:
		otaz = np.arange(dh.n_internal_zones)+1

	# auto chunk size calculation
	n_chunks_per_job = 0
	chunk_size = np.inf
	while chunk_size > max_chunk_size:
		n_chunks_per_job += 1
		chunk_size = int(np.ceil(len(otaz) / n_jobs / n_chunks_per_job))
		if chunk_size == 1:
			break

	otaz_chunks = [otaz[i:i + chunk_size] for i in range(0, len(otaz), chunk_size)]
	inits = [None for _ in range(0,min(len(otaz),n_jobs))]

	import joblib

	with joblib.Parallel(n_jobs=n_jobs, verbose=100) as parallel:
		if init_step:
			log.info("joblib model init starting")
			_ = parallel(
				joblib.delayed(choice_simulator_initialize)(dh, False)
				for _ in inits
			)
			log.info("joblib model init complete")
		else:
			log.info("joblib model body starting")
		parts = parallel(
			joblib.delayed(choice_simulator_trips)(dh, otaz_chunk, purposes=purposes)
			for otaz_chunk in otaz_chunks
		)
		log.info("joblib model body complete")
	return pd.concat(parts, axis=1)



