import numpy as np
import pandas as pd
import os
import larch
from larch.util import piece
from larch.util.dataframe import columnize
import re
from addict import Dict
import pyarrow as pa
import pyarrow.feather as pf

from .tnc_costs import taxi_cost, tnc_solo_cost, tnc_pool_cost
from .transit_approach import transit_approach
from .choice_model import model_builder, alt_codes_and_names
from .random_states import check_random_state
from .data_handlers import DataHandler

from .cmap_logging import getLogger
log = getLogger()

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})
n_modes = len(mode5codes)

av = {}


def _data_for_application_1(dh, otaz=1, peak=True, replication=None):
	"""

	Parameters
	----------
	dh : DataHandler
	otaz : int
	peak : bool
	replication : int, optional

	Returns
	-------
	pd.DataFrame
	"""
	global av

	n_zones = dh.n_internal_zones
	if len(av) != n_zones * 5:
		av = {}
		num = 5
		for i in range(n_zones):
			av[num + mode5codes.AUTO] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TNC1] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TNC2] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TAXI] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TRANSIT] = f"altdest{i + 1:04d}_transit_avail_{timeperiod_names[t]}" # TODO make avail non-purpose-special
			num += 5

	if replication is None:
		replication = dh.cfg.get('n_replications', 50)

	dtaz = pd.Index(np.arange(n_zones) + 1)
	otaz_series = pd.Series(otaz, index=dtaz)

	df1 = {
		'dtaz': dtaz,
	}

	attractions_mapping = {
		'HBWH': 'hwahi',
		'HBWL': 'hwalo',
		'HBO': 'hoa',
		'NHB': 'nha',
	}
	for purpose in attractions_mapping:
		with np.errstate(divide='ignore'):
			df1[f'log_attractions_{purpose}'] = np.log(
				dh.trip_attractions[attractions_mapping[purpose]]
			)
		df1[f'log_attractions_{purpose} > -666'] = (df1[f'log_attractions_{purpose}'] > -666)

	df1['o_zone == dtaz'] = (otaz == dtaz)

	df1['auto_time_PEAK'   ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
	df1['auto_time_OFFPEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_time']][otaz - 1, :n_zones]

	df1['auto_dist_PEAK'   ] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
	df1['auto_dist_OFFPEAK'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_dist']][otaz - 1, :n_zones]

	tskims = {
		'OFFPEAK'  :dh.skims.transit_op,
		'PEAK'     :dh.skims.transit_pk,
	}

	peak_tnc_pricing = {
		'OFFPEAK'  :0,
		'PEAK'     :1,
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

	for t in ['PEAK', 'OFFPEAK']:
		df1[f'piece(auto_dist_{t},None,5)'] = piece(df1[f'auto_dist_{t}'],None,5)
		df1[f'piece(auto_dist_{t},5,10)'] = piece(df1[f'auto_dist_{t}'],5,10)
		df1[f'piece(auto_dist_{t},10,None)'] = piece(df1[f'auto_dist_{t}'],10,None)

		df1[f'transit_ivtt_{t}'] = tskims[t].raw[tskims[t].col_mapping['ivtt']][otaz - 1, :n_zones]
		df1[f'transit_ovtt_{t}'] = tskims[t].raw[tskims[t].col_mapping['ovtt']][otaz - 1, :n_zones]
		df1[f'transit_fare_{t}'] = tskims[t].raw[tskims[t].col_mapping['fare']][otaz - 1, :n_zones]

		df1[f'taxi_fare_{t}'] = taxi_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], otaz_series, df1['dtaz'],
		)
		df1[f'tnc_solo_fare_{t}'] = tnc_solo_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], otaz_series, df1['dtaz'], peak_tnc_pricing[t],
		)
		df1[f'tnc_pool_fare_{t}'] = tnc_pool_cost(
			dh, df1[f'auto_time_{t}'], df1[f'auto_dist_{t}'], otaz_series, df1['dtaz'], peak_tnc_pricing[t],
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

	df1 = pd.DataFrame(df1, index=dtaz)

	df2 = pd.concat([df1] * replication).reset_index(drop=True)
	df2_ = {}

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

	# TODO peak and offpeak
	df2_['transit_approach_drivetime'] = trapp['drivetime'].T.reshape(-1)
	df2_['transit_approach_waittime'] = trapp['waittime'].T.reshape(-1)
	df2_['transit_approach_walktime'] = trapp['walktime'].T.reshape(-1)
	df2_['transit_approach_cost'] = trapp['cost'].T.reshape(-1)

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
	for purpose in attractions_mapping:
		df2_[f'auto_parking_cost_{purpose}'] = 0.0

	df2_['samp_wgt'] = 1.0
	df2_['log(1/samp_wgt)'] = 0.0

	for t in ['PEAK', 'OFFPEAK']:
		df2_[f'transit_avail_{t}'] = (
			(df2[f'transit_ivtt_{t}'] < 999)
			& (df2_['transit_approach_walktime'] < 999)
			& (df2_['transit_approach_drivetime'] < 999)
			#& (df2[f'log_attractions_{purpose}'] > -9998)
		)
	# df2_['auto_avail'] = (
	# 	df2[f'log_attractions_{purpose}'] > -9998
	# )

	df2 = pd.concat([df2, pd.DataFrame(df2_, index=df2.index)], axis=1)



	df3 = pd.DataFrame(
		df2.to_numpy(dtype=np.float64).reshape(replication, -1),
		columns=pd.MultiIndex.from_product([
			[f"altdest{x:04d}" for x in range(1,n_zones+1)],
			df2.columns,
		])
	)

	#df3.columns = [f"{j[0]}_{j[1]}" for j in df3.columns]
	df3['o_zone'] = np.float64(otaz)
	_fix_column_names(dh, df3)

	return df3



def _fix_column_names(dh, dfx):
	column_2_replacement = getattr(dh, 'column_2_replacement', [])
	if len(dfx.columns) != len(column_2_replacement):
		columns = [f"{j[0]}_{j[1]}" for j in dfx.columns]
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

		column_2_replacement = [s4f(s3f(s2f(s1f(s0f(j))))) for j in columns]
		dh.column_2_replacement = column_2_replacement

	dfx.columns = column_2_replacement


def _data_for_application_2(dh, df2, filename):
	#_fix_column_names(dh, df2)

	n_zones = dh.n_internal_zones
	alt_codes, alt_names = alt_codes_and_names(
		n_sampled_dests=n_zones,
		include_actual_dest=False,
	)

	dfas = larch.DataFrames(
		co=df2.astype(np.float64),
		alt_codes=alt_codes,
		av=columnize(df2, av, inplace=False, dtype=np.int8)
	)

	dfas.to_feathers(filename)

	# return dfas

	# tb = pa.table([df2.to_numpy(dtype=np.float64).T.reshape(-1)], ['data_co'])
	# pf.write_feather(tb, str(filename)+".data_co.f2")
	#
	# avail = columnize(df2, av, inplace=False, dtype=np.int8)
	# tb = pa.table([avail.to_numpy(dtype=np.int8).T.reshape(-1)], ['data_av'])
	# pf.write_feather(tb, str(filename)+".data_av.f2")
	#
	return _reload_data_for_application_2(dh, filename)

def _reload_data_for_application_2(dh, filename):

	# n_zones = dh.n_internal_zones
	# alt_codes, alt_names = alt_codes_and_names(
	# 	n_sampled_dests=n_zones,
	# 	include_actual_dest=False,
	# )
	#
	# tb = pf.read_table(
	# 	str(filename)+".data_co.f2"
	# )
	# df2 = pd.DataFrame(
	# 	tb['data_co'].to_numpy().reshape(len(dh.column_2_replacement), -1).T,
	# 	columns=dh.column_2_replacement,
	# )
	#
	# tb = pf.read_table(
	# 	str(filename)+".data_av.f2"
	# )
	# avail = pd.DataFrame(
	# 	tb['data_av'].to_numpy().reshape(len(alt_codes), -1).T,
	# 	columns=alt_codes,
	# )
	#
	# dfas = larch.DataFrames(
	# 	co=df2,
	# 	alt_codes=alt_codes,
	# 	av=avail,
	# )
	# return dfas
	return larch.DataFrames.from_feathers(filename)



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
		df2 = _data_for_application_1(dh, otaz=otaz, peak=peak, replication=replication)
		filename = dh.filenames.cache_dir / f"data_for_application_{otaz}"
	else:
		df2_list = [
			_data_for_application_1(dh, otaz=z, peak=peak, replication=replication)
			for z in otaz
		]
		df2 = pd.concat(df2_list)
		filename = dh.filenames.cache_dir / f"data_for_application_{otaz[0]}_{otaz[-1]}"
	return _data_for_application_2(dh, df2, filename)



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
	log.debug(f"blockwise_mean")
	n_blocks = a.shape[0]//blocksize + (1 if a.shape[0]%blocksize else 0)
	mean = np.zeros([n_blocks, *a.shape[1:]])
	for j in range(n_blocks):
		mean[j] = a[j*blocksize:(j+1)*blocksize].mean(0)
	return mean


choice_simulator_global = Dict()

def choice_simulator_initialize(dh, return_simulators=True, n_threads=1):

	log.debug("initializing choice simulator")
	auto_cost_per_mile = dh.cfg.auto.cost.per_mile
	n_zones = dh.n_internal_zones
	choice_model_params = dh.choice_model_params

	if (auto_cost_per_mile, n_zones) in choice_simulator_global:
		import time, os
		with open(f"/tmp/log{os.getpid()}.log", 'at') as f:
			f.write(f"{time.strftime('%Y-%m-%d %I:%M:%S %p')} choice_simulator_initialize existing\n")
		choice_simulator = choice_simulator_global[(auto_cost_per_mile, n_zones)]
	else:
		import time, os
		with open(f"/tmp/log{os.getpid()}.log", 'at') as f:
			f.write(f"{time.strftime('%Y-%m-%d %I:%M:%S %p')} choice_simulator_initialize fresh\n")
		choice_simulator = Dict()
		for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
			choice_simulator[purpose] = model_builder(
				purpose=purpose,
				include_actual_dest=False,
				n_sampled_dests=n_zones,  # 3632,
				parameter_values=choice_model_params[purpose],
				constraints=False,
				n_threads=n_threads,
			)
		choice_simulator_global[(auto_cost_per_mile, n_zones)] = choice_simulator

	if return_simulators:
		return choice_simulator


def attach_dataframes(sim, purpose, dfa):
	if sim.dataframes is None:
		log.debug(f"choice_simulator_prob {purpose} attach dataframes new")
		sim.dataframes = dfa
	else:
		log.debug(f"choice_simulator_prob {purpose} attach dataframes direct injection")
		sim.set_dataframes(dfa, False, direct_injection=True)


def _sim_prob(purpose, sim):
	log.debug(f"choice_simulator_prob {purpose} simulate probability")
	sim_pr = sim.probability()
	return sim_pr


def choice_simulator_prob(dh, otaz, n_threads=1):
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

	choice_simulator = choice_simulator_initialize(dh, n_threads=n_threads)
	simulated_probability = {}

	for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
		sim = choice_simulator[purpose]
		attach_dataframes(sim, purpose, dfa)
		# log.debug(f"choice_simulator_prob {purpose} attach dataframes")
		# if sim.dataframes is None:
		# 	sim.dataframes = dfa
		# else:
		# 	sim.set_dataframes(dfa, False)
		# log.debug(f"choice_simulator_prob {purpose} simulate probability")
		sim_pr = _sim_prob(purpose, sim)
		simulated_probability[purpose] = blockwise_mean(sim_pr, replication)

	log.debug("choice_simulator_prob complete")
	return simulated_probability #.reshape([sim_pr.shape[0],-1,5])


def choice_simulator_data_prep(dh, otaz, n_threads=1):
	"""

	Parameters
	----------
	otaz : int or array-like

	Returns
	-------

	"""
	log.debug("choice_simulator_data_prep data_for_application")
	dfa = data_for_application(dh, otaz=otaz, peak=True) # TODO no more peak

	log.debug("choice_simulator_prob settings")
	replication = dh.cfg.get('n_replications', 50)

	choice_simulator = choice_simulator_initialize(dh, n_threads=n_threads)
	simulated_probability = {}

	for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
		sim = choice_simulator[purpose]
		attach_dataframes(sim, purpose, dfa)
		# log.debug(f"choice_simulator_prob {purpose} attach dataframes")
		# if sim.dataframes is None:
		# 	sim.dataframes = dfa
		# else:
		# 	sim.set_dataframes(dfa, False)
		# log.debug(f"choice_simulator_prob {purpose} simulate probability")
		sim_pr = _sim_prob(purpose, sim)
		simulated_probability[purpose] = blockwise_mean(sim_pr, replication)

	log.debug("choice_simulator_prob complete")
	return simulated_probability #.reshape([sim_pr.shape[0],-1,5])




def choice_simulator_trips(
		dh,
		otaz,
		purposes=None,
		random_state=None,
		n_threads=1,
		save_dir=None,
):
	"""

	Parameters
	----------
	otaz : int or array-like
	purposes : Collection, optional

	Returns
	-------

	"""
	import time, os
	with open(f"/tmp/log{os.getpid()}.log", 'at') as f:
		f.write(f"{time.strftime('%Y-%m-%d %I:%M:%S %p')} choice_simulator_trips\n")

	if purposes is None:
		purposes = ['HBWH', 'HBWL', 'HBO', 'NHB']

	if isinstance(otaz, int):
		otaz = [otaz]
	simulated_probability = choice_simulator_prob(
		dh,
		otaz=otaz,
		n_threads=n_threads,
	)
	simulated_choices = {}

	random_state = check_random_state(random_state or otaz[0])
	for purpose in purposes:
		choices_data = {}

		for n, _o in enumerate(otaz):
			p = simulated_probability[purpose][n]
			c = random_state.choice(p.size, size=dh.zone_productions.loc[_o,purpose], p=p)
			choices_data[_o] = pd.DataFrame(dict(
				mode=(c % n_modes) + 1,
				zone=(c // n_modes) + 1,
			)).value_counts().sort_index().rename(_o).astype(np.int16)

		full_index = pd.MultiIndex.from_product(
			[
				np.arange(n_modes) + 1,
				np.arange(dh.n_internal_zones) + 1,
			],
			names=[
				'mode',
				'a_zone',
			],
		)

		simulated_choices[purpose] = pd.concat(
			choices_data,
			axis=1,
		).reindex(full_index).fillna(0).astype(np.int16)
		simulated_choices[purpose].columns.name = 'p_zone'

	if save_dir is None:
		return pd.concat(simulated_choices)
	else:
		concise = pd.DataFrame(pd.concat(simulated_choices).stack().rename("trips"))
		concise.to_parquet(os.path.join(save_dir, f"choice_simulator_trips_{otaz[0]}_{otaz[-1]}.pq"))


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
	save_dir = dh.filenames.cache_dir / "choice_simulator_trips"
	os.makedirs(save_dir, exist_ok=True)
	n_threads = max(joblib.cpu_count() // n_jobs, 1)

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
			joblib.delayed(choice_simulator_trips)(
				dh,
				otaz_chunk,
				purposes=purposes,
				save_dir=save_dir,
				n_threads=n_threads,
			)
			for otaz_chunk in otaz_chunks
		)
		log.info("joblib model body complete")
	try:
		result = pd.concat(parts, axis=1)
	except ValueError:
		return
	else:
		return result



