import numpy as np
import pandas as pd
import larch
from larch.util import piece
from larch.util.dataframe import columnize
import re
from addict import Dict

from .tnc_costs import taxi_cost, tnc_solo_cost, tnc_pool_cost
from .transit_approach import transit_approach
from .choice_model import model_builder
from .random_states import check_random_state
from .data_handlers import DataHandler

from .cmap_logging import getLogger
log = getLogger()

mode5codes = Dict({'AUTO': 1, 'TAXI': 2, 'TNC1': 3, 'TNC2': 4, 'TRANSIT': 5})

av = {}


def _data_for_application_1(dh, otaz=1, peak=True, purpose='HBWH', replication=None):
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
	if len(av) != n_zones * 5:
		av = {}
		num = 5
		for i in range(n_zones):
			av[num + mode5codes.AUTO] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TNC1] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TNC2] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TAXI] = f"altdest{i + 1:04d}_auto_avail"
			av[num + mode5codes.TRANSIT] = f"altdest{i + 1:04d}_transit_avail"
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
	df1['log(attractions)'] = np.log(df1['attractions'])
	df1['attractions > 1e-290'] = (df1['attractions'] > 1e-290)

	df1['o_zone == dtaz'] = (otaz == df1['dtaz'])

	if peak:
		df1['auto_time'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_time']][otaz - 1, :n_zones]
		df1['auto_dist'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['am_dist']][otaz - 1, :n_zones]
		tskims = dh.skims.transit_pk
	else:
		df1['auto_time'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_time']][otaz - 1, :n_zones]
		df1['auto_dist'] = dh.skims.auto.raw[dh.skims.auto.col_mapping['md_dist']][otaz - 1, :n_zones]
		tskims = dh.skims.transit_op

	df1['piece(auto_dist,None,5)'] = piece(df1['auto_dist'],None,5)
	df1['piece(auto_dist,5,10)'] = piece(df1['auto_dist'],5,10)
	df1['piece(auto_dist,10,None)'] = piece(df1['auto_dist'],10,None)

	df1['transit_ivtt'] = tskims.raw[tskims.col_mapping['ivtt']][otaz - 1, :n_zones]
	df1['transit_ovtt'] = tskims.raw[tskims.col_mapping['ovtt']][otaz - 1, :n_zones]
	df1['transit_fare'] = tskims.raw[tskims.col_mapping['fare']][otaz - 1, :n_zones]

	df1['taxi_fare'] = taxi_cost(
		dh, df1['auto_time'], df1['auto_dist'], df1['otaz'], df1['dtaz'],
	)
	df1['tnc_solo_fare'] = tnc_solo_cost(
		dh, df1['auto_time'], df1['auto_dist'], df1['otaz'], df1['dtaz'], peak,
	)
	df1['tnc_pool_fare'] = tnc_pool_cost(
		dh, df1['auto_time'], df1['auto_dist'], df1['otaz'], df1['dtaz'], peak,
	)
	if peak:
		df1['tnc_solo_wait_time'] = dh.m01['tnc_solo_wait_pk'][otaz]
		df1['tnc_pool_wait_time'] = dh.m01['tnc_pool_wait_pk'][otaz]
		df1['taxi_wait_time'] = dh.m01['taxi_wait_pk'][otaz]
	else:
		df1['tnc_solo_wait_time'] = dh.m01['tnc_solo_wait_op'][otaz]
		df1['tnc_pool_wait_time'] = dh.m01['tnc_pool_wait_op'][otaz]
		df1['taxi_wait_time'] = dh.m01['taxi_wait_op'][otaz]

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

	df2['samp_wgt'] = 1.0
	df2['log(1/samp_wgt)'] = 0.0

	df2['transit_avail'] = (
		(df2['transit_ivtt'] < 999)
		& (df2['transit_approach_walktime'] < 999)
		& (df2['transit_approach_drivetime'] < 999)
		& (df2['attractions'] > 1e-290)
	)
	df2['auto_avail'] = (
		df2['attractions'] > 1e-290
	)

	df2['rep'] = np.repeat(np.arange(replication)+1, len(df1))
	df2['altdest'] = df2['dtaz'].apply(lambda x: f"altdest{x:04d}")
	df2 = df2.set_index(['rep','altdest']).unstack()
	df2['o_zone'] = otaz

	return df2

def _data_for_application_2(dh, df2):

	columns = [f"{j[1]}_{j[0]}" for j in df2.columns]

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
	alt_codes = np.arange(1, n_zones * 5 + 1) + 5

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


def choice_simulator_prob(dh, purpose, otaz):
	"""

	Parameters
	----------
	purpose : str
	otaz : int or array-like
	peak : bool

	Returns
	-------

	"""
	dfa = data_for_application(dh, otaz=otaz, peak=('HBW' in purpose))

	auto_cost_per_mile = dh.cfg.auto.cost.per_mile
	n_sampled_dests = dh.n_internal_zones
	choice_model_params = dh.choice_model_params
	replication = dh.cfg.get('n_replications', 50)

	if (auto_cost_per_mile, n_sampled_dests) in choice_simulator_global:
		choice_simulator = choice_simulator_global[(auto_cost_per_mile, n_sampled_dests)]
	else:
		choice_simulator = Dict()
		for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
			choice_simulator[purpose] = model_builder(
				purpose=purpose,
				include_actual_dest=False,
				n_sampled_dests=len(dh.m01),  # 3632,
				parameter_values=choice_model_params[purpose],
			)
		choice_simulator_global[(auto_cost_per_mile, n_sampled_dests)] = choice_simulator

	sim = choice_simulator[purpose]
	if sim.dataframes is None:
		sim.dataframes = dfa
	else:
		sim.set_dataframes(dfa, False)
	sim_pr = sim.probability()
	sim_pr = blockwise_mean(sim_pr, replication)
	return sim_pr #.reshape([sim_pr.shape[0],-1,5])



def choice_simulator_trips(dh, purpose, otaz, random_state=None):
	"""

	Parameters
	----------
	purpose : str
	otaz : int or array-like
	peak : bool

	Returns
	-------

	"""

	if isinstance(otaz, int):
		otaz = [otaz]
	sim_pr = choice_simulator_prob(
		dh,
		purpose=purpose,
		otaz=otaz,
	)

	random_state = check_random_state(random_state or otaz[0])
	choices_data = {}

	for n, _o in enumerate(otaz):
		p = sim_pr[n]
		c = random_state.choice(p.size, size=dh.zone_productions.loc[_o,purpose], p=p)
		choices_data[_o] = pd.DataFrame(dict(
			mode=(c % 5) + 1,
			zone=(c // 5) + 1,
		)).value_counts().sort_index().rename(_o)

	full_index = pd.MultiIndex.from_product(
		[np.arange(5) + 1, np.arange(len(dh.m01)) + 1],
		names=['mode', 'zone'],
	)

	return pd.concat(
		choices_data,
		axis=1,
	).reindex(full_index).fillna(0).astype(int).T



def choice_simulator_trips_many(dh, purpose, otaz, max_chunk_size=20, n_jobs=5, init_step=False):

	# auto chunk size calculation
	n_chunks_per_job = 0
	chunk_size = np.inf
	while chunk_size > max_chunk_size:
		n_chunks_per_job += 1
		chunk_size = int(np.ceil(len(otaz) / n_jobs / n_chunks_per_job))
		if chunk_size == 1:
			break

	otaz_chunks = [otaz[i:i + chunk_size] for i in range(0, len(otaz), chunk_size)]
	init_chunks = [otaz[i:i+1] for i in range(0,min(len(otaz),n_jobs))]

	import joblib

	with joblib.Parallel(n_jobs=n_jobs) as parallel:
		if init_step:
			log.info("joblib model init starting")
			_ = parallel(
				joblib.delayed(choice_simulator_trips)(dh, purpose, otaz_chunk)
				for otaz_chunk in init_chunks
			)
			log.info("joblib model init complete")
		else:
			log.info("joblib model body starting")
		parts = parallel(
			joblib.delayed(choice_simulator_trips)(dh, purpose, otaz_chunk)
			for otaz_chunk in otaz_chunks
		)
		log.info("joblib model body complete")
	return pd.concat(parts)



