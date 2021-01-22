import numpy as np
import pandas as pd
import os
from addict import Dict
import larch
from larch import P,X
from larch.util.figures import distribution_figure, share_figure
from larch.util.data_expansion import piecewise_linear
from IPython.display import display, HTML

import logging
log = logging.getLogger('CMAP')

n_sampled_dests = 5

def L(*args):
	if len(args)==1 and isinstance(args[0], str) and args[0][0]=="#":
		log.info(args[0])
	else:
		s = "\n".join(str(i) for i in args)
		s = "\n"+s
		log.info(s.replace("\n","\n    "))


from ..transit_approach import transit_approach
from .est_data import dh
from ..choice_model import model_builder

from .est_config import mode_modeled
from .est_survey import trips
from .est_sample_dest import sample_dest_zones_and_data
from ..util import resource_usage


L(
	"###### data statistics ######",
	trips['tripPurpose'].value_counts(),
	trips.mode3.value_counts()
)

L("###### sample_dest_zones_and_data ######")

mode5codes = Dict(zip(
	trips.mode5.cat.categories,
	np.arange(len(trips.mode5.cat.categories)) + 1,
))

# Null out invalid data # now in survey_data
# transit_cols = ['ivtt','ovtt','headway','fare','firstmode','lastmode']
# for a in transit_cols:
#     to_nan = trips[f'transit_{a}'] > 999
#     trips.loc[to_nan,f'transit_{a}'] = np.nan


L("## ae_approach_los ##")
from .est_survey import ae_approach_los

trip_approach_distances, ae_los = ae_approach_los(trips)

L("## sample_dest_zones_and_data ##")
TRIP_ALTS_CACHE_FILE = "trip_alts_v5"

trip_alt_df = dh.filenames.load(TRIP_ALTS_CACHE_FILE)
if trip_alt_df is None:
	trip_alt_df = sample_dest_zones_and_data(
		trips,
		n_zones=dh.n_internal_zones,
		n_sampled_dests=n_sampled_dests,
		keep_trips_cols=[
			'd_zone',
			'mode5',
			'mode5code',
			'incomeLevel',
			'tripCat',
			'tripPurpose',
			'auto_dist',
			'auto_time',
			'auto_parking_cost',
			'transit_fare',
			'transit_ivtt',
			'transit_ovtt',
			'transit_headway',
			'transit_approach_cost',
			'transit_approach_drivetime',
			'transit_approach_walktime',
			'transit_approach_waittime',
			'taxi_wait_time',
			'taxi_fare',
			'tnc_solo_wait_time',
			'tnc_pool_wait_time',
			'tnc_solo_fare',
			'tnc_pool_fare',
			'hhinc_dollars',
			'timeperiod',
			'mode_and_time',
			'actualdest',
		]
	)

	#
	# display(HTML(f"<h4>auto_dist statistics</h4>"))
	# display(trip_alt_df['auto_dist'].statistics())
	#
	# display(HTML(f"<h4>altdest0001_auto_dist statistics</h4>"))
	# display(trip_alt_df['altdest0001_auto_dist'].statistics())
	#


	L("## invalid_walktime ##")
	for purpose3 in ['HW', 'HO', 'NH']:
		invalid_walktime = trip_alt_df[f'actualdest_transit_approach_walktime_{purpose3}'] > 180
		trip_alt_df.loc[invalid_walktime, f'actualdest_transit_approach_walktime_{purpose3}'] = np.nan

	L("## invalid_drivetime ##")
	for purpose3 in ['HW', 'HO', 'NH']:
		invalid_drivetime = trip_alt_df[f'actualdest_transit_approach_drivetime_{purpose3}'] > 180
		trip_alt_df.loc[invalid_drivetime, f'actualdest_transit_approach_drivetime_{purpose3}'] = np.nan

	for i in range(n_sampled_dests):
		for purpose in ['HW', 'HO', 'NH']:
			L(f"# approach simulate for altdest{i + 1:04d} {purpose} ")
			q = (trips.tripPurpose == purpose)
			_trips_by_purpose = trip_alt_df[q]
			result_purpose = transit_approach(
				dh,
				_trips_by_purpose.o_zone,
				_trips_by_purpose[f'altdest{i + 1:04d}'],
				purpose,
				replication=1,
				approach_distances=None,
				trace=False,
				random_state=123 + i,
			)
			for key in ['drivetime', 'walktime', 'cost', 'waittime']:
				v = result_purpose[key][:, 0].astype(float)
				if key in ['drivetime', 'walktime', 'waittime']:
					v[v > 180] = np.nan
				trip_alt_df.loc[q, f'altdest{i + 1:04d}_transit_approach_{key}_{purpose}'] = v

	dh.filenames.save(TRIP_ALTS_CACHE_FILE, trip_alt_df)

base_mode_names = list(trips.mode5.cat.categories)

from ..choice_model import alt_codes_and_names

alt_codes, alt_names = alt_codes_and_names(
	n_sampled_dests=n_sampled_dests,
	modenames=None,
	include_actual_dest=True,
)


dats = Dict()
mods = Dict()
mods_preload = Dict()


purposes = [
	('HBWH', 'hwahi'),
	('HBWL', 'hwalo'),
	('HBO', 'hoa'),
	('NHB', 'nha'),
]


altdest_tags = lambda suffix: [
	f'altdest{i + 1:04d}_{suffix}'
	for i in range(n_sampled_dests)
]

# `ca_folds` defines how ca_folded (see below) is built
ca_folds = {
	"nAttractions_HBWH": ['actualdest_log_attractions_HBWH'] + altdest_tags("log_attractions_HBWH"),
	"nAttractions_HBWL": ['actualdest_log_attractions_HBWL'] + altdest_tags("log_attractions_HBWL"),
	"nAttractions_HBO": ['actualdest_log_attractions_HBO'] + altdest_tags("log_attractions_HBO"),
	"nAttractions_NHB": ['actualdest_log_attractions_NHB'] + altdest_tags("log_attractions_NHB"),
	"auto_dist": ['auto_dist'] + altdest_tags("auto_dist"),
	"auto_time": ['auto_time'] + altdest_tags("auto_time"),
}

# `ca_folded` will contain, by purpose, destination-specific data for later analysis
ca_folded = Dict()

cached_model_filename = lambda purpose: dh.filenames.cache_dir / f"choicemodel_{purpose}_hc.xlsx"
cached_model_filereport = lambda purpose: dh.filenames.cache_dir / f"choicemodel_{purpose}_hc_report.xlsx"

for purpose, purpose_a in purposes:

	# define filter for this trip purpose
	q = (trip_alt_df['tripCat'] == purpose)

	# assemble quantitative (size) factors
	#   We use the attractions defined in the model's Trip Generation step.
	log.debug(f"size_of_altdests for {purpose}")
	size_of_altdests = [
		np.log(np.fmax(
			dh.trip_attractions.loc[
				trip_alt_df[q][f"altdest{i + 1:04d}"],
				purpose_a,
			].reset_index(
				drop=True
			),
			1e-300,  # nonzero but tiny
		)).rename(
			f'altdest{i + 1:04d}_log_attractions_{purpose}'
		)
		for i in range(n_sampled_dests)
	]

	log.debug(f"actualdest_log_attractions for {purpose}")
	_df = pd.concat(
		[
			trip_alt_df[q].reset_index(),
			np.log(np.fmax(
				dh.trip_attractions.loc[
					trips[q].d_zone,
					purpose_a,
				].reset_index(
					drop=True
				),
				1e-300,  # nonzero but tiny
			)).rename(
				f'actualdest_log_attractions_{purpose}'
			),
		] + size_of_altdests,
		axis=1,
	)

	purpose4to3 = {
		'HBWH': 'HW',
		'HBWL': 'HW',
		'HBO': 'HO',
		'NHB': 'NH',
	}

	peaky = 'PEAK' if 'W' in purpose else 'OFFPEAK'
	for i in range(n_sampled_dests):
		positive_attractions = (_df[f'altdest{i + 1:04d}_log_attractions_{purpose}'] > -666)
		_df[f'altdest{i + 1:04d}_transit_avail_{purpose}'] = (
			(  _df[f'altdest{i + 1:04d}_transit_ivtt_{peaky}'] < 999)
			& (_df[f'altdest{i + 1:04d}_transit_approach_walktime_{purpose4to3[purpose]}'] < 999)
			& (_df[f'altdest{i + 1:04d}_transit_approach_drivetime_{purpose4to3[purpose]}'] < 999)
			& positive_attractions
		)
		_df[f'altdest{i + 1:04d}_auto_avail_{purpose}'] = positive_attractions
	# Build IDCA folded data for later analysis
	_ca_folded = {}
	for k, v in ca_folds.items():
		if 'nAttractions' in k:
			if purpose not in k:
				continue
			folder = np.exp(_df[v])
		else:
			folder = _df[v]
		folder.columns = range(len(v))
		_ca_folded[k] = folder.stack().rename(k)
	ca_folded[purpose] = pd.DataFrame(_ca_folded)

	# Model's actual data
	dfs = dats[purpose] = larch.DataFrames(
		co=_df,
		alt_codes=alt_codes,
		alt_names=alt_names,
		ch='mode5code',
		# av=pd.DataFrame({
		# 	k: _df.eval(v)
		# 	for k, v in av.items()
		# }).astype(np.int8),
	)

	cached_model_file = cached_model_filename(purpose)
	if os.path.exists(cached_model_file) and False:
		m = mods[purpose] = larch.Model.load(
			cached_model_file
		)
		mods_preload[purpose] = True
	else:
		m = mods[purpose] = model_builder(
			purpose,
			include_actual_dest=True,
			n_sampled_dests=n_sampled_dests,
			parameter_values=None,
			constraints=False,
		)
		mods_preload[purpose] = False
	m.dataservice = dfs
	m.load_data()
	m.diagnosis = m.doctor(repair_ch_av="-")


with open(dh.filenames.choice_model_param_file, 'w', encoding="utf-8") as cmp_yaml:
	print("---", file=cmp_yaml)
	for purpose, purpose_a in purposes:
		print(f"{purpose}:", file=cmp_yaml)
		for k, v in mods[purpose].pf.value.items():
			if ':' in k:
				k = f'"{k}"'
			print(f'  {k:24s}: {v}', file=cmp_yaml)
		print("", file=cmp_yaml)
	print("...", file=cmp_yaml)


## TODO for Mode/Dest Model ##

"""

- treatments for tolling, HOV usage

- treatment for vehicle occupancy

- For TNC LoS:
	- We will be working with only 2 time periods, peak and off-peak,
	  so the relationship between distance & time and cost needs to be
	  aggregated to just these two buckets
	- We will need to split out the fare into the TNC actual fare
	  as a function of time and distance, plus the locale-specific taxes 
	  and fees.  This is not important for estimation but it will be for 
	  application so let's do it up front. The data is 2019, but there's
	  already a different tax structure in the city and CMAP may want to
	  examine changing it even more.

- Check per-mile auto operating cost is consistent with rest of model

- Parking price at destination
	- mirror treatment per unit time from current model?

- Intra-zonal factors?  Vary by mode?

- Quantity(Size) terms on destinations

- treatment of income levels for home-based trips


"""

L("## model parameter estimation ##")

Pr = Dict()

nests_per_dest = 2 # CHANGE ME if/when the number of nests per destination is altered in `model_builder`



figures = Dict()

def dest_profiler(
	purpose,
):
	_offset = (n_sampled_dests + 1) * 5+nests_per_dest-1
	_ch = mods[purpose].dataframes.data_ch_cascade(mods[purpose].graph)\
		      .iloc[:, _offset:-1:nests_per_dest].stack().values
	_av = mods[purpose].dataframes.data_av_cascade(mods[purpose].graph)\
		      .iloc[:, _offset:-1:nests_per_dest].stack().values

	figdef = Dict()
	figdef['auto_dist'].bins = 50
	figdef['auto_dist'].range = (0, 50)
	figdef['auto_time'].bins = 60
	figdef['auto_time'].range = (0, 60)

	for x in figdef:

		figures.distribution[purpose][x] = distribution_figure(
			ca_folded[purpose][x],
			probability=Pr.ByDest[purpose].stack().values,
			choices=_ch,
			availability=_av,
			xlabel=None,
			ylabel='Relative Frequency',
			style='hist',
			bins=figdef[x].bins,
			pct_bins=20,
			range=figdef[x].range,
			prob_label="Modeled",
			obs_label="Observed",
			bw_method=None,
			discrete=None,
			ax=None,
			format='png',
			#header=f"{purpose} / {x} Distribution",
			accumulator=True,
		)
		display(figures.distribution[purpose][x])


def mode_share_profiler(
		purpose,
):
	d_codes = np.arange(n_sampled_dests+1)

	_pr = Pr.ByMode[purpose].copy()
	_pr.columns = pd.MultiIndex.from_product([
		d_codes,
		['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT'],
	])
	# _pr = _pr.stack([0,2]).sum(1).unstack()

	_ch = mods[purpose].dataframes.data_ch.copy()
	_ch.columns=pd.MultiIndex.from_product([
		d_codes,
		['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT'],
	])
	# _ch = _ch.stack([0,2]).sum(1).unstack()

	figdef = Dict()
	figdef['auto_dist'].bins = np.logspace(np.log10(1),np.log10(51),10)-1
	# np.concatenate([
	# 	np.arange(0, 10, 1),  # first 1 mile bins to 10 miles
	# 	np.arange(10, 20, 2), # then 2 mile bins to 20 miles
	# 	np.arange(20, 51, 5), # then 5 mile bins to 50 miles
	# ])

	for x in figdef:

		figures.share[purpose][x] = share_figure(
			x=ca_folded[purpose][x],
			probability=_pr.fillna(0),
			choices=_ch,
			style='stacked',
			bins=figdef[x].bins,
			format='png',
			#header=f"{purpose} Mode Share by {x}"
			xscale='symlog',
			xmajorticks=[0, 1, 2, 5, 10, 20, 50],
			xminorticks=np.concatenate([
				np.arange(0, 10),
				np.arange(10, 20, 2),
				np.arange(20, 50, 3),
			]),
		)
		display(figures.share[purpose][x])

# def timeshare_profiler(purpose):
# 	d_codes = np.arange(n_sampled_dests + 1)
#
# 	_pr = Pr.ByMode[purpose].copy()
# 	_pr.columns = pd.MultiIndex.from_product([
# 		d_codes,
# 		timeperiod_names,
# 		['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT'],
# 	])
# 	_pr1 = _pr.stack([1, 2]).sum(1).unstack([1, 2]).sum().unstack().reindex(timeperiod_names)
#
# 	_ch = mods[purpose].dataframes.data_ch.copy()
# 	_ch.columns = pd.MultiIndex.from_product([
# 		d_codes,
# 		timeperiod_names,
# 		['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT'],
# 	])
# 	_ch1 = _ch.stack([1, 2]).sum(1).unstack([1, 2]).sum().unstack().reindex(timeperiod_names)
#
# 	_pr_detail = _pr1.drop(columns=['AUTO'])
# 	_pr_gross = pd.DataFrame([
# 		_pr1.AUTO, _pr_detail.sum(1).rename("OTHER"),
# 	]).T
#
# 	_ch_detail = _ch1.drop(columns=['AUTO'])
# 	_ch_gross = pd.DataFrame([
# 		_ch1.AUTO, _ch_detail.sum(1).rename("OTHER"),
# 	]).T
#
# 	from matplotlib import pyplot as plt
# 	fig, axs = plt.subplots(
# 		2, 2,
# 		figsize=(12, 10),
# 		sharey='row', sharex='col',
# 		gridspec_kw={'wspace': 0.1, 'hspace': 0.1}
# 	)
# 	_ch_gross.plot(kind='bar', stacked=True, ax=axs[0][1])
# 	_pr_gross.plot(kind='bar', stacked=True, ax=axs[0][0])
# 	_ch_detail.plot(kind='bar', stacked=True, ax=axs[1][1])
# 	_pr_detail.plot(kind='bar', stacked=True, ax=axs[1][0])
# 	axs[0][0].set_title("Modeled Time Periods")
# 	axs[0][1].set_title("Observed Time Periods")
# 	axs[0][0].set_ylabel("Relative Frequency")
# 	axs[1][0].set_ylabel("Relative Frequency")
# 	axs[0][0].set_yticks([])
# 	axs[1][0].set_yticks([])
#
# 	from larch.util.png import make_png
# 	result = make_png(fig)
# 	fig.clf()
# 	plt.close(fig)
#
# 	figures.timeshare[purpose] = result
# 	display(result)


def mode_choice_summary(m):
	ch_av_summary = m.dataframes.choice_avail_summary().iloc[:-1]
	for k in ['available', 'available weighted', 'available unweighted']:
		try:
			ch_av_summary[k] = ch_av_summary[k].astype(int)
		except KeyError:
			pass
	pr_summary = m.probability(return_dataframe=True).sum()
	ch_av_summary['model prob'] = pr_summary
	ch_av_summary.index = pd.MultiIndex.from_product(
		[
			np.arange(n_sampled_dests + 1),
			['AUTO', 'TAXI', 'TNC1', 'TNC2', 'TRANSIT'],
		],
		names=['dest', 'mode'],
	)
	result = ch_av_summary.groupby('mode').sum()
	display(HTML(f"<h4>{m.title} Mode Choices Summary</h4>"))
	display(result)
	return result

def estimation():
	for purpose, m in mods.items():

		# if purpose != 'HBWH': continue # TODO stop short circuit

		m.dataframes.autoscale_weights()
		display(HTML(f"<h3>{m.title}</h3>"))
		if not mods_preload[purpose]:
			m.maximize_loglike()
			m.calculate_parameter_covariance()
			m.to_xlsx(
				cached_model_filename(purpose)
			)
			summary = m.most_recent_estimation_result.copy()
			summary.pop('x', None)
			summary.pop('d_loglike', None)
			summary.pop('nit', None)
			display(larch.util.dictx(summary).__xml__())
			display(m.estimation_statistics())
		_pr = m.probability(return_dataframe='names', include_nests=True)
		n_elemental_alts = (n_sampled_dests+1)*5
		Pr.ByDest[purpose] = _pr.iloc[:,n_elemental_alts+nests_per_dest-1:-1:nests_per_dest]
		Pr.ByMode[purpose] = _pr.iloc[:,:n_elemental_alts]

		display(HTML(f"<h2>{m.title}</h2>"))
		display(m.parameter_summary())

		try:
			mode_choice_summary(m)
		except:
			log.exception("exception in mode_choice_summary")
			mode_choice_summary_success = False
		else:
			mode_choice_summary_success = True

		try:
			dest_profiler(purpose)
		except:
			log.exception("exception in dest_profiler")
			dest_profiler_success = False
		else:
			dest_profiler_success = True

		try:
			mode_share_profiler(purpose)
		except:
			log.exception("exception in mode_share_profiler")
			mode_share_profiler_success = False
		else:
			mode_share_profiler_success = True

		xl = m.to_xlsx(
			cached_model_filereport(purpose),
			save_now=False
		)
		if dest_profiler_success:
			xl.add_content_tab(
				figures.distribution[purpose]['auto_dist'],
				sheetname="Figures",
				heading="Destination Probabilities by Distance",
			)
			xl.add_content_tab(
				figures.distribution[purpose]['auto_time'],
				sheetname="Figures",
				heading="Destination Probabilities by Auto Travel Time",
			)
		if mode_share_profiler_success:
			xl.add_content_tab(
				figures.share[purpose]['auto_dist'],
				sheetname="Figures",
				heading="Mode Choice by Distance",
			)
		xl.save()

	return mods

resource_usage.check()
L("## est_choice complete ##")
