import pandas as pd
from addict import Dict

def load_tg(filenames):

	tripgen = pd.read_sas(filenames.tripgen_sas)
	tripgen['zone17'] = tripgen['zone17'].astype(int)

	trip_attractions = tripgen.groupby("zone17")[['hwalo','hwahi','hoa','nha', 'area']].sum()

	trip_productions = tripgen.groupby("zone17")[['hwplo','hwphi','hop','nhp']].sum()

	zone_productions = trip_productions.round().astype(int)
	zone_productions = zone_productions.rename(columns=dict(
		hwplo='HBWL',
		hwphi='HBWH',
		hop='HBO',
		nhp='NHB',
	))

	tg = Dict()
	tg.trip_attractions = trip_attractions
	tg.trip_productions = trip_productions
	tg.zone_productions = zone_productions
	return tg
