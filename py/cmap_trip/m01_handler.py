
import logging
log = logging.getLogger('CMAP')
import pandas as pd
from addict import Dict
from .filepaths import filenames

m01 = Dict()

def read_m01(filename):
	raw = pd.read_csv(filename, header=None, index_col=0)

	columns = [
		'zone_type',
		'pnr_parking_cost',
		'zone_income',
		'pnr_flag',
		'first_wait_bus_peak',
		'first_wait_bus_offpeak',
		'first_wait_feeder_peak',
		'first_wait_feeder_offpeak',
	]

	# autocc column only appears in HW files
	if len(raw.columns) == len(columns)+1:
		columns.append('autocc')

	raw.columns = columns
	raw.index.name = 'zone'
	return raw


m01.HW = read_m01(filenames.PDHW_M01)
m01.HO = read_m01(filenames.PDHO_M01)
m01.NH = read_m01(filenames.PDNH_M01)

