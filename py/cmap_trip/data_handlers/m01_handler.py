
import logging
log = logging.getLogger('CMAP')
import pandas as pd
from addict import Dict


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


def load_m01(filenames):
	m01 = Dict()

	m01.HW = read_m01(filenames.PDHW_M01)
	# There is only one unique m01 file
	#   the others are simply copies
	# m01.HO = read_m01(filenames.PDHO_M01)
	# m01.NH = read_m01(filenames.PDNH_M01)

	m01.HW['taxi_wait_pk'] = m01.HW.zone_type.map(filenames.cfg.taxi.wait_time.peak)
	m01.HW['taxi_wait_op'] = m01.HW.zone_type.map(filenames.cfg.taxi.wait_time.offpeak)

	m01.HW['tnc_solo_wait_pk'] = m01.HW.zone_type.map(filenames.cfg.tnc.wait_time.peak)
	m01.HW['tnc_solo_wait_op'] = m01.HW.zone_type.map(filenames.cfg.tnc.wait_time.offpeak)
	m01.HW['tnc_pool_wait_pk'] = m01.HW.zone_type.map(filenames.cfg.tnc_pooled.wait_time.peak)
	m01.HW['tnc_pool_wait_op'] = m01.HW.zone_type.map(filenames.cfg.tnc_pooled.wait_time.offpeak)

	return m01.HW
