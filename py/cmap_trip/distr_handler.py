import logging
log = logging.getLogger('CMAP')
import pandas as pd
from addict import Dict
from .filepaths import filenames

distr = Dict()

def read_distr(filename):
	raw = pd.read_csv(filename, header=None, index_col=0)
	raw.columns = pd.MultiIndex.from_product(
		[
			['metra','ctarail','bus','feederbus','pnr'],
			['p1','p2','p3'],
		],
		names = ['submode', 'param']
	)
	raw.index.name = 'zone'
	result = raw.stack('submode')
	return result

distr.HW = read_distr(filenames.PDHW_DISTR)
distr.HO = read_distr(filenames.PDHO_DISTR)
distr.NH = read_distr(filenames.PDNH_DISTR)
