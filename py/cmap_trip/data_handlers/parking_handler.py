import numpy as np
import pandas as pd
from addict import Dict

def load_cbd_parking(filenames):

	cbd_parking = pd.read_csv(
		filenames.HW_CBDPARK,
		header=None,
		names=['ZoneID', 'CumProb', 'ThresholdPrice', 'SavePrice', 'WalkSeconds'],
	)
	cbd_parking.CumProb /= 10000.
	# cbd_parking['SumPrice'] = cbd_parking.ThresholdPrice + cbd_parking.SavePrice
	cbd_parking['rownum'] = cbd_parking.groupby(['ZoneID']).cumcount()
	_z = cbd_parking.ZoneID.value_counts().sort_index().index
	CBD_PARKING_ZONES = dict(zip(_z, np.arange(len(_z))))


	def decumulate(x):
		x_ = np.array(x)
		x_[1:] -= x[:-1]
		return x_
	cbd_parking['Prob'] = cbd_parking.groupby("ZoneID")['CumProb'].transform(decumulate)
	cbd_parking['WeightedPrice'] = cbd_parking['Prob'] * cbd_parking['ThresholdPrice']
	cbd_parking_prices = cbd_parking.set_index(["ZoneID",'rownum']).ThresholdPrice.unstack()
	cbd_parking_price_prob = cbd_parking.set_index(["ZoneID",'rownum']).Prob.unstack()

	cbd_parking2 = pd.read_csv(
		filenames.HW_CBDPARK2,
		header=None,
		names=[
			'IncomeCeiling',
			'FreeParkingPct',
			'TransitPct',
			'AutoOcc1Pct',
			'AutoOcc2Pct',
			'AutoOcc3Pct',
			'AutoOcc4Pct',
		],
	)

	parking = Dict()
	parking.cbd_parking = cbd_parking
	parking.cbd_parking_prices = cbd_parking_prices
	parking.cbd_parking_price_prob = cbd_parking_price_prob
	parking.cbd_parking2 = cbd_parking2
	parking.CBD_PARKING_ZONES = CBD_PARKING_ZONES
	return parking
