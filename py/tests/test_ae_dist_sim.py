import pytest
from pytest import approx
import numpy as np

def test_simulate_ae_dist():
	from cmap_trip.ae_distance_sim import simulate_ae_dist
	p1 = [0.89, 0.85, 0.9 , 0.83, 0.75]
	p2 = [0.2, 0.2, 0.2, 0.2, 0.2]
	p3 = [101, 101, 101, 101, 101]
	assert simulate_ae_dist(p1, p2, p3, random_state=1).reshape(-1) == approx(
		[1.214869, 0.72764874, 0.79436564, 0.6154063, 0.9230815]
	)
	assert simulate_ae_dist(p1, p2, p3, random_state=1, replication=5) == approx(
		np.array([
			[1.214869  , 0.72764874, 0.79436564, 0.6154063 , 0.9230815 ],
			[0.42969227, 1.1989623 , 0.7477586 , 0.8938078 , 0.70012593],
			[1.1824216 , 0.43797186, 0.8355166 , 0.75318915, 0.9767539 ],
			[0.6700218 , 0.8155144 , 0.7244283 , 0.83844274, 0.866563  ],
			[0.66987616, 1.0789447 , 1.0803181 , 0.93049884, 0.9301712 ],
		]).T
	)

	p1 = [0.1, 0.1, 0.1, 0.1, 10.1]
	p2 = [0.2, 0.2, 0.2, 0.2, 10.2]
	p3 = [0.25, 0.369, 0.802, 0.846, 0.913]
	assert simulate_ae_dist(p1, p2, p3, random_state=1).reshape(-1) == approx(
		[0.1564524, 0.17953482, 0.10001285, 0.13205005, 10.1152638]
	)
	assert simulate_ae_dist(p1, p2, p3, random_state=1, replication=5) == approx(
		np.array([
			[ 0.1564524 ,  0.18119404,  0.10002858,  0.14508852,  0.12630762],
			[ 0.1151631 ,  0.12789729,  0.14600554,  0.1511932 ,  0.16444318],
			[ 0.14463477,  0.1707938 ,  0.12235227,  0.18889633,  0.10306524],
			[ 0.16883636,  0.14378384,  0.15790248,  0.11510868,  0.12120398],
			[10.18078   , 10.19696   , 10.132338  , 10.170184  , 10.188115  ],
		])
	)
