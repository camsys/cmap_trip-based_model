from test_transit_approach import test_transit_approach

from cmap_trip.transit_approach import transit_approach
from cmap_trip.distr_handler import distr

out = transit_approach(
	ozone=[ 844,  844,  844,  863,  955,  955,  817,  817,  844,    7,    7,
        330,  330,   33,   33,  644, 1482,  652,  652,  644,  644, 1482,
        653,  644, 2583, 2583, 2583, 2570, 2583, 2583, 2583,  202,  201,
        202,  202,  201,  202,  621, 1549, 1491, 1491,  368,  368,  368,
        368, 1008, 1008, 1008, 1008, 1083],
	dzone=[ 955,  955,  863,  864,  864,  817,  817,  791,  791,  343,  343,
        283,  283,   57,   57, 1482,  669,  669, 1627, 1627, 1482,  653,
        643,  643,  958, 2570,  958, 2580, 2580,  958,  958,  201, 1114,
        201, 1114, 1114, 1114, 1549, 1491, 1491,  621,  418,  418,  327,
        327,  364,  364,   28,   28, 2422],
	TPTYPE='HW',
	replication=10,
	approach_distances=None,
	trace=False,
	random_state=789,
)

print(distr.HW.loc[[123,456]])


print(out.approach_distances.mean(1))

print(out.drivetime.mean(1))
print(out.walktime.mean(1))
print(out.cost.mean(1))
print(out.waittime.mean(1))

q = 4

test_transit_approach()

