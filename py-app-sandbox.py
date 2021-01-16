import numpy as np
import pandas as pd

from cmap_trip.util import resource_usage, start_memory_tracing, ping_memory_tracing


from cmap_trip.estimation.est_logging import *
from cmap_trip.estimation.est_config import *
from cmap_trip.estimation.est_data import dh
from cmap_trip.application import choice_simulator_trips, choice_simulator_trips_many

L("#### Single process application test ####")
sim_trips = choice_simulator_trips(
    dh,
    otaz=np.arange(1, 1+50),
)


sim_trips

L("#### Multiprocess application test ####")
sim_trips5 = choice_simulator_trips_many(
    dh,
    max_chunk_size=50,
    n_jobs=2,
)

L("#### application test complete ####")


