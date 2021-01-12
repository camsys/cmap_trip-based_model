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
    purpose="HBWH",
    otaz=np.arange(1, 1+10),
)


L("#### Multiprocess application test ####")
sim_trips5 = choice_simulator_trips_many(
    dh,
    purpose="HBWH",
    otaz=np.arange(1, 1+360),
    chunk_size=15,
    n_jobs=8,
)

L("#### application test complete ####")


