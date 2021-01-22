import numpy as np
import pandas as pd

from cmap_trip.util import resource_usage, start_memory_tracing, ping_memory_tracing


from cmap_trip.estimation.est_logging import *
from cmap_trip.estimation.est_config import *
from cmap_trip.estimation.est_data import dh
from cmap_trip.application import choice_simulator_trips, choice_simulator_trips_many, choice_simulator_initialize

# from cmap_trip.application import data_for_application, _data_for_application_1
#
# otaz = np.arange(1,5)
#
# log.debug("choice_simulator_prob data_for_application")
# dfa = data_for_application(dh, otaz=otaz, peak=True) # TODO no more peak
#
# log.debug("choice_simulator_prob data_for_application")
# dfa = data_for_application(dh, otaz=np.arange(1,11), peak=True) # TODO no more peak
# log.debug("choice_simulator_prob data_for_application .")
#
#
# # +
# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
#
# # code you want to profile
# dfa = data_for_application(dh, otaz=np.arange(1,6), peak=True) # TODO no more peak
# #df1 = _data_for_application_1(dh, otaz=1, peak=True, purpose='HBWH', replication=50, debug1=True)
#
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True))
#
# # +
# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
#
# # code you want to profile
# dfa = data_for_application(dh, otaz=np.arange(1,6), peak=True) # TODO no more peak
# #df1 = _data_for_application_1(dh, otaz=1, peak=True, purpose='HBWH', replication=50, debug1=True)
#
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True))
#
# # +
# from pyinstrument import Profiler
# profiler = Profiler()
# profiler.start()
#
# # code you want to profile
# # dfa = data_for_application(dh, otaz=np.arange(1,11), peak=True) # TODO no more peak
# df1 = _data_for_application_1(dh, otaz=1, peak=True, purpose='HBWH', replication=50, debug1=True)
#
# profiler.stop()
# print(profiler.output_text(unicode=True, color=True))
#
# # +
# # log.debug("choice_simulator_prob settings")
# # replication = dh.cfg.get('n_replications', 50)
#
# # choice_simulator = choice_simulator_initialize(dh)
# # simulated_probability = {}
#
# # for purpose in ['HBWH', 'HBWL', 'HBO', 'NHB']:
# #     sim = choice_simulator[purpose]
# #     log.debug(f"choice_simulator_prob {purpose} attach dataframes")
# #     if sim.dataframes is None:
# #         sim.dataframes = dfa
# #     else:
# #         sim.set_dataframes(dfa, False)
# #     log.debug(f"choice_simulator_prob {purpose} simulate probability")
# #     sim_pr = sim.probability()
# #     log.debug(f"choice_simulator_prob {purpose} blockwise_mean")
# #     simulated_probability[purpose] = blockwise_mean(sim_pr, replication)
#
# # log.debug("choice_simulator_prob complete")
# # return simulated_probability #.reshape([sim_pr.shape[0],-1,5])
# # -
#
# from cmap_trip.application import _data_for_application_1
# peak = True
# purpose = 'HBWH'
# replication = 50
# n_zones = dh.n_internal_zones
# otaz = [1,2,3]
#
# d1 = [
#     _data_for_application_1(dh, otaz=z, peak=peak, purpose=purpose, replication=replication, debug1=True)
#     for z in otaz
# ]
#
# df1
#
# df2 = d1[0].set_index(['rep', 'altdest'])
#
# df3 = pd.DataFrame(
# 		df2.to_numpy(dtype=np.float64).reshape(replication, -1),
# 		columns=pd.MultiIndex.from_product([
# 			[f"altdest{x:04d}" for x in range(1,n_zones+1)],
# 			df2.columns,
# 		])
# 	)
# df3.columns = [f"{j[0]}_{j[1]}" for j in df3.columns]
#
# df3
#
# d1[0].set_index(['rep', 'altdest']).unstack()
#
# d1[0]
#
# d3 = d1[0].set_index(['rep'])
#
# qq = dict(list(d3.groupby('altdest')))
#
# qq['altdest0001']
#
# # +
# # pd.concat([f for (_,f) in qq[:3]])
#
# pd.concat(qq.values(), keys=qq.keys(), axis=1)
#
#
# # +
# # d3.groupby?
# # -
#
# def stash(i, *args, **kwargs):
#     df = _data_for_application_1(*args, **kwargs)
#     df.to_parquet(f"/tmp/cmap/{i}.pq")
#
#
#
# # +
# import joblib
# n_jobs = 5
#
# with joblib.Parallel(n_jobs=n_jobs, verbose=100) as parallel:
# #     if init_step:
# #         log.info("joblib model init starting")
# #         _ = parallel(
# #             joblib.delayed(choice_simulator_initialize)(dh, False)
# #             for _ in inits
# #         )
# #         log.info("joblib model init complete")
# #     else:
# #         log.info("joblib model body starting")
#     parts = parallel(
#         joblib.delayed(stash)(z, dh, otaz=z, peak=peak, purpose=purpose, replication=replication)
#         for z in np.arange(1,51)
#     )
#     log.info("joblib model body complete")
#
# # -
#
#

from cmap_trip.application import choice_simulator_initialize
m = choice_simulator_initialize(dh).HBO

# +
L("#### Single process application test ####")
from cmap_trip.util import profiler

with profiler():
    sim_trips = choice_simulator_trips(
        dh,
        otaz=np.arange(1, 1+60),
        n_threads=-1,
    )

# +
L("#### Single process application test ####")
from cmap_trip.util import profiler

with profiler():
    sim_trips = choice_simulator_trips(
        dh,
        otaz=np.arange(61, 61+60),
        n_threads=-1,
    )
# -


STOP

L("#### Multiprocess application test ####")
sim_trips5 = choice_simulator_trips_many(
    dh,
    otaz = np.arange(360)+1,
    max_chunk_size=20,
    n_jobs=8,
)

L("#### application test complete ####")

# +
from cmap_trip.application import data_for_application

otaz = np.arange(1,3)
peak=True
purpose='HBWH'
replication=50

# +
from cmap_trip.util import profiler

with profiler():

    s1 = data_for_application(dh, otaz=1, peak=True, purpose='HBWH', replication=None) # 11.815
# -

with profiler():

    s2 = data_for_application(dh, otaz=2, peak=True, purpose='HBWH', replication=None) # 7.713

with profiler():

    s3 = data_for_application(dh, otaz=np.arange(3,9), peak=True, purpose='HBWH', replication=None) # 14.136

# +

s1.to_feathers( "/tmp/dataframes_s111.dfs")
s3.to_feathers( "/tmp/dataframes_s3.dfs")
# -

s1.data_co['altdest1234_auto_time_NIGHT'].iloc[0], s2.data_co['altdest1234_auto_time_NIGHT'].iloc[0]

s2.inject_feathers("/tmp/dataframes_s3.dfs")

s1.data_co['altdest1234_auto_time_NIGHT'].iloc[0], s2.data_co['altdest1234_auto_time_NIGHT'].iloc[0]

# +
filename='/private/tmp/dataframes_s3.dfs'

tb = pf.read_table(str(filename)+".metadata")
tb
# -

'altcodes' in tb.column_names
tb['altcodes'].to_numpy()

import pyarrow.feather as pf

np.exp(-666), np.log(1e-321)

int(b'123')


def segment_out(seg):
    array_ = getattr(self, f'array_{seg}')()
    if array_ is not None and seg in components:
        tb = pa.table([array_.T.reshape(-1)], [f'data_{seg}'])
        if 'meta' in components:
            metadata = tb.schema.metadata or {}
            data_ = getattr(self, f'data_{seg}')
            columns = pickle.dumps(data_.columns)
            metadata[b'COLUMNS'] = pa.compress(columns, asbytes=True)
            metadata[b'COLUMNSBYTES'] = str(len(columns))
            new_schema = tb.schema.with_metadata(metadata)
            tb = tb.cast(new_schema)
        pf.write_feather(tb, str(filename)+f".data_{seg}")



s2.__getattribute__("data_co")

s2.array_co()[:] = 0

s2_data_co['altdest1234_auto_time_NIGHT'].iloc[0]


def inject_feathers(self, filename, components=None):
    import pyarrow.feather as pf
    import os

    if components is None:
        components = {'co','ca','ce','wt','av','ch'}
    else:
        components = set(components)

    # core data
    if self.array_co() is not None and 'co' in components:
        filename_co = str(filename)+".data_co"
        tb = pf.read_table(filename_co)
        arr1 = tb['data_co'].to_numpy().reshape(self.array_co().shape[1], self.array_co().shape[0]).T
        self.array_co()[:] = arr1[:]
    if self.array_av() is not None and 'av' in components:
        filename_av = str(filename)+".data_av"
        tb = pf.read_table(filename_av)
        arr = tb['data_av'].to_numpy().reshape(self.array_av().shape[1], self.array_av().shape[0]).T
        self.array_av()[:] = arr[:]

    return arr1


arr1 = inject_feathers(s2, "/tmp/dataframes_s111.dfs")

arr1

s1.data_co.columns.get_loc('altdest1234_auto_time_NIGHT')

s1.data_co.iloc[0,171392]

arr1[0,171392]

s2.array_co()[0,171392]

s2.data_co.iloc[0,171392]

s2.data_co['altdest1234_auto_time_NIGHT'].iloc[0]

from larch.util.dataframe import columnize
from cmap_trip.application import av



# +
# %%prun -s cumulative

columnize(s3.data_co, av, inplace=False)
# -

with profiler():
    jj = eval_many(av, s3.data_co)

jj


def eval_many(d, df):
    dict_of_numpy_arrays = {}
    renaming = {}
    for k in sorted(d.keys()):
        v = d[k]
        if v in df:
            dict_of_numpy_arrays[str(k)] = df[v].to_numpy()
        else:
            dict_of_numpy_arrays[str(k)] = df.eval(v).to_numpy()
        renaming[str(k)] = k
            
    out = pa.table(dict_of_numpy_arrays).to_pandas()
    out.rename(columns=renaming, inplace=True)
    return out


import pyarrow as pa
import pyarrow.feather as pf


tb

with pa.OSFile("/tmp/buch.f2", 'w') as f:
    pf.write_feather(tb, f)
    pf.write_feather(tb2, f)

with pa.OSFile("/tmp/buch.f2", 'r') as f:
    r1 = pf.read_feather(f)
    r2 = pf.read_feather(f)

s3.array_av()

ssss = s3.data_co.columns.to_numpy()

pa.table({
    'altcodes': s3.alternative_codes(),
    'altnames': [f"alt{a}" for a in s3.alternative_codes()],
})


float.fromhex(hh)

hh = s3.weight_normalization.hex()
import struct
struct.unpack('!f', bytes.fromhex(hh))[0]

by

s3.alternative_names()

pa.table([])

import pickle
ppp = pyarrow.compress(pickle.dumps(s3.data_co.columns))
len(ppp)

pickle.loads(ppp)

bbb = s3.data_co.columns.to_numpy().astype('U').tobytes()
len(bbb), pyarrow.compress(bbb).size

np.frombuffer(bbb, dtype='U500')

s3b = (s3.data_co.columns.to_numpy().tobytes())
s3z = pyarrow.compress(s3b)

s3o = pyarrow.decompress(s3z, len(s3b))

s3ob = s3o.to_pybytes()

np.frombuffer(s3ob, dtype='s')

tb = pa.table([s3.data_co.to_numpy(dtype=np.float64).T.reshape(-1)], ['data_co'])
tb2 = pa.table([s3.data_av.to_numpy(dtype=np.int8).T.reshape(-1)], ['data_av'])
# pf.write_feather(tb, str(filename)+".data_co.f2")


tb = pf.read_table(
    "/Users/jeffnewman/Cambridge Systematics/PROJ CMAP Trip-Based - General/Estimation/cache/data_for_application_1.data_av.f2"
)

tb['data_av'].to_numpy().dtype

pd.DataFrame(
    tb['data_co'].to_numpy().reshape(len(dh.column_2_replacement)+1, -1).T,
    columns = dh.column_2_replacement + ['otaz'],
)

25242450 / 504849


