
import numba
import numpy as np
from numpy import log, exp, log1p, expm1
from sharrow.maths import piece, hard_sigmoid
from .extra_funcs import *
from .extra_vars import *

@numba.njit(cache=True, error_model='numpy', boundscheck=False)
def log_attractions_HBS____666_L3ZMVRT365KFSBLEDVABFEGL(
    _args, 
    _inputs, 
    _outputs,
    __attractions__HBS
):
    return log(__attractions__HBS[_args[1],]) > -666
