
import numba
import numpy as np
from numpy import log, exp, log1p, expm1
from sharrow.maths import piece, hard_sigmoid



@numba.njit(cache=True, error_model='numpy', boundscheck=False)
def auto_avail_HBS(
    _args, 
    _inputs, 
    _outputs,
    
):
    return _inputs[3] > -9998
