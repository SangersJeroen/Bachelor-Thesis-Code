from numba import njit, guvectorize, float32, int32
from numba.core.decorators import jit_module
from numba import vectorize
import numpy as np


@njit
def get_mask(r1: int, radii3d: np.ndarray):
    a = np.where( radii3d<r1, True, False)
    return a

@njit
def entries(mask):
    entries = np.sum( np.sum(mask, axis=2), axis=1)
    return entries

#@guvectorize(['float32[:](float32[:,:,:], boolean[:,:,:])', '(m,n,n),(m,n,n)->(m)'])
@njit
def sum(stack, mask):
    area = np.where(mask, stack, 0)
    integral = np.sum( np.sum( area, axis=2), axis=1)
    return integral

def lis(r1, stack, radii3d):
    mask = get_mask(r1, radii3d)
    sum_ = sum(stack, mask)
    entries_ = entries(mask)
    integral = sum_ / entries_
    return integral