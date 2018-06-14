import numba.plures.typing
from numba import jit
from xnd import xnd


@jit(nopython=True)
def hi(x):
    return x


print(hi(xnd(10)))

