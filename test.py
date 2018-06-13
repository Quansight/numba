import numba.plures.typing
from numba import jit
from xnd import xnd

@jit(nopython=True)
def hi(x):
    print("has error", x.err_occurred)
    print("type", x.type)
    print("value", x)
    return x

print(hi(xnd([10])))
