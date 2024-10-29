import numpy as np

try:
    import cupy as ph
except:
    import numpy as ph

# Returns a given array as a powerhouse array
def aspha(a):
    if isinstance(a, ph.ndarray):
        return a
    return ph.asarray(a)

# Returns a given array as a numpy array
def asnpa(a):
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)
