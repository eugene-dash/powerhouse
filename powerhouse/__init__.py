import numpy as np
try:
    import cupy as cp
    def get(a):# cp.ndarry -> np.ndarray
        return a.get()
    def give(a):# np.ndarry -> cp.ndarray
        return cp.asarray(a)
except:
    cp = np
    def get(a):# np.ndarry -> np.ndarray
        return a
    def give(a):# np.ndarry -> np.ndarray
        return a
def iscp(a):
    if isinstance(a, cp.ndarray):
        return True
    return False
def isnp(a):
    if isinstance(a, np.ndarray):
        return True
    return False
