#=----=#
#------#
#=----=#
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
#',',',',',',',',',',-==-,',',',',',',',',','|
#''''''''''''''''''''''''''''''''''''''''''''|
import numpy as np
np.seterr(divide="raise")
try:
    import cupy as cp
except:
    import numpy as cp
# /\ /\ /\ /\ /\ /\ /\ /\ /\|
#//////////IMPORTS//////////|
#'''''''''''''''''''''''''''|

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
#//////////PRIMARY_FUNCTIONS//////////|
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ |
cpdef a2cp(a):
    if isinstance(a, cp.ndarray):
        return a
    return cp.asarray(a)
cpdef a2np(a):
    if isinstance(a, np.ndarray):
        return a
    elif isinstance(a, cp.ndarray):
        return a.get()
    return np.asarray(a)
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ |
#//////////PRIMARY_FUNCTIONS//////////|
#'''''''''''''''''''''''''''''''''''''|

