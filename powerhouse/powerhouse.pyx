#=----=#
#------#
#=----=#
#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
#',',',',',',',',',',-==-,',',',',',',',',','|
#''''''''''''''''''''''''''''''''''''''''''''|
import numpy as np
#np.seterr(divide="raise")
try:
    import cupy as powerhouse
except:
    import numpy as powerhouse
# /\ /\ /\ /\ /\ /\ /\ /\ /\|
#//////////IMPORTS//////////|
#'''''''''''''''''''''''''''|

#,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,|
#//////////PRIMARY_FUNCTIONS//////////|
# \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ |
cpdef aspha(a):
    if isinstance(a, powerhouse.ndarray):
        return a
    return powerhouse.asarray(a)
cpdef asnpa(a):
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)
# /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ |
#//////////PRIMARY_FUNCTIONS//////////|
#'''''''''''''''''''''''''''''''''''''|

