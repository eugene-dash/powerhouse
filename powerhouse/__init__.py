# Returns a given array as a powerhouse array
def aspha(a):
    if isinstance(a, powerhouse.ndarray):
        return a
    return powerhouse.asarray(a)

# Returns a given array as a numpy array
def asnpa(a):
    if isinstance(a, np.ndarray):
        return a
    return np.asarray(a)
