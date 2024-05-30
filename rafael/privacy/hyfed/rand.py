import secrets
import sys
from functools import reduce
import numpy as np


STRONG_SECRET = secrets.SystemRandom()
# INT64_MAX = sys.maxsize
# INT64_MIN = -sys.maxsize - 1


# def randint64(n: int):
#     """generate random 64-bit integers"""
#     x = [STRONG_SECRET.randint(INT64_MIN, INT64_MAX) for _ in range(n)]
#     return np.array(x)

# def rand(a: float = 0., b: float = 1., n: int = 1, dtype=np.float32):
#     x = [STRONG_SECRET.uniform(dtype(a), dtype(b)) for _ in range(n)]
#     return np.array(x)

# def randn(mu: float = 0., sigma: float = 1., n: int = 1, dtype=np.float32):
#     x = [STRONG_SECRET.gauss(dtype(mu), dtype(sigma)) for _ in range(n)]
#     return np.array(x)

def randint(low, high=None, size=None):
    """ Similar to np.random.randint(), but the default dtype is int64 """
    if size is None:
        n_elements = 1
    else:
        n_elements = reduce(lambda x, y: x*y, size)
    
    if high is None:
        low = 0
        high = low
    
    x = [STRONG_SECRET.randint(low, high) for _ in range(n_elements)]
    return np.array(x).reshape(*size)

def rand(*ds):
    """ Similar to np.random.rand(), the dtype is float32 """
    if len(ds) == 0:
        n_elements = 1
        ds = (1,)
    else:
        n_elements = reduce(lambda x, y: x*y, ds)

    x = [STRONG_SECRET.uniform(0., 1.) for _ in range(n_elements)]
    return np.array(x).reshape(*ds)

def randn(*ds):
    if len(ds) == 0:
        n_elements = 1
        ds = (1,)
    else:
        n_elements = reduce(lambda x, y: x*y, ds)
        
    x = [STRONG_SECRET.gauss(0., 1.) for _ in range(n_elements)]
    return np.array(x).reshape(*ds)
