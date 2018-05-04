import numpy as np
from numba import vectorize, complex64, boolean, jit
from PIL import Image
import os, sys, itertools, time
import itertools

import matplotlib.pyplot as plt
from matplotlib import colors
%matplotlib inline

@jit
def f(z, c):   #iterating function
    return z * z + c
@jit
def does_diverge(z, c, iters):  #checks if a given pixel diverges
#     iters = 3
    for _ in range(iters):
        z = f(z, c)
        if abs(z) > 2: # Diverges
            return 1
    return 0
# Note no @jit tag here because 
""" Credit to https://aboveintelligent.com/what-do-deep-neural-networks-understand-of-fractals-2ae354911601
    for the starter code of generation. Our changes include, speed optimizations (including jit), which reduce   
    the computation time per image by ~.1 seconds, and the ability to set the c parameter.
"""
def generate_julia_set(size, c=False, iterations=50):
    x = np.linspace(-2, 2, size * 3)
    y = np.linspace(-1, 1, size * 2)
    X = np.array(list(itertools.product(y, x)))[:, (1, 0)]
    if not c:
        output = np.array([does_diverge(complex(*sample), complex(*sample), iterations) for sample in X])
    else:
        output = np.array([does_diverge(complex(*sample), c, iterations) for sample in X])
    return X, output.reshape((size * 2, size * 3)) #image format