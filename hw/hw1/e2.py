# version 0.1 execrcise 2 homework 1
""" 
generate dataset for func. y(x), x \in [0, 1] defined by random uniform numbers.
y is a quadratic polynomial, w/ stochastic noise from normal distribution N(0,1). 
"""
import numpy as np

x = np.random.rand(100, 1)
y = 5*x^2 + 0.1*np.random.randn(100, 1)

""" 
https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/Regression-bs.html
code generated using information found here. 

"""
