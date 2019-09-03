# version 0.1 execrcise 2 homework 1
""" 
imports taken from liquid drop example on slides from regression. link under 
task 1.
"""
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import os

project_root_dir = "results"
figure_id = "results/figurefiles"
data_id = "datafiles"
if not os.path.exists(project_root_dir):
    os.mkdir(project_root_dir)

""" 
generate dataset for func. y(x), x \in [0, 1] defined by random uniform numbers.
y is a quadratic polynomial, w/ stochastic noise from normal distribution N(0,1). 
"""
import numpy as np

x = np.random.rand(100, 1)
y = 5*x^2 + 0.1*np.random.randn(100, 1)

""" 
1.
https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/Regression-bs.html
code generated using information found here. 

compute parameterization of data set fitting 2.order polynomial.
"""

