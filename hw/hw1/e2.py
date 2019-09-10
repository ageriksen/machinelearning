# version 0.1 execrcise 2 homework 1
""" 
generate dataset for func. y(x), x \in [0, 1] defined by random uniform numbers.
y is a quadratic polynomial, w/ stochastic noise from normal distribution N(0,1). 
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(100, 1)
y = 5*x*x + 0.1*np.random.randn(100, 1)

""" 
1.
https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/Regression-bs.html
code generated using information found here. 

compute parameterization of data set fitting 2.order polynomial.
"""
X = np.c_[np.ones(len(x)), x, x**2]

beta = np.linalg.inv( X.T @ X ) @ X.T @ y

y_pred = X @ beta


""" 
MSE( y_pred, y) 

plt.scatter(x, y, label='true')
plt.scatter(x, y_pred, label='predicted')
plt.legend()
plt.show()
"""
MSE = 1/len(y_pred) * np.linalg.norm(y - y_pred)**2
print("MSE is: ", MSE)

#plot
plt.scatter(x, y, label='true')
plt.scatter(x, y_pred, label='predicted')
plt.legend()
plt.show()
