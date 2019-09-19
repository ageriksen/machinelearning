""" 
Start code from code in project 1 description on github
"""
import numpy as np
from random import random, seed

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Frankefunction(x, y):
    term1 =   0.75  *np.exp(    -0.25*(9*x - 2)**2  -   0.25*(9*y - 2)**2   )
    term2 =   0.75  *np.exp(    -(9*x + 1)**2/49.   -   0.1*(9*y - 1)       )
    term3 =   0.5   *np.exp(    -0.25*(9*x - 7)**2  -   0.25*(9*y - 3)**2   )
    term4 =  -0.2   *np.exp(    -(9*x - 4)**2       -   (9*y - 7)**2        )
    return term1 + term2 + term3 + term4

def Designmatrix(x, y, n=5):
    """ 
    create a design matrix dependent on the polynomial grade you want, with a base of 3.
    want the collumns of X to be [1, x, y, x^2, xy, y^2, x^3, x^2y, xy^2, y^3]
    and so on. 
    """
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int( (n+1)*(n+2)/2 )    # nr. of elements in beta
    X = np.ones((N,l))

    for i in range(1, n+1):
        q = int( (i)*(i+1)/2 )
        for k in range(i+1):
            X[:, q+k] = x**(i-k) * y**k
    
    return X


# make data
x       =   np.arange(0, 1, 0.05)
y       =   np.arange(0, 1, 0.05)
x, y    =   np.meshgrid(x,y)
print( 'shape(x) = ', x.shape, ', shape(y) = ', y.shape )
z       =   Frankefunction(x, y) 

#flatten
x = x.ravel()
y = y.ravel()
z = z.ravel()
z.reshape(20, 20)

#design matrix
n = 5
X = Designmatrix(x, y, n)


#///////////////////
""" from hw1 1.2:
x = np.random.rand(100, 1)
y = 5*x*x + 0.1*np.random.randn(100, 1)

X = np.c_[np.ones(len(x)), x, x**2]

beta = np.linalg.inv( X.T @ X ) @ X.T @ y

y_pred = X @ beta


MSE = 1/len(y_pred) * np.linalg.norm(y - y_pred)**2
print("MSE is: ", MSE)
"""
#////////////////////////

# Linear regression
beta = np.linalg.inv( X.T @ X ) @ X.T @ z
z_pred = X @ beta

#Error
MSE = 1/len(z_pred) * np.linalg.norm( y - z_pred )**2
print( "MSE is: ", MSE )

fig = plt.figure()


#ax = fig.gca(projection='3d')
ax = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax.plot_surface(
    x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Franke')

ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(
    x, y, z_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf2, shrink=0.5, aspect=5)
plt.title('Fitted Franke')

ax.legend()
plt.show()
