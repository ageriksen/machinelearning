""" 
ols of franke function using sklearn
"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.linalg as scl
from matplotlib import cm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression


def Frankefunction(x, y):
    term1   =   0.75  *np.exp(    -0.25*(9*x - 2)**2  -   0.25*(9*y - 2)**2   )
    term2   =   0.75  *np.exp(    -(9*x + 1)**2/49.   -   0.1*(9*y - 1)       )
    term3   =   0.5   *np.exp(    -0.25*(9*x - 7)**2  -   0.25*(9*y - 3)**2   )
    term4   =  -0.2   *np.exp(    -(9*x - 4)**2       -   (9*y - 7)**2        )
    return term1 + term2 + term3 + term4


np.random.seed(2018)

nrow = 100
ncol = 200
rand_row        =       np.random.uniform(0, 1, size=nrow)
rand_col        =       np.random.uniform(0, 1, size=ncol)

sortrowindex    =       np.argsort(rand_row)
sortcolindex    =       np.argsort(rand_col)

rowsort         =       rand_row[sortrowindex]
colsort         =       rand_col[sortcolindex]

colmat, rowmat  =       np.meshgrid(colsort, rowsort)

noiseSTR        =       0.1
noise           =       np.random.randn(nrow, ncol)

zmat_nonoise    =       Frankefunction(rowmat, colmat)
zmat            =       zmat_nonoise + noiseSTR*noise

rowarr          =       rowmat.ravel()
colarr          =       colmat.ravel()
zarr            =       zmat.ravel()#.reshape(-1, 1)

deg = 5

model   =   make_pipeline( PolynomialFeatures(degree=deg), LinearRegression( fit_intercept=False ) )

model.fit(np.column_stack((colarr, rowarr)), zarr)
zarr_pred =   model.predict(np.column_stack((colarr, rowarr))).ravel()
zmat_pred   =   zarr_pred.reshape(colmat.shape)


#   ///////   Error   ///////   
#CImin, CImax    =       confidence(beta, X)
#MSE             =       1/len(zarr_pred) * np.linalg.norm( zarr - zarr_pred )**2
#RR              =       Rsquared(zarr, zarr_pred)
#print(  "\nMSE is: ",       MSE, 
#        '\nR^2 is: ',       RR, 
#        '\nCI_min:\n',      CImin, 
#        '\nCI_max:\n',      CImax   )
#   ///////     /////// 

#   ///////   Plot   ///////   
fig = plt.figure()

ax      =   fig.add_subplot(1, 2, 1, projection='3d')
surf    =   ax.plot_surface(
            rowmat, colmat, zmat, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Measurement')

ax      =   fig.add_subplot(1, 2, 2, projection='3d')
surf    =   ax.plot_surface(
            rowmat, colmat, zmat_pred, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('OLS fit')

#plt.figure()
#plt.plot(beta, label=r'$\beta$')
#plt.plot(CImin, label=r'$\beta_{min}$')
#plt.plot(CImax, label=r'$\beta_{max}$')
#plt.legend()

plt.show()
