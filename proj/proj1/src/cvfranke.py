""" 
sketch using example code mainly
vim marks: 
'i for imports
'f for functions
'c for calculations
'p for plots
"""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

def function(X):
    return np.cos(1.5 * np.pi * X)

np.random.seed(2018)

noise       =   0.1
N           =   1000#=   int(1e3)
k           =   5

x           =   np.sort(np.random.uniform(0,1,N)).reshape(-1,1)
y           =   function(x).reshape(-1,1) + np.random.randn(len(x)).reshape(-1,1)*noise
y_nonoise   =   function(x)

degrees     =   np.arange(1, 16)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

X_trainz, X_testz, y_trainz, y_testz = train_test_split(x, y, test_size=1./k)
array_size_thingy=len(y_testz)

err     =   []
bi      =   []
vari    =   []

for deg in degrees:
    y_pred  =   np.empty(   (array_size_thingy, k)  )
    j   =   0
    model   =   make_pipeline( PolynomialFeatures(degree=deg), LinearRegression( fit_intercept=False ) )
#
    for train_inds, test_inds in kfold.split(x):
        xtrain      =   x[train_inds]
        ytrain      =   y[train_inds]

        xtest       =   x[test_inds]
        ytest       =   y[test_inds]

        y_pred[:,j] =   model.fit(xtrain,ytrain).predict(xtest).ravel()
        j+=1

    error   =   np.mean( np.mean((ytest - y_pred)**2, axis=1, keepdims=True) )
    bias    =   np.mean( (ytest - np.mean(y_pred, axis=1, keepdims=True))**2 )
    variance=   np.mean(    np.var(y_pred, axis=1, keepdims=True)   )
    err.append(error)
    bi.append(bias)
    vari.append(variance)

max_pd  =   12  #max polynomial degree to plot
plt.figure()
plt.plot(degrees[:max_pd], err[:max_pd], 'k', label='MSE')
plt.plot(degrees[:max_pd], bi[:max_pd], 'b', label='Bias^2')
plt.plot(degrees[:max_pd], vari[:max_pd], 'y', label='Var')

summ=np.zeros(len(vari))

for i in range(len(err)):
    summ[i]=vari[i]+bi[i]
plt.plot(degrees[:max_pd],summ[:max_pd], 'ro', label='sum')

plt.xlabel('Polynomial degree')
plt.ylabel('MSE CV')
plt.legend()
plt.show()
#       ///////////////////////////////
