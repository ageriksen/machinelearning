"""
Least Absolute Shrinkage and Selection Operator (LASSO) 
regression of simulated terrain data from Franke's function 
with noise.
"""

import seaborn as sns
import numpy as np
import sklearn.linear_model as skl

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.linalg as scl
from matplotlib import cm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#   ///////  Functions   /////// 
def Frankefunction(x, y):
    term1   =   0.75  *np.exp(    -0.25*(9*x - 2)**2  -   0.25*(9*y - 2)**2   )
    term2   =   0.75  *np.exp(    -(9*x + 1)**2/49.   -   0.1*(9*y - 1)       )
    term3   =   0.5   *np.exp(    -0.25*(9*x - 7)**2  -   0.25*(9*y - 3)**2   )
    term4   =  -0.2   *np.exp(    -(9*x - 4)**2       -   (9*y - 7)**2        )
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
def SVDinv(A):
    """
    Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    
    taken from Regressinon slides at https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/Regression.html
    with some modifications
    """
    U, s, VT = np.linalg.svd(A)
    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    return VT.T @ ( np.linalg.inv(D) @ U.T )
#   /////// !Functions   /////// 

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
#   ///////     /////// 

#   ///////   flatten   ///////   
rowarr          =       rowmat.ravel()
colarr          =       colmat.ravel()
zarr            =       zmat.ravel()
#   ///////     /////// 

k           =   5
degrees     =   np.arange(1, 16)
_lambda     =   np.logspace(-1.7, -1)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

#////////////
zarr_trainz, zarr_testz = train_test_split(zarr, test_size=1./k)
arrsze=len(zarr_testz)
#///////////

# sklearn & lecture slides calls this "clf". uncertain what this means



error       =   np.zeros((len(_lambda), len(degrees)))
bias        =   np.zeros((len(_lambda), len(degrees)))
var         =   np.zeros((len(_lambda), len(degrees)))


for lmbd in range(len(_lambda)):
    for deg in degrees:
        zpred   = np.empty( (arrsze, k) )
        z_test  = np.empty( (arrsze, k) )
        X = Designmatrix(rowarr, colarr, deg) 
        j = 0
        for traininds, testinds in kfold.split(X):

            ztrain          =   zarr[traininds]
            ztest           =   zarr[testinds]

            Xtrain          =   X[traininds]
            Xtest           =   X[testinds]
            XTX             =   Xtrain.T @ Xtrain
            

            model = skl.Lasso(alpha=_lambda[lmbd])
            model.fit(Xtrain, ztrain)
            #print('model coefficients:\n', model.coef_)
            zpred[:,j] = model.predict(Xtest)


            #beta            =   SVDinv(XTX + _lambda[lmbd]*np.identity(len(XTX))) @ Xtrain.T @ ztrain
            #zpred[:,j]      =   Xtest @ beta
            z_test[:,j]     =   ztest
            j += 1

        error[  lmbd, deg-1 ]    =   np.mean(    np.mean( (z_test - zpred)**2, axis=1, keepdims=True )   ) # mean of k MSE's 
        bias[   lmbd, deg-1 ]     =   np.mean(    (z_test - np.mean(zpred, axis=1, keepdims=True))**2     )
        var[    lmbd, deg-1 ]      =   np.mean(    np.var(zpred, axis=1, keepdims=True)                    )

print('len lambda: ', len(_lambda), '\nlen degrees: ', len(degrees), '\nerror.shape: ', error.shape)

ax = sns.heatmap(error)
plt.xlabel('complexity')
plt.ylabel(r'$\lambda$')
plt.title(r'MSE for complexity and $\lambda$')
plt.show()
ax = sns.heatmap(bias)
plt.xlabel('complexity')
plt.ylabel(r'$\lambda$')
plt.title(r'bias for complexity and $\lambda$')
plt.show()
ax = sns.heatmap(var)
plt.xlabel('complexity')
plt.ylabel(r'$\lambda$')
plt.title(r'variance for complexity and $\lambda$')
plt.show()




