import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.linalg as scl
from matplotlib import cm

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

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

def ols_svd(X: np.ndarray, z: np.ndarray) -> np.ndarray:
    u, s, v = scl.svd(X)
    return v.T @ scl.pinv(scl.diagsvd(s, u.shape[0], v.shape[0])) @ u.T @ z

def confidence(beta, X, confidence=1.96):
    weight = np.sqrt( np.diag( np.linalg.inv( X.T @ X ) ) )*confidence
    betamin = beta - weight
    betamax = beta + weight
    return betamin, betamax

def Rsquared(y, y_pred):
    return 1 - ( np.sum( (y - y_pred)**2 )/np.sum( (y - np.mean(y))**2 ) )
#   /////// !Functions   /////// 

#   ///////   make random data   ///////   
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
zarr            =       zmat.ravel().reshape(-1,1)
#   ///////     /////// 

#   ///////   design matrix   ///////   
n               =       5
X               =       Designmatrix(rowarr, colarr, n)
#print('\nX.shape: ', X.shape, '\nX:\n', X)
Xarr = X.ravel().reshape(-1,1)
#print('\nXarr.shape: ', Xarr.shape, '\nX:\n', Xarr)
#   ///////     /////// 

#   ///////    Linear regression   ///////   
#beta            =       np.linalg.inv( X.T @ X ) @ X.T @ zarr
#zarr_pred       =       X @ beta
#zmat_pred       =       zarr_pred.reshape(nrow, ncol)
#   ///////     /////// 

#   ///////   Error   ///////   
#CImin, CImax    =       confidence(beta, X)
#MSE             =       1/len(zarr_pred) * np.linalg.norm( zarr - zarr_pred )**2
#RR              =       Rsquared(zarr, zarr_pred)

#   // kfold //
#k           =   16
k           =   5
#degrees     =   np.arange(1, 27)
degrees     =   np.arange(1, 16)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

#////////////
X_trainz, X_testz, zarr_trainz, zarr_testz = train_test_split(X, zarr, test_size=1./k)
testarrsze=len(zarr_testz)
Xarr_testz = X_testz.reshape(-1,1)
zarr_testz_pred  =   np.empty(   (len(Xarr_testz), len(degrees))  )
#///////////

testerr     =   []
testbi      =   []
testvari    =   []

err         =   []

for deg in degrees:
    ztest_pred  =   np.empty(   (testarrsze, k)  )
    j   =   0
    model   =   make_pipeline( PolynomialFeatures(degree=deg), LinearRegression( fit_intercept=False ) )

    for traininds, testinds in kfold.split(X):

        Xtrain      =   Xarr[traininds]
        ztrain      =   zarr[traininds]

        Xtest       =   Xarr[testinds]
        ztest       =   zarr[testinds]
        
        ztest_pred[:,j] =   model.fit(Xtrain,ztrain).predict(Xtest).ravel()
        #ztrain_pred[:,j] =   model.fit(Xtrain,ztrain).predict(Xtrain).ravel()
        j+=1
    error_test      =   np.mean(    np.mean((ztest - ztest_pred)**2, axis=1, keepdims=True) )
    bias_test       =   np.mean(    (ztest - np.mean(ztest_pred, axis=1, keepdims=True))**2 )
    variance_test   =   np.mean(    np.var(ztest_pred, axis=1, keepdims=True)               )
    testerr.append(error_test)
    testbi.append(bias_test)
    testvari.append(variance_test)
    
    zarr_testz_pred[:,deg-1] = model.predict(Xarr_testz).ravel()
    #err.append(np.mean(    np.mean((zarr_testz - zarr_testz_pred)**2, axis=1, keepdims=True) ))



plt.figure()
plt.plot(degrees, testerr, label='training set')
#plt.plot(degrees, err, label='test set')
plt.xlabel('complexity')
plt.ylabel('MSE')
plt.title('OLS resampling k-fold CV')
plt.legend()


#fig = plt.figure()

#ax      =   fig.add_subplot(1, 2, 1, projection='3d')
#surf    =   ax.plot_surface(
#            rowmat, colmat, zmat, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Measurement')

#ax      =   fig.add_subplot(1, 2, 2, projection='3d')
#surf    =   ax.plot_surface(
#            rowmat, colmat, , cmap=cm.coolwarm, linewidth=0, antialiased=False   )
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('OLS fit')
plt.show()
