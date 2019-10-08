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
zarr            =       zmat.ravel()#.reshape(-1, 1)
#   ///////     /////// 

k           =   5
degrees     =   np.arange(1, 10)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

#////////////
zarr_trainz, zarr_testz = train_test_split(zarr, test_size=1./k)
testarrsze=len(zarr_testz)
#///////////

err     =   []
bi      =   []
vari    =   []

zshape = zarr.reshape(-1,1)
for deg in degrees:
    ztest_pred  =   np.empty(   (testarrsze, k)  )
    j   =   0
    model   =   make_pipeline( PolynomialFeatures(degree=deg), LinearRegression( fit_intercept=False ) )

    for traininds, testinds in kfold.split(rowarr):

        coltrain    =   colarr[traininds]
        rowtrain    =   rowarr[traininds]
        ztrain      =   zarr[traininds]

        coltest     =   colarr[testinds]
        rowtest     =   rowarr[testinds]
        ztest       =   zshape[testinds]
        
        #print( '\n===================\ncoltrain.shape: ', coltrain.shape, '\nrowtrain.shape: ', rowtrain.shape, '\nztrain.shape: ', ztrain.shape, '\n=====================\n')
        
        model.fit(np.column_stack((coltrain, rowtrain)), ztrain)
        matpred =   model.predict(np.column_stack((coltest, rowtest)))
        ztest_pred[:,j] =   matpred.ravel()
        j+=1

    error_test      =   np.mean(    np.mean((ztest - ztest_pred)**2, axis=1, keepdims=True) )
    bias_test       =   np.mean(    (ztest - np.mean(ztest_pred, axis=1, keepdims=True))**2 )
    variance_test   =   np.mean(    np.var(ztest_pred, axis=1, keepdims=True)               )
    err.append(error_test)
    bi.append(bias_test)
    vari.append(variance_test)
    

plt.figure()
plt.plot(degrees, err, label='training set')
plt.xlabel('complexity')
plt.ylabel('MSE')
plt.title('OLS resampling k-fold CV')
plt.legend()


#fig = plt.figure()
#
#ax      =   fig.add_subplot(1, 2, 1, projection='3d')
#surf    =   ax.plot_surface(
#            rowmat, colmat, zmat, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Measurement')
#
#matpred.reshape(rowmat.shape)
#ax      =   fig.add_subplot(1, 2, 2, projection='3d')
#surf    =   ax.plot_surface(
#            rowmat, colmat, matpred, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('OLS fit')
plt.show()
