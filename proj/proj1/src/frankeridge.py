import numpy as np
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
    ''' Takes as input a numpy matrix A and returns inv(A) based on singular value decomposition (SVD).
    SVD is numerically more stable than the inversion algorithms provided by
    numpy and scipy.linalg at the cost of being slower.
    
    taken from Regressinon slides at https://compphysics.github.io/MachineLearning/doc/pub/Regression/html/Regression.html
    '''
    U, s, VT = np.linalg.svd(A)
#    print('test U')
#    print( (np.transpose(U) @ U - U @np.transpose(U)))
#    print('test VT')
#    print( (np.transpose(VT) @ VT - VT @np.transpose(VT)))
    print(U)
    print(s)
    print(VT)

    D = np.zeros((len(U),len(VT)))
    for i in range(0,len(VT)):
        D[i,i]=s[i]
    UT = np.transpose(U); V = np.transpose(VT); invD = np.linalg.inv(D)
    #return np.matmul(V,np.matmul(invD,UT))
    return V @ ( invD @ UT )
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
zarr            =       zmat.ravel()
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
_lambda =   0.1

for deg in degrees:
    ztest_pred  =   np.empty(   (testarrsze, k)  )
    ztests  =   np.empty(   (testarrsze, k)  )

    j   =   0
    X = Designmatrix(rowarr, colarr, deg) 
    for traininds, testinds in kfold.split(X):

        ztrain          =   zarr[traininds]
        ztest           =   zarr[testinds]

        Xtrain          =   X[traininds]
        Xtest           =   X[testinds]
        XTX             =   Xtrain.T @ Xtrain
        beta            =   np.linalg.inv(XTX + _lambda*np.identity(len(XTX))) @ Xtrain.T @ ztrain
        ztest_pred[:,j] =   Xtest @ beta
        ztests[:,j]     =   ztest

        j+=1

    error_test      =   np.mean(    np.mean((ztests - ztest_pred)**2, axis=1, keepdims=True) )
    bias_test       =   np.mean(    (ztests - np.mean(ztest_pred, axis=1, keepdims=True))**2 )
    variance_test   =   np.mean(    np.var(ztest_pred, axis=1, keepdims=True)               )
    err.append(error_test)
    bi.append(bias_test)
    vari.append(variance_test)




#plt.figure()
#plt.plot(degrees, err, label='Error')
#plt.plot(degrees, bi, label='bias')
#plt.plot(degrees, vari, label='variance')
##plt.plot(degrees, bi+vari, 'o', label='bias + variance')
#plt.xlabel('complexity')
#plt.ylabel('MSE')
#plt.title('OLS resampling k-fold CV')
#plt.legend()
#plt.show()
