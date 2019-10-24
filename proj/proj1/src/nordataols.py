import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

import scipy.linalg as scl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

#   ///////  Functions   /////// 

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
#   /////// !Functions   /////// 

# Load the terrain
terrain1 = imread('../resources/SRTM_data_Norway_1.tif')
# Show the terrain
#plt.figure()
#plt.title('Terrain over Norway 1')
#plt.imshow(terrain1, cmap='gray')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()

# make training arrays 
leny, lenx = terrain1.shape
#print('lenx: ', lenx, '\nleny: ', leny)
xarr = np.linspace(0, 1, lenx)
yarr = np.linspace(0, 1, leny)

xmat, ymat = np.meshgrid(xarr, yarr)

#taking a slice of data for computing time
np.random.seed(2039)#this gives an ok approx for MSE, etc. 
N = 100
start = np.random.randint(0, 100, 1)[0]
#print('start at: ', start)
xmat = xmat[ start:start+N, start:start+N]
ymat = ymat[ start:start+N, start:start+N]
zmat = terrain1[ start:start+N, start:start+N]
x = xmat.ravel()
y = ymat.ravel()
z = zmat.ravel()


k           =   5
degrees     =   np.arange(1, 10)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

#////////////
ztrain, ztest = train_test_split(z, test_size=1./k)
testarrsze=len(ztest)
#///////////

err     =   []
bi      =   []
vari    =   []

for deg in degrees:
    zpred  =   np.empty(   (testarrsze, k)  )
    ztests  =   np.empty(   (testarrsze, k)  )

    j   =   0
    X = Designmatrix(x, y, deg) 
    for traininds, testinds in kfold.split(X):

        ztrain      =   z[traininds]
        ztest       =   z[testinds]

        Xtrain  =   X[traininds]
        Xtest   =   X[testinds]
        
        beta = np.linalg.inv(Xtrain.T @ Xtrain) @ Xtrain.T @ ztrain
        zpred[:,j] =   Xtest @ beta
        ztests[:,j] =   ztest
        j+=1

    error_test      =   np.mean(    np.mean((ztests - zpred)**2, axis=1, keepdims=True) )
    bias_test       =   np.mean(    (ztests - np.mean(zpred, axis=1, keepdims=True))**2 )
    variance_test   =   np.mean(    np.var(zpred, axis=1, keepdims=True)               )
    err.append(error_test)
    bi.append(bias_test)
    vari.append(variance_test)
    

plt.figure()
plt.plot(degrees, err, label='Error')
plt.plot(degrees, bi, label='bias')
plt.plot(degrees, vari, label='variance')
#plt.plot(degrees, bi+vari, 'o', label='bias + variance')
plt.xlabel('complexity')
plt.ylabel('MSE')
plt.title('OLS resampling k-fold CV')
plt.legend()
plt.show()

# Seems like the best complexity lies in 2. thus, 

X       =   Designmatrix(x, y, 5) 
beta    =   np.linalg.inv(X.T @ X) @ X.T @ z
zpred   =   X @ beta
zpred   =   np.reshape(zpred, xmat.shape)
print('xmat.shape: ', xmat.shape)
print('zpred.shape: ', zpred.shape)

fig = plt.figure()

ax      =   fig.add_subplot(1, 2, 1, projection='3d')
surf    =   ax.plot_surface(
            xmat, ymat, zmat, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('Measurement')

ax      =   fig.add_subplot(1, 2, 2, projection='3d')
surf    =   ax.plot_surface(
            xmat, ymat, zpred, cmap=cm.coolwarm, linewidth=0, antialiased=False   )
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.title('OLS fit')
plt.show()
