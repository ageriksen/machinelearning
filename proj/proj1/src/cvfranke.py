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

def traintest(X, k=2, shuffle=True):
    #foldsze = len(X)//k 
    if len(X.shape) > 1:
        X = np.ravel(X)
    indices = np.arange(len(X))
    if shuffle:
        #print('indices pre shuffle:\n', indices)
        np.random.shuffle(indices)
        #print('indices post shuffle:\n', indices)
    folds = np.split(indices, k)
    return folds

np.random.seed(2018)

noise       =   0.1
N           =   1000#=   int(1e3)
k           =   5

X           =   np.sort(np.random.uniform(0,1,N)).reshape(-1,1)
y           =   function(X).reshape(-1,1) + np.random.randn(len(X)).reshape(-1,1)*noise
y_nonoise   =   function(X)

degrees     =   np.arange(1, 16)

kfold       =   KFold(  n_splits=k, shuffle=True, random_state=5  )

X_trainz, X_testz, y_trainz, y_testz = train_test_split(X, y, test_size=1./k)
arrsze=len(y_testz)

err     =   []
bi      =   []
vari    =   []

folds = traintest(X,k)

for deg in degrees:
    y_pred  =   np.empty(   (arrsze, k)  )
    j   =   0
    model   =   make_pipeline( PolynomialFeatures(degree=deg), LinearRegression( fit_intercept=False ) )

#    for train_inds, test_inds in kfold.split(X):
    for fold in range(len(folds)):
        first = True
        for i in range(len(folds)):
            if i != fold:
                if first:
                    traininds = folds[i]
                    first = False
                else:
                    concd = np.concatenate((traininds, folds[i]))
                    traininds = concd
        testinds = folds[fold]

        Xtrain      =   X[train_inds]
        ytrain      =   y[train_inds]

        Xtest       =   X[test_inds]
        ytest       =   y[test_inds]

##        print('lenghts:\nXtrain: ', len(Xtrain), 
##                '\nytrain: ', len(ytrain),
##                '\nXtest: ', len(Xtest) ,
##                '\ny_pred[:,',j,']: ', len(y_pred[:,j]))
##        input("press enter to proceed")

        y_pred[:,j] =   model.fit(Xtrain,ytrain).predict(Xtest).ravel()
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
