err     =   []
bi      =   []
vari    =   []

for lmbd in range(len(_lambda)):
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
            
            #beta            =   np.linalg.inv(XTX + _lambda(lmbd)*np.identity(len(XTX))) @ Xtrain.T @ ztrain
            beta            =   SVDinv(XTX + _lambda(lmbd)*np.identity(len(XTX))) @ Xtrain.T @ ztrain
            ztest_pred[:,j] =   Xtest @ beta
            ztests[:,j]     =   ztest

            j+=1

        error_test      =   np.mean(    np.mean((ztests - ztest_pred)**2, axis=1, keepdims=True) )
        bias_test       =   np.mean(    (ztests - np.mean(ztest_pred, axis=1, keepdims=True))**2 )
        variance_test   =   np.mean(    np.var(ztest_pred, axis=1, keepdims=True)               )
        err.append(error_test)
        bi.append(bias_test)
        vari.append(variance_test)

