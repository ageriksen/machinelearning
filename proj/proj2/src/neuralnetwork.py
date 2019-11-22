
class neural:
    
    def __init__(self):
        self

    def train(self, dat, epochs, hidden, batchsze, seed):
        np.random.seed(seed)
        self.layers = hidden
        epochs = int(epochs)
        
        self.dat = dat

        layers = np.concatenate(
            [ [self.dat.P], np.array(hidden, dtype=np.int32), [self.dat.Q] ]
                )

        rndlen = ( self.dat.N//batchsze )*batchsze
        batches = self.dat.N//batchsze
        
        X = self.dat.Xtrain[:rndlen, :, np.newaxis]
        Y = self.dat.Ytrain[:rndlen, :, np.newaxis]
        
        Xbatch = np.array(np.split(X, batches))
        Ybatch = np.array(np.split(Y, batches))

        self.weights = []
        self.bias = []
        for i in range(len(self.layers)-1):
            self.weights.append(
                np.random.normal(0, 0.5, (1, self.layers[i+1], layers[i]) )
                )
            self.bias.append(
                np.random.normal(0, 0.5, (1, self.layers[i+1], 1) )
                )
        
        endcount = epochs*len(Xbatch) 
        timers = np.zeros(endcount)
        count = 0
        t0 = time()
        dt = 0
        print(f"\t{0:>3d}%", end = "")
        
        for e in range(epochs):
            for n in range(len(Xbatch)):
                X = Xbatch[n]
                Y = Ybatch[n]
                Z = []
                Z.append(X)
                for i in self.layers:
                    Z.append(np.zeros( (batchsze, i, 1) ))

                for m in range(len(self.weights)):
                    Z[m+1] = self.weights[m] @ Z[m] + self.bias[m]
                    Z[m+1] = self.activation(Z[m+1])
    
                delta = 2.*( Z[-1] - Y ) 
                delta *= self.dat.activation(Z[-1])



if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import sys
    import os
    from time import time
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    sys.path.append("..")

    from data import Data

    dat = Data(2019)
    dat.source('../rsrc/defaultofcreditcardclients.xls')
#    p = dat.Features
    epochs = 2
    hidden = []
    batchsze = 2
    nn = neural()
    nn.train(dat, epochs, hidden, batchsze, 2019)

