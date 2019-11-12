
class neural:
    
    def __init__(self):
        self

    def setup(self, layerlst, seed):
        self.layers = layerlst
        self.weights = []
        self.bias = []
        np.random.seed(seed)
        for i in range(len(self.layers)-1):
            self.weights.append(
                np.random.normal(0, 0.5, (1, self.layers[i+1], layers[i]) )
                )
            self.bias.append(
                np.random.normal(0, 0.5, (1, self.layers[i+1], 1) )
                )
    
    def train(self, epochs, batchsze):
        """
        epochs = integer, # of forward feed. back propagation "runs" on the training set
        """
        for e in epochs:
            for n in batches:
                X = dims(batchsze, features)
                Y = dim(batchsze, labels)
                z = []
                z.append(X)
                for i in self.layers:
                    z.append(np.zeros( (batchsze, i, 1) ))
                for m in range(len(self.weights)):
                    z[m+1] = self.weights[m] @ Z[m] + self.bias[m]
                    z[m+1] = self.activation(z[m+1])



    def activation(z):
        """
        softmax function
        """
        return 1/( 1 + np.exp(-x) )


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import sys
    import os
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split

    sys.path.append("..")

    from data import Data

    dat = Data()
    dat.source('../rsrc/defaultofcreditcardclients.xls')
#    p = dat.Features
    p = 95
    q = 2
    layers = [p, q]
    nn = neural()
    nn.setup(layers, 2019)

