# importing data, as specified

class Data:
    def __init__(self, seed):
        """
        self.X, 
        self.Y, 
        self.M, 
        self.P, 
        self.layers, 
        self.normcoeff
        """

        self.std = [] #lists of feature scalers
        self.mean = []

        self.seed = seed

    def source(self,dataname):
        import numpy as np
        import os
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.model_selection import train_test_split

        cwd = os.getcwd() #process dir
        filename = cwd + '/' + dataname
        nanDict = {}
        df = pd.read_excel(
            filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
        df.rename(
            index=str, 
            columns={"default payment next month": "defaultPaymentNextMonth"},
            inplace=True)
        
        #print("\ndatafile:\n", df)

        self.X = df.loc[ :, df.columns != 'defaultPaymentNextMonth'].values
        self.Y = df.loc[ :, df.columns == 'defaultPaymentNextMonth'].values

        #print("\nInitial design matrix X:\n", self.X)
        onehotencoder = OneHotEncoder(categories="auto")
        
        onehotinds = [1, 2, 3, 5, 6, 7, 8, 9, 10]
        self.X = ColumnTransformer(
            [("", onehotencoder, onehotinds),],
            remainder="passthrough"
        ).fit_transform(self.X).todense()

        self.Y.shape
    
        trainshare=0.5
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest = train_test_split(
                self.X, self.Y, train_size=trainshare, random_state=self.seed)

        #scaling continuous variables fitted to training data
        
        #sc = StandardScaler()
        #self.Xtrain[:,-12:] = sc.fit_transform(self.Xtrain[:,-12:])
        #self.Xtest[:,-12:] = sc.transform(self.Xtest[:,-12:])
        for i in range(12, 0, -1):
            self.std.append(np.std(self.Xtrain[:,-i]))
            self.mean.append(np.mean(self.Xtrain[:,-i]))
            self.Xtrain[:,-i] = (self.Xtrain[:,-i] - self.mean[-1])/self.std[-1] 
            self.Xtest[:,-i] = (self.Xtest[:,-i] - self.mean[-1])/self.std[-1] 

        #print("\ntraining data, X shape:", self.Xtrain.shape, "\nXtrain[0:3, -6:]:\n", self.Xtrain[0:3, -6:])
        self.N, self.M = self.X.shape[0], self.Y.shape[0]
        self.P, self.Q = self.X.shape[1], self.Y.shape[1]
        
    def activation(self, Z):
        """
        sigmoid function
        """
        from numpy import exp
        try:
            return 1/( 1 + exp(-Z) )
        except
        #return Z*( 1 - Z )


#//////////////////////////
if __name__ == "__main__":

    dat = Data(2019)
    dat.source('../rsrc/defaultofcreditcardclients.xls')
    
