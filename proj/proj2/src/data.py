# importing data, as specified

class Data:
    def __init__(self):
        """
        self.X, 
        self.Y, 
        self.M, 
        self.P, 
        self.layers, 
        self.normcoeff
        """

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
        self.Xtrain, self.Xtest, ytrain, ytest = train_test_split(
                self.X, self.Y, train_size=trainshare)

        #scaling continuous variables fitted to training data
        sc = StandardScaler()
        self.Xtrain[:,-12:] = sc.fit_transform(self.Xtrain[:,-12:])
        self.Xtest[:,-12:] = sc.transform(self.Xtest[:,-12:])

        print("\ntraining data, X shape:", self.Xtrain.shape, "\nXtrain[0:3, -6:]:\n", self.Xtrain[0:3, -6:])
        
    def MSE(x, xpred):
        self.MSE = 1/len(xpred) * np.linalg.norm( x - xpred )**2
        return self.MSE

#def scaler():
#    
#    for i in self.Xtrain[:,-12:]:
#        mean = np.std(X[

#//////////////////////////
if __name__ == "__main__":

    dat = Data()
    dat.source('../rsrc/defaultofcreditcardclients.xls')
    
