# importing data, as specified

class Data:
    def __init__(self):
        """
            necessary imports when calling:
            pandas, 
            os, 
            numpy, 
            sklearn.preprocessing.OneHotEncoder, 
            sklearn.compose.ColumnTransformer,
            sklearn.model_selection.train_test_split

            for finddata:
        """
        self

    def source(self,dataname):
        import numpy as np
        import os
        import pandas as pd
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.compose import ColumnTransformer
        from sklearn.model_selection import train_test_split

        self.cwd = os.getcwd() #process dir
        self.filename = self.cwd + '/' + dataname
        nanDict = {}
        self.df = pd.read_excel(
            self.filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
        self.df.rename(
            index=str, 
            columns={"default payment next month": "defaultPaymentNextMonth"},
            inplace=True)
        
        print("\ndatafile:\n", self.df)

        self.X = self.df.loc[ :, self.df.columns != 'defaultPaymentNextMonth'].values
        self.y = self.df.loc[ :, self.df.columns == 'defaultPaymentNextMonth'].values

        print("\nInitial design matrix X:\n", self.X)
        onehotencoder = OneHotEncoder(categories="auto")
        
        onehotinds = [1, 2, 3, 5, 6, 7, 8, 9, 10]
        self.X = ColumnTransformer(
            [("", onehotencoder, onehotinds),],
            remainder="passthrough"
        ).fit_transform(self.X).todense()

        self.y.shape
    
        trainshare=0.5
        self.Xtrain, self.Xtest, ytrain, ytest = train_test_split(
                self.X, self.y, train_size=trainshare)

        #scaling continuous variables fitted to training data
        self.sc = StandardScaler()
        self.Xtrain[-12:] = self.sc.fit_transform(self.Xtrain[-12:])
        self.Xtest[-12:] = self.sc.transform(self.Xtest[-12:])

        print("\ntraining data, X shape:", self.Xtrain.shape, "\nlast 8 columns, 3 rows:\n", self.Xtrain[0:3, -8:])
        
    def MSE(x, xpred):
        self.MSE = 1/len(xpred) * np.linalg.norm( x - xpred )**2
        return self.MSE

#//////////////////////////
if __name__ == "__main__":

    dat = Data()
    dat.source('../rsrc/defaultofcreditcardclients.xls')
    print("\n==================================================\n")
    print("dat.Xtrain: ", dat.Xtrain)
    print("\n==================================================\n")
    
