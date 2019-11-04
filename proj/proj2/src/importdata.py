# importing data, as specified

class dataimport:
    def __init__(self, dataname):
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

        self.cwd = os.getcwd() #process dir
        self.filename = self.cwd + '/' + dataname
        nanDict = {}
        self.df = pd.read_excel(
            self.filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
        self.df.rename(
            index=str, 
            columns={"default payment next month": "defaultPaymentNextMonth"},
            inplace=True)

        
    def finddata(self, trainshare=0.5, seed="NaN"):
        #features and targets
        self.X = self.df.loc[ :, self.df.columns != 'defaultPaymentNextMonth'].values
        self.y = self.df.loc[ :, self.df.columns == 'defaultPaymentNextMonth'].values
        
        onehotencoder = OneHotEncoder(categories="auto")
        self.X = ColumnTransformer(
            [("", onehotencoder, [3]),],
            remainder="passthrough"
        ).fit_transform(self.X)
        
        self.y.shape

        if type(seed) != str:
            self.Xtrain, self.Xtest, ytrain, ytest = train_test_split(
                    self.X, self.y, train_size=trainshare, random_state=seed)
        else:
            self.Xtrain, self.Xtest, ytrain, ytest = train_test_split(
                    self.X, self.y, train_size=trainshare)
        
        self.df = self.df.drop(self.df[
            (self.df.BILL_AMT1 == 0) &
            (self.df.BILL_AMT2 == 0) &
            (self.df.BILL_AMT3 == 0) &
            (self.df.BILL_AMT4 == 0) &
            (self.df.BILL_AMT5 == 0) &
            (self.df.BILL_AMT6 == 0)].index
            )
            
        self.df = self.df.drop(self.df[
            (self.df.PAY_AMT1 == 0) &
            (self.df.PAY_AMT2 == 0) &
            (self.df.PAY_AMT3 == 0) &
            (self.df.PAY_AMT4 == 0) &
            (self.df.PAY_AMT5 == 0) &
            (self.df.PAY_AMT6 == 0)].index
            )
            
    def MSE(x, xpred):
        self.MSE = 1/len(xpred) * np.linalg.norm( x - xpred )**2
        return self.MSE


    def printhello(self):
        print("Hello world")



#//////////////////////////
if __name__ == "__main__":
    import pandas as pd
    import os
    import numpy as np
    
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    
    import numpy as np

    dat = dataimport('../rsrc/defaultofcreditcardclients.xls')
    dat.finddata()
