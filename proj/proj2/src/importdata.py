# importing data, as specified

class dataimport:
    def __init__(self, dataname):
        import pandas as pd
        import os
        import numpy as np

        self.cwd = os.getcwd() #process dir
        self.filename = self.cwd + '/' + dataname
        nanDict = {}
        self.df = pd.read_excel(
            filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
        self.df.rename(
            index=str, 
            columns={"default payment next month": "defaultPaymentNextMonth"},
            inplace=True)
        #features and targets
        self.X = self.df.loc[ :, self.df.columns != 'defaultPaymentNextMonth'].values
        self.y = self.df.loc[ :, self.df.columns == 'defaultPaymentNextMonth'].values



    def printhello(self):
        print("Hello world")



#//////////////////////////
if __name__ == "__main__":
    dat = dataimport()
    dat.printhello()
